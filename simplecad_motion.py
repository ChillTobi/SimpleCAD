"""
SimpleCAD Motion — optional add-on: motion studies of mechanisms in the Assembly window
(kinematics: how parts move, without forces).

SimpleCAD shows the "Motion" button in the Assembly toolbar only while this file sits next to
SimpleCAD.py (or is bundled into the executable). Motion is a mode of the Assembly window: the
toolbar row swaps to the motion tools, the right panel gets a Motion tab (joints, couplings,
drivers, trace points, settings, results) and a time slider sits under the 3-D view.

  * Joints connect two parts and leave them some freedom: revolute (hinge), slider,
    cylindrical, ball, planar and fixed. They are made from picked references like alignments
    (a hole rim or circular edge = point + axis, a flat face = point + normal, a straight edge =
    midpoint + direction, a corner = point). The first pick is on the base part, the second on
    the moving part; the joint is defined in the assembled position, so nothing jumps.
  * Couplings tie two joint coordinates: gear pair, belt / chain, rack and pinion, worm drive
    (ratio from the Gear tool's teeth when the parts were made with it), screw (lead) or a
    ratio of your own.
  * Drivers move a joint: constant speed, oscillation, or a position over time (table / CSV).
  * Enabled alignments of the assembly hold their two parts together during the motion
    (unless a joint connects the same two parts).
  * Solver: each moving part has a position and an orientation; the joint equations are solved
    by Newton iteration with least squares at every time step (closed loops such as four-bar
    linkages and crank-sliders work, redundant hinge equations are fine). The degrees of freedom
    are reported like in the sketcher; dead points, lock-ups and joint limits are reported.
  * Results: playback, trace curves of picked points, plots of position, speed and
    acceleration, interference between parts per frame (red), range of motion; export as GIF /
    MP4 video and CSV; Apply position takes over the shown frame as the assembled placement.
Units: mm, degrees, seconds.
"""
import os, math, time, copy, shutil, tempfile, subprocess, traceback

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QToolBar, QAction, QLabel, QDoubleSpinBox, QSpinBox, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QMessageBox, QPushButton, QComboBox,
    QCheckBox, QSlider, QStackedWidget, QPlainTextEdit, QListWidget, QListWidgetItem, QMenu,
    QFileDialog, QInputDialog, QScrollArea, QFrame, QProgressBar, QSizePolicy, QToolButton,
    QApplication, QStyle, QHeaderView)
from PyQt5.QtCore import Qt, QObject, QThread, QTimer, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QIcon, QColor, QPainter, QPen, QFont, QFontMetricsF, QPolygonF, QImage
from OpenGL import GL as gl

import SimpleCAD as cad

ADDON_TITLE = "Motion"
ADDON_VERSION = "1.0"

# ── data ──────────────────────────────────────────────────────────────────────
JOINT_KINDS = {           # kind: (title, joint coordinates, needs an axis)
    "revolute": ("Revolute (hinge)", ("angle",), True),
    "slider": ("Slider", ("slide",), True),
    "cylindrical": ("Cylindrical", ("angle", "slide"), True),
    "planar": ("Planar", ("angle",), True),
    "ball": ("Ball", (), False),
    "fixed": ("Fixed", (), False),
}
JOINT_HINTS = {
    "revolute": "Turns about an axis, like a hinge or a wheel on its shaft. Pick a hole rim or a "
                "circular edge on the base part, then on the moving part.",
    "slider": "Slides along an axis without turning, like a piston in its cylinder or a carriage "
              "on a rail. Pick a straight edge or a flat face (its normal) for the direction.",
    "cylindrical": "Turns about and slides along an axis, like a shaft in a plain bearing.",
    "planar": "Slides on a plane and turns about its normal, like a part lying on a table. Pick "
              "the flat face on the base part, then the face on the moving part.",
    "ball": "Turns in every direction about a point, like a ball joint. Pick the two centres.",
    "fixed": "Holds the two parts together rigidly.",
}
WORLD = ""                       # joint base "": fixed in space (a bearing / guide without a frame part)
WORLD_TITLE = "World (fixed in space)"
WORLD_COMPONENT = {"name": WORLD, "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0],
                   "pivot": [0.0, 0.0, 0.0], "bodies": [], "visible": True}
FIXED_IN_SPACE = {"revolute": ("Bearing — turns in place",
                               "The part turns about its own axis and stays where it is, like a "
                               "wheel or gear in a bearing. Click the part's hole rim or circular "
                               "edge; on a gear made with the Gear tool any face will do."),
                  "slider": ("Guide — slides in place",
                             "The part slides along a line and does not turn, like a rack in its "
                             "guide. Click a straight edge or a flat face (its normal) of the part; "
                             "on a rack made with the Gear tool any face will do.")}
COUPLING_KINDS = {"gear": "Gear pair", "belt": "Belt / chain", "rack": "Rack and pinion",
                  "worm": "Worm drive", "screw": "Screw (lead)", "ratio": "Ratio"}
DRIVER_KINDS = {"speed": "Constant speed", "oscillate": "Oscillation",
                "table": "Position over time (table)"}
COORD_UNITS = {"angle": "°", "slide": "mm"}
PLAYBACK_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
TRACE_COLORS = ((1.0, 0.85, 0.2), (0.35, 0.9, 1.0), (1.0, 0.45, 0.8), (0.5, 1.0, 0.45),
                (1.0, 0.6, 0.3), (0.75, 0.6, 1.0))
GEAR_MESHING = ("spur", "helical", "herringbone", "bevel", "internal", "planetary")
BELT_KINDS = ("pulley", "sprocket")
INTERFERENCE_SHARE = 1e-3      # overlap volume (share of the smaller part) that counts as a clash


def study_new(name="Motion 1"):
    return {"name": name, "joints": [], "couplings": [], "drivers": [], "traces": [],
            "ground": [], "duration_s": 2.0, "steps": 120, "interference": True,
            "alignments_rigid": True}


def _vec(value, default=(0.0, 0.0, 0.0)):
    try:
        out = [float(x) for x in value][:3]
        return out if len(out) == 3 and all(math.isfinite(x) for x in out) else list(default)
    except (TypeError, ValueError):
        return list(default)


def _num(value, default, low=None, high=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _ref(data):
    data = data if isinstance(data, dict) else {}
    return {"origin": _vec(data.get("origin")), "axis": _vec(data.get("axis"), (0.0, 0.0, 1.0)),
            "perp": _vec(data.get("perp"), (1.0, 0.0, 0.0)), "label": str(data.get("label", ""))}


def _limit(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low, high = _num(value[0], None), _num(value[1], None)
    return None if low is None or high is None or high <= low else [low, high]


def study_settings(data):
    """A validated motion study."""
    data = data if isinstance(data, dict) else {}
    study = study_new(str(data.get("name", "Motion 1")))
    for item in data.get("joints", []):
        if not isinstance(item, dict) or item.get("kind") not in JOINT_KINDS:
            continue
        study["joints"].append({
            "name": str(item.get("name", "Joint")), "kind": item["kind"],
            "enabled": bool(item.get("enabled", True)),
            "base": str(item.get("base", "")), "moving": str(item.get("moving", "")),
            "base_ref": _ref(item.get("base_ref")), "moving_ref": _ref(item.get("moving_ref")),
            "angle_limits": _limit(item.get("angle_limits")),
            "slide_limits": _limit(item.get("slide_limits"))})
    for item in data.get("couplings", []):
        if not isinstance(item, dict) or item.get("kind") not in COUPLING_KINDS:
            continue
        study["couplings"].append({
            "name": str(item.get("name", "Coupling")), "kind": item["kind"],
            "enabled": bool(item.get("enabled", True)),
            "a": str(item.get("a", "")), "a_coord": "slide" if item.get("a_coord") == "slide" else "angle",
            "b": str(item.get("b", "")), "b_coord": "slide" if item.get("b_coord") == "slide" else "angle",
            "ratio": _num(item.get("ratio"), 1.0), "auto": bool(item.get("auto", True)),
            "reverse": bool(item.get("reverse", False)),
            "lead_mm": _num(item.get("lead_mm"), 1.0, 1e-9)})
    for item in data.get("drivers", []):
        if not isinstance(item, dict) or item.get("kind") not in DRIVER_KINDS:
            continue
        table = []
        for row in item.get("table") or []:
            try:
                table.append([float(row[0]), float(row[1])])
            except (TypeError, ValueError, IndexError):
                pass
        study["drivers"].append({
            "name": str(item.get("name", "Driver")), "kind": item["kind"],
            "enabled": bool(item.get("enabled", True)), "joint": str(item.get("joint", "")),
            "coord": "slide" if item.get("coord") == "slide" else "angle",
            "speed": _num(item.get("speed"), 180.0), "amplitude": _num(item.get("amplitude"), 30.0),
            "period": _num(item.get("period"), 1.0, 1e-6), "table": sorted(table)})
    for item in data.get("traces", []):
        if not isinstance(item, dict):
            continue
        study["traces"].append({"name": str(item.get("name", "Trace")),
                                "enabled": bool(item.get("enabled", True)),
                                "component": str(item.get("component", "")),
                                "point": _vec(item.get("point")), "label": str(item.get("label", ""))})
    study["ground"] = [str(name) for name in data.get("ground", []) if isinstance(name, str)]
    study["duration_s"] = _num(data.get("duration_s"), 2.0, 1e-4)
    study["steps"] = int(_num(data.get("steps"), 120, 1, 10000))
    study["interference"] = bool(data.get("interference", True))
    study["alignments_rigid"] = bool(data.get("alignments_rigid", True))
    return study


_CHECKED = []          # motion data objects already validated (edited in place afterwards)


def motion_of(assembly):
    """The motion studies of an assembly ({"studies": [...], "active": index}), validated once
    (loaded or restored data) and created when missing."""
    data = assembly.get("motion")
    if not any(data is checked for checked in _CHECKED):
        raw = data if isinstance(data, dict) else {}
        studies = [study_settings(item) for item in raw.get("studies", []) if isinstance(item, dict)]
        data = {"studies": studies or [study_new()],
                "active": int(_num(raw.get("active"), 0, 0))}
        data["active"] = min(data["active"], len(data["studies"]) - 1)
        assembly["motion"] = data
        _CHECKED.append(data)
        del _CHECKED[:-64]
    return data


def _strip(data):
    """Motion data as plain values (for comparing)."""
    return {key: value for key, value in data.items() if not key.startswith("_")}


def unique_name(names, base):
    number = 1
    while f"{base} {number}" in names:
        number += 1
    return f"{base} {number}"


def rename_component(assembly, old, new):
    motion = assembly.get("motion")
    if not isinstance(motion, dict):
        return
    for study in motion.get("studies", []):
        for joint in study.get("joints", []):
            for key in ("base", "moving"):
                if joint.get(key) == old:
                    joint[key] = new
        for trace in study.get("traces", []):
            if trace.get("component") == old:
                trace["component"] = new
        study["ground"] = [new if name == old else name for name in study.get("ground", [])]


def remove_component(assembly, name):
    motion = assembly.get("motion")
    if not isinstance(motion, dict):
        return
    for study in motion.get("studies", []):
        gone = {j["name"] for j in study.get("joints", []) if name in (j.get("base"), j.get("moving"))}
        study["joints"] = [j for j in study.get("joints", []) if j["name"] not in gone]
        study["couplings"] = [c for c in study.get("couplings", [])
                              if c.get("a") not in gone and c.get("b") not in gone]
        study["drivers"] = [d for d in study.get("drivers", []) if d.get("joint") not in gone]
        study["traces"] = [t for t in study.get("traces", []) if t.get("component") != name]
        study["ground"] = [g for g in study.get("ground", []) if g != name]


# ── geometry helpers ──────────────────────────────────────────────────────────
def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _rodrigues(phi):
    angle = float(np.linalg.norm(phi))
    if angle < 1e-14:
        return np.eye(3) + _skew(phi)
    k = _skew(phi / angle)
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * k @ k


def _orthonormalise(R):
    u, _s, vt = np.linalg.svd(R)
    out = u @ vt
    if np.linalg.det(out) < 0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def _perpendicular(axis):
    axis = np.asarray(axis, float)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = np.cross(axis, helper)
    return perp / np.linalg.norm(perp)


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def component_pose(component):
    """(R, t) of a component: world = R · local + t."""
    R = cad._import_rotation_matrix(component["rotation"])
    pivot = np.asarray(component["pivot"], float)
    t = pivot + np.asarray(component["position"], float) - R @ pivot
    return R, t


def pose_placement(component, R, t):
    """(position, rotation) of a component placed at the pose (R, t)."""
    pivot = np.asarray(component["pivot"], float)
    return ([float(x) for x in t - pivot + R @ pivot],
            [float(x) for x in cad._import_euler_angles(R)])


def reference_of(item):
    """(world point, world axis or None, label) of a picked reference."""
    point, label = cad.assembly_reference_point(item)
    axis = None
    if item["kind"] == "face":
        axis = item.get("normal")
    elif item["kind"] == "edge":
        axis = cad.assembly_reference_axis(item)
        if axis is None and item.get("axis") is not None:
            axis = item["axis"][2]
    if axis is not None:
        axis = np.asarray(axis, float)
        norm = float(np.linalg.norm(axis))
        axis = axis / norm if norm > 1e-12 else None
    return np.asarray(point, float), axis, label


def make_joint(assembly, study, kind, base, moving, base_pick, moving_pick):
    """A joint from two picked references (first on the base part, second on the moving part)
    in the assembled position: (joint, note) or (None, problem). `base` None = the world (fixed
    in space; then both picks are on the moving part)."""
    point_a, axis_a, label_a = base_pick
    point_b, axis_b, label_b = moving_pick
    needs_axis = JOINT_KINDS[kind][2]
    axis = axis_a if axis_a is not None else axis_b
    note = ""
    if needs_axis and axis is None:
        return None, ("this joint needs an axis: pick a hole rim, a circular or straight edge, or a "
                      "flat face (its normal)")
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0])
    if axis_a is not None and axis_b is not None and abs(float(axis_a @ axis_b)) < 0.98:
        note = "the two references are not parallel: the joint uses the base part's axis"
    origin = point_a
    tolerance = 0.05 + 1e-3 * _model_size(assembly)
    offset = point_b - origin
    if kind in ("revolute", "cylindrical") and \
            float(np.linalg.norm(offset - (offset @ axis) * axis)) > tolerance:
        return None, ("the two references are not on one axis, so the moving part would swing "
                      "around the base part's axis. Pick the same hole / shaft on both parts. For "
                      "a part that turns about its own axis (a gear, a wheel) use Joint ▾ → "
                      "Bearing, and connect meshing gears with Coupling ▾ → Gear pair (or Joint ▾ → "
                      "Gears automatically).")
    if kind == "ball" and float(np.linalg.norm(offset)) > tolerance:
        return None, "the two centres do not coincide: pick the ball centre on both parts"
    perp = _perpendicular(axis)

    def local(component, label):
        if component is None:                     # the world: world coordinates
            return {"origin": [float(x) for x in origin], "axis": [float(x) for x in axis],
                    "perp": [float(x) for x in perp], "label": label}
        return {"origin": [float(x) for x in cad.assembly_local_point(component, origin)],
                "axis": [float(x) for x in cad.assembly_local_direction(component, axis)],
                "perp": [float(x) for x in cad.assembly_local_direction(component, perp)],
                "label": label}
    names = {j["name"] for j in study["joints"]}
    title = FIXED_IN_SPACE[kind][0].split(" ")[0] if base is None and kind in FIXED_IN_SPACE else \
        JOINT_KINDS[kind][0].split(" ")[0]
    joint = {"name": unique_name(names, title), "kind": kind,
             "enabled": True, "base": WORLD if base is None else base["name"], "moving": moving["name"],
             "base_ref": local(base, label_a), "moving_ref": local(moving, label_b),
             "angle_limits": None, "slide_limits": None}
    return joint, note


def gear_reference(component):
    """(world origin, world axis, world x axis) of a component's gear, or None."""
    gear = component_gear(component)
    if not gear or gear.get("origin") is None or gear.get("axis") is None:
        return None
    origin = cad.assembly_world_point(component, gear["origin"])
    axis = cad.assembly_world_direction(component, gear["axis"])
    x_axis = cad.assembly_world_direction(component, gear.get("x_axis") or _perpendicular(gear["axis"]))
    return np.asarray(origin, float), axis / np.linalg.norm(axis), x_axis / np.linalg.norm(x_axis)


def fixed_in_space(assembly, study, kind, moving, pick=None):
    """A bearing (revolute) or guide (slider) of one part on the world: from the part's gear data
    when it has some, else from the picked reference. (joint, note) or (None, problem)."""
    reference = gear_reference(moving)
    if reference is not None:
        origin, axis, x_axis = reference
        gear = component_gear(moving)
        if kind == "slider":
            if gear["kind"] != "rack":
                return None, "a guide on a gear: use a Bearing (a gear turns, a rack slides)"
            axis = x_axis                                # a rack slides along its teeth
        label = f"{gear['kind']} gear axis" if kind == "revolute" else "rack direction"
        return make_joint(assembly, study, kind, None, moving, (origin, axis, label), (origin, axis, label))
    if pick is None:
        return None, "pick a hole rim, a circular or straight edge or a flat face of the part"
    return make_joint(assembly, study, kind, None, moving, pick, pick)


def auto_gears(assembly, study):
    """Bearings for the gears (Gear-tool parts), gear / rack couplings for the pairs that mesh and
    a driver: (list of what was done, notes)."""
    gears = [c for c in assembly["components"] if c.get("visible", True) and gear_reference(c)]
    done, notes = [], []
    if len(gears) < 1:
        return done, ["No gears made with the Gear tool in this assembly."]
    names = {c["name"] for c in gears}
    # joints between two gears are never right for meshing gears: they go
    wrong = [j for j in study["joints"] if j["base"] in names and j["moving"] in names]
    for joint in wrong:
        study["joints"].remove(joint)
        study["couplings"] = [c for c in study["couplings"] if joint["name"] not in (c["a"], c["b"])]
        study["drivers"] = [d for d in study["drivers"] if d["joint"] != joint["name"]]
        done.append(f"removed {joint['name']} (it joined two gears directly)")
    study["ground"] = [g for g in study["ground"] if g not in names]
    joint_of = {}
    for component in gears:
        existing = next((j for j in study["joints"] if j["moving"] == component["name"]), None)
        if existing is not None:
            joint_of[component["name"]] = existing
            continue
        kind = "slider" if component_gear(component)["kind"] == "rack" else "revolute"
        joint, note = fixed_in_space(assembly, study, kind, component)
        if joint is None:
            notes.append(f"{component['name']}: {note}")
            continue
        study["joints"].append(joint)
        joint_of[component["name"]] = joint
        done.append(f"{joint['name']} on {component['name']}")
    # meshing pairs: parallel axes at the centre distance m (z1 ± z2) / 2
    for n, first in enumerate(gears):
        for second in gears[n + 1:]:
            ja, jb = joint_of.get(first["name"]), joint_of.get(second["name"])
            if ja is None or jb is None or any({c["a"], c["b"]} == {ja["name"], jb["name"]}
                                               for c in study["couplings"]):
                continue
            ga, gb = component_gear(first), component_gear(second)
            oa, ua, xa = gear_reference(first)
            ob, ub, xb = gear_reference(second)
            if abs(float(ua @ ub)) < 0.98:
                continue
            module = ga.get("module") or gb.get("module")
            if not module:
                continue
            kinds = {ga["kind"], gb["kind"]}
            if "rack" in kinds:
                rack, pinion = (first, second) if ga["kind"] == "rack" else (second, first)
                o_r, _u, x_r = gear_reference(rack)
                o_p, _u2, _x2 = gear_reference(pinion)
                g_p = component_gear(pinion)
                offset = o_p - o_r
                across = offset - (offset @ x_r) * x_r - (offset @ ua) * ua
                if float(np.linalg.norm(across)) > module * (g_p["teeth"] / 2.0 + 3.0):
                    continue
                a_joint, b_joint = joint_of[pinion["name"]], joint_of[rack["name"]]
                kind, coords = "rack", ("angle", "slide")
            else:
                if not (ga.get("teeth") and gb.get("teeth")):
                    continue
                offset = ob - oa
                distance = float(np.linalg.norm(offset - (offset @ ua) * ua))
                internal = "internal" in kinds
                wanted = module * (abs(gb["teeth"] - ga["teeth"]) if internal else ga["teeth"] + gb["teeth"]) / 2.0
                if abs(distance - wanted) > 0.6 * module + 0.01 * wanted:
                    continue
                a_joint, b_joint = ja, jb
                kind, coords = "gear", ("angle", "angle")
            coupling_names = {c["name"] for c in study["couplings"]}
            coupling = {"name": unique_name(coupling_names, "Gear" if kind == "gear" else "Rack"),
                        "kind": kind, "enabled": True, "a": a_joint["name"], "a_coord": coords[0],
                        "b": b_joint["name"], "b_coord": coords[1], "ratio": 1.0, "auto": True,
                        "reverse": False, "lead_mm": 1.0}
            value, why = auto_ratio(assembly, study, coupling)
            if value is not None:
                coupling["ratio"] = value
            study["couplings"].append(coupling)
            done.append(f"{coupling['name']}: {first['name']} ↔ {second['name']} ({why})")
    if not study["drivers"]:
        first = next((joint_of[c["name"]] for c in gears if c["name"] in joint_of
                      and joint_of[c["name"]]["kind"] == "revolute"), None)
        if first is not None:
            names_d = {d["name"] for d in study["drivers"]}
            study["drivers"].append({"name": unique_name(names_d, "Driver"), "kind": "speed",
                                     "enabled": True, "joint": first["name"], "coord": "angle",
                                     "speed": 180.0, "amplitude": 30.0, "period": 1.0, "table": []})
            done.append(f"driver on {first['name']} (180 °/s)")
    return done, notes


def _model_size(assembly):
    boxes = [box for box in map(cad.assembly_bounds, assembly["components"]) if box is not None]
    if not boxes:
        return 100.0
    low = np.min([b[0] for b in boxes], axis=0)
    high = np.max([b[1] for b in boxes], axis=0)
    return max(float(np.max(high - low)), 1e-3)


def component_gear(component):
    """Gear data of a component (the first body made with the Gear tool), or None."""
    for body in component.get("bodies", []):
        if body.get("gear"):
            return body["gear"]
    return None


def auto_ratio(assembly, study, coupling):
    """(ratio in display units — b change per a change —, explanation) from the gears of the two
    joints' moving parts, or (None, why not)."""
    joints = {j["name"]: j for j in study["joints"]}
    ja, jb = joints.get(coupling["a"]), joints.get(coupling["b"])
    if ja is None or jb is None:
        return None, "choose both joints"
    kind = coupling["kind"]
    if kind == "screw":
        return coupling["lead_mm"] / 360.0, f"lead {coupling['lead_mm']:g} mm per turn"
    if kind == "ratio":
        return None, "ratio of your own"
    ca = cad.assembly_component(assembly, ja["moving"])
    cb = cad.assembly_component(assembly, jb["moving"])
    ga = component_gear(ca) if ca else None
    gb = component_gear(cb) if cb else None
    if kind == "rack":
        pinion = ga if ga and ga.get("kind") != "rack" else gb
        if not pinion or not pinion.get("teeth") or not pinion.get("module"):
            return None, "the pinion was not made with the Gear tool: enter the ratio (mm per degree)"
        radius = pinion["module"] * pinion["teeth"] / 2.0
        return math.pi * radius / 180.0, (f"pinion {pinion['teeth']:g} teeth, module "
                                          f"{pinion['module']:g}: {2 * math.pi * radius:.4g} mm per turn")
    if not ga or not gb or not ga.get("teeth") or not gb.get("teeth"):
        return None, "the parts were not made with the Gear tool: enter the ratio"
    za, zb = ga["teeth"], gb["teeth"]
    if kind == "worm":
        worm, wheel = (ga, gb) if ga["kind"] == "worm" else (gb, ga) if gb["kind"] == "worm" else (ga, gb)
        starts = worm.get("starts") or 1.0
        teeth = wheel.get("teeth") or zb
        ratio = starts / teeth if worm is ga else teeth / starts
        return ratio, f"worm with {starts:g} start{'s' if starts != 1 else ''}, wheel {teeth:g} teeth"
    if kind == "belt":
        return za / zb, f"{za:g} → {zb:g} teeth, same direction"
    internal = "internal" in (ga["kind"], gb["kind"])
    ratio = (za / zb) if internal else -(za / zb)
    return ratio, f"{za:g} → {zb:g} teeth" + (", ring gear (same direction)" if internal else
                                               ", opposite directions")


# ── the solver ────────────────────────────────────────────────────────────────
class MotionError(Exception):
    pass


class MotionCancelled(Exception):
    pass


class Mechanism:
    """The parts joined by a study's joints (plus alignments), solved for every time step."""

    def __init__(self, assembly, study):
        self.assembly, self.study = assembly, study
        components = {c["name"]: c for c in assembly["components"]}
        components[WORLD] = WORLD_COMPONENT
        self.L = _model_size(assembly)
        joints = [j for j in study["joints"] if j.get("enabled", True) and j["base"] in components
                  and j["moving"] in components and j["base"] != j["moving"]]
        self.messages = []
        pairs = {frozenset((j["base"], j["moving"])) for j in joints}
        welds = []
        if study.get("alignments_rigid", True):
            for constraint in assembly.get("constraints", []):
                a, b = constraint.get("target"), constraint.get("moving")
                if not constraint.get("enabled", True) or a not in components or b not in components \
                        or a == b or frozenset((a, b)) in pairs:
                    continue
                welds.append((a, b))
                pairs.add(frozenset((a, b)))
        involved = []
        for a, b in [(j["base"], j["moving"]) for j in joints] + welds:
            for name in (a, b):
                if name not in involved:
                    involved.append(name)
        if not joints:
            raise MotionError("The study has no joints yet: add one (Joint ▾ in the Motion toolbar).")
        ground = [name for name in study.get("ground", []) if name in components]
        if not ground:
            ground = [joints[0]["base"]]
        self.ground = set(ground) | {WORLD}
        for name in ground:
            if name not in involved:
                involved.append(name)
        self.names = involved
        self.components = [components[name] for name in involved]
        self.index = {name: i for i, name in enumerate(involved)}
        self.fixed = np.array([name in self.ground for name in involved])
        self.R0, self.t0 = [], []
        for component in self.components:
            R, t = component_pose(component)
            self.R0.append(R); self.t0.append(t)
        self.R = [r.copy() for r in self.R0]
        self.t = [t.copy() for t in self.t0]
        self.pivot = [np.asarray(c["pivot"], float) for c in self.components]
        self.unknown = {}
        for i in range(len(involved)):
            if not self.fixed[i]:
                self.unknown[i] = len(self.unknown)
        self.n = len(self.unknown)
        if not self.n:
            raise MotionError("Every part of the joints is fixed: nothing can move. Untick a part "
                              "under Fixed parts.")
        # runtime joints (local vectors), welds as fixed joints at the assembled position
        self.joints = []
        for joint in joints:
            a, b = self.index[joint["base"]], self.index[joint["moving"]]
            ra, rb = joint["base_ref"], joint["moving_ref"]
            self.joints.append({"name": joint["name"], "kind": joint["kind"], "a": a, "b": b,
                                "oa": np.asarray(ra["origin"], float), "ob": np.asarray(rb["origin"], float),
                                "ua": self._unit(ra["axis"]), "ub": self._unit(rb["axis"]),
                                "ea": self._unit(ra["perp"]), "eb": self._unit(rb["perp"]),
                                "limits": {"angle": joint.get("angle_limits"),
                                           "slide": joint.get("slide_limits")}})
        for a_name, b_name in welds:
            a, b = self.index[a_name], self.index[b_name]
            origin = self.R[b] @ self.pivot[b] + self.t[b]
            local = lambda i, v, point=False: (self.R[i].T @ (v - self.t[i])) if point else self.R[i].T @ v
            z, x = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
            self.joints.append({"name": f"alignment {a_name}–{b_name}", "kind": "fixed", "a": a, "b": b,
                                "oa": local(a, origin, True), "ob": local(b, origin, True),
                                "ua": local(a, z), "ub": local(b, z), "ea": local(a, x), "eb": local(b, x),
                                "limits": {}, "weld": True})
        self.by_name = {j["name"]: j for j in self.joints}
        # couplings and drivers on joint coordinates
        self.couplings = []
        for coupling in study["couplings"]:
            if not coupling.get("enabled", True):
                continue
            ja, jb = self.by_name.get(coupling["a"]), self.by_name.get(coupling["b"])
            if ja is None or jb is None:
                self.messages.append(f"Coupling '{coupling['name']}' is left out: a joint is missing.")
                continue
            for joint, coord in ((ja, coupling["a_coord"]), (jb, coupling["b_coord"])):
                if coord not in JOINT_KINDS[joint["kind"]][1]:
                    raise MotionError(f"Coupling '{coupling['name']}': joint '{joint['name']}' has no "
                                      f"{'turning' if coord == 'angle' else 'sliding'} motion.")
            ratio = coupling["ratio"]
            if coupling.get("auto", True) and coupling["kind"] != "ratio":
                value, _why = auto_ratio(assembly, study, coupling)
                if value is not None:
                    ratio = value
                    if coupling["kind"] in ("gear", "belt") and coupling["a_coord"] == "angle" == coupling["b_coord"]:
                        # angles are measured about each joint's own axis
                        da = self.R[ja["a"]] @ ja["ua"]
                        db = self.R[jb["a"]] @ jb["ua"]
                        if float(da @ db) < -0.5:
                            ratio = -ratio
            if coupling.get("reverse"):
                ratio = -ratio
            scale = (math.pi / 180.0 if coupling["b_coord"] == "angle" else 1.0) / \
                (math.pi / 180.0 if coupling["a_coord"] == "angle" else 1.0)
            # planetary gears: a gear sitting on a turning carrier meshes relative to it,
            # (ω_b − ω_carrier) = r (ω_a − ω_carrier)
            carrier, on_a, on_b = None, 0, 0
            if coupling["kind"] in ("gear", "belt") and coupling["a_coord"] == "angle" == coupling["b_coord"]:
                for other in self.joints:
                    if other is ja or other is jb or other.get("weld") or \
                            other["kind"] not in ("revolute", "cylindrical") or self.fixed[other["b"]]:
                        continue
                    if other["b"] in (ja["a"], jb["a"]):
                        carrier, on_a, on_b = other, int(ja["a"] == other["b"]), int(jb["a"] == other["b"])
                        break
                if on_a and on_b:                      # both gears on the carrier: plain ratio
                    carrier = None
            self.couplings.append({"name": coupling["name"], "a": ja, "a_coord": coupling["a_coord"],
                                   "b": jb, "b_coord": coupling["b_coord"], "ratio": ratio * scale,
                                   "display_ratio": ratio, "carrier": carrier, "on_a": on_a, "on_b": on_b})
        self.drivers = []
        for driver in study["drivers"]:
            if not driver.get("enabled", True):
                continue
            joint = self.by_name.get(driver["joint"])
            if joint is None:
                self.messages.append(f"Driver '{driver['name']}' is left out: its joint is missing.")
                continue
            if driver["coord"] not in JOINT_KINDS[joint["kind"]][1]:
                raise MotionError(f"Driver '{driver['name']}': joint '{joint['name']}' has no "
                                  f"{'turning' if driver['coord'] == 'angle' else 'sliding'} motion.")
            self.drivers.append(dict(driver, _joint=joint))
        # unwrapped angles: (raw at the last accepted step, continuous value)
        self.state = {}
        for joint in self.joints:
            if "angle" in JOINT_KINDS[joint["kind"]][1]:
                raw = self._angle(joint)[0]
                self.state[joint["name"]] = (raw, raw)
        self.start = {joint["name"]: self.coords(joint) for joint in self.joints}

    @staticmethod
    def _unit(v):
        v = np.asarray(v, float)
        return v / max(float(np.linalg.norm(v)), 1e-300)

    # world quantities -------------------------------------------------------------------
    def _world(self, joint):
        Ra, ta, Rb, tb = self.R[joint["a"]], self.t[joint["a"]], self.R[joint["b"]], self.t[joint["b"]]
        return (Ra @ joint["oa"] + ta, Rb @ joint["ob"] + tb, Ra @ joint["ua"], Rb @ joint["ub"],
                Ra @ joint["ea"], Rb @ joint["eb"])

    def _centre(self, i):
        return self.R[i] @ self.pivot[i] + self.t[i]

    def _angle(self, joint):
        _oa, _ob, ua, _ub, ea, eb = self._world(joint)
        s = float(ua @ np.cross(ea, eb))
        c = float(ea @ eb)
        return math.atan2(s, c), (s, c, ua, ea, eb)

    def coords(self, joint):
        """{"angle": continuous radians, "slide": mm} of a joint."""
        out = {}
        if "angle" in JOINT_KINDS[joint["kind"]][1]:
            raw = self._angle(joint)[0]
            last_raw, last = self.state.get(joint["name"], (raw, raw))
            out["angle"] = last + _wrap(raw - last_raw)
        if "slide" in JOINT_KINDS[joint["kind"]][1]:
            oa, ob, ua, _ub, _ea, _eb = self._world(joint)
            out["slide"] = float((ob - oa) @ ua)
        return out

    def set_poses(self, poses, restart=True):
        """Place parts at {name: (R, t)}; with `restart` the joint coordinates count from there."""
        for name, (R, t) in poses.items():
            i = self.index.get(name)
            if i is not None and not self.fixed[i]:
                self.R[i], self.t[i] = np.array(R, float), np.array(t, float)
        if not restart:
            return
        for joint in self.joints:
            if joint["name"] in self.state:
                raw = self._angle(joint)[0]
                self.state[joint["name"]] = (raw, raw)
        self.start = {joint["name"]: self.coords(joint) for joint in self.joints}

    def poses(self):
        return {self.names[i]: (self.R[i].copy(), self.t[i].copy()) for i in self.unknown}

    def accept(self):
        """The current poses are a solved step: the angles unwrap from here."""
        for joint in self.joints:
            if joint["name"] in self.state:
                raw = self._angle(joint)[0]
                last_raw, last = self.state[joint["name"]]
                self.state[joint["name"]] = (raw, last + _wrap(raw - last_raw))

    # equations ---------------------------------------------------------------------------
    def _jpoint(self, i, X):
        """d(world point X on part i) / d(unknowns of i): (3, 6); rotations scaled by L."""
        return np.hstack([np.eye(3), -_skew(X - self._centre(i)) / self.L])

    def _jdir(self, d):
        return np.hstack([np.zeros((3, 3)), -_skew(d) / self.L])

    def equations(self, time_value=None, drivers=True):
        """(residuals f, Jacobian J) of the joints, couplings and (at `time_value`) drivers."""
        rows, values = [], []
        L = self.L

        def add(value, parts):
            row = np.zeros(6 * self.n)
            for i, grad in parts:
                if i in self.unknown:
                    k = 6 * self.unknown[i]
                    row[k:k + 6] += grad
            rows.append(row)
            values.append(value)

        for joint in self.joints:
            a, b, kind = joint["a"], joint["b"], joint["kind"]
            oa, ob, ua, ub, ea, eb = self._world(joint)
            n2 = np.cross(ua, ea)
            d = ob - oa
            if kind in ("revolute", "ball", "fixed"):
                Ja, Jb = self._jpoint(a, oa), self._jpoint(b, ob)
                for k in range(3):
                    add(d[k] / L, [(b, Jb[k] / L), (a, -Ja[k] / L)])
            if kind in ("slider", "cylindrical"):
                for n in (ea, n2):
                    add(float(d @ n) / L, [(b, n @ self._jpoint(b, ob) / L),
                                           (a, (-n @ self._jpoint(a, oa) + d @ self._jdir(n)) / L)])
            if kind == "planar":
                add(float(d @ ua) / L, [(b, ua @ self._jpoint(b, ob) / L),
                                        (a, (-ua @ self._jpoint(a, oa) + d @ self._jdir(ua)) / L)])
            if kind in ("revolute", "slider", "cylindrical", "planar", "fixed"):
                for n in (ea, n2):                    # axes parallel
                    add(float(ub @ n), [(b, n @ self._jdir(ub)), (a, ub @ self._jdir(n))])
            if kind in ("slider", "fixed"):          # no turning about the axis
                add(float(eb @ n2), [(b, n2 @ self._jdir(eb)), (a, eb @ self._jdir(n2))])

        def coordinate(joint, coord):
            """(value, [(part, gradient)], scale) of a joint coordinate."""
            a, b = joint["a"], joint["b"]
            if coord == "slide":
                oa, ob, ua, _ub, _ea, _eb = self._world(joint)
                d = ob - oa
                return float(d @ ua), [(b, ua @ self._jpoint(b, ob)),
                                       (a, -ua @ self._jpoint(a, oa) + d @ self._jdir(ua))], 1.0 / L
            raw, (s, c, ua, ea, eb) = self._angle(joint)
            last_raw, last = self.state[joint["name"]]
            value = last + _wrap(raw - last_raw)
            r2 = max(s * s + c * c, 1e-300)
            g_ua = c * np.cross(ea, eb) / r2
            g_ea = (c * np.cross(eb, ua) - s * eb) / r2
            g_eb = (c * np.cross(ua, ea) - s * ea) / r2
            return value, [(a, g_ua @ self._jdir(ua) + g_ea @ self._jdir(ea)),
                           (b, g_eb @ self._jdir(eb))], 1.0

        for coupling in self.couplings:
            qa, ga, _sa = coordinate(coupling["a"], coupling["a_coord"])
            qb, gb, sb = coordinate(coupling["b"], coupling["b_coord"])
            qa0 = self.start[coupling["a"]["name"]][coupling["a_coord"]]
            qb0 = self.start[coupling["b"]["name"]][coupling["b_coord"]]
            r = coupling["ratio"]
            if coupling["carrier"] is None:
                add(((qb - qb0) - r * (qa - qa0)) * sb,
                    [(i, g * sb) for i, g in gb] + [(i, -r * g * sb) for i, g in ga])
            else:                                       # angles relative to the carrier
                qc, gc, _sc = coordinate(coupling["carrier"], "angle")
                dc = qc - self.start[coupling["carrier"]["name"]]["angle"]
                kb, ka = coupling["on_b"] - 1.0, coupling["on_a"] - 1.0
                kc = kb - r * ka
                add(((qb - qb0) + kb * dc - r * ((qa - qa0) + ka * dc)) * sb,
                    [(i, g * sb) for i, g in gb] + [(i, -r * g * sb) for i, g in ga] +
                    [(i, kc * g * sb) for i, g in gc])
        if drivers and time_value is not None:
            for driver in self.drivers:
                q, g, scale = coordinate(driver["_joint"], driver["coord"])
                add((q - self.target(driver, time_value)) * scale, [(i, gi * scale) for i, gi in g])
        if not rows:
            return np.zeros(0), np.zeros((0, 6 * self.n))
        return np.asarray(values, float), np.vstack(rows)

    def target(self, driver, time_value):
        """The driven joint coordinate (radians / mm) at a time."""
        unit = math.pi / 180.0 if driver["coord"] == "angle" else 1.0
        start = self.start[driver["_joint"]["name"]][driver["coord"]]
        if driver["kind"] == "speed":
            return start + driver["speed"] * unit * time_value
        if driver["kind"] == "oscillate":
            return start + driver["amplitude"] * unit * math.sin(2.0 * math.pi * time_value / driver["period"])
        if driver["kind"] == "manual":                  # dragged in the view
            return start + driver["value"] * unit
        table = driver.get("table") or []
        if not table:
            return start
        points = np.asarray(table, float)
        return float(np.interp(time_value, points[:, 0], points[:, 1])) * unit

    def apply(self, dx):
        for i, k in self.unknown.items():
            step = dx[6 * k:6 * k + 6]
            centre = self._centre(i) + step[:3]
            turn = _rodrigues(step[3:] / self.L)
            self.R[i] = turn @ self.R[i]
            self.t[i] = centre - self.R[i] @ self.pivot[i]

    def solve(self, time_value, iterations=40, tolerance=1e-10):
        """Newton iteration with least-squares steps; (converged, residual, iterations)."""
        error = 0.0
        for iteration in range(iterations):
            f, J = self.equations(time_value)
            error = float(np.abs(f).max()) if len(f) else 0.0
            if error < tolerance:
                return True, error, iteration
            dx = np.linalg.lstsq(J, -f, rcond=None)[0]
            largest = float(np.abs(dx).max())
            if largest > 0.25 * self.L:              # no wild jumps
                dx *= 0.25 * self.L / largest
            self.apply(dx)
        for i in self.unknown:
            self.R[i] = _orthonormalise(self.R[i])
        f, _J = self.equations(time_value)
        error = float(np.abs(f).max()) if len(f) else 0.0
        return error < 1e-7, error, iterations

    def analyse(self):
        """Degrees of freedom: {"dof", "redundant", "drivers", "free", "text", "ok"}."""
        f, J = self.equations(None, drivers=False)
        unknowns = 6 * self.n
        tol = 1e-8
        rank = 0
        if len(f):
            s = np.linalg.svd(J, compute_uv=False)
            rank = int((s > tol * max(1.0, s[0])).sum())
        dof = unknowns - rank
        redundant = len(f) - rank
        f2, J2 = self.equations(0.0)
        rank2 = rank
        if len(f2):
            s2 = np.linalg.svd(J2, compute_uv=False)
            rank2 = int((s2 > tol * max(1.0, s2[0])).sum())
        effective = rank2 - rank
        free = unknowns - rank2
        parts = [name for name in self.names if name != WORLD]
        fixed = [name for name in parts if name in self.ground]
        lines = [f"{len(parts)} part{'s' if len(parts) != 1 else ''} "
                 f"({len(fixed)} fixed), {len(self.joints)} joint{'s' if len(self.joints) != 1 else ''}"
                 + (f", {len(self.couplings)} coupling{'s' if len(self.couplings) != 1 else ''}"
                    if self.couplings else "") +
                 f": {dof} degree{'s' if dof != 1 else ''} of freedom, {len(self.drivers)} "
                 f"driver{'s' if len(self.drivers) != 1 else ''}"]
        if redundant:
            lines.append(f"{redundant} redundant equation{'s' if redundant != 1 else ''} (usual for "
                         "planar linkages made of hinges)")
        ok = True
        if not self.drivers:
            lines.append("⚠ No driver: add one to a joint (Driver in the Motion toolbar).")
            ok = False
        elif effective < len(self.drivers):
            lines.append("⚠ Over-driven: a driver acts on a joint that cannot move, or drivers fight "
                         "each other.")
            ok = False
        if free > 0 and self.drivers:
            lines.append(f"{free} degree{'s' if free != 1 else ''} of freedom not driven: those parts "
                         "stay as still as possible.")
        if dof == 0:
            lines.append("⚠ The mechanism cannot move (0 degrees of freedom).")
            ok = False
        return {"dof": dof, "redundant": redundant, "drivers": len(self.drivers), "free": free,
                "text": "; ".join(lines[:1]) + ("\n" + "\n".join(lines[1:]) if len(lines) > 1 else ""),
                "ok": ok}

    def trace_points(self):
        out = []
        for trace in self.study["traces"]:
            if trace.get("enabled", True) and trace["component"] in self.index:
                out.append((trace, self.index[trace["component"]], np.asarray(trace["point"], float)))
        return out

    def run(self, progress=None, cancelled=None):
        """Solve every time step; the result dict (poses, joint coordinates, traces, messages)."""
        started = time.perf_counter()
        study = self.study
        steps = int(study["steps"])
        times = np.linspace(0.0, float(study["duration_s"]), steps + 1)
        analysis = self.analyse()
        if not analysis["ok"]:
            raise MotionError(analysis["text"].replace("⚠ ", ""))
        movers = [i for i in range(len(self.components)) if not self.fixed[i]]
        poses_R = np.zeros((len(times), len(movers), 3, 3))
        poses_t = np.zeros((len(times), len(movers), 3))
        coords = {j["name"]: {c: np.full(len(times), np.nan) for c in JOINT_KINDS[j["kind"]][1]}
                  for j in self.joints if not j.get("weld")}
        traces = self.trace_points()
        trace_paths = {t["name"]: np.full((len(times), 3), np.nan) for t, _i, _p in traces}
        messages = list(self.messages)
        limit_hits, dead_point = {}, None
        valid = 0
        previous = None
        sigma0 = None
        for k, time_value in enumerate(times):
            if cancelled is not None and cancelled():
                raise MotionCancelled()
            if progress is not None and k % 5 == 0:
                progress(int(80 * k / max(len(times) - 1, 1)), f"Motion: step {k} of {len(times) - 1}")
            if k >= 2 and previous is not None:      # predictor: repeat the last change
                (R1, t1), (R2, t2) = previous
                for j, i in enumerate(movers):
                    turn = R2[j] @ R1[j].T
                    centre = 2.0 * (R2[j] @ self.pivot[i] + t2[j]) - (R1[j] @ self.pivot[i] + t1[j])
                    self.R[i] = _orthonormalise(turn @ R2[j])
                    self.t[i] = centre - self.R[i] @ self.pivot[i]
            ok, error, _its = self.solve(time_value)
            if not ok:
                if k == 0:
                    raise MotionError("The joints do not fit the assembled position (residual "
                                      f"{error:.2g}). Check the joints and the alignments.")
                messages.append(f"⚠ The mechanism locks up at t = {time_value:.4g} s: it cannot follow "
                                "its driver any further (a dead point or a joint that blocks). The "
                                "motion ends there.")
                break
            if k == 0 and any(float(np.linalg.norm(self.t[i] - self.t0[i])) > 1e-6 * self.L
                              for i in movers):
                messages.append("The parts were moved slightly to fit the joints at the start.")
            self.accept()
            # dead point: the driven system becomes singular
            if analysis["free"] == 0:
                f, J = self.equations(time_value)
                if J.shape[0] >= J.shape[1] and J.shape[1]:
                    sigma = np.linalg.svd(J, compute_uv=False)[J.shape[1] - 1]
                    sigma0 = sigma if sigma0 is None else sigma0
                    if dead_point is None and sigma < 1e-3 * sigma0:
                        dead_point = time_value
            for j, i in enumerate(movers):
                poses_R[k, j] = self.R[i]
                poses_t[k, j] = self.t[i]
            for joint in self.joints:
                if joint.get("weld"):
                    continue
                values = self.coords(joint)
                for coord, value in values.items():
                    shown = math.degrees(value) if coord == "angle" else value
                    coords[joint["name"]][coord][k] = shown
                    limits = joint["limits"].get(coord)
                    if limits and not (limits[0] - 1e-9 <= shown <= limits[1] + 1e-9) and \
                            (joint["name"], coord) not in limit_hits:
                        limit_hits[(joint["name"], coord)] = time_value
            for trace, i, point in traces:
                trace_paths[trace["name"]][k] = self.R[i] @ point + self.t[i]
            previous = ((poses_R[k - 1].copy(), poses_t[k - 1].copy()), (poses_R[k].copy(), poses_t[k].copy())) \
                if k >= 1 else None
            valid = k + 1
        for (name, coord), when in limit_hits.items():
            messages.append(f"⚠ Joint '{name}' passes its {'angle' if coord == 'angle' else 'slide'} "
                            f"limit at t = {when:.4g} s.")
        if dead_point is not None:
            messages.append(f"⚠ Near a dead point at t = {dead_point:.4g} s: the driver can hardly "
                            "move the mechanism there (e.g. a crank-slider driven from the slider).")
        summary = [f"Motion study '{study['name']}': {analysis['text']}",
                   f"{valid - 1} of {steps} time steps over {times[valid - 1]:.4g} s solved in "
                   f"{time.perf_counter() - started:.1f} s"]
        for coupling in self.couplings:
            summary.append(f"Coupling '{coupling['name']}': ratio {coupling['display_ratio']:.6g} "
                           f"({COORD_UNITS[coupling['b_coord']]} per {COORD_UNITS[coupling['a_coord']]})" +
                           (f", relative to the carrier of '{coupling['carrier']['name']}'"
                            if coupling["carrier"] is not None else ""))
        for name, values in coords.items():
            for coord, series in values.items():
                good = series[:valid]
                if len(good) and np.isfinite(good).any():
                    unit = COORD_UNITS[coord]
                    summary.append(f"Joint '{name}' {coord}: {np.nanmin(good):.4g} … {np.nanmax(good):.4g} {unit}"
                                   f" (range {np.nanmax(good) - np.nanmin(good):.4g} {unit})")
        for name, path in trace_paths.items():
            good = path[:valid]
            if len(good) > 1:
                length = float(np.linalg.norm(np.diff(good, axis=0), axis=1).sum())
                size = good.max(0) - good.min(0)
                summary.append(f"Trace '{name}': path {length:.4g} mm long, spans "
                               f"{size[0]:.4g} × {size[1]:.4g} × {size[2]:.4g} mm")
        return {"study": study["name"], "times": times, "valid": valid,
                "movers": [self.names[i] for i in movers], "R": poses_R, "t": poses_t,
                "coords": coords, "traces": trace_paths,
                "trace_colors": {t["name"]: TRACE_COLORS[n % len(TRACE_COLORS)]
                                 for n, (t, _i, _p) in enumerate(traces)},
                "analysis": analysis, "interference": [], "clash_frames": {},
                "summary": summary, "messages": messages,
                "seconds": time.perf_counter() - started}


def run_study(assembly, study, progress=None, cancelled=None):
    """Solve a study and (if switched on) check the parts for interference."""
    mechanism = Mechanism(assembly, study)
    result = mechanism.run(progress, cancelled)
    if study.get("interference", True):
        check_interference(assembly, study, mechanism, result, progress, cancelled)
    return result


# ── interference ──────────────────────────────────────────────────────────────
def _component_manifold(component):
    """manifold3d solid of a component's bodies (own coordinates) and its volume, or None."""
    try:
        import manifold3d as m3
    except Exception:
        return None
    solids = []
    for body in component["bodies"]:
        if body.get("fused") or not body["triangles"]:
            continue
        try:
            mesh = m3.Mesh(vert_properties=np.ascontiguousarray(body["vertices"], np.float32),
                           tri_verts=np.ascontiguousarray(body["triangles"], np.uint32))
            mesh.merge()
            solid = m3.Manifold(mesh)
            if solid.status() == m3.Error.NoError and not solid.is_empty():
                solids.append(solid)
        except Exception:
            continue
    if not solids:
        return None
    solid = solids[0]
    for other in solids[1:]:
        solid = solid + other
    return solid, float(solid.volume())


def check_interference(assembly, study, mechanism, result, progress=None, cancelled=None):
    """Overlapping parts per frame: result["interference"] = [(first frame, a, b, largest
    volume mm³)], result["clash_frames"] = {frame: {part names}}."""
    names = [c["name"] for c in assembly["components"] if c.get("visible", True)]
    components = {c["name"]: c for c in assembly["components"]}
    jointed = {frozenset((j["base"], j["moving"])) for j in study["joints"] if j.get("enabled", True)}
    jointed |= {frozenset((c["target"], c["moving"])) for c in assembly.get("constraints", [])
                if c.get("enabled", True)}
    movers = set(result["movers"])
    pairs = [(a, b) for n, a in enumerate(names) for b in names[n + 1:]
             if (a in movers or b in movers) and frozenset((a, b)) not in jointed]
    if not pairs:
        return
    solids = {}
    missing = []
    for name in {x for pair in pairs for x in pair}:
        solids[name] = _component_manifold(components[name])
        if solids[name] is None:
            missing.append(name)
    if missing:
        result["messages"].append("No interference check for " + ", ".join(f"'{m}'" for m in sorted(missing)) +
                                  " (manifold3d missing or the mesh is not closed).")
    pairs = [p for p in pairs if solids.get(p[0]) and solids.get(p[1])]
    mover_index = {name: j for j, name in enumerate(result["movers"])}
    boxes = {}
    for name in {x for pair in pairs for x in pair}:
        vertices = np.concatenate([b["vertices"] for b in components[name]["bodies"] if len(b["vertices"])])
        boxes[name] = (vertices.min(0), vertices.max(0))

    def pose(name, frame):
        if name in mover_index:
            j = mover_index[name]
            return result["R"][frame, j], result["t"][frame, j]
        return component_pose(components[name])

    def world_box(name, R, t):
        low, high = boxes[name]
        corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1])
                            for z in (low[2], high[2])])
        world = corners @ R.T + t
        return world.min(0), world.max(0)

    found = {}
    frames = range(result["valid"])
    for frame in frames:
        if cancelled is not None and cancelled():
            raise MotionCancelled()
        if progress is not None and frame % 5 == 0:
            progress(80 + int(20 * frame / max(result["valid"], 1)),
                     f"Interference: frame {frame} of {result['valid'] - 1}")
        placed = {}
        for a, b in pairs:
            Ra, ta = pose(a, frame)
            Rb, tb = pose(b, frame)
            la, ha = world_box(a, Ra, ta)
            lb, hb = world_box(b, Rb, tb)
            if np.any(ha < lb) or np.any(hb < la):
                continue
            for name, R, t in ((a, Ra, ta), (b, Rb, tb)):
                if name not in placed:
                    placed[name] = solids[name][0].transform(np.hstack([R, t[:, None]]))
            try:
                volume = float((placed[a] ^ placed[b]).volume())
            except Exception:
                continue
            if volume > INTERFERENCE_SHARE * min(solids[a][1], solids[b][1]):
                result["clash_frames"].setdefault(frame, set()).update((a, b))
                key = (a, b)
                if key not in found:
                    found[key] = [frame, volume]
                else:
                    found[key][1] = max(found[key][1], volume)
    for (a, b), (frame, volume) in found.items():
        result["interference"].append((frame, a, b, volume))
        when = result["times"][frame]
        result["messages"].append(f"⚠ '{a}' and '{b}' overlap from t = {when:.4g} s (up to "
                                  f"{volume:.3g} mm³): shown red during playback.")
    if not found:
        result["summary"].append("Interference: no parts overlap during the motion.")


# ── background run ────────────────────────────────────────────────────────────
class MotionWorker(QThread):
    progressed = pyqtSignal(int, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, assembly, study):
        super().__init__()
        self._assembly, self._study = assembly, study
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            result = run_study(self._assembly, self._study, self.progressed.emit, lambda: self._cancel)
        except MotionCancelled:
            self.failed.emit("")
        except MotionError as ex:
            self.failed.emit(str(ex))
        except Exception as ex:
            cad._trace("Motion error:\n" + traceback.format_exc())
            self.failed.emit(f"{type(ex).__name__}: {ex}")
        else:
            self.succeeded.emit(result)


# ── plot ──────────────────────────────────────────────────────────────────────
_PLOT_COLORS = ("#ff6b6b", "#4fc3f7", "#ffd54f", "#81c784", "#ba68c8", "#ff8a65")


class MotionPlot(QWidget):
    """Curves over time with a cursor at the shown time; a click selects a time."""
    timeClicked = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(170)
        self._times, self._curves, self._unit, self._cursor = None, [], "", None

    def set_data(self, times, curves, unit):
        self._times = None if times is None else np.asarray(times, float)
        self._curves = [(name, np.asarray(values, float), _PLOT_COLORS[i % len(_PLOT_COLORS)])
                        for i, (name, values) in enumerate(curves)]
        self._unit = unit
        self.update()

    def set_cursor(self, value):
        self._cursor = value
        self.update()

    def _frame(self):
        return QRectF(52.0, 20.0, max(10.0, self.width() - 62.0), max(10.0, self.height() - 56.0))

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#171a1f"))
        painter.setPen(QColor("#3b424d"))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if self._times is None or len(self._times) < 2 or not self._curves:
            painter.setPen(QColor("#7d8590"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Compute the motion to see plots")
            return
        frame = self._frame()
        values = np.concatenate([c[1][np.isfinite(c[1])] for c in self._curves] or [np.zeros(1)])
        low, high = (float(values.min()), float(values.max())) if len(values) else (0.0, 1.0)
        pad = max(1e-9, 0.06 * (high - low)) if high > low else max(1.0, abs(high) * 0.1)
        low, high = low - pad, high + pad
        t0, t1 = float(self._times[0]), float(self._times[-1]) or 1.0
        x_of = lambda t: frame.left() + (t - t0) / max(t1 - t0, 1e-12) * frame.width()
        y_of = lambda v: frame.bottom() - (v - low) / max(high - low, 1e-12) * frame.height()
        font = QFont(self.font()); font.setPointSizeF(max(7.0, font.pointSizeF() - 1.0))
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        for i in range(5):
            value = low + (high - low) * i / 4
            y = y_of(value)
            painter.setPen(QPen(QColor("#2c323b"), 1.0))
            painter.drawLine(QPointF(frame.left(), y), QPointF(frame.right(), y))
            painter.setPen(QColor("#aab2bd"))
            shown = 0.0 if abs(value) < 1e-9 * max(abs(high - low), 1e-30) else value
            painter.drawText(QRectF(0, y - 8, frame.left() - 4, 16), Qt.AlignRight | Qt.AlignVCenter,
                             f"{shown:.3g}")
        for fraction in (0.0, 0.5, 1.0):
            t = t0 + (t1 - t0) * fraction
            text = f"{t:.3g} s"
            x = min(max(x_of(t) - metrics.horizontalAdvance(text) / 2, frame.left() - 4),
                    self.width() - metrics.horizontalAdvance(text) - 4)
            painter.drawText(QPointF(x, frame.bottom() + metrics.ascent() + 3), text)
        for name, curve, colour in self._curves:
            painter.setPen(QPen(QColor(colour), 1.8))
            points = [QPointF(x_of(t), y_of(v)) for t, v in zip(self._times, curve) if np.isfinite(v)]
            if len(points) > 1:
                painter.drawPolyline(QPolygonF(points))
        if self._cursor is not None:
            painter.setPen(QPen(QColor("#e2e5e9"), 1.0, Qt.DashLine))
            x = x_of(self._cursor)
            painter.drawLine(QPointF(x, frame.top()), QPointF(x, frame.bottom()))
        x, y = 6.0, self.height() - 6.0
        painter.setPen(QColor("#aab2bd"))
        painter.drawText(QPointF(6.0, 13.0), self._unit)
        for name, _curve, colour in self._curves:
            width = metrics.horizontalAdvance(name) + 22
            if x + width > self.width() - 4:
                break
            painter.fillRect(QRectF(x, y - 7, 10, 3), QColor(colour))
            painter.setPen(QColor("#e2e5e9"))
            painter.drawText(QPointF(x + 13, y - 2), name)
            x += width

    def mousePressEvent(self, event):
        if self._times is not None and len(self._times) > 1 and event.buttons() & Qt.LeftButton:
            frame = self._frame()
            fraction = min(1.0, max(0.0, (event.pos().x() - frame.left()) / frame.width()))
            self.timeClicked.emit(float(self._times[0] + fraction * (self._times[-1] - self._times[0])))

    mouseMoveEvent = mousePressEvent


# ── the Motion mode of the Assembly window ───────────────────────────────────
def attach(dialog):
    """Called by the Assembly window (SimpleCAD.AssemblyDialog) once its UI is built."""
    return MotionController(dialog)


def _hint(text):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color:#aab2bd;font-size:9px;")
    return label


def _spin(low, high, suffix, step=1.0, decimals=3, width=None):
    spin = QDoubleSpinBox()
    spin.setLocale(cad.QLocale.c())
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setSuffix(suffix)
    spin.setKeyboardTracking(False)
    if width:
        spin.setFixedWidth(width)
    return spin


class MotionController(QObject):
    """Motion mode of one Assembly window: toolbar row, Motion tab, playback bar, view overlay."""

    def __init__(self, dialog):
        super().__init__(dialog)
        self.dialog = dialog
        self.view = dialog.view
        self.active = False
        self._loading = False
        self._result = None
        self._result_key = None
        self._frame = None               # frame shown in the view (None = assembled)
        self._recolored = {}             # render entry id → (entry, original colour)
        self._pick = None
        self._worker = None
        self._selection = None           # ("joint" | "coupling" | "driver" | "trace", name)
        self._build()
        self.view.overlay_painters.append(self._paint_overlay)
        self.view.mouse_handlers.append(self)
        self._display = {}               # part name → (R, t) while not at its assembled placement
        self._drag = None
        self._hover = None
        self.view.sigMeasureChanged.connect(self._on_picks)

    # ── building the UI ───────────────────────────────────────────────────────
    def _icon(self, name):
        return QIcon(os.path.join(cad._LOG_DIR, "icons", name))

    def _compute_icon(self):
        """The motion icon with the cyan play badge of Start Simulation (bottom right)."""
        from PyQt5.QtGui import QPixmap, QBrush
        pixmap = QPixmap(os.path.join(cad._LOG_DIR, "icons", "motion_general.png"))
        if pixmap.isNull():
            return self._icon("start_thermal_simulation.png")
        scale = pixmap.width() / 512.0
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        centre = QPointF(400.0 * scale, 402.0 * scale)
        painter.setBrush(QBrush(QColor("#1e2531")))
        painter.drawEllipse(centre, 80.0 * scale, 80.0 * scale)
        painter.setBrush(QBrush(QColor("#1ec8ff")))
        painter.drawEllipse(centre, 64.0 * scale, 64.0 * scale)
        painter.setBrush(QBrush(QColor("#1e2531")))
        x, y = centre.x(), centre.y()
        painter.drawPolygon(QPolygonF([QPointF(x - 20 * scale, y - 32 * scale),
                                       QPointF(x - 20 * scale, y + 32 * scale),
                                       QPointF(x + 34 * scale, y)]))
        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def _toolbar_item(toolbar, action):
        """The toolbar item that shows `action` (its own, or the container of its button)."""
        for item in toolbar.actions():
            if item is action:
                return item
            widget = toolbar.widgetForAction(item)
            if widget is not None and any(button.defaultAction() is action
                                          for button in widget.findChildren(QToolButton)):
                return item
        return None

    def _tidy_separators(self):
        """No doubled separators where a whole group of the toolbar is hidden."""
        toolbar = self.dialog._toolbar
        items = toolbar.actions()
        previous_separator = True                  # (none at the start)
        for index, item in enumerate(items):
            if item.isSeparator():
                later = any(not other.isSeparator() and other.isVisible() for other in items[index + 1:])
                item.setVisible(not previous_separator and later)
                if item.isVisible():
                    previous_separator = True
            elif item.isVisible():
                previous_separator = False

    def _build(self):
        d = self.dialog
        toolbar = d._toolbar
        self.act_motion = QAction(self._icon("motion_general.png"), "Motion", d, checkable=True,
                                  toolTip="Motion — make the assembly move: joints, couplings and "
                                          "drivers, then play the motion (Assemble ⇄ Motion)")
        self.act_motion.toggled.connect(self.set_active)
        d._tools_menu.addSeparator()
        d._tools_menu.addAction(self.act_motion)
        self.joint_menu = QMenu("Joint", d)
        for kind, (title, _coords, _axis) in JOINT_KINDS.items():
            act = self.joint_menu.addAction(title, lambda _c=False, k=kind: self._start_pick("joint", k))
            act.setToolTip(JOINT_HINTS[kind])
        self.joint_menu.addSeparator()
        for kind, (title, hint) in FIXED_IN_SPACE.items():
            act = self.joint_menu.addAction(title, lambda _c=False, k=kind: self._start_pick("space", k))
            act.setToolTip(hint)
        for kind, title, hint in (
                ("revolute", "Bearing about an axis",
                 "The part turns about an axis you choose, fixed in space: first click the axis — a "
                 "world axis (X / Y / Z lines through the origin) or a hole rim, edge or face of any "
                 "part, e.g. the sun gear — then the part, e.g. a planet carrier."),
                ("slider", "Guide along an axis",
                 "The part slides along a line you choose: first click the line (a world axis or an "
                 "edge / face of any part), then the part.")):
            act = self.joint_menu.addAction(title, lambda _c=False, k=kind: self._start_pick("axis", k))
            act.setToolTip(hint)
        self.joint_menu.addSeparator()
        act = self.joint_menu.addAction("Gears automatically", lambda _c=False: self._auto_gears())
        act.setToolTip("A bearing on every gear made with the Gear tool, a gear coupling for each "
                       "pair that meshes (and rack and pinion), and a driver")
        self.joint_menu.setToolTipsVisible(True)
        self.coupling_menu = QMenu("Coupling", d)
        for kind, title in COUPLING_KINDS.items():
            self.coupling_menu.addAction(title, lambda _c=False, k=kind: self._add_coupling(k))
        self.export_menu = QMenu("Export", d)
        self.export_menu.addAction("Video (GIF / MP4)…", lambda _c=False: self._export_video())
        self.export_menu.addAction("Values over time (CSV)…", lambda _c=False: self._export_csv())
        self.act_joint = QAction(self._icon("joint.png"), "Joint", d,
                                 toolTip="Joint — connect two parts: choose the kind, then pick a "
                                         "reference on the base part and one on the moving part")
        self.act_joint.setMenu(self.joint_menu)
        self.act_coupling = QAction(self._icon("add_coupling.png"), "Coupling", d,
                                    toolTip="Coupling — tie two joints: gears, belt / chain, rack, "
                                            "worm, screw or a ratio")
        self.act_coupling.setMenu(self.coupling_menu)
        self.act_driver = QAction(self._icon("driver.png"), "Driver", d,
                                  toolTip="Driver — a motor on the selected joint (constant speed, "
                                          "oscillation or a table)")
        self.act_driver.triggered.connect(lambda _c=False: self._add_driver())
        self.act_trace = QAction(self._icon("trace_point.png"), "Trace point", d,
                                 toolTip="Trace point — pick a point on a part; its path is drawn "
                                         "and plotted")
        self.act_trace.triggered.connect(lambda _c=False: self._start_pick("trace"))
        self.act_compute = QAction(self._compute_icon(), "Compute", d,
                                   toolTip="Compute the motion (F5), then play it with the slider "
                                           "under the view")
        self.act_compute.setShortcut("F5")
        self.act_compute.triggered.connect(lambda _c=False: self._compute())
        self.act_reset = QAction(self._icon("motion_reset.png"), "Reset", d,
                                 toolTip="Reset — show the assembled position again")
        self.act_reset.triggered.connect(lambda _c=False: self.reset_view(True))
        self.act_apply = QAction(self._icon("apply_position.png"), "Apply position", d,
                                 toolTip="Apply position — take over the shown frame (or the "
                                         "dragged position) as the assembled placement of the "
                                         "parts (Undo takes it back)")
        self.act_apply.triggered.connect(lambda _c=False: self._apply_position())
        self.act_export_motion = QAction(self._icon("video.png"), "Export", d,
                                         toolTip="Export — the motion as a GIF / MP4 video, or the "
                                                 "values over time as CSV")
        self.act_export_motion.setMenu(self.export_menu)
        self.act_remove_item = QAction(self._icon("delete_part.png"), "Remove", d,
                                       toolTip="Remove the selected joint, coupling, driver or "
                                               "trace point (Motion tab)")
        self.act_remove_item.triggered.connect(lambda _c=False: self._remove_item())
        self.act_remove_item.setEnabled(False)
        self._motion_actions = (self.act_joint, self.act_coupling, self.act_driver, self.act_trace,
                                self.act_remove_item, self.act_compute, self.act_reset,
                                self.act_apply, self.act_export_motion)
        # Motion mode swaps the toolbar row: the motion tools stand where the assembly tools
        # are (after Undo / Redo); the Motion switch comes last
        before = self._toolbar_item(toolbar, d.act_measure)
        for act in (self.act_joint, self.act_coupling, self.act_driver, self.act_trace,
                    self.act_remove_item, None, self.act_compute, self.act_reset, self.act_apply,
                    self.act_export_motion, None):
            if act is None:
                toolbar.insertSeparator(before) if before is not None else toolbar.addSeparator()
            else:
                act.setVisible(False)
                toolbar.insertAction(before, act) if before is not None else toolbar.addAction(act)
        toolbar.addSeparator()
        toolbar.addAction(self.act_motion)
        cad.enlarge_toolbar_icons(toolbar)                # the same large buttons as the others
        for button in toolbar.findChildren(QToolButton):
            if button.defaultAction() in (self.act_joint, self.act_coupling, self.act_export_motion):
                button.setPopupMode(QToolButton.InstantPopup)
        self._tidy_separators()
        # playback bar under the view
        self.playback = QWidget()
        row = QHBoxLayout(self.playback)
        row.setContentsMargins(4, 0, 4, 2)
        self.btn_play = QPushButton()
        self.btn_play.setIcon(d.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setFixedWidth(34)
        self.btn_play.setToolTip("Play / pause")
        self.btn_play.clicked.connect(self._toggle_play)
        self.sl_time = QSlider(Qt.Horizontal)
        self.sl_time.setRange(0, 0)
        self.sl_time.valueChanged.connect(self._on_slider)
        self.lbl_time = QLabel("t = —")
        self.lbl_time.setMinimumWidth(130)
        self.cb_speed = QComboBox()
        for speed in PLAYBACK_SPEEDS:
            self.cb_speed.addItem(f"{speed:g}×", speed)
        self.cb_speed.setCurrentIndex(PLAYBACK_SPEEDS.index(1.0))
        self.cb_speed.setToolTip("Playback speed")
        self.chk_loop = QCheckBox("Loop")
        self.chk_loop.setChecked(True)
        for widget in (self.btn_play, self.sl_time, self.lbl_time, self.cb_speed, self.chk_loop):
            row.addWidget(widget, 1 if widget is self.sl_time else 0)
        self.playback.hide()
        cad._style_checkbox_like_explorer(self.chk_loop)
        d._panel_view.layout().addWidget(self.playback)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._play_step)
        self._build_tab()

    def _build_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        study_row = QHBoxLayout()
        self.cb_study = QComboBox()
        self.cb_study.setToolTip("Motion studies of this assembly (e.g. \"open\" and \"close\")")
        self.cb_study.currentIndexChanged.connect(self._on_study_chosen)
        study_row.addWidget(QLabel("Study:"))
        study_row.addWidget(self.cb_study, 1)
        for text, tip, slot in (("+", "New motion study", self._new_study),
                                ("Rename", "Rename the motion study", self._rename_study),
                                ("✖", "Delete the motion study", self._delete_study)):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.setFixedWidth(28 if len(text) == 1 else 62)
            button.clicked.connect(lambda _c=False, s=slot: s())
            study_row.addWidget(button)
        layout.addLayout(study_row)
        self.tree = cad.ExplorerTree()
        self.tree.setHeaderHidden(True)
        self.tree.setTextElideMode(Qt.ElideNone)          # whole names, scroll sideways instead
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setStretchLastSection(False)
        self.tree.setFixedHeight(190)
        self.tree.itemSelectionChanged.connect(self._on_tree_selected)
        self.tree.itemChanged.connect(self._on_tree_changed)
        layout.addWidget(self.tree)
        buttons = QHBoxLayout()
        self.btn_remove = QPushButton(self._icon("delete_part.png"), "Remove")
        self.btn_remove.setToolTip("Remove the selected joint, coupling, driver or trace point")
        self.btn_remove.clicked.connect(self._remove_item)
        buttons.addWidget(self.btn_remove)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.pages = QStackedWidget()
        self.pages.addWidget(QWidget())                  # 0: nothing selected
        self.pages.addWidget(self._joint_page())         # 1
        self.pages.addWidget(self._coupling_page())      # 2
        self.pages.addWidget(self._driver_page())        # 3
        self.pages.addWidget(self._trace_page())         # 4
        layout.addWidget(self.pages)
        layout.addWidget(self._settings_box())
        self.lbl_dof = _hint("")
        self.lbl_dof.setStyleSheet("color:#9fd0ff;font-size:9px;")
        layout.addWidget(self.lbl_dof)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()
        layout.addWidget(self.progress)
        layout.addWidget(QLabel("<b>Results</b>"))
        self.txt_messages = QPlainTextEdit()
        self.txt_messages.setReadOnly(True)
        self.txt_messages.setFont(QFont("Consolas", 8))
        self.txt_messages.setMinimumHeight(110)
        layout.addWidget(self.txt_messages)
        plot_row = QHBoxLayout()
        self.cb_series = QComboBox()
        self.cb_series.setToolTip("What the plot shows")
        self.cb_quantity = QComboBox()
        for key, title in (("position", "Position"), ("speed", "Speed"), ("acceleration", "Acceleration")):
            self.cb_quantity.addItem(title, key)
        for combo in (self.cb_series, self.cb_quantity):
            combo.currentIndexChanged.connect(lambda _i: self._update_plot())
        plot_row.addWidget(self.cb_series, 1)
        plot_row.addWidget(self.cb_quantity)
        layout.addLayout(plot_row)
        self.plot = MotionPlot()
        self.plot.timeClicked.connect(self._on_plot_clicked)
        layout.addWidget(self.plot)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(panel)
        self.tab = scroll
        self._style_checks(panel)

    def _style_checks(self, widget):
        """Check boxes, tree and list indicators in the style of the object explorer."""
        for check in widget.findChildren(QCheckBox):
            cad._style_checkbox_like_explorer(check)
        indicator = ("::indicator { width: 15px; height: 15px; border: 2px solid rgb(28, 198, 251);"
                     " border-radius: 3px; background-color: #171a1f; }"
                     "::indicator:checked { background-color: rgb(28, 198, 251); image: none; }"
                     "::indicator:unchecked { image: none; }"
                     "::indicator:unchecked:hover { background-color: #263744; }")
        for view, kind in ((self.tree, "QTreeWidget"), (self.lst_ground, "QListWidget")):
            view.setStyleSheet("".join(f"{kind}{rule}}}" for rule in indicator.split("}") if rule.strip()))

    def _form(self):
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(0, 4, 0, 4)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        return page, form

    def _joint_page(self):
        page, form = self._form()
        self.lbl_joint_name = QLabel()
        form.addRow("<b>Joint</b>", self.lbl_joint_name)
        self.cb_joint_kind = QComboBox()
        for kind, (title, _c, _a) in JOINT_KINDS.items():
            self.cb_joint_kind.addItem(title, kind)
        self.cb_joint_kind.currentIndexChanged.connect(self._joint_edited)
        form.addRow("Kind:", self.cb_joint_kind)
        self.lbl_joint_parts = QLabel()
        self.lbl_joint_parts.setWordWrap(True)
        form.addRow("Parts:", self.lbl_joint_parts)
        self.chk_angle_limits = QCheckBox("Angle limits")
        self.sp_angle_min = _spin(-1e6, 1e6, "°", 5.0, 2, 90)
        self.sp_angle_max = _spin(-1e6, 1e6, "°", 5.0, 2, 90)
        self.chk_slide_limits = QCheckBox("Slide limits")
        self.sp_slide_min = _spin(-1e6, 1e6, " mm", 1.0, 3, 90)
        self.sp_slide_max = _spin(-1e6, 1e6, " mm", 1.0, 3, 90)
        self._angle_row, self._slide_row = QWidget(), QWidget()
        for row_widget, check, low, high in ((self._angle_row, self.chk_angle_limits, self.sp_angle_min,
                                              self.sp_angle_max),
                                             (self._slide_row, self.chk_slide_limits, self.sp_slide_min,
                                              self.sp_slide_max)):
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(check); row.addWidget(low); row.addWidget(QLabel("…")); row.addWidget(high)
            check.toggled.connect(self._joint_edited)
            low.valueChanged.connect(self._joint_edited)
            high.valueChanged.connect(self._joint_edited)
            form.addRow(row_widget)
        self.lbl_joint_hint = _hint("")
        form.addRow(self.lbl_joint_hint)
        return page

    def _coupling_page(self):
        page, form = self._form()
        self.lbl_coupling_name = QLabel()
        form.addRow("<b>Coupling</b>", self.lbl_coupling_name)
        self.cb_coupling_kind = QComboBox()
        for kind, title in COUPLING_KINDS.items():
            self.cb_coupling_kind.addItem(title, kind)
        form.addRow("Kind:", self.cb_coupling_kind)
        self.cb_coupling_a, self.cb_coupling_b = QComboBox(), QComboBox()
        self.cb_coord_a, self.cb_coord_b = QComboBox(), QComboBox()
        for combo in (self.cb_coord_a, self.cb_coord_b):
            combo.addItem("turning", "angle")
            combo.addItem("sliding", "slide")
        for joint_combo, coord_combo, label in ((self.cb_coupling_a, self.cb_coord_a, "Input joint:"),
                                                (self.cb_coupling_b, self.cb_coord_b, "Output joint:")):
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(joint_combo, 1); row.addWidget(coord_combo)
            form.addRow(label, row_widget)
        self.chk_coupling_auto = QCheckBox("Ratio from the gears / lead")
        form.addRow(self.chk_coupling_auto)
        self.sp_ratio = _spin(-1e6, 1e6, "", 0.01, 6, 130)
        self.sp_ratio.setToolTip("Output change per input change (° per °, mm per ° or ° per mm)")
        form.addRow("Ratio:", self.sp_ratio)
        self.sp_lead = _spin(1e-6, 1e6, " mm", 0.1, 4, 130)
        self.sp_lead.setToolTip("Travel per turn (pitch × starts)")
        self._lead_label = QLabel("Lead:")
        form.addRow(self._lead_label, self.sp_lead)
        self.chk_coupling_reverse = QCheckBox("Other direction")
        form.addRow(self.chk_coupling_reverse)
        self.lbl_coupling_info = _hint("")
        form.addRow(self.lbl_coupling_info)
        for widget in (self.cb_coupling_kind, self.cb_coupling_a, self.cb_coupling_b, self.cb_coord_a,
                       self.cb_coord_b):
            widget.currentIndexChanged.connect(self._coupling_edited)
        for widget in (self.chk_coupling_auto, self.chk_coupling_reverse):
            widget.toggled.connect(self._coupling_edited)
        for widget in (self.sp_ratio, self.sp_lead):
            widget.valueChanged.connect(self._coupling_edited)
        return page

    def _driver_page(self):
        page, form = self._form()
        self.lbl_driver_name = QLabel()
        form.addRow("<b>Driver</b>", self.lbl_driver_name)
        self.cb_driver_joint = QComboBox()
        form.addRow("Joint:", self.cb_driver_joint)
        self.cb_driver_coord = QComboBox()
        self.cb_driver_coord.addItem("Turning (angle)", "angle")
        self.cb_driver_coord.addItem("Sliding (position)", "slide")
        form.addRow("Moves:", self.cb_driver_coord)
        self.cb_driver_kind = QComboBox()
        for kind, title in DRIVER_KINDS.items():
            self.cb_driver_kind.addItem(title, kind)
        form.addRow("Motion:", self.cb_driver_kind)
        self.sp_driver_speed = _spin(-1e7, 1e7, " °/s", 10.0, 3, 130)
        self._speed_label = QLabel("Speed:")
        form.addRow(self._speed_label, self.sp_driver_speed)
        self.sp_driver_amplitude = _spin(-1e7, 1e7, "°", 1.0, 3, 130)
        self._amplitude_label = QLabel("Amplitude:")
        form.addRow(self._amplitude_label, self.sp_driver_amplitude)
        self.sp_driver_period = _spin(1e-4, 1e6, " s", 0.1, 4, 130)
        self._period_label = QLabel("Period:")
        form.addRow(self._period_label, self.sp_driver_period)
        self.txt_driver_table = QPlainTextEdit()
        self.txt_driver_table.setFont(QFont("Consolas", 8))
        self.txt_driver_table.setFixedHeight(90)
        self.txt_driver_table.setPlaceholderText("time s, position (° or mm) — one pair per line")
        self._table_label = QLabel("Table:")
        form.addRow(self._table_label, self.txt_driver_table)
        self.btn_driver_csv = QPushButton("Import CSV…")
        self.btn_driver_csv.clicked.connect(self._import_driver_csv)
        form.addRow("", self.btn_driver_csv)
        self.lbl_driver_hint = _hint("")
        form.addRow(self.lbl_driver_hint)
        for widget in (self.cb_driver_joint, self.cb_driver_coord, self.cb_driver_kind):
            widget.currentIndexChanged.connect(self._driver_edited)
        for widget in (self.sp_driver_speed, self.sp_driver_amplitude, self.sp_driver_period):
            widget.valueChanged.connect(self._driver_edited)
        self.txt_driver_table.textChanged.connect(self._driver_table_edited)
        return page

    def _trace_page(self):
        page, form = self._form()
        self.lbl_trace_name = QLabel()
        form.addRow("<b>Trace point</b>", self.lbl_trace_name)
        self.lbl_trace_part = QLabel()
        self.lbl_trace_part.setWordWrap(True)
        form.addRow("On:", self.lbl_trace_part)
        form.addRow(_hint("Its path is drawn in the view; the plot shows its position, speed and "
                          "acceleration."))
        return page

    def _settings_box(self):
        box = QWidget()
        form = QFormLayout(box)
        form.setContentsMargins(0, 6, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.addRow(QLabel("<b>Settings</b>"))
        self.sp_duration = _spin(1e-4, 1e6, " s", 0.5, 4, 110)
        form.addRow("Duration:", self.sp_duration)
        self.sp_steps = QSpinBox()
        self.sp_steps.setRange(1, 10000)
        self.sp_steps.setSingleStep(10)
        self.sp_steps.setKeyboardTracking(False)
        self.sp_steps.setFixedWidth(110)
        self.sp_steps.setToolTip("Time steps (frames) of the motion")
        form.addRow("Steps:", self.sp_steps)
        self.lst_ground = QListWidget()
        self.lst_ground.setFixedHeight(80)
        self.lst_ground.setToolTip("Fixed parts (the frame of the mechanism); without one, the "
                                   "base part of the first joint stays fixed")
        self.lst_ground.itemChanged.connect(self._settings_edited)
        form.addRow("Fixed parts:", self.lst_ground)
        self.chk_rigid = QCheckBox("Aligned parts move together")
        self.chk_rigid.setToolTip("Enabled alignments hold their two parts together (unless a joint "
                                  "connects the same two parts)")
        self.chk_interference = QCheckBox("Check interference (red)")
        for widget in (self.chk_rigid, self.chk_interference):
            form.addRow(widget)
            widget.toggled.connect(self._settings_edited)
        self.sp_duration.valueChanged.connect(self._settings_edited)
        self.sp_steps.valueChanged.connect(self._settings_edited)
        return box

    # ── data access ───────────────────────────────────────────────────────────
    def motion(self):
        return motion_of(self.dialog.assembly)

    def study(self):
        motion = self.motion()
        return motion["studies"][motion["active"]]

    def _item(self, kind=None, name=None):
        if kind is None:
            if self._selection is None:
                return None
            kind, name = self._selection
        collection = {"joint": "joints", "coupling": "couplings", "driver": "drivers",
                      "trace": "traces"}[kind]
        return next((item for item in self.study()[collection] if item["name"] == name), None)

    def _edited(self, status=None):
        """Store a motion edit in the assembly (undo, unsaved state) and refresh."""
        self.motion()                                  # (edited in place)
        self.dialog._changed()
        self._refresh_dof()
        if status:
            self.dialog._set_status(status)

    # ── hooks from the Assembly window ────────────────────────────────────────
    def set_active(self, on):
        d = self.dialog
        if on:
            for act in (d.act_align, d.act_measure, d.act_section):
                if act.isChecked():
                    act.setChecked(False)
            if d._explosion_offsets:
                d._fit_together()
        self.active = bool(on)
        if self.act_motion.isChecked() != self.active:
            self.act_motion.blockSignals(True)
            self.act_motion.setChecked(self.active)
            self.act_motion.blockSignals(False)
        for act in (d.act_import, d.act_export, d.act_measure, d.act_remove, d.act_align,
                    d.act_exploded, d.act_fit_together, d.act_section, d.act_update_selected,
                    d.act_update_all):
            act.setVisible(not on)
        for spin in d.sp_pos + d.sp_rot:
            spin.setEnabled(not on)
        for act in self._motion_actions:
            act.setVisible(on)
        self._tidy_separators()
        self.playback.setVisible(on)
        d._show_tool_tab(self.tab, "Motion", on)
        if on:
            self._fill_tab()
            self._refresh_dof()
            d._set_status("Motion: add joints (Joint ▾), a driver and Compute (F5). The assembled "
                          "placement stays as it is; Apply position takes over a frame.")
        else:
            self._end_pick()
            self.reset_view(False)
            d._set_status("Assemble mode")
        d._update_gizmo()
        self.view.update()

    def scene_rebuilt(self):
        """The window re-made its render entries (undo, open, new): they show the assembled
        placement again."""
        self._frame = None
        self._display.clear()
        self._drag = None
        self._recolored.clear()
        if self.active:
            self._fill_tab()
            self._refresh_dof()

    def assembly_changed(self):
        if self._result is not None and self._result_key != self._key():
            self._discard_result()

    def component_renamed(self, old, new):
        rename_component(self.dialog.assembly, old, new)

    def component_removed(self, name):
        remove_component(self.dialog.assembly, name)

    def closing(self):
        self._timer.stop()
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(3000)
        self._end_pick()
        self.reset_view(False)

    # ── the tab ───────────────────────────────────────────────────────────────
    def _key(self):
        """What a result depends on: the motion data and the placements."""
        assembly = self.dialog.assembly
        return repr((_strip(self.motion()), [(c["name"], c["position"], c["rotation"], c["visible"])
                                             for c in assembly["components"]],
                     [(c.get("moving"), c.get("target"), c.get("enabled", True))
                      for c in assembly.get("constraints", [])]))

    def _fill_tab(self):
        self._loading = True
        try:
            motion = self.motion()
            self.cb_study.clear()
            for study in motion["studies"]:
                self.cb_study.addItem(study["name"])
            self.cb_study.setCurrentIndex(motion["active"])
            study = self.study()
            self.sp_duration.setValue(study["duration_s"])
            self.sp_steps.setValue(study["steps"])
            self.chk_rigid.setChecked(study["alignments_rigid"])
            self.chk_interference.setChecked(study["interference"])
            self.lst_ground.clear()
            ground = study["ground"] or ([study["joints"][0]["base"]] if study["joints"] else [])
            for component in self.dialog.assembly["components"]:
                item = QListWidgetItem(component["name"])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if component["name"] in ground else Qt.Unchecked)
                self.lst_ground.addItem(item)
            self._fill_tree()
        finally:
            self._loading = False
        self._show_page()

    def _fill_tree(self):
        study = self.study()
        was = self._loading
        self._loading = True
        try:
            self.tree.clear()
            groups = (("joint", "Joints", study["joints"],
                       lambda j: f"{j['name']}  ·  {j['moving']} on {j['base'] or 'World'}"),
                      ("coupling", "Couplings", study["couplings"],
                       lambda c: f"{c['name']}  ·  {c['a']} → {c['b']}"),
                      ("driver", "Drivers", study["drivers"],
                       lambda d: f"{d['name']}  ·  {self._driver_text(d)} on {d['joint']}"),
                      ("trace", "Trace points", study["traces"],
                       lambda t: f"{t['name']}  ·  on {t['component']}"))
            font = QFont(self.tree.font()); font.setItalic(True)
            selected = None
            for kind, title, items, text in groups:
                group = QTreeWidgetItem([f"{title} ({len(items)})"])
                group.setFont(0, font)
                group.setFlags(group.flags() & ~Qt.ItemIsSelectable)
                self.tree.addTopLevelItem(group)
                for entry in items:
                    item = QTreeWidgetItem([text(entry)])
                    item.setToolTip(0, text(entry))
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(0, Qt.Checked if entry.get("enabled", True) else Qt.Unchecked)
                    item.setData(0, Qt.UserRole, (kind, entry["name"]))
                    group.addChild(item)
                    if self._selection == (kind, entry["name"]):
                        selected = item
                group.setExpanded(True)
            if selected is not None:
                self.tree.setCurrentItem(selected)
        finally:
            self._loading = was

    @staticmethod
    def _driver_text(driver):
        unit = "°" if driver["coord"] == "angle" else " mm"
        if driver["kind"] == "speed":
            return f"{driver['speed']:g}{unit}/s"
        if driver["kind"] == "oscillate":
            return f"±{driver['amplitude']:g}{unit}, {driver['period']:g} s"
        return f"table, {len(driver['table'])} points"

    def _on_study_chosen(self, index):
        if self._loading or index < 0:
            return
        self.motion()["active"] = index
        self._selection = None
        self._discard_result()
        self._edited()
        self._fill_tab()

    def _new_study(self):
        motion = self.motion()
        names = {s["name"] for s in motion["studies"]}
        motion["studies"].append(study_new(unique_name(names, "Motion")))
        motion["active"] = len(motion["studies"]) - 1
        self._selection = None
        self._discard_result()
        self._edited("New motion study")
        self._fill_tab()

    def _rename_study(self):
        study = self.study()
        name, ok = QInputDialog.getText(self.dialog, "Rename", "Motion study:", text=study["name"])
        name = name.strip()
        if ok and name and name != study["name"]:
            study["name"] = name
            self._edited()
            self._fill_tab()

    def _delete_study(self):
        motion = self.motion()
        if len(motion["studies"]) <= 1:
            motion["studies"] = [study_new()]
        else:
            del motion["studies"][motion["active"]]
        motion["active"] = 0
        self._selection = None
        self._discard_result()
        self._edited("Motion study deleted")
        self._fill_tab()

    def _on_tree_selected(self):
        if self._loading:
            return
        items = self.tree.selectedItems()
        data = items[0].data(0, Qt.UserRole) if items else None
        self._selection = tuple(data) if data else None
        self._sync_remove()
        self._show_page()
        self.view.update()

    def _sync_remove(self):
        """Remove follows the selection (the uncoupling icon for a coupling)."""
        kind = self._selection[0] if self._selection else None
        self.act_remove_item.setEnabled(kind is not None)
        self.btn_remove.setEnabled(kind is not None)
        icon = self._icon("uncoupling.png" if kind == "coupling" else "delete_part.png")
        self.act_remove_item.setIcon(icon)
        self.btn_remove.setIcon(icon)
        self.act_remove_item.setToolTip(f"Remove {self._selection[1]}" if kind else
                                        "Remove the selected joint, coupling, driver or trace point")

    def _on_tree_changed(self, item, _column):
        if self._loading:
            return
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        entry = self._item(*data)
        if entry is not None:
            entry["enabled"] = item.checkState(0) == Qt.Checked
            self._edited()

    def _remove_item(self):
        if self._selection is None:
            return
        kind, name = self._selection
        study = self.study()
        collection = {"joint": "joints", "coupling": "couplings", "driver": "drivers",
                      "trace": "traces"}[kind]
        study[collection] = [item for item in study[collection] if item["name"] != name]
        if kind == "joint":
            study["couplings"] = [c for c in study["couplings"] if name not in (c["a"], c["b"])]
            study["drivers"] = [d for d in study["drivers"] if d["joint"] != name]
        self._selection = None
        self._edited(f"Removed {name}")
        self._fill_tree()
        self._show_page()

    def _show_page(self):
        self._sync_remove()
        entry = self._item() if self._selection else None
        if entry is None:
            self._set_page(0)
            return
        kind = self._selection[0]
        self._loading = True
        try:
            if kind == "joint":
                self._set_page(1)
                self.lbl_joint_name.setText(entry["name"])
                self.cb_joint_kind.setCurrentIndex(max(0, self.cb_joint_kind.findData(entry["kind"])))
                self.lbl_joint_parts.setText(f"{entry['moving']} moves on {entry['base'] or WORLD_TITLE}\n"
                                             f"({entry['base_ref'].get('label') or 'reference'} / "
                                             f"{entry['moving_ref'].get('label') or 'reference'})")
                coords = JOINT_KINDS[entry["kind"]][1]
                self._angle_row.setVisible("angle" in coords)
                self._slide_row.setVisible("slide" in coords)
                for check, low, high, key in ((self.chk_angle_limits, self.sp_angle_min,
                                               self.sp_angle_max, "angle_limits"),
                                              (self.chk_slide_limits, self.sp_slide_min,
                                               self.sp_slide_max, "slide_limits")):
                    limits = entry.get(key)
                    check.setChecked(limits is not None)
                    low.setValue(limits[0] if limits else -90.0)
                    high.setValue(limits[1] if limits else 90.0)
                    low.setEnabled(limits is not None)
                    high.setEnabled(limits is not None)
                self.lbl_joint_hint.setText(JOINT_HINTS[entry["kind"]] + " Angles and slides count "
                                            "from the position the joint was made in.")
            elif kind == "coupling":
                self._set_page(2)
                self.lbl_coupling_name.setText(entry["name"])
                self.cb_coupling_kind.setCurrentIndex(max(0, self.cb_coupling_kind.findData(entry["kind"])))
                names = [j["name"] for j in self.study()["joints"]]
                for combo, value in ((self.cb_coupling_a, entry["a"]), (self.cb_coupling_b, entry["b"])):
                    combo.clear()
                    for name in names:
                        combo.addItem(name, name)
                    combo.setCurrentIndex(max(0, combo.findData(value)))
                self.cb_coord_a.setCurrentIndex(self.cb_coord_a.findData(entry["a_coord"]))
                self.cb_coord_b.setCurrentIndex(self.cb_coord_b.findData(entry["b_coord"]))
                self.chk_coupling_auto.setChecked(entry["auto"])
                self.chk_coupling_auto.setVisible(entry["kind"] != "ratio")
                self.chk_coupling_reverse.setChecked(entry["reverse"])
                self.sp_lead.setValue(entry["lead_mm"])
                self.sp_lead.setVisible(entry["kind"] == "screw")
                self._lead_label.setVisible(entry["kind"] == "screw")
                value, why = auto_ratio(self.dialog.assembly, self.study(), entry)
                automatic = entry["auto"] and value is not None and entry["kind"] != "ratio"
                self.sp_ratio.setValue(value if automatic else entry["ratio"])
                self.sp_ratio.setEnabled(not automatic)
                ua, ub = COORD_UNITS[entry["a_coord"]], COORD_UNITS[entry["b_coord"]]
                self.lbl_coupling_info.setText(
                    (f"From the parts: {why}. " if value is not None and entry["kind"] != "ratio"
                     else f"{why[0].upper() + why[1:]}. " if why else "") +
                    f"Ratio = output {ub} per input {ua}; negative = opposite direction. Angles are "
                    "measured about each joint's own axis.")
            elif kind == "driver":
                self._set_page(3)
                self.lbl_driver_name.setText(entry["name"])
                self.cb_driver_joint.clear()
                for joint in self.study()["joints"]:
                    if JOINT_KINDS[joint["kind"]][1]:
                        self.cb_driver_joint.addItem(joint["name"], joint["name"])
                self.cb_driver_joint.setCurrentIndex(max(0, self.cb_driver_joint.findData(entry["joint"])))
                self.cb_driver_coord.setCurrentIndex(self.cb_driver_coord.findData(entry["coord"]))
                self.cb_driver_kind.setCurrentIndex(self.cb_driver_kind.findData(entry["kind"]))
                unit = "°" if entry["coord"] == "angle" else " mm"
                self.sp_driver_speed.setSuffix(unit + "/s")
                self.sp_driver_amplitude.setSuffix(unit)
                self.sp_driver_speed.setValue(entry["speed"])
                self.sp_driver_amplitude.setValue(entry["amplitude"])
                self.sp_driver_period.setValue(entry["period"])
                if self.txt_driver_table.toPlainText().strip() != self._table_text(entry["table"]):
                    self.txt_driver_table.setPlainText(self._table_text(entry["table"]))
                kind_ = entry["kind"]
                for widgets, show in (((self._speed_label, self.sp_driver_speed), kind_ == "speed"),
                                      ((self._amplitude_label, self.sp_driver_amplitude, self._period_label,
                                        self.sp_driver_period), kind_ == "oscillate"),
                                      ((self._table_label, self.txt_driver_table, self.btn_driver_csv),
                                       kind_ == "table")):
                    for widget in widgets:
                        widget.setVisible(show)
                self.lbl_driver_hint.setText(
                    {"speed": "Turns / slides at a constant speed from the assembled position "
                              "(360 °/s = 60 rpm).",
                     "oscillate": "Swings to and fro about the assembled position: amplitude · "
                                  "sin(2π t / period).",
                     "table": "The joint position at each time (counted from where the joint was "
                              "made), linear in between."}[kind_])
            else:
                self._set_page(4)
                self.lbl_trace_name.setText(entry["name"])
                self.lbl_trace_part.setText(f"{entry['component']} ({entry.get('label') or 'point'})")
        finally:
            self._loading = False

    def _set_page(self, index):
        """Show one property page; the others take no room (the stack fits the shown page)."""
        self.pages.setCurrentIndex(index)
        for i in range(self.pages.count()):
            policy = QSizePolicy.Preferred if i == index else QSizePolicy.Ignored
            self.pages.widget(i).setSizePolicy(policy, policy)
        self.pages.adjustSize()

    @staticmethod
    def _table_text(table):
        return "\n".join(f"{t:g}, {v:g}" for t, v in table)

    def _joint_edited(self, *_args):
        joint = self._item()
        if self._loading or joint is None or self._selection[0] != "joint":
            return
        joint["kind"] = self.cb_joint_kind.currentData()
        joint["angle_limits"] = [self.sp_angle_min.value(), self.sp_angle_max.value()] \
            if self.chk_angle_limits.isChecked() and self.sp_angle_max.value() > self.sp_angle_min.value() else None
        joint["slide_limits"] = [self.sp_slide_min.value(), self.sp_slide_max.value()] \
            if self.chk_slide_limits.isChecked() and self.sp_slide_max.value() > self.sp_slide_min.value() else None
        study = self.study()
        coords = JOINT_KINDS[joint["kind"]][1]
        for driver in study["drivers"]:          # drivers / couplings follow the joint's motions
            if driver["joint"] == joint["name"] and driver["coord"] not in coords and coords:
                driver["coord"] = coords[0]
        self._edited()
        self._fill_tree()
        self._show_page()

    def _coupling_edited(self, *_args):
        coupling = self._item()
        if self._loading or coupling is None or self._selection[0] != "coupling":
            return
        coupling["kind"] = self.cb_coupling_kind.currentData()
        coupling["a"] = self.cb_coupling_a.currentData() or ""
        coupling["b"] = self.cb_coupling_b.currentData() or ""
        coupling["a_coord"] = self.cb_coord_a.currentData()
        coupling["b_coord"] = self.cb_coord_b.currentData()
        if coupling["kind"] in ("rack", "screw") and self.sender() is self.cb_coupling_kind:
            coupling["a_coord"], coupling["b_coord"] = "angle", "slide"
        coupling["auto"] = self.chk_coupling_auto.isChecked()
        coupling["reverse"] = self.chk_coupling_reverse.isChecked()
        coupling["lead_mm"] = float(self.sp_lead.value())
        if self.sp_ratio.isEnabled():
            coupling["ratio"] = float(self.sp_ratio.value())
        self._edited()
        self._fill_tree()
        self._show_page()

    def _driver_edited(self, *_args):
        driver = self._item()
        if self._loading or driver is None or self._selection[0] != "driver":
            return
        driver["joint"] = self.cb_driver_joint.currentData() or driver["joint"]
        driver["coord"] = self.cb_driver_coord.currentData()
        driver["kind"] = self.cb_driver_kind.currentData()
        driver["speed"] = float(self.sp_driver_speed.value())
        driver["amplitude"] = float(self.sp_driver_amplitude.value())
        driver["period"] = float(self.sp_driver_period.value())
        self._edited()
        self._fill_tree()
        self._show_page()

    def _driver_table_edited(self):
        driver = self._item()
        if self._loading or driver is None or self._selection[0] != "driver":
            return
        table = []
        for line in self.txt_driver_table.toPlainText().replace(";", ",").splitlines():
            parts = [p for p in line.replace("\t", ",").split(",") if p.strip()]
            try:
                table.append([float(parts[0]), float(parts[1])])
            except (ValueError, IndexError):
                continue
        if sorted(table) != driver["table"]:
            driver["table"] = sorted(table)
            self._edited()
            self._fill_tree()

    def _import_driver_csv(self):
        driver = self._item()
        if driver is None:
            return
        path, _ = QFileDialog.getOpenFileName(self.dialog, "Driver table", "", "CSV files (*.csv *.txt)")
        if not path:
            return
        table = []
        try:
            with open(path, "r", encoding="utf-8-sig") as stream:
                for line in stream:
                    parts = [p for p in line.replace(";", ",").replace("\t", ",").split(",") if p.strip()]
                    try:
                        table.append([float(parts[0]), float(parts[1])])
                    except (ValueError, IndexError):
                        continue
        except OSError as ex:
            QMessageBox.warning(self.dialog, "Driver table", str(ex))
            return
        driver["table"] = sorted(table)
        driver["kind"] = "table"
        self._edited(f"{len(table)} points read from {os.path.basename(path)}")
        self._fill_tree()
        self._show_page()

    def _settings_edited(self, *_args):
        if self._loading:
            return
        study = self.study()
        study["duration_s"] = float(self.sp_duration.value())
        study["steps"] = int(self.sp_steps.value())
        study["alignments_rigid"] = self.chk_rigid.isChecked()
        study["interference"] = self.chk_interference.isChecked()
        study["ground"] = [self.lst_ground.item(row).text() for row in range(self.lst_ground.count())
                           if self.lst_ground.item(row).checkState() == Qt.Checked]
        self._edited()

    def _refresh_dof(self):
        if not self.active:
            return
        try:
            mechanism = Mechanism(self.dialog.assembly, self.study())
            text = mechanism.analyse()["text"]
        except MotionError as ex:
            text = str(ex)
        except Exception as ex:
            text = f"{type(ex).__name__}: {ex}"
        self.lbl_dof.setText(text)

    # ── making joints, couplings, drivers, traces ─────────────────────────────
    def _start_pick(self, purpose, kind=None):
        d = self.dialog
        if not self.active:
            self.set_active(True)
        self.reset_view(False)
        for act in (d.act_align, d.act_measure):
            if act.isChecked():
                act.setChecked(False)
        self._pick = {"purpose": purpose, "kind": kind}
        self.view.alignment_picking = True
        self.view.set_measure_mode(True)
        self.view.measure_clear()
        if purpose == "trace":
            d._set_status("Trace point: click a corner, an edge, a hole rim or a surface on the part "
                          "whose path you want to see (Esc cancels)")
        elif purpose == "space":
            d._set_status(f"{FIXED_IN_SPACE[kind][0]}: {FIXED_IN_SPACE[kind][1]} Or click a world "
                          "axis first to turn about it.")
        elif purpose == "axis":
            d._set_status(f"{'Bearing about an axis' if kind == 'revolute' else 'Guide along an axis'}: "
                          "click the axis — a world axis (X / Y / Z through the origin) or a hole rim, "
                          "edge or face of any part — then the part that moves.")
        else:
            d._set_status(f"{JOINT_KINDS[kind][0]}: click the reference on the base part (it carries "
                          "the joint) — or a world axis for a joint fixed in space — then on the "
                          "moving part. " + JOINT_HINTS[kind])

    def _end_pick(self):
        if self._pick is None:
            return
        self._pick = None
        self.view.alignment_picking = False
        self.view.set_measure_mode(False)
        self.view.update()

    def _on_picks(self):
        if self._pick is None:
            return
        picks = list(self.view._meas_sel)
        d = self.dialog
        if not picks:
            return
        study = self.study()
        purpose = self._pick["purpose"]
        if purpose == "axis" and self._pick.get("axis_ref") is None:
            point, axis, label = reference_of(picks[0])
            owner = picks[0]["entry"].get("_component")
            if axis is None:
                self.view.measure_clear()
                d._set_status("Pick something with an axis: a hole rim, a circular or straight edge, a "
                              "flat face, or a world axis")
                return
            self._pick["axis_ref"] = (point, axis, f"axis of {owner['name']} ({label})" if owner else label)
            self.view.measure_clear()
            d._set_status(f"Axis: {self._pick['axis_ref'][2]} — now click the part that moves about it")
            return
        if self._pick.get("axis_ref") is not None and purpose in ("axis", "space", "joint"):
            item = picks[-1]
            component = item["entry"].get("_component")
            if component is None:
                return
            kind = self._pick["kind"]
            axis_ref = self._pick["axis_ref"]
            moving_ref = reference_of(item) if purpose == "joint" else axis_ref
            joint, note = make_joint(d.assembly, study, kind, None, component, axis_ref, moving_ref)
            if joint is None:
                self.view.measure_clear()
                d._set_status(f"{JOINT_KINDS[kind][0]}: {note}")
                QMessageBox.information(d, JOINT_KINDS[kind][0], note[0].upper() + note[1:])
                return
            study["joints"].append(joint)
            study["ground"] = [g for g in study["ground"] if g != component["name"]]
            added_driver = ""
            if not study["drivers"] and JOINT_KINDS[kind][1]:
                study["drivers"].append(self._driver_new(joint["name"], JOINT_KINDS[kind][1][0]))
                added_driver = f"; driver {study['drivers'][-1]['name']} added on it"
            self._end_pick()
            self._selection = ("joint", joint["name"])
            self._edited(f"{joint['name']}: {component['name']} "
                         f"{'turns about' if kind != 'slider' else 'slides along'} {axis_ref[2]}" + added_driver)
            self._fill_tab()
            return
        if purpose == "space":
            item = picks[0]
            component = item["entry"].get("_component")
            if component is None:
                return
            kind = self._pick["kind"]
            joint, note = fixed_in_space(d.assembly, study, kind, component, reference_of(item))
            if joint is None:
                self.view.measure_clear()
                d._set_status(f"{FIXED_IN_SPACE[kind][0]}: {note}")
                return
            study["joints"].append(joint)
            study["ground"] = [g for g in study["ground"] if g != component["name"]]
            added_driver = ""
            if not study["drivers"]:
                study["drivers"].append(self._driver_new(joint["name"], JOINT_KINDS[kind][1][0]))
                added_driver = f"; driver {study['drivers'][-1]['name']} added on it"
            self._end_pick()
            self._selection = ("joint", joint["name"])
            self._edited(f"{joint['name']}: {component['name']} {'turns' if kind == 'revolute' else 'slides'} "
                         f"in place ({joint['moving_ref']['label']})" + added_driver +
                         (f" — {note}" if note else ""))
            self._fill_tab()
            return
        if self._pick["purpose"] == "trace":
            item = picks[0]
            component = item["entry"].get("_component")
            if component is None:
                return
            point, _axis, label = reference_of(item)
            names = {t["name"] for t in study["traces"]}
            trace = {"name": unique_name(names, "Trace"), "enabled": True,
                     "component": component["name"],
                     "point": [float(x) for x in cad.assembly_local_point(component, point)],
                     "label": label}
            study["traces"].append(trace)
            self._end_pick()
            self._selection = ("trace", trace["name"])
            self._edited(f"{trace['name']} on {component['name']} ({label}) — Compute to see its path")
            self._fill_tree()
            self._show_page()
            return
        if len(picks) == 1:
            base = picks[0]["entry"].get("_component")
            d._set_status(f"Base part: {base['name']} — now click the reference on the moving part")
            return
        base, moving = picks[0]["entry"].get("_component"), picks[1]["entry"].get("_component")
        if base is None or moving is None:
            self.view.measure_clear()
            return
        if base is moving:
            self.view._meas_sel.pop()
            self.view.update()
            d._set_status("The moving part must be a different part")
            return
        kind = self._pick["kind"]
        joint, note = make_joint(d.assembly, study, kind, base, moving, reference_of(picks[0]),
                                 reference_of(picks[1]))
        if joint is None:
            self.view._meas_sel.pop()
            self.view.update()
            d._set_status(f"{JOINT_KINDS[kind][0]}: {note}")
            QMessageBox.information(d, JOINT_KINDS[kind][0], note[0].upper() + note[1:])
            return
        study["joints"].append(joint)
        if not study["ground"]:
            study["ground"] = [base["name"]]
        added_driver = ""
        if not study["drivers"] and JOINT_KINDS[kind][1]:
            coord = JOINT_KINDS[kind][1][0]
            study["drivers"].append(self._driver_new(joint["name"], coord))
            added_driver = f"; driver {study['drivers'][-1]['name']} added on it"
        self._end_pick()
        self._selection = ("joint", joint["name"])
        self._edited(f"{joint['name']}: {moving['name']} on {base['name']}" + added_driver +
                     (f" — {note}" if note else ""))
        self._fill_tab()

    def _auto_gears(self):
        if not self.active:
            self.set_active(True)
        self._end_pick()
        self.reset_view(False)
        done, notes = auto_gears(self.dialog.assembly, self.study())
        if not done:
            QMessageBox.information(self.dialog, "Gears automatically", "\n".join(
                notes or ["Nothing to do: every gear has a joint and its couplings already."]))
            return
        self._selection = None
        self._edited("Gears: " + "; ".join(done))
        self._fill_tab()
        if notes:
            QMessageBox.information(self.dialog, "Gears automatically", "\n".join(notes))

    def _driver_new(self, joint, coord):
        names = {d["name"] for d in self.study()["drivers"]}
        return {"name": unique_name(names, "Driver"), "kind": "speed", "enabled": True,
                "joint": joint, "coord": coord, "speed": 180.0 if coord == "angle" else 10.0,
                "amplitude": 30.0 if coord == "angle" else 5.0, "period": 1.0, "table": []}

    def _movable_joints(self):
        return [j for j in self.study()["joints"] if JOINT_KINDS[j["kind"]][1]]

    def _add_driver(self):
        if not self.active:
            self.set_active(True)
        joints = self._movable_joints()
        if not joints:
            self.dialog._set_status("Driver: make a revolute, slider or cylindrical joint first")
            return
        chosen = self._item() if self._selection and self._selection[0] == "joint" else None
        joint = chosen if chosen is not None and JOINT_KINDS[chosen["kind"]][1] else joints[0]
        driver = self._driver_new(joint["name"], JOINT_KINDS[joint["kind"]][1][0])
        self.study()["drivers"].append(driver)
        self._selection = ("driver", driver["name"])
        self._edited(f"{driver['name']} on {joint['name']}: set its speed or motion in the Motion tab")
        self._fill_tree()
        self._show_page()

    def _add_coupling(self, kind):
        if not self.active:
            self.set_active(True)
        joints = self._movable_joints()
        if len(joints) < 2:
            self.dialog._set_status("Coupling: make the two joints first (e.g. a revolute joint "
                                    "for each gear)")
            return
        selected = self._item() if self._selection and self._selection[0] == "joint" else None
        first = selected if selected in joints else joints[0]
        second = next((j for j in joints if j is not first), joints[1])
        a_coord = "angle" if "angle" in JOINT_KINDS[first["kind"]][1] else "slide"
        b_coord = "slide" if kind in ("rack", "screw") and "slide" in JOINT_KINDS[second["kind"]][1] \
            else ("angle" if "angle" in JOINT_KINDS[second["kind"]][1] else "slide")
        if kind == "rack" and "slide" not in JOINT_KINDS[second["kind"]][1]:
            slider = next((j for j in joints if "slide" in JOINT_KINDS[j["kind"]][1]), None)
            if slider is not None:
                second, b_coord = slider, "slide"
        names = {c["name"] for c in self.study()["couplings"]}
        coupling = {"name": unique_name(names, COUPLING_KINDS[kind].split(" ")[0]), "kind": kind,
                    "enabled": True, "a": first["name"], "a_coord": a_coord, "b": second["name"],
                    "b_coord": b_coord, "ratio": 1.0, "auto": kind != "ratio", "reverse": False,
                    "lead_mm": 1.0}
        value, why = auto_ratio(self.dialog.assembly, self.study(), coupling)
        if value is not None:
            coupling["ratio"] = value
        self.study()["couplings"].append(coupling)
        self._selection = ("coupling", coupling["name"])
        self._edited(f"{coupling['name']}: {first['name']} → {second['name']} ({why})")
        self._fill_tree()
        self._show_page()

    # ── computing ─────────────────────────────────────────────────────────────
    def _compute(self):
        if self._worker is not None:
            return
        if not self.active:
            self.set_active(True)
        self._end_pick()
        self.reset_view(False)
        study = copy.deepcopy(self.study())
        assembly = self.dialog.assembly
        snapshot = {"components": assembly["components"], "constraints": assembly.get("constraints", [])}
        try:
            Mechanism(snapshot, study)           # quick checks before the thread
        except MotionError as ex:
            QMessageBox.information(self.dialog, "Motion", str(ex))
            return
        self._key_at_start = self._key()
        worker = MotionWorker(snapshot, study)
        worker.progressed.connect(self._on_progress)
        worker.succeeded.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        self.progress.setValue(0)
        self.progress.show()
        self.act_compute.setEnabled(False)
        self.dialog._set_status("Computing the motion...")
        worker.start()

    def _on_progress(self, percent, text):
        self.progress.setValue(percent)
        self.dialog._set_status(text)

    def _on_done(self, result):
        self._result = result
        self._result_key = self._key_at_start
        self.txt_messages.setPlainText("\n".join(result["summary"] + [""] + result["messages"]))
        self.sl_time.blockSignals(True)
        self.sl_time.setRange(0, max(0, result["valid"] - 1))
        self.sl_time.setValue(0)
        self.sl_time.blockSignals(False)
        self._fill_series()
        self.show_frame(0)
        clashes = len(result["interference"])
        self.dialog._set_status(
            f"Motion computed: {result['valid'] - 1} steps over {result['times'][result['valid'] - 1]:.4g} s"
            + (f", {clashes} overlap{'s' if clashes != 1 else ''} (red)" if clashes else "") +
            " — press Play")

    def _on_failed(self, text):
        if text:
            self.txt_messages.setPlainText(text)
            QMessageBox.warning(self.dialog, "Motion", text)
            self.dialog._set_status("Motion not possible: " + text.splitlines()[0])
        else:
            self.dialog._set_status("Motion cancelled")

    def _on_finished(self):
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()
        self.progress.hide()
        self.act_compute.setEnabled(True)

    def _discard_result(self):
        self._timer.stop()
        self.btn_play.setIcon(self.dialog.style().standardIcon(QStyle.SP_MediaPlay))
        if self._frame is not None or self._display:
            self.reset_view(False)
        self._result = None
        self._result_key = None
        self.sl_time.blockSignals(True)
        self.sl_time.setRange(0, 0)
        self.sl_time.blockSignals(False)
        self.lbl_time.setText("t = —")
        self.plot.set_data(None, [], "")
        self.txt_messages.setPlainText("The setup changed — Compute again." if self.active else "")

    # ── showing frames ────────────────────────────────────────────────────────
    def _show_poses(self, poses):
        """Put parts at {name: (R, t)} in the view (the assembled placement is not changed)."""
        components = {c["name"]: c for c in self.dialog.assembly["components"]}
        for name, (R, t) in poses.items():
            component = components.get(name)
            if component is None:
                continue
            position, rotation = pose_placement(component, R, t)
            placed = dict(component, position=position, rotation=rotation)
            for body in component["bodies"]:
                entry = body.get("_render")
                if entry is None:
                    continue
                world = cad.assembly_world_vertices(placed, body["vertices"])
                entry["solid"] = (list(map(tuple, world.tolist())), body["triangles"])
                entry["faces"] = cad.assembly_world_faces(placed, body["faces"])
                for key in cad._ASSEMBLY_CACHE_KEYS:
                    entry.pop(key, None)
            self._display[name] = (np.array(R, float), np.array(t, float))

    def show_frame(self, frame):
        """Put the parts where they are at a frame (the assembled placement is not changed)."""
        result = self._result
        if result is None:
            return
        frame = int(min(max(frame, 0), result["valid"] - 1))
        components = {c["name"]: c for c in self.dialog.assembly["components"]}
        self._reset_display(keep=set(result["movers"]))
        self._show_poses({name: (result["R"][frame, j], result["t"][frame, j])
                          for j, name in enumerate(result["movers"])})
        self._restore_colors()
        for name in result["clash_frames"].get(frame, ()):
            component = components.get(name)
            if component is None:
                continue
            for body in component["bodies"]:
                entry = body.get("_render")
                if entry is not None:
                    self._recolored[id(entry)] = (entry, entry.get("color"))
                    entry["color"] = (1.0, 0.18, 0.18)
        self._frame = frame
        time_value = float(result["times"][frame])
        self.lbl_time.setText(f"t = {time_value:.3f} s / {result['times'][result['valid'] - 1]:.3g} s")
        if self.sl_time.value() != frame:
            self.sl_time.blockSignals(True)
            self.sl_time.setValue(frame)
            self.sl_time.blockSignals(False)
        self.plot.set_cursor(time_value)
        self.view.update()

    def _restore_colors(self):
        for entry, colour in self._recolored.values():
            entry["color"] = colour
        self._recolored.clear()

    def _reset_display(self, keep=()):
        """Parts shown away from their placement (except `keep`) go back in the view."""
        components = {c["name"]: c for c in self.dialog.assembly["components"]}
        for name in [n for n in self._display if n not in keep]:
            component = components.get(name)
            if component is not None:
                self.dialog._apply_transform(component)
            del self._display[name]

    def reset_view(self, status=True):
        """Back to the assembled placement in the view."""
        self._timer.stop()
        self._drag = None
        self.btn_play.setIcon(self.dialog.style().standardIcon(QStyle.SP_MediaPlay))
        self._restore_colors()
        self._reset_display()
        self._frame = None
        if self._result is not None:
            self.sl_time.blockSignals(True)
            self.sl_time.setValue(0)
            self.sl_time.blockSignals(False)
            self.lbl_time.setText("t = — (assembled)")
            self.plot.set_cursor(None)
        if status:
            self.dialog._set_status("Assembled position")
        self.view.update()

    def _on_slider(self, value):
        if self._result is not None:
            self.show_frame(value)

    def _on_plot_clicked(self, time_value):
        if self._result is not None:
            frame = int(np.argmin(np.abs(self._result["times"][:self._result["valid"]] - time_value)))
            self.show_frame(frame)

    def _toggle_play(self):
        if self._result is None:
            if self.active:
                self._compute()
            return
        if self._timer.isActive():
            self._timer.stop()
            self.btn_play.setIcon(self.dialog.style().standardIcon(QStyle.SP_MediaPlay))
            return
        if self._frame is None or self._frame >= self._result["valid"] - 1:
            self.show_frame(0)
        dt = float(self._result["times"][1] - self._result["times"][0]) if self._result["valid"] > 1 else 0.05
        speed = float(self.cb_speed.currentData() or 1.0)
        self._play_stride = max(1, int(math.ceil(0.03 / max(dt / speed, 1e-9))))
        self._timer.setInterval(max(15, int(1000.0 * dt * self._play_stride / speed)))
        self._timer.start()
        self.btn_play.setIcon(self.dialog.style().standardIcon(QStyle.SP_MediaPause))

    def _play_step(self):
        result = self._result
        if result is None:
            self._timer.stop()
            return
        frame = (self._frame or 0) + getattr(self, "_play_stride", 1)
        if frame >= result["valid"]:
            if self.chk_loop.isChecked():
                frame = 0
            else:
                self.show_frame(result["valid"] - 1)
                self._timer.stop()
                self.btn_play.setIcon(self.dialog.style().standardIcon(QStyle.SP_MediaPlay))
                return
        self.show_frame(frame)

    def _apply_position(self):
        if not self._display:
            self.dialog._set_status("Apply position: show a frame of the motion or drag a handle first")
            return
        components = {c["name"]: c for c in self.dialog.assembly["components"]}
        moved = []
        for name, (R, t) in self._display.items():
            component = components.get(name)
            if component is None:
                continue
            component["position"], component["rotation"] = pose_placement(component, R, t)
            moved.append(component)
        when = f"t = {float(self._result['times'][self._frame]):.4g} s" \
            if self._result is not None and self._frame is not None else "the dragged position"
        self._restore_colors()
        self._display.clear()
        self._frame = None
        for component in moved:
            self.dialog._apply_transform(component)
        self.dialog._show_properties()
        self.dialog._changed()
        self.dialog._set_status(f"The parts now stand as at {when} (Undo takes it back)")

    # ── dragging a joint in the view ──────────────────────────────────────────
    ANGLE_COLOR = (1.0, 0.6, 0.15)
    SLIDE_COLOR = (0.25, 0.8, 1.0)
    HOT_COLOR = (1.0, 0.92, 0.3)

    def _pose(self, name):
        if name == WORLD:
            return np.eye(3), np.zeros(3)
        if name in self._display:
            return self._display[name]
        component = cad.assembly_component(self.dialog.assembly, name)
        return component_pose(component) if component is not None else (np.eye(3), np.zeros(3))

    def _handles(self):
        """Drag handles of the joints that can move: [{joint, coord, origin, axis, perp, size}]."""
        if not self.active or self._pick is not None or self._timer.isActive():
            return []
        names = {c["name"] for c in self.dialog.assembly["components"]}
        size = 0.09 * _model_size(self.dialog.assembly)
        out = []
        for joint in self.study()["joints"]:
            coords = JOINT_KINDS[joint["kind"]][1]
            if not joint.get("enabled", True) or not coords or joint["moving"] not in names or \
                    (joint["base"] != WORLD and joint["base"] not in names):
                continue
            R, t = self._pose(joint["base"])
            ref = joint["base_ref"]
            origin = R @ np.asarray(ref["origin"], float) + t
            axis = R @ np.asarray(ref["axis"], float)
            perp = R @ np.asarray(ref["perp"], float)
            axis /= np.linalg.norm(axis)
            perp /= np.linalg.norm(perp)
            for coord in coords:
                out.append({"joint": joint, "coord": coord, "origin": origin, "axis": axis,
                            "perp": perp, "size": size * (1.25 if coord == "slide" else 1.0)})
        return out

    @staticmethod
    def _ring(handle, count=48):
        """The arc of a rotation handle (20 to 290 degrees, as the Assembly's rotate rings)."""
        n2 = np.cross(handle["axis"], handle["perp"])
        angles = np.radians(np.linspace(20.0, 290.0, count + 1))
        return handle["origin"] + handle["size"] * (np.cos(angles)[:, None] * handle["perp"] +
                                                    np.sin(angles)[:, None] * n2)

    @staticmethod
    def _arrow(handle):
        return handle["origin"] - handle["axis"] * handle["size"], handle["origin"] + handle["axis"] * handle["size"]

    def _project(self, points):
        """Pixel positions of world points (n, 2), or None before the first paint."""
        view = self.view
        if view._mvp is None:
            return None
        points = np.atleast_2d(points)
        clip = np.hstack([points, np.ones((len(points), 1))]) @ np.asarray(view._mvp).T
        ndc = clip[:, :3] / clip[:, 3:4]
        return np.stack([(ndc[:, 0] + 1.0) * 0.5 * view.width(), (1.0 - ndc[:, 1]) * 0.5 * view.height()], 1)

    def _handle_at(self, pos, tolerance=9.0):
        mouse = np.array([pos.x(), pos.y()], float)
        best, best_distance = None, tolerance
        for handle in self._handles():
            line = self._ring(handle) if handle["coord"] == "angle" else np.array(self._arrow(handle))
            pixels = self._project(line)
            if pixels is None:
                return None
            for a, b in zip(pixels[:-1], pixels[1:]):
                d = b - a
                u = 0.0 if d @ d < 1e-12 else min(1.0, max(0.0, float((mouse - a) @ d / (d @ d))))
                distance = float(np.linalg.norm(mouse - (a + u * d)))
                if distance < best_distance:
                    best, best_distance = handle, distance
        return best

    def _drag_param(self, handle, pos):
        """The mouse position along the handle: an angle (radians) about the axis, or a slide
        (mm) along it; None when the view looks along the ring's plane."""
        origin, direction = self.view._ray(pos)
        direction = direction / np.linalg.norm(direction)
        axis, centre = handle["axis"], handle["origin"]
        if handle["coord"] == "angle":
            denominator = float(direction @ axis)
            if abs(denominator) < 1e-6:
                return None
            hit = origin + float((centre - origin) @ axis) / denominator * direction
            v = hit - centre
            n2 = np.cross(axis, handle["perp"])
            return math.atan2(float(v @ n2), float(v @ handle["perp"]))
        w = origin - centre                       # closest point of the axis line to the ray
        b = float(axis @ direction)
        denominator = 1.0 - b * b
        if denominator < 1e-9:
            return None
        return float((axis @ w) - b * (direction @ w)) / denominator

    def _world_axes(self):
        """(name, direction, colour) of the world axes drawn while picking."""
        return (("X", np.array([1.0, 0.0, 0.0]), (1.0, 0.35, 0.35)),
                ("Y", np.array([0.0, 1.0, 0.0]), (0.4, 1.0, 0.4)),
                ("Z", np.array([0.0, 0.0, 1.0]), (0.45, 0.6, 1.0)))

    def _world_axis_at(self, pos, tolerance=10.0):
        length = 0.6 * _model_size(self.dialog.assembly)
        mouse = np.array([pos.x(), pos.y()], float)
        best, best_distance = None, tolerance
        for name, direction, _colour in self._world_axes():
            pixels = self._project(np.array([-length * direction, length * direction]))
            if pixels is None:
                return None
            a, b = pixels
            d = b - a
            u = 0.0 if d @ d < 1e-12 else min(1.0, max(0.0, float((mouse - a) @ d / (d @ d))))
            distance = float(np.linalg.norm(mouse - (a + u * d)))
            if distance < best_distance:
                best, best_distance = (name, direction), distance
        return best

    def press(self, view, event):
        if event.button() == Qt.RightButton:
            self._right_press = (event.pos().x(), event.pos().y(), bool(
                self._pick is not None and (self._pick.get("axis_ref") is not None or self.view._meas_sel)))
            return False
        if event.button() == Qt.LeftButton and self._pick is not None and \
                self._pick["purpose"] in ("joint", "space", "axis") and self._pick.get("axis_ref") is None:
            axis = self._world_axis_at(event.pos())
            first = list(self.view._meas_sel)
            if axis is None or (self._pick["purpose"] != "joint" and first) or len(first) > 1:
                return False
            name, direction = axis
            self._pick["axis_ref"] = (np.zeros(3), direction, f"the world {name} axis")
            if first:                                  # a part first, then the world axis: fixed in space
                self._on_picks()
                return True
            self.view.measure_clear()
            self.view.update()
            self.dialog._set_status(f"World {name} axis — now click the part that "
                                    f"{'slides along' if self._pick['kind'] == 'slider' else 'turns about'} it")
            return True
        if event.button() != Qt.LeftButton or not self.active:
            return False
        handle = self._handle_at(event.pos())
        if handle is None:
            return False
        study = copy.deepcopy(self.study())
        joint = handle["joint"]
        study["drivers"] = [{"name": "drag", "kind": "manual", "enabled": True, "joint": joint["name"],
                             "coord": handle["coord"], "speed": 0.0, "amplitude": 0.0, "period": 1.0,
                             "table": [], "value": 0.0}]
        study["couplings"] = [c for c in study["couplings"]]
        try:
            mechanism = Mechanism(self.dialog.assembly, study)
        except MotionError as ex:
            self.dialog._set_status(f"Cannot move {joint['name']}: {ex}")
            return True
        if self._display:
            mechanism.set_poses(self._display)
        start = self._drag_param(handle, event.pos())
        if start is None:
            return True
        self._restore_colors()
        self._frame = None
        self._drag = {"handle": handle, "mechanism": mechanism, "start": start, "last": start,
                      "total": 0.0, "value": 0.0}
        self.view.setCursor(Qt.ClosedHandCursor)
        return True

    def move(self, view, event):
        drag = self._drag
        if drag is None:
            if not event.buttons() and self.active:
                hover = self._handle_at(event.pos())
                if (hover is None) != (self._hover is None) or (hover is not None and (
                        hover["joint"]["name"], hover["coord"]) != (self._hover["joint"]["name"], self._hover["coord"])):
                    self._hover = hover
                    self.view.setCursor(Qt.OpenHandCursor if hover else Qt.ArrowCursor)
                    self.view.update()
            return False
        if not event.buttons() & Qt.LeftButton:
            return self.release(view, event)
        handle, mechanism = drag["handle"], drag["mechanism"]
        param = self._drag_param(handle, event.pos())
        if param is None:
            return True
        if handle["coord"] == "angle":
            drag["total"] += _wrap(param - drag["last"])
            drag["last"] = param
            value = math.degrees(drag["total"])
            if not event.modifiers() & Qt.ControlModifier:
                value = round(value)                    # whole degrees; Ctrl = free
        else:
            value = param - drag["start"]
            if not event.modifiers() & Qt.ControlModifier:
                value = round(value * 10.0) / 10.0      # 0.1 mm; Ctrl = free
        mechanism.drivers[0]["value"] = value
        saved = mechanism.poses()
        ok, _error, _its = mechanism.solve(0.0)
        if not ok:
            mechanism.set_poses(saved, restart=False)
            self.dialog._set_status(f"{handle['joint']['name']}: the mechanism cannot move further "
                                    "this way")
            return True
        mechanism.accept()
        drag["value"] = value
        self._show_poses(mechanism.poses())
        unit = "°" if handle["coord"] == "angle" else " mm"
        self.dialog._set_status(f"{handle['joint']['name']}: {value:+.1f}{unit} (Ctrl = free; Apply "
                                "position keeps it, Reset goes back)")
        self.view.update()
        return True

    def _right_click(self, had_pick=False):
        """A right click (no drag) takes back the last pick, ends a pick, or clears the selection."""
        if self._pick is not None:
            if had_pick or self._pick.get("axis_ref") is not None or self.view._meas_sel:
                self._pick.pop("axis_ref", None)
                self.view.measure_clear()
                self.view.update()
                self.dialog._set_status("Pick cleared - click the reference again (right click again "
                                        "or Esc cancels)")
            else:
                self._end_pick()
                self.dialog._set_status("Pick cancelled")
            return
        if self._selection is not None:
            self._selection = None
            self.tree.clearSelection()
            self.tree.setCurrentItem(None)
            self._sync_remove()
            self._show_page()
            self.view.update()

    def release(self, view, event):
        if event.button() == Qt.RightButton:
            start, self._right_press = getattr(self, "_right_press", None), None
            if start is not None and self.active and                     abs(event.pos().x() - start[0]) + abs(event.pos().y() - start[1]) < 5:
                self._right_click(start[2])
            return False
        if self._drag is None:
            return False
        self._drag = None
        self.view.setCursor(Qt.OpenHandCursor if self._hover else Qt.ArrowCursor)
        self.view.update()
        return True

    # ── plots ─────────────────────────────────────────────────────────────────
    def _fill_series(self):
        result = self._result
        self.cb_series.blockSignals(True)
        self.cb_series.clear()
        if result is not None:
            for name, values in result["coords"].items():
                for coord in values:
                    self.cb_series.addItem(f"{name} — {'angle' if coord == 'angle' else 'slide'}",
                                           ("joint", name, coord))
            for name in result["traces"]:
                self.cb_series.addItem(f"{name} — path", ("trace", name, None))
        self.cb_series.blockSignals(False)
        self._update_plot()

    def _update_plot(self):
        result = self._result
        data = self.cb_series.currentData()
        if result is None or data is None:
            self.plot.set_data(None, [], "")
            return
        valid = result["valid"]
        times = result["times"][:valid]
        quantity = self.cb_quantity.currentData()
        kind, name, coord = data

        def derive(values, order):
            out = np.asarray(values, float)
            for _ in range(order):
                out = np.gradient(out, times) if len(out) > 1 else out * 0.0
            return out
        order = {"position": 0, "speed": 1, "acceleration": 2}[quantity]
        if kind == "joint":
            unit = COORD_UNITS[coord]
            values = result["coords"][name][coord][:valid]
            curves = [(f"{name} {coord}", derive(values, order))]
            unit = unit + ("" if order == 0 else "/s" if order == 1 else "/s²")
        else:
            path = result["traces"][name][:valid]
            if order == 0:
                curves = [(axis, path[:, n]) for n, axis in enumerate("XYZ")]
                unit = "mm"
            else:
                vector = np.stack([derive(path[:, n], order) for n in range(3)], axis=1)
                curves = [(f"|{'v' if order == 1 else 'a'}|", np.linalg.norm(vector, axis=1))] + \
                    [(axis, vector[:, n]) for n, axis in enumerate("XYZ")]
                unit = "mm/s" if order == 1 else "mm/s²"
        self.plot.set_data(times, curves, unit)
        if self._frame is not None:
            self.plot.set_cursor(float(result["times"][self._frame]))

    # ── overlay: joints and trace curves ──────────────────────────────────────
    def _paint_overlay(self):
        if not self.active:
            return
        assembly = self.dialog.assembly
        study = self.study()
        size = _model_size(assembly)
        gl.glPushAttrib(gl.GL_ENABLE_BIT | gl.GL_CURRENT_BIT | gl.GL_LINE_BIT | gl.GL_POINT_BIT)
        try:
            gl.glDisable(gl.GL_LIGHTING)
            gl.glDisable(gl.GL_DEPTH_TEST)
            result, frame = self._result, self._frame
            if result is not None:
                for name, path in result["traces"].items():
                    path = path[:result["valid"]]
                    colour = result["trace_colors"].get(name, (1.0, 0.85, 0.2))
                    gl.glLineWidth(1.5)
                    gl.glColor3f(*[0.45 * c for c in colour])
                    gl.glBegin(gl.GL_LINE_STRIP)
                    for point in path:
                        gl.glVertex3f(*[float(x) for x in point])
                    gl.glEnd()
                    if frame is not None:
                        gl.glLineWidth(2.5)
                        gl.glColor3f(*colour)
                        gl.glBegin(gl.GL_LINE_STRIP)
                        for point in path[:frame + 1]:
                            gl.glVertex3f(*[float(x) for x in point])
                        gl.glEnd()
                        gl.glPointSize(9.0)
                        gl.glBegin(gl.GL_POINTS)
                        gl.glVertex3f(*[float(x) for x in path[frame]])
                        gl.glEnd()
            if self._pick is not None and self._pick["purpose"] in ("joint", "space", "axis"):
                length = 0.6 * size
                chosen = self._pick.get("axis_ref")
                for name, direction, colour in self._world_axes():
                    hot = chosen is not None and float(abs(chosen[1] @ direction)) > 0.999 and \
                        not float(np.linalg.norm(chosen[0]))
                    gl.glLineWidth(4.0 if hot else 1.6)
                    gl.glColor3f(*colour)
                    gl.glBegin(gl.GL_LINES)
                    gl.glVertex3f(*[float(x) for x in -length * direction])
                    gl.glVertex3f(*[float(x) for x in length * direction])
                    gl.glEnd()
                if chosen is not None and float(np.linalg.norm(chosen[0])):
                    gl.glLineWidth(4.0)                 # an axis picked on a part
                    gl.glColor3f(*self.HOT_COLOR)
                    gl.glBegin(gl.GL_LINES)
                    for point in (chosen[0] - 0.3 * size * chosen[1], chosen[0] + 0.3 * size * chosen[1]):
                        gl.glVertex3f(*[float(x) for x in point])
                    gl.glEnd()
            # drag handles: a ring where a joint turns, a double arrow where it slides
            selected = self._selection[1] if self._selection and self._selection[0] == "joint" else None
            hot = self._drag["handle"] if self._drag else self._hover
            for handle in self._handles():
                is_hot = hot is not None and hot["joint"]["name"] == handle["joint"]["name"] and \
                    hot["coord"] == handle["coord"]
                colour = self.HOT_COLOR if is_hot else \
                    (self.ANGLE_COLOR if handle["coord"] == "angle" else self.SLIDE_COLOR)
                active = is_hot or handle["joint"]["name"] == selected
                if is_hot:
                    colour = tuple(0.45 + 0.55 * c for c in colour)
                pixel = self.view._gizmo_pixel(handle["origin"])
                height, radius = 16.0 * pixel, 9.0 * pixel
                if handle["coord"] == "angle":
                    ring = self._ring(handle)
                    n2 = np.cross(handle["axis"], handle["perp"])
                    a0, a1 = math.radians(20.0), math.radians(290.0)
                    tangent0 = math.sin(a0) * handle["perp"] - math.cos(a0) * n2          # backwards
                    tangent1 = -math.sin(a1) * handle["perp"] + math.cos(a1) * n2
                    cones = ((ring[0], tangent0), (ring[-1], tangent1))
                else:
                    a, b = self._arrow(handle)
                    ring = np.array([a, b])
                    cones = ((a, -handle["axis"]), (b, handle["axis"]))
                for outline in (True, False):               # dark outline first, then the colour
                    gl.glLineWidth((11.0 if active else 8.0) if outline else (6.0 if active else 4.0))
                    gl.glColor3f(*((0.07, 0.08, 0.09) if outline else colour))
                    gl.glBegin(gl.GL_LINE_STRIP)
                    for point in ring:
                        gl.glVertex3f(*[float(x) for x in point])
                    gl.glEnd()
                    for tip, direction in cones:
                        self.view._draw_cone(tip, direction, height, radius if outline else radius * 0.67)
            if frame is None and not self._display:
                components = {c["name"]: c for c in assembly["components"]}
                for n, trace in enumerate(study["traces"]):
                    component = components.get(trace["component"])
                    if component is None:
                        continue
                    point = cad.assembly_world_point(component, trace["point"])
                    gl.glPointSize(8.0)
                    gl.glColor3f(*TRACE_COLORS[n % len(TRACE_COLORS)])
                    gl.glBegin(gl.GL_POINTS)
                    gl.glVertex3f(*[float(x) for x in point])
                    gl.glEnd()
        finally:
            gl.glPopAttrib()

    # ── export ────────────────────────────────────────────────────────────────
    def _export_base(self):
        path = getattr(self.dialog._window, "_assembly_file", None)
        base = os.path.splitext(path)[0] if path else os.path.join(os.path.expanduser("~"), "motion")
        return base + "_" + self.study()["name"].replace(" ", "_")

    def _export_csv(self, path=None):
        result = self._result
        if result is None:
            self.dialog._set_status("Export: compute the motion first")
            return
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self.dialog, "Values over time",
                                                  self._export_base() + ".csv", "CSV files (*.csv)")
            if not path:
                return
        valid = result["valid"]
        times = result["times"][:valid]
        header, columns = ["Time s"], [times]
        for name, values in result["coords"].items():
            for coord, series in values.items():
                unit = COORD_UNITS[coord]
                series = series[:valid]
                speed = np.gradient(series, times) if valid > 1 else series * 0
                header += [f"{name} {coord} {unit}", f"{name} {coord} speed {unit}/s",
                           f"{name} {coord} acceleration {unit}/s²"]
                columns += [series, speed, np.gradient(speed, times) if valid > 1 else speed * 0]
        for name, path_points in result["traces"].items():
            path_points = path_points[:valid]
            velocity = np.stack([np.gradient(path_points[:, n], times) for n in range(3)], 1) \
                if valid > 1 else path_points * 0
            header += [f"{name} {axis} mm" for axis in "XYZ"] + [f"{name} speed mm/s"]
            columns += [path_points[:, n] for n in range(3)] + [np.linalg.norm(velocity, axis=1)]
        clash = [", ".join(sorted(result["clash_frames"].get(k, ()))) for k in range(valid)]
        import csv
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(header + ["Overlapping parts"])
                for k in range(valid):
                    writer.writerow([f"{column[k]:.6g}" for column in columns] + [clash[k]])
        except OSError as ex:
            QMessageBox.critical(self.dialog, "Export failed", f"Could not write\n{path}\n\n{ex}")
            return
        self.dialog._set_status(f"Values over time saved to {path}")

    def _frame_images(self, stride=1, width=None):
        """QImages of the frames as shown in the view."""
        images = []
        for frame in range(0, self._result["valid"], stride):
            self.show_frame(frame)
            self.view.updateGL()
            image = self.view.grabFrameBuffer()
            if width and image.width() > width:
                from PyQt5.QtCore import Qt as _Qt
                image = image.scaledToWidth(width, _Qt.SmoothTransformation)
            images.append(image)
        return images

    def _export_video(self, path=None):
        result = self._result
        if result is None:
            self.dialog._set_status("Export: compute the motion first")
            return
        ffmpeg = shutil.which("ffmpeg")
        if path is None:
            filters = "Animated GIF (*.gif)" + (";;MP4 video (*.mp4)" if ffmpeg else "")
            path, _ = QFileDialog.getSaveFileName(self.dialog, "Export the motion as a video",
                                                  self._export_base() + ".gif", filters)
            if not path:
                return
        mp4 = path.lower().endswith(".mp4")
        if mp4 and not ffmpeg:
            QMessageBox.information(self.dialog, "Video", "MP4 needs ffmpeg on the PATH; a GIF works "
                                    "without it.")
            return
        valid = result["valid"]
        dt = float(result["times"][1] - result["times"][0]) if valid > 1 else 0.05
        stride = max(1, int(math.ceil(valid / 300)))       # at most about 300 frames
        frame_ms = max(20, int(round(1000.0 * dt * stride)))
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            images = self._frame_images(stride, 960)
            if mp4:
                folder = tempfile.mkdtemp(prefix="simplecad_motion_")
                try:
                    for n, image in enumerate(images):
                        image.save(os.path.join(folder, f"frame_{n:05d}.png"))
                    fps = max(1.0, 1000.0 / frame_ms)
                    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{fps:.4f}",
                                    "-i", os.path.join(folder, "frame_%05d.png"), "-pix_fmt", "yuv420p",
                                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", path], check=True)
                finally:
                    shutil.rmtree(folder, ignore_errors=True)
            else:
                write_gif(path, images, frame_ms)
        except Exception as ex:
            QApplication.restoreOverrideCursor()
            cad._trace("Motion video export failed:\n" + traceback.format_exc())
            QMessageBox.critical(self.dialog, "Export failed", f"Could not write\n{path}\n\n{ex}")
            return
        QApplication.restoreOverrideCursor()
        self.dialog._set_status(f"Video saved to {path} ({len(images)} frames)")


def write_gif(path, images, frame_ms):
    """An animated GIF from QImages (Pillow when available, else one PNG per frame)."""
    try:
        from PIL import Image
    except Exception:
        folder = os.path.splitext(path)[0] + "_frames"
        os.makedirs(folder, exist_ok=True)
        for n, image in enumerate(images):
            image.save(os.path.join(folder, f"frame_{n:05d}.png"))
        raise RuntimeError(f"Pillow is not installed: the frames were saved as PNG files in {folder}")
    frames = []
    for image in images:
        image = image.convertToFormat(QImage.Format_RGB888)
        width, height = image.width(), image.height()
        pointer = image.constBits()
        pointer.setsize(image.byteCount())
        array = np.frombuffer(pointer, np.uint8).reshape(height, image.bytesPerLine())[:, :3 * width]
        frames.append(Image.fromarray(array.reshape(height, width, 3).copy()).convert(
            "P", palette=Image.ADAPTIVE, colors=255))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0,
                   optimize=False, disposal=2)
