"""
SimpleCAD Thermal — optional add-on (Phase 1: steady-state heat conduction, FEM).

SimpleCAD shows the "Thermal" toolbar button only while this file sits next to
SimpleCAD.py (or is bundled into the executable). The window works like the Assembly window:
parts come from the open project, from STEP / STL / OBJ files or from an assembly; block and
cylinder heat sources are placed with XYZ fields or the move / rotate arrows. Every body needs a
material, and any body can carry a heat power in watts.

Solver "Conduction (FEM)":
  * all visible bodies are voxelised on one Cartesian grid (cell-centre inside test, majority of
    three ray directions, so slightly open meshes still work). Bodies that touch share grid
    nodes, i.e. their contact is bonded. Later bodies win where bodies overlap; placed heat
    sources always win.
  * trilinear 8-node hexahedral finite elements; laminates (FR-4, G10, CFK) are orthotropic:
    the through-plane axis is the thinnest extent of the body.
  * exposed faces lose heat by convection (natural-convection correlations for vertical,
    upward- and downward-facing surfaces with gravity along -Z, or a fixed h) and by radiation
    to the ambient. Air enclosed in a housing conducts heat as still air.
  * the non-linear terms (natural convection, radiation) are handled by Picard iteration; each
    linear system is solved by conjugate gradients with a Jacobi preconditioner (scipy).
  * every run reports the heat balance: power in = convection + radiation.
Units in the window: mm, W, degrees Celsius. The solver works in SI units.
"""
import os, math, json, copy, time, traceback, collections

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QDialog, QWidget, QToolBar, QAction, QLabel, QDoubleSpinBox, QSpinBox,
    QTreeWidget, QTreeWidgetItem, QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QMessageBox, QPushButton, QFrame, QInputDialog, QListWidget, QListWidgetItem, QMenu, QMenuBar,
    QShortcut, QPlainTextEdit, QToolButton, QCheckBox, QTableWidget, QTableWidgetItem, QComboBox,
    QSizePolicy, QSplitter, QProgressBar, QHeaderView, QDialogButtonBox, QAbstractItemView,
    QScrollArea, QLineEdit, QStackedWidget, QSlider, QStyle)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QPoint, QRectF, QTimer, QEvent
from PyQt5.QtGui import (QIcon, QKeySequence, QColor, QPainter, QLinearGradient, QFont, QPen,
                         QFontMetricsF, QPolygonF, QCursor)
from PyQt5.QtCore import QPointF
from OpenGL import GL as gl
import ctypes

import SimpleCAD as cad        # the running application (SimpleCAD.load_addon registers it)

ADDON_TITLE = "Thermal"
ADDON_VERSION = "1.0"          # add-on release, shown in SimpleCAD's Help → About
THERMAL_FORMAT = "simplecad-thermal-study"
THERMAL_EXTENSION = ".thml"
THERMAL_DEFAULT_NAME = "Untitled"
HEAT_SOURCE_COLOR = (1.0, 0.45, 0.15)
PRIMITIVE_KINDS = ("block", "cylinder")
COMPONENT_KINDS = ("project", "file", "assembly") + PRIMITIVE_KINDS

# ── materials ─────────────────────────────────────────────────────────────────
# Thermal data at about 20 °C, typical handbook values (actual grades vary). Densities come from
# SimpleCAD's MATERIAL_DENSITIES_G_CM3, so both lists use the same material names.
#   k           thermal conductivity, W/(m·K); through-plane value for laminates
#   k_plane     in-plane conductivity of laminates (optional, makes the body orthotropic)
#   cp          specific heat capacity, J/(kg·K)  (used by later transient runs)
#   emissivity  of the plain surface (can be overridden per body, e.g. anodised or painted)
#   max_c       highest continuous service temperature in °C; None = no warning
MATERIAL_THERMAL = {
    "Aluminium": {"k": 167.0, "cp": 896.0, "emissivity": 0.09, "max_c": None,
                  "note": "alloy 6061 / 6082, bare surface (pure aluminium 237 W/m·K); for "
                          "heatsinks see Aluminium Black Anodized"},
    "Aluminium Black Anodized": {
        "k": 200.0, "cp": 900.0, "emissivity": 0.85, "max_c": None,
        "note": "heatsink extrusion alloy 6063-T5/T6 with black anodised surface; the coating "
                "raises the emissivity from ≈ 0.1 to ≈ 0.85, so radiation removes far more heat"},
    "Copper": {"k": 398.0, "cp": 385.0, "emissivity": 0.05, "max_c": None,
               "note": "pure copper; oxidised surface ε ≈ 0.6"},
    "Tin": {"k": 66.8, "cp": 228.0, "emissivity": 0.05, "max_c": 200.0,
            "note": "melts at 232 °C"},
    "Brass": {"k": 109.0, "cp": 380.0, "emissivity": 0.10, "max_c": None, "note": "CuZn37"},
    "Bronze": {"k": 50.0, "cp": 380.0, "emissivity": 0.10, "max_c": None,
               "note": "tin bronze (CuSn8)"},
    "Steel": {"k": 50.0, "cp": 490.0, "emissivity": 0.30, "max_c": None,
              "note": "carbon steel; stainless steel only ≈ 16 W/m·K"},
    "ABS": {"k": 0.17, "cp": 1400.0, "emissivity": 0.92, "max_c": 80.0,
            "note": "softens near 95 °C"},
    "PLA": {"k": 0.13, "cp": 1800.0, "emissivity": 0.92, "max_c": 50.0,
            "note": "glass transition ≈ 60 °C"},
    "PET": {"k": 0.20, "cp": 1200.0, "emissivity": 0.90, "max_c": 70.0,
            "note": "PET / PETG, glass transition ≈ 80 °C"},
    "Micarta": {"k": 0.30, "cp": 1400.0, "emissivity": 0.90, "max_c": 120.0,
                "note": "phenolic laminate"},
    "G10": {"k": 0.29, "k_plane": 0.81, "cp": 1100.0, "emissivity": 0.90, "max_c": 140.0,
            "note": "glass-epoxy laminate"},
    "FR-4": {"k": 0.29, "k_plane": 0.81, "cp": 1100.0, "emissivity": 0.90, "max_c": 130.0,
             "note": "glass-epoxy PCB laminate without copper; glass transition Tg ≈ 130–140 °C; "
                     "thermal resistivity ≈ 3.4 m·K/W through the board"},
    "GFK": {"k": 0.30, "cp": 1000.0, "emissivity": 0.90, "max_c": 120.0,
            "note": "glass-fibre reinforced plastic"},
    "CFK": {"k": 0.80, "k_plane": 5.0, "cp": 1000.0, "emissivity": 0.85, "max_c": 120.0,
            "note": "carbon-fibre laminate, quasi-isotropic lay-up"},
    "Wood (Oak)": {"k": 0.17, "cp": 2000.0, "emissivity": 0.90, "max_c": 100.0,
                   "note": "across the grain"},
    "Wood (Pine)": {"k": 0.12, "cp": 2300.0, "emissivity": 0.90, "max_c": 100.0,
                    "note": "across the grain"},
    "Wood (Beech)": {"k": 0.16, "cp": 2100.0, "emissivity": 0.90, "max_c": 100.0,
                     "note": "across the grain"},
    "Wood (Birch)": {"k": 0.14, "cp": 1900.0, "emissivity": 0.90, "max_c": 100.0,
                     "note": "across the grain"},
    "Wood (Walnut)": {"k": 0.14, "cp": 1900.0, "emissivity": 0.90, "max_c": 100.0,
                      "note": "across the grain"},
}
AIR_CONDUCTIVITY = 0.026            # W/(m·K), still air at about 25 °C (air enclosed in a housing)
AIR_DENSITY = 1.18                  # kg/m³
AIR_HEAT_CAPACITY = 1005.0          # J/(kg·K)
STEFAN_BOLTZMANN = 5.670374419e-8   # W/(m²·K⁴)


def material_names():
    """Materials that have thermal data, in the order of SimpleCAD's material list."""
    names = [name for name in cad.MATERIAL_DENSITIES_G_CM3 if name in MATERIAL_THERMAL]
    return names + [name for name in MATERIAL_THERMAL if name not in names]


def material_properties(name):
    """Thermal data of a material incl. density in kg/m³, or None for an unknown material."""
    data = MATERIAL_THERMAL.get(name)
    if data is None:
        return None
    props = dict(data)
    density = cad.MATERIAL_DENSITIES_G_CM3.get(name)
    props["density"] = None if density is None else density * 1000.0
    return props


def material_summary(name):
    props = material_properties(name)
    if props is None:
        return "No material set — choose one before the simulation."
    k = f"k {props['k']:g} W/m·K"
    if props.get("k_plane"):
        k = f"k {props['k_plane']:g} W/m·K in-plane, {props['k']:g} through-plane"
    parts = [k, f"cp {props['cp']:g} J/kg·K"]
    if props["density"]:
        parts.append(f"ρ {props['density']:g} kg/m³")
    parts.append(f"ε {props['emissivity']:g}")
    if props.get("max_c") is not None:
        parts.append(f"max {props['max_c']:g} °C")
    return " · ".join(parts)


# ── study data model ──────────────────────────────────────────────────────────
# A study holds components like an assembly (cad.assembly_* helpers work on it): a placed snapshot
# of project parts or of a file (also when it came in through an assembly), or a block / cylinder
# heat source. Each body additionally carries "material" (name or None), "material_auto" (True =
# taken from the source, so Update brings a changed source material along; False = chosen in the
# study and kept), "power_w" and "emissivity" (None = the material's value).
DEFAULT_SETTINGS = {
    "ambient_c": 25.0,         # ambient air and surroundings
    "convection": "natural",   # "natural" (correlations, gravity -Z) or "fixed"
    "h": 10.0,                 # W/(m²·K) for "fixed"
    "radiation": True,
    "cell_mm": 0.0,            # 0 = automatic
    "max_cells": 400000,
    "analysis": "steady",      # "steady" (final temperatures) or "transient" (over time)
    "duration_s": 600.0,       # transient: simulated time from switching on
    "duration_unit": "min",    # unit the duration is shown in: "s", "min" or "h"
    "time_steps": 100,         # transient: number of implicit time steps
    "initial_ambient": True,   # transient: everything starts at the ambient temperature
    "initial_c": 25.0,         # transient: start temperature when not at ambient
}
TIME_UNITS = {"s": 1.0, "min": 60.0, "h": 3600.0}
MAX_FRAMES = 100               # transient: stored temperature fields for the time slider
TRANSIENT_TOLERANCE = 1e-7     # relative residual of each time step (well below the step error)
COARSE_BLOCK = 4               # grid nodes per edge of a coarse block of the preconditioner


def _coarse_space(node_index, block):
    """Piecewise-constant coarse space: every node belongs to the block of grid nodes around it
    (block³ nodes). Returns the prolongation matrix (nodes × blocks)."""
    import scipy.sparse as sp
    cells = node_index // block
    dims = cells.max(0) + 1
    keys = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
    _unique, coarse = np.unique(keys, return_inverse=True)
    n = len(node_index)
    return sp.csr_matrix((np.ones(n), (np.arange(n), coarse.ravel())),
                         shape=(n, int(coarse.max()) + 1))


_COMPONENT_KEYS =("name", "kind", "visible", "position", "rotation", "pivot", "source",
                   "source_name", "signature", "path", "file_stamp", "file_root", "shape")


def study_new(name=THERMAL_DEFAULT_NAME):
    return {"name": name, "components": [], "settings": dict(DEFAULT_SETTINGS),
            "conditions": [], "probes": []}


def thermal_body(name, vertices, triangles, faces, color=cad.DEFAULT_SOLID_COLOR, opacity=1.0,
                 material=None, power_w=0.0, emissivity=None, fused=False, material_auto=False):
    body = cad.assembly_body(name, vertices, triangles, faces, color, opacity, fused=fused)
    body["material"] = material if material in MATERIAL_THERMAL else None
    body["material_auto"] = bool(material_auto) and body["material"] is not None
    body["power_w"] = max(0.0, float(power_w or 0.0))
    body["emissivity"] = None if emissivity is None else min(1.0, max(0.0, float(emissivity)))
    return body


def thermal_body_from_assembly(body, material=None):
    """A study body from an assembly body snapshot (its material when the snapshot has one).
    Meshes without CAD faces (STL, OBJ) are split into faces at sharp edges, so face conditions
    can be put on single faces."""
    material = body.get("material") or material
    faces = body.get("faces", [])
    if _needs_segmentation(faces, len(body["triangles"])):
        faces = segment_faces(body["vertices"], body["triangles"])
    return thermal_body(body["name"], body["vertices"], body["triangles"], faces,
                        body.get("color", cad.DEFAULT_SOLID_COLOR), body.get("opacity", 1.0),
                        material, fused=body.get("fused", False), material_auto=material is not None)


def _needs_segmentation(faces, triangle_count):
    """True for a mesh whose only face is the whole (curved) surface."""
    if triangle_count < 2:
        return False
    if not faces:
        return True
    return len(faces) == 1 and not faces[0].get("planar") and \
        len(faces[0].get("tris", ())) >= triangle_count


def segment_faces(vertices, triangles, angle_deg=30.0):
    """Face records of a plain triangle mesh: neighbouring triangles whose normals differ by less
    than `angle_deg` belong to one face (a box gives 6 faces, a cylinder its side and 2 caps)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    V = np.asarray(vertices, float).reshape(-1, 3)
    T = np.asarray(triangles, np.int64).reshape(-1, 3)
    m = len(T)
    scale = max(float(np.ptp(V, axis=0).max()) if len(V) else 0.0, 1e-9)
    _keys, weld = np.unique(np.round(V / (scale * 1e-7)).astype(np.int64), axis=0,
                            return_inverse=True)
    W = weld.ravel()[T]
    P = V[T]
    normal = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    edges = np.sort(np.concatenate([W[:, [0, 1]], W[:, [1, 2]], W[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(m), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    shared = np.all(edges[1:] == edges[:-1], axis=1)
    a, b = owner[:-1][shared], owner[1:][shared]
    smooth = (normal[a] * normal[b]).sum(1) >= math.cos(math.radians(angle_deg))
    graph = coo_matrix((np.ones(int(smooth.sum())), (a[smooth], b[smooth])), shape=(m, m))
    _count, labels = connected_components(graph, directed=False)
    order = np.argsort(labels, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1)
    faces = []
    for tris in groups:
        mean = normal[tris].sum(0)
        mean /= max(float(np.linalg.norm(mean)), 1e-30)
        planar = bool((normal[tris] @ mean).min() > math.cos(math.radians(0.5)))
        faces.append({"planar": planar, "tris": [int(t) for t in tris]})
    return faces


SOURCE_KINDS = ("project", "file")     # components with a source that Update can re-read


def component_status(component, objects):
    """"ok", "outdated" (the source changed after the snapshot) or "missing" (source gone);
    heat sources placed in the study have no source and are always "ok"."""
    if component.get("kind") not in SOURCE_KINDS:
        return "ok"
    return cad.assembly_component_status(component, objects)


def study_refresh_component(component, objects, window=None):
    """Re-read the source of a component like the Assembly's Update: the placement stays, and so
    do power, emissivity and materials chosen in the study (bodies are matched by name, else by
    order). A material that came from the source follows the source."""
    if component.get("kind") not in SOURCE_KINDS:
        raise ValueError("a heat source placed in the study has no source to update")
    old_bodies = component["bodies"]
    cad.assembly_refresh_component(component, objects, window)
    by_name = {body["name"]: body for body in old_bodies}
    bodies = []
    for index, snapshot in enumerate(component["bodies"]):
        old = by_name.get(snapshot["name"]) or (old_bodies[index] if index < len(old_bodies) else None)
        body = thermal_body_from_assembly(snapshot)
        if old is not None:
            if old.get("material") and not old.get("material_auto"):
                body["material"], body["material_auto"] = old["material"], False
            elif body["material"] is None and old.get("material"):
                body["material"], body["material_auto"] = old["material"], False
            body["power_w"] = old.get("power_w", 0.0)
            body["emissivity"] = old.get("emissivity")
        bodies.append(body)
    component["bodies"] = bodies


# ── face conditions and probes ────────────────────────────────────────────────
# A face condition applies to faces picked on the bodies; faces are referenced by part name,
# body index and face index plus a signature (centre, normal, area in part coordinates), so a
# condition follows its faces when the part is moved, updated or resized. Faces without a
# condition exchange heat with the environment (Simulation panel). When a face is in several
# conditions, the last one counts.
CONDITION_KINDS = {             # kind: (title, colour in the 3-D view)
    "fixed": ("Fixed temperature", (0.25, 0.55, 1.0)),
    "flux": ("Heat flux", (1.0, 0.3, 0.8)),        # magenta: apart from orange heat sources
    "convection": ("Convection", (0.2, 0.85, 0.85)),
    "insulated": ("Insulated", (0.8, 0.8, 0.8)),
}
CONDITION_HINTS = {
    "fixed": "The faces are held at this temperature, e.g. a part clamped to a cold plate or a "
             "water-cooled surface.",
    "flux": "This heat enters through the faces (e.g. a heating foil or a known load). The faces "
            "still exchange heat with the air.",
    "convection": "Own heat transfer coefficient and air temperature for these faces, e.g. a fan-"
                  "cooled side (h ≈ 20–100 W/m²K). Radiation stays as set for the environment.",
    "insulated": "No heat crosses these faces (no convection, no radiation), e.g. a face against "
                 "insulation or a mounting surface that hardly conducts.",
}


def condition_new(kind, name):
    return {"name": name, "kind": kind, "enabled": True, "faces": [],
            "temperature_c": 25.0,                                  # fixed
            "flux_mode": "total", "power_w": 1.0, "flux_w_m2": 1000.0,  # flux
            "h": 25.0, "fluid_ambient": True, "fluid_c": 25.0}      # convection


def condition_settings(data):
    """A validated condition from loaded data."""
    kind = data.get("kind")
    if kind not in CONDITION_KINDS:
        raise ValueError(f"invalid face condition kind: {kind!r}")
    condition = condition_new(kind, str(data.get("name", CONDITION_KINDS[kind][0])))
    condition["enabled"] = bool(data.get("enabled", True))
    for key in ("temperature_c", "power_w", "flux_w_m2", "h", "fluid_c"):
        if key in data:
            condition[key] = float(data[key])
    condition["flux_mode"] = "density" if data.get("flux_mode") == "density" else "total"
    condition["fluid_ambient"] = bool(data.get("fluid_ambient", True))
    condition["faces"] = [{"component": str(ref["component"]), "body": int(ref["body"]),
                           "face": int(ref["face"]),
                           "signature": [float(x) for x in ref.get("signature", [])]}
                          for ref in data.get("faces", [])]
    return condition


def face_signature(body, face_index):
    """[centre xyz, unit normal xyz, area] of a body face in part coordinates."""
    V = np.asarray(body["vertices"], float)
    T = np.asarray(body["triangles"], np.int64).reshape(-1, 3)
    tris = np.asarray(body["faces"][face_index]["tris"], np.int64)
    if not len(tris):
        return [0.0] * 7
    P = V[T[tris]]
    cross = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2.0
    total = max(float(area.sum()), 1e-12)
    centre = (P.mean(1) * area[:, None]).sum(0) / total
    normal = cross.sum(0)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    return [float(x) for x in (*centre, *normal)] + [float(area.sum())]


def condition_toggle_face(condition, component, body_index, face_index):
    """Add the face to the condition, or remove it when it is there. True = added."""
    for ref in condition["faces"]:
        if ref["component"] == component["name"] and ref["body"] == body_index and \
                ref["face"] == face_index:
            condition["faces"].remove(ref)
            return False
    condition["faces"].append({"component": component["name"], "body": int(body_index),
                               "face": int(face_index),
                               "signature": face_signature(component["bodies"][body_index], face_index)})
    return True


def _match_face(body, signature):
    """Index of the body face that fits a stored signature best, or None."""
    if len(signature) != 7 or not body.get("faces"):
        return None
    centre, normal, area = np.array(signature[:3]), np.array(signature[3:6]), signature[6]
    best, best_score = None, float("inf")
    for index in range(len(body["faces"])):
        other = face_signature(body, index)
        if float(np.dot(normal, other[3:6])) < 0.9:
            continue
        distance = float(np.linalg.norm(centre - np.array(other[:3])))
        score = distance + 0.05 * abs(math.sqrt(max(area, 0.0)) - math.sqrt(max(other[6], 0.0)))
        if score < best_score:
            best, best_score = index, score
    if best is None or best_score > max(0.5, 0.25 * math.sqrt(max(area, 1e-12))):
        return None
    return best


def remap_condition_faces(study, component):
    """After Update or a size change: find the referenced faces of the part again by their
    signatures. Returns the number of faces that could not be found (they are removed)."""
    lost = 0
    for condition in study.get("conditions", []):
        kept = []
        for ref in condition["faces"]:
            if ref["component"] != component["name"]:
                kept.append(ref)
                continue
            if not 0 <= ref["body"] < len(component["bodies"]):
                lost += 1
                continue
            body = component["bodies"][ref["body"]]
            index = ref["face"] if 0 <= ref["face"] < len(body["faces"]) else None
            if index is None or _match_face(body, ref["signature"]) != index:
                index = _match_face(body, ref["signature"])
            if index is None:
                lost += 1
                continue
            ref["face"] = index
            ref["signature"] = face_signature(body, index)
            if not any(r["component"] == ref["component"] and r["body"] == ref["body"] and
                       r["face"] == index for r in kept):
                kept.append(ref)
        condition["faces"] = kept
    return lost


def refresh_signatures(study, component):
    """New signatures for faces that keep their index (a resized block or cylinder)."""
    for condition in study.get("conditions", []):
        kept = []
        for ref in condition["faces"]:
            if ref["component"] == component["name"]:
                if not 0 <= ref["body"] < len(component["bodies"]) or \
                        not 0 <= ref["face"] < len(component["bodies"][ref["body"]]["faces"]):
                    continue
                ref["signature"] = face_signature(component["bodies"][ref["body"]], ref["face"])
            kept.append(ref)
        condition["faces"] = kept


def rename_component_refs(study, old, new):
    for condition in study.get("conditions", []):
        for ref in condition["faces"]:
            if ref["component"] == old:
                ref["component"] = new
    for probe in study.get("probes", []):
        if probe["component"] == old:
            probe["component"] = new


def drop_component_refs(study, name):
    for condition in study.get("conditions", []):
        condition["faces"] = [ref for ref in condition["faces"] if ref["component"] != name]
    study["probes"] = [probe for probe in study.get("probes", []) if probe["component"] != name]


def probe_settings(data):
    return {"name": str(data["name"]), "component": str(data["component"]),
            "body": int(data.get("body", 0)), "point": [float(x) for x in data["point"]]}


def probe_world_point(study, probe):
    component = next((c for c in study["components"] if c["name"] == probe["component"]), None)
    if component is None:
        return None
    return cad.assembly_world_point(component, probe["point"])


def study_conditions(study, jobs):
    """Face conditions for the solver: [{..condition values.., "targets": {job index: [face
    indices]}}] for the enabled conditions, in order (later ones win on shared faces)."""
    job_of = {job["key"]: index for index, job in enumerate(jobs)}
    component_index = {c["name"]: i for i, c in enumerate(study["components"])}
    out = []
    for condition in study.get("conditions", []):
        if not condition.get("enabled", True):
            continue
        targets = collections.defaultdict(list)
        for ref in condition["faces"]:
            key = (component_index.get(ref["component"]), ref["body"])
            if key in job_of:
                targets[job_of[key]].append(ref["face"])
        entry = {key: value for key, value in condition.items() if key != "faces"}
        entry["targets"] = dict(targets)
        out.append(entry)
    return out


def _block_mesh(size):
    hx, hy, hz = (max(1e-3, float(value)) / 2.0 for value in size)
    vertices = [(x, y, z) for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)]
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    triangles, faces = [], []
    for a, b, c, d in quads:                     # each quad: outward, counter-clockwise
        faces.append({"planar": True, "tris": [len(triangles), len(triangles) + 1]})
        triangles += [(a, b, c), (a, c, d)]
    return vertices, triangles, faces


def _cylinder_mesh(diameter, height, axis="Z", segments=48):
    r, hz = max(1e-3, float(diameter)) / 2.0, max(1e-3, float(height)) / 2.0
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    ring = [(r * math.cos(a), r * math.sin(a)) for a in angles]
    vertices = [(x, y, -hz) for x, y in ring] + [(x, y, hz) for x, y in ring] + \
               [(0.0, 0.0, -hz), (0.0, 0.0, hz)]
    bottom_centre, top_centre = 2 * segments, 2 * segments + 1
    side, top, bottom = [], [], []
    for i in range(segments):
        j = (i + 1) % segments
        side += [(i, j, segments + j), (i, segments + j, segments + i)]
        top.append((top_centre, segments + i, segments + j))
        bottom.append((bottom_centre, j, i))
    triangles = side + top + bottom
    n = len(side)
    faces = [{"planar": False, "tris": list(range(n))},
             {"planar": True, "tris": list(range(n, n + segments))},
             {"planar": True, "tris": list(range(n + segments, n + 2 * segments))}]
    # local Z → requested axis by a cyclic permutation (a rotation, so the orientation stays)
    order = {"Z": (0, 1, 2), "X": (2, 0, 1), "Y": (1, 2, 0)}[axis if axis in ("X", "Y", "Z") else "Z"]
    vertices = [(v[order[0]], v[order[1]], v[order[2]]) for v in vertices]
    return vertices, triangles, faces


def primitive_mesh(kind, shape):
    if kind == "block":
        return _block_mesh(shape.get("size", (10.0, 10.0, 2.0)))
    return _cylinder_mesh(shape.get("diameter", 8.0), shape.get("height", 12.0),
                          shape.get("axis", "Z"))


def primitive_extent(kind, shape):
    """Size (x, y, z) in mm of an unrotated primitive."""
    if kind == "block":
        return [float(value) for value in shape.get("size", (10.0, 10.0, 2.0))]
    d, h = float(shape.get("diameter", 8.0)), float(shape.get("height", 12.0))
    axis = "XYZ".index(shape.get("axis", "Z")) if shape.get("axis", "Z") in "XYZ" else 2
    return [h if i == axis else d for i in range(3)]


def primitive_set_shape(component, shape):
    """Replace the mesh of a heat-source primitive after a size change."""
    component["shape"] = copy.deepcopy(shape)
    vertices, triangles, faces = primitive_mesh(component["kind"], shape)
    old = component["bodies"][0]
    body = thermal_body(old["name"], vertices, triangles, faces, old["color"], old["opacity"],
                        old.get("material"), old.get("power_w", 0.0), old.get("emissivity"),
                        material_auto=old.get("material_auto", False))
    component["bodies"][0] = body
    component["pivot"] = [0.0, 0.0, 0.0]
    return body


def study_add_primitive(study, kind, shape, name, position, power_w, material):
    vertices, triangles, faces = primitive_mesh(kind, shape)
    body = thermal_body(name, vertices, triangles, faces, HEAT_SOURCE_COLOR,
                        material=material, power_w=power_w)
    component = cad.assembly_add_component(study, name, kind, [body], shape=copy.deepcopy(shape),
                                           source=f"{kind.capitalize()} placed in the study")
    body["name"] = component["name"]
    component["pivot"] = [0.0, 0.0, 0.0]
    component["position"] = [float(value) for value in position]
    return component


def component_bounds(component):
    return cad.assembly_bounds(component)


def body_volume_mm3(body):
    """Volume of the body's (closed) mesh in mm³, cached on the body."""
    cached = body.get("_volume")
    if cached is None or cached[0] is not body["vertices"]:
        V = np.asarray(body["vertices"], float)
        T = np.asarray(body["triangles"], np.int64).reshape(-1, 3)
        volume = abs(_mesh_volume(V, T)) if len(T) else 0.0
        cached = (body["vertices"], volume)
        body["_volume"] = cached
    return cached[1]


def _mesh_volume(V, T):
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _mesh_area(V, T):
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2.0)


# persistence -------------------------------------------------------------------
def study_settings(data):
    settings = dict(DEFAULT_SETTINGS)
    for key, default in DEFAULT_SETTINGS.items():
        if key not in (data or {}):
            continue
        value = data[key]
        try:
            if isinstance(default, bool):
                settings[key] = bool(value)
            elif isinstance(default, int):
                settings[key] = int(value)
            elif isinstance(default, float):
                settings[key] = float(value)
            else:
                settings[key] = str(value)
        except (TypeError, ValueError):
            pass
    if settings["convection"] not in ("natural", "fixed"):
        settings["convection"] = "natural"
    if settings["analysis"] not in ("steady", "transient"):
        settings["analysis"] = "steady"
    if settings["duration_unit"] not in TIME_UNITS:
        settings["duration_unit"] = "min"
    settings["duration_s"] = max(0.001, settings["duration_s"])
    settings["time_steps"] = min(10000, max(1, settings["time_steps"]))
    return settings


def format_time(seconds):
    """1 h 05 min, 12 min 30 s, 45.0 s, 0.25 s."""
    seconds = float(seconds)
    if seconds >= 3600.0:
        hours, rest = divmod(int(round(seconds)), 3600)
        return f"{hours} h {rest // 60:02d} min"
    if seconds >= 60.0:
        minutes, rest = divmod(int(round(seconds)), 60)
        return f"{minutes} min {rest:02d} s"
    return f"{seconds:.3g} s" if seconds < 1.0 else f"{seconds:.1f} s"


def estimate_time_constant(study):
    """Rough thermal time constant in seconds (heat capacity / (h · surface)) of the visible
    bodies, or None when a material is missing. The temperatures are within about 1 % of their
    final values after five time constants."""
    settings = study["settings"]
    capacity, area, emissive = 0.0, 0.0, 0.0
    for job in study_jobs(study):
        props = material_properties(job["material"])
        if props is None or not props.get("density"):
            return None
        V, T = job["V"], job["T"]
        volume = abs(_mesh_volume(V, T)) * 1e-9
        surface = _mesh_area(V, T) * 1e-6
        capacity += props["density"] * props["cp"] * volume
        area += surface
        emissivity = props["emissivity"] if job.get("emissivity") is None else job["emissivity"]
        emissive += emissivity * surface
    if capacity <= 0 or area <= 0:
        return None
    h = settings["h"] if settings["convection"] == "fixed" else 7.0
    if settings["radiation"]:
        h += 6.0 * emissive / area            # ≈ 4 σ T³ ε near room temperature
    return capacity / max(h * area, 1e-12)


def study_to_data(study):
    components = []
    for component in study["components"]:
        data = {key: copy.deepcopy(component[key]) for key in _COMPONENT_KEYS if key in component}
        data["bodies"] = [{
            "name": body["name"], "color": list(body["color"]), "opacity": body["opacity"],
            "fused": bool(body.get("fused", False)), "material": body.get("material"),
            "material_auto": bool(body.get("material_auto", False)),
            "power_w": float(body.get("power_w", 0.0)), "emissivity": body.get("emissivity"),
            "vertices": np.asarray(body["vertices"], float).tolist(),
            "triangles": [list(t) for t in body["triangles"]],
            "faces": [{"planar": bool(face.get("planar", False)),
                       "tris": [int(i) for i in face.get("tris", ())]}
                      for face in body.get("faces", [])]}
            for body in component["bodies"]]
        components.append(data)
    return {"name": study["name"], "settings": dict(study["settings"]), "components": components,
            "conditions": copy.deepcopy(study.get("conditions", [])),
            "probes": copy.deepcopy(study.get("probes", []))}


def study_from_data(data):
    study = study_new()
    if not data:
        return study
    study["name"] = str(data.get("name", study["name"]))
    study["settings"] = study_settings(data.get("settings"))
    for item in data.get("components", []):
        if item.get("kind") not in COMPONENT_KINDS:
            raise ValueError(f"invalid study component kind: {item.get('kind')!r}")
        component = {
            "name": str(item["name"]), "kind": item["kind"],
            "visible": bool(item.get("visible", True)),
            "position": [float(x) for x in item.get("position", (0, 0, 0))],
            "rotation": [float(x) for x in item.get("rotation", (0, 0, 0))],
            "pivot": [float(x) for x in item.get("pivot", (0, 0, 0))],
            "bodies": [thermal_body(body["name"], body["vertices"], body["triangles"],
                                    body.get("faces", []), body.get("color", cad.DEFAULT_SOLID_COLOR),
                                    body.get("opacity", 1.0), body.get("material"),
                                    body.get("power_w", 0.0), body.get("emissivity"),
                                    body.get("fused", False), body.get("material_auto", False))
                       for body in item.get("bodies", [])]}
        for key in ("source", "source_name", "signature", "path", "file_stamp", "file_root",
                    "shape"):
            if key in item:
                component[key] = copy.deepcopy(item[key])
        study["components"].append(component)
    study["conditions"] = [condition_settings(item) for item in data.get("conditions", [])]
    study["probes"] = [probe_settings(item) for item in data.get("probes", [])]
    return study


def study_save_file(study, path):
    if not os.fspath(path).lower().endswith(THERMAL_EXTENSION):
        raise ValueError(f"thermal study files must end in {THERMAL_EXTENSION}")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump({"format": THERMAL_FORMAT, "version": 1, "study": study_to_data(study)}, stream,
                  default=cad._json_default)


def study_load_file(path):
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or data.get("format") != THERMAL_FORMAT:
        raise ValueError("not a SimpleCAD thermal study file")
    return study_from_data(data.get("study"))


# light snapshots for undo / unsaved-changes tracking: they share the mesh arrays, which are
# never changed in place (a size change of a heat source creates new arrays)
_SHARED_BODY_KEYS = ("vertices", "triangles", "faces")


def study_state(study):
    components = []
    for component in study["components"]:
        data = {key: copy.deepcopy(value) for key, value in component.items()
                if key != "bodies" and not key.startswith("_")}
        data["bodies"] = [{key: (value if key in _SHARED_BODY_KEYS else copy.deepcopy(value))
                           for key, value in body.items() if not key.startswith("_")}
                          for body in component["bodies"]]
        components.append(data)
    return {"name": study["name"], "settings": dict(study["settings"]), "components": components,
            "conditions": copy.deepcopy(study.get("conditions", [])),
            "probes": copy.deepcopy(study.get("probes", []))}


def study_from_state(state):
    restored = study_state(state)          # copy, so the stored state stays untouched
    return restored


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return ("array", id(value))
    return value


def state_key(state):
    if state is None:
        return None
    components = []
    for component in state["components"]:
        bodies = tuple(tuple(sorted((key, ("shared", id(value)) if key in _SHARED_BODY_KEYS
                                     else _freeze(value)) for key, value in body.items()))
                       for body in component["bodies"])
        rest = _freeze({key: value for key, value in component.items() if key != "bodies"})
        components.append((rest, bodies))
    return (state["name"], _freeze(state["settings"]), tuple(components),
            _freeze(state.get("conditions", [])), _freeze(state.get("probes", [])))


# ── simulation input ──────────────────────────────────────────────────────────
def study_jobs(study):
    """The bodies the solver sees: every visible body (Bridge sources are inside their Bridge
    and left out) in world coordinates. Placed heat sources come last so they win overlaps."""
    jobs, primitives = [], []
    for ci, component in enumerate(study["components"]):
        if not component.get("visible", True):
            continue
        for bi, body in enumerate(component["bodies"]):
            if body.get("fused") or not len(body["triangles"]):
                continue
            label = component["name"] if len(component["bodies"]) == 1 else \
                f"{component['name']} / {body['name']}"
            job = {"key": (ci, bi), "name": label, "material": body.get("material"),
                   "power": float(body.get("power_w", 0.0)), "emissivity": body.get("emissivity"),
                   "V": cad.assembly_world_vertices(component, body["vertices"]),
                   "T": np.asarray(body["triangles"], np.int64).reshape(-1, 3),
                   "faces": [np.asarray(face.get("tris", ()), np.int64)
                             for face in body.get("faces", [])],
                   "primitive": component["kind"] in PRIMITIVE_KINDS}
            (primitives if job["primitive"] else jobs).append(job)
    return jobs + primitives


def _nice_up(value):
    """Round up to two significant digits (0.437 → 0.44)."""
    if value <= 0:
        return value
    scale = 10.0 ** (math.floor(math.log10(value)) - 1)
    return math.ceil(value / scale - 1e-9) * scale


def _nice(value):
    if value <= 0:
        return value
    scale = 10.0 ** (math.floor(math.log10(value)) - 1)
    return max(scale, round(value / scale) * scale)


GRID_CELL_LIMIT = 30_000_000        # cells of the whole grid incl. the air around the parts
AUTO_TARGET_CELLS = 150_000


def choose_cell_size(volumes, extents, span, settings):
    """(cell size in mm, notes). `volumes` mm³ and `extents` (bounding-box sizes) per body,
    `span` the size of all bodies together."""
    notes = []
    total = float(sum(volumes)) or 1.0
    max_cells = max(1000, int(settings.get("max_cells", DEFAULT_SETTINGS["max_cells"])))
    requested = float(settings.get("cell_mm", 0.0) or 0.0)
    if requested > 0:
        h = requested
    else:
        thin = [float(min(e)) for e in extents if min(e) > 1e-3]
        h = (total / AUTO_TARGET_CELLS) ** (1.0 / 3.0)
        if thin:
            h = min(h, min(thin) / 3.0)
        h = _nice(h)
    low = (total / max_cells) ** (1.0 / 3.0)
    for _ in range(3):
        grid = float(np.prod(np.asarray(span, float) + 3.0 * max(h, low)))
        low = max(low, (grid / GRID_CELL_LIMIT) ** (1.0 / 3.0))
    if h < low:
        h = _nice_up(low)
        notes.append(f"Cell size raised to {h:g} mm to stay within {max_cells:,} cells "
                     "(Max cells in the settings).".replace(",", " "))
    return h, notes


def estimate_cells(study):
    """(cell size mm, estimated solid cells) for the current settings, or None."""
    volumes, extents, lows, highs = [], [], [], []
    for job in study_jobs(study):
        V = job["V"]
        low, high = V.min(0), V.max(0)
        volume = abs(_mesh_volume(V, job["T"]))
        volumes.append(volume if volume > 0 else float(np.prod(high - low)))
        extents.append(high - low); lows.append(low); highs.append(high)
    if not volumes:
        return None
    span = np.max(highs, axis=0) - np.min(lows, axis=0)
    h, _ = choose_cell_size(volumes, extents, span, study["settings"])
    return h, int(sum(volumes) / h ** 3)


# ── solver ────────────────────────────────────────────────────────────────────
class ThermalCancelled(Exception):
    pass


class ThermalError(Exception):
    """A problem with the study setup, shown to the user as it is."""


_S1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
_M1 = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0
# conduction matrices of a unit cube (multiply by k·h); corner c = 4·x + 2·y + z
_KX = np.kron(_S1, np.kron(_M1, _M1))
_KY = np.kron(_M1, np.kron(_S1, _M1))
_KZ = np.kron(_M1, np.kron(_M1, _S1))
_CORNERS = np.array([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)])
_FACE_MASS = np.kron(_M1, _M1)         # bilinear quad of unit area
_OFFSETS = np.array([(ox, oy, oz) for ox in (-1, 0, 1) for oy in (-1, 0, 1) for oz in (-1, 0, 1)])
_RAY_JITTER = (0.0012345, 0.0023457, 0.0034569)   # cell units; keeps rays off mesh edges


def _offset_index(delta):
    return int((delta[0] + 1) * 9 + (delta[1] + 1) * 3 + (delta[2] + 1))


def _voxel_parity(P, shape, axis, pair_limit=4_000_000):
    """Cells of a grid of `shape` whose centres lie inside the closed triangle soup P (m, 3, 3),
    given in cell units (centre of cell (i, j, k) at (i, j, k)), by counting the crossings of
    rays parallel to `axis`."""
    a, b = [ax for ax in range(3) if ax != axis]
    A = P[:, :, a] - _RAY_JITTER[a]
    B = P[:, :, b] - _RAY_JITTER[b]
    C = P[:, :, axis]
    area = (A[:, 1] - A[:, 0]) * (B[:, 2] - B[:, 0]) - (A[:, 2] - A[:, 0]) * (B[:, 1] - B[:, 0])
    keep = np.abs(area) > 1e-12
    A, B, C, area = A[keep], B[keep], C[keep], area[keep]
    toggles = np.zeros(shape, np.uint8)
    if not len(area):
        return toggles.astype(bool)
    ia0 = np.maximum(np.ceil(A.min(1)), 0).astype(np.int64)
    ia1 = np.minimum(np.floor(A.max(1)), shape[a] - 1).astype(np.int64)
    ib0 = np.maximum(np.ceil(B.min(1)), 0).astype(np.int64)
    ib1 = np.minimum(np.floor(B.max(1)), shape[b] - 1).astype(np.int64)
    na, nb = ia1 - ia0 + 1, ib1 - ib0 + 1
    count = np.where((na > 0) & (nb > 0), na * nb, 0)
    cum = np.cumsum(count)
    start = 0
    while start < len(count):
        base = int(cum[start - 1]) if start else 0
        end = int(np.searchsorted(cum, base + pair_limit, side="right"))
        end = max(end, start + 1)
        tri = np.repeat(np.arange(start, end), count[start:end])
        start, total = end, len(tri)
        if not total:
            continue
        offsets = base + np.arange(total) - (cum[tri] - count[tri])   # pair number per triangle
        ia = ia0[tri] + offsets // nb[tri]
        ib = ib0[tri] + offsets % nb[tri]
        a0, a1, a2 = A[tri, 0], A[tri, 1], A[tri, 2]
        b0, b1, b2 = B[tri, 0], B[tri, 1], B[tri, 2]
        w0 = (a1 - a0) * (ib - b0) - (b1 - b0) * (ia - a0)     # weight of vertex 2
        w1 = (a2 - a1) * (ib - b1) - (b2 - b1) * (ia - a1)     # weight of vertex 0
        w2 = (a0 - a2) * (ib - b2) - (b0 - b2) * (ia - a2)     # weight of vertex 1
        inside = ((w0 > 0) & (w1 > 0) & (w2 > 0)) | ((w0 < 0) & (w1 < 0) & (w2 < 0))
        if not inside.any():
            continue
        tri, ia, ib = tri[inside], ia[inside], ib[inside]
        w0, w1, w2 = w0[inside], w1[inside], w2[inside]
        t = (w1 * C[tri, 0] + w2 * C[tri, 1] + w0 * C[tri, 2]) / area[tri]
        k = np.maximum(np.floor(t).astype(np.int64) + 1, 0)
        valid = k < shape[axis]
        index = [None, None, None]
        index[a], index[b], index[axis] = ia[valid], ib[valid], k[valid]
        np.add.at(toggles, tuple(index), 1)
    # a cell is inside when an odd number of crossings lies below its centre (uint8 wrap-around
    # keeps the parity)
    return (np.cumsum(toggles, axis=axis, dtype=np.uint8) & 1).astype(bool)


def voxelize(V_cells, shape):
    """Inside cells of a mesh (vertices in cell units, see _voxel_parity): majority of three
    ray directions, so a mesh with small holes or doubled faces still gives a solid."""
    votes = np.zeros(shape, np.uint8)
    for axis in range(3):
        votes += _voxel_parity(V_cells, shape, axis)
    return votes >= 2


def _natural_h(dT, length, orientation):
    """Natural-convection coefficient W/(m²·K) of air (simplified correlations, Holman):
    orientation 0 = vertical, 1 = hot surface facing up, 2 = facing down. `length` in m."""
    dT = np.maximum(dT, 0.05)
    ra = 9.3e7 * dT * length ** 3                     # Rayleigh number of air near 300 K
    vertical = np.where(ra < 1e9, 1.42 * (dT / length) ** 0.25, 1.31 * dT ** (1.0 / 3.0))
    up = np.where(ra < 1e7, 1.32 * (dT / length) ** 0.25, 1.52 * dT ** (1.0 / 3.0))
    down = 0.59 * (dT / length) ** 0.25
    return np.select([orientation == 0, orientation == 1], [vertical, up], down)


def solve_thermal(jobs, settings, progress=None, cancelled=None, conditions=()):
    """Temperatures of the study bodies: steady state, or over time (settings "analysis").
    `conditions` are the face conditions of study_conditions(). Returns a result dict (see the
    end); temperature fields are stored as frames of the result-surface nodes (one frame for
    steady state)."""
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        import scipy.ndimage as ndi
        from scipy.spatial import cKDTree
    except ImportError as ex:
        raise ThermalError("The thermal solver needs scipy. Install it with:  pip install scipy") from ex
    started = time.perf_counter()
    messages = []

    def step(percent, text):
        if cancelled is not None and cancelled():
            raise ThermalCancelled()
        if progress is not None:
            progress(int(percent), text)

    if not jobs:
        raise ThermalError("There is nothing to simulate: import or add visible parts first.")
    settings = study_settings(settings)
    ambient = settings["ambient_c"]
    step(2, "Preparing bodies...")

    # 1. bodies, materials, cell size -------------------------------------------------
    bodies = []
    for job in jobs:
        props = material_properties(job["material"])
        if props is None:
            raise ThermalError(f"'{job['name']}' has no material. Choose one in the properties.")
        V, T = np.asarray(job["V"], float), np.asarray(job["T"], np.int64)
        low, high = V.min(0), V.max(0)
        extent = high - low
        volume = abs(_mesh_volume(V, T))
        bbox_volume = float(np.prod(np.maximum(extent, 1e-6)))
        if volume <= 1e-9 or volume > 1.01 * bbox_volume:
            messages.append(f"⚠ '{job['name']}' is not a closed mesh; its volume is estimated.")
            volume = 0.0
        k = np.full(3, props["k"])
        if props.get("k_plane"):
            k[:] = props["k_plane"]
            k[int(np.argmin(extent))] = props["k"]          # through-plane = thinnest axis
        emissivity = props["emissivity"] if job.get("emissivity") is None else float(job["emissivity"])
        bodies.append({"job": job, "name": job["name"], "material": job["material"],
                       "props": props, "V": V, "T": T, "low": low, "high": high,
                       "extent": extent, "volume": volume, "bbox_volume": bbox_volume,
                       "area": _mesh_area(V, T), "k": k, "emissivity": emissivity,
                       "power": max(0.0, float(job.get("power", 0.0)))})
    total_power = sum(body["power"] for body in bodies)
    conditions = [condition for condition in (conditions or ())]
    has_fixed = any(c["kind"] == "fixed" and c["targets"] for c in conditions)
    has_flux = any(c["kind"] == "flux" and c["targets"] for c in conditions)
    if total_power <= 0 and not has_flux and not has_fixed:
        raise ThermalError("No heat source: give a body a power (W), add a heat source, or put a "
                           "heat flux or fixed temperature on faces.")
    if settings["convection"] == "fixed" and settings["h"] <= 0 and not settings["radiation"] \
            and not has_fixed:
        raise ThermalError("Without convection (h = 0) and radiation the heat cannot leave.")
    low = np.min([body["low"] for body in bodies], axis=0)
    high = np.max([body["high"] for body in bodies], axis=0)
    h_mm, notes = choose_cell_size([body["volume"] or body["bbox_volume"] for body in bodies],
                                   [body["extent"] for body in bodies], high - low, settings)
    messages += notes
    h = h_mm / 1000.0
    origin = low - h_mm                        # one cell of air around everything
    shape = tuple(int(n) for n in np.ceil((high - origin) / h_mm).astype(int) + 1)

    # 2. voxels ---------------------------------------------------------------------------
    owner = np.full(shape, -1, np.int16)
    for index, body in enumerate(bodies):
        step(5 + 30 * index / len(bodies), f"Meshing {body['name']}...")
        first = np.maximum(np.ceil((body["low"] - origin) / h_mm - 0.5).astype(int), 0)
        last = np.minimum(np.floor((body["high"] - origin) / h_mm - 0.5).astype(int),
                          np.array(shape) - 1)
        body["cells"] = 0
        if np.any(last < first):
            continue
        sub_shape = tuple(int(n) for n in last - first + 1)
        P = (body["V"][body["T"]] - (origin + (first + 0.5) * h_mm)) / h_mm
        inside = voxelize(P, sub_shape)
        region = tuple(slice(int(f), int(l) + 1) for f, l in zip(first, last))
        sub = owner[region]
        replaced = sub[inside]
        replaced = replaced[replaced >= 0]
        if len(replaced):
            for other, cells in enumerate(np.bincount(replaced, minlength=len(bodies))):
                if cells:
                    messages.append(f"'{bodies[other]['name']}' and '{body['name']}' overlap "
                                    f"({cells * h_mm ** 3:.4g} mm³); the overlap belongs to "
                                    f"'{body['name']}'.")
        sub[inside] = index
    step(36, "Finding enclosed air...")
    solid = owner >= 0
    labels, _count = ndi.label(~solid)
    border = np.unique(np.concatenate([labels[0].ravel(), labels[-1].ravel(), labels[:, 0].ravel(),
                                       labels[:, -1].ravel(), labels[:, :, 0].ravel(),
                                       labels[:, :, -1].ravel()]))
    cavity = (labels > 0) & ~np.isin(labels, border)
    del labels
    air_index = len(bodies)
    cavity_cells = int(cavity.sum())
    if cavity_cells:
        owner[cavity] = air_index
        messages.append(f"Air enclosed in a housing ({cavity_cells * h_mm ** 3 / 1000.0:.4g} cm³) "
                        "conducts heat as still air (no airflow in Conduction mode).")
    del cavity, solid
    I, J, K = np.nonzero(owner >= 0)
    element_body = owner[I, J, K].astype(np.int64)
    n_elements = len(I)
    cells = np.bincount(element_body, minlength=air_index + 1)
    for index, body in enumerate(bodies):
        body["cells"] = int(cells[index])
        if not body["cells"]:
            if body["power"] > 0:
                raise ThermalError(f"The heat source '{body['name']}' is thinner than one cell "
                                   f"({h_mm:g} mm). Choose a smaller cell size.")
            messages.append(f"⚠ '{body['name']}' is thinner than one cell ({h_mm:g} mm) and is "
                            "left out. Choose a smaller cell size.")
            continue
        if body["volume"] > 0:
            error = body["cells"] * h_mm ** 3 / body["volume"] - 1.0
            body["volume_error"] = error
            if abs(error) > 0.2:
                messages.append(f"⚠ '{body['name']}' is coarsely resolved (volume {error:+.0%}); "
                                "a smaller cell size gives better results.")
    if n_elements == 0:
        raise ThermalError("The bodies are thinner than one cell. Choose a smaller cell size.")

    # 3. finite elements ----------------------------------------------------------------
    step(40, f"Assembling {n_elements:,} elements...".replace(",", " "))
    ny1, nz1 = shape[1] + 1, shape[2] + 1
    corner_offset = (_CORNERS[:, 0] * ny1 + _CORNERS[:, 1]) * nz1 + _CORNERS[:, 2]
    base = (I.astype(np.int64) * ny1 + J) * nz1 + K
    nodes_global, conn = np.unique(base[:, None] + corner_offset[None, :], return_inverse=True)
    conn = conn.reshape(n_elements, 8)
    n_nodes = len(nodes_global)
    k_table = np.array([body["k"] for body in bodies] + [[AIR_CONDUCTIVITY] * 3])
    kx, ky, kz = (k_table[element_body, axis] * h for axis in range(3))
    pairs = collections.defaultdict(list)
    for a in range(8):
        for b in range(8):
            pairs[_offset_index(_CORNERS[b] - _CORNERS[a])].append((a, b))
    present = np.zeros((27, n_nodes), bool)
    values = np.zeros((27, n_nodes))
    for offset, offset_pairs in pairs.items():
        rows = np.concatenate([conn[:, a] for a, _b in offset_pairs])
        weights = np.concatenate([kx * _KX[a, b] + ky * _KY[a, b] + kz * _KZ[a, b]
                                  for a, b in offset_pairs])
        values[offset] = np.bincount(rows, weights, n_nodes)
        present[offset] = np.bincount(rows, minlength=n_nodes) > 0
    rank = (np.cumsum(present, axis=0) - present).astype(np.int32)    # entries before an offset
    indptr = np.zeros(n_nodes + 1, np.int64)
    np.cumsum(present.sum(0), out=indptr[1:])
    nnz = int(indptr[-1])
    indices = np.empty(nnz, np.int32)
    data_k = np.empty(nnz)
    global_offset = (_OFFSETS[:, 0] * ny1 + _OFFSETS[:, 1]) * nz1 + _OFFSETS[:, 2]
    for offset in range(27):
        rows = np.nonzero(present[offset])[0]
        slots = indptr[rows] + rank[offset, rows]
        indices[slots] = np.searchsorted(nodes_global, nodes_global[rows] + global_offset[offset])
        data_k[slots] = values[offset, rows]
    del values
    # heat sources: the power of a body is spread evenly over its cells
    element_power = np.zeros(n_elements)
    for index, body in enumerate(bodies):
        if body["power"] > 0 and body["cells"]:
            element_power[element_body == index] = body["power"] / body["cells"]
    load_source = np.bincount(conn.ravel(), np.repeat(element_power / 8.0, 8), n_nodes)

    # exposed faces (towards outside air)
    padded = np.pad(owner, 1, constant_values=-1)
    face_element, face_dir = [], []
    stair_area = np.zeros(air_index + 1)
    for direction in range(6):
        axis, side = divmod(direction, 2)
        step_vector = np.zeros(3, int); step_vector[axis] = 1 if side else -1
        neighbour = padded[I + 1 + step_vector[0], J + 1 + step_vector[1], K + 1 + step_vector[2]]
        stair_area += np.bincount(element_body[neighbour != element_body], minlength=air_index + 1)
        exposed = np.nonzero(neighbour == -1)[0]
        face_element.append(exposed); face_dir.append(np.full(len(exposed), direction))
    del padded
    face_element = np.concatenate(face_element)
    face_dir = np.concatenate(face_dir)
    face_body = element_body[face_element]
    face_corners = np.array([[c for c in range(8) if _CORNERS[c][axis] == side]
                             for axis in range(3) for side in (0, 1)])       # (6, 4)
    face_nodes = conn[face_element[:, None], face_corners[face_dir]]          # (nf, 4)
    # matrix slots of the 16 node pairs of every face
    face_slots = np.empty((len(face_element), 16), np.int64)
    for direction in range(6):
        selected = np.nonzero(face_dir == direction)[0]
        corners = face_corners[direction]
        for p in range(4):
            rows = face_nodes[selected, p]
            for q in range(4):
                offset = _offset_index(_CORNERS[corners[q]] - _CORNERS[corners[p]])
                face_slots[selected, 4 * p + q] = indptr[rows] + rank[offset, rows]
    diagonal_slots = indptr[:-1] + rank[_offset_index((0, 0, 0))]     # heat capacity goes here
    del rank, present
    # staircase correction: the voxel surface of a slanted or curved face is larger than the
    # real one; each body's exposed faces are scaled by (mesh area / voxel surface area)
    correction = np.ones(air_index + 1)
    for index, body in enumerate(bodies):
        if stair_area[index] > 0 and body["area"] > 0:
            correction[index] = min(1.5, max(0.3, body["area"] / (stair_area[index] * h_mm ** 2)))
    face_area = h * h * correction[face_body]                                  # m²
    orientation = np.where(face_dir == 5, 1, np.where(face_dir == 4, 2, 0))   # +Z up, -Z down
    vertical_length = np.array([max(body["extent"][2], 5.0) / 1000.0 for body in bodies] + [1.0])
    horizontal_length = np.array([
        max(body["extent"][0] * body["extent"][1] /
            (2.0 * (body["extent"][0] + body["extent"][1]) + 1e-12), 2.0) / 1000.0
        for body in bodies] + [1.0])
    face_length = np.where(orientation == 0, vertical_length[face_body],
                           horizontal_length[face_body])
    face_emissivity = np.array([body["emissivity"] for body in bodies] + [0.0])[face_body]
    exposed_area = np.bincount(face_body, face_area, air_index + 1)
    if exposed_area.sum() <= 0:
        raise ThermalError("No surface is exposed to the air.")
    ambient_k = ambient + 273.15

    # face conditions: every exposed cell face belongs to the nearest mesh face of its body that
    # looks the same way, and takes the condition of that face (the last condition wins)
    n_faces = len(face_element)
    face_condition = np.full(n_faces, -1)
    if conditions:
        step(38, "Applying face conditions...")
        face_normal = np.zeros((n_faces, 3))
        face_normal[np.arange(n_faces), face_dir // 2] = np.where(face_dir % 2 == 1, 1.0, -1.0)
        cell = np.stack([I, J, K], axis=1)[face_element]
        face_centre = origin + (cell + 0.5) * h_mm + 0.5 * h_mm * face_normal
        mesh_face = np.full(n_faces, -1)
        for b in sorted({b for condition in conditions for b in condition["targets"]}):
            body = bodies[b]
            selected = np.nonzero(face_body == b)[0]
            if not len(selected):
                continue
            tri_face = np.full(len(body["T"]), -1)
            for face_index, tris in enumerate(body["job"].get("faces", [])):
                tri_face[np.asarray(tris, np.int64)] = face_index
            P = body["V"][body["T"]]
            normal = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
            samples, _triangles, source = refine_triangles(body["V"], body["T"], h_mm,
                                                           with_source=True)
            count = min(8, len(samples))
            distance, nearest = cKDTree(samples).query(face_centre[selected], k=count)
            distance, nearest = distance.reshape(len(selected), -1), nearest.reshape(len(selected), -1)
            fits = ((normal[source[nearest]] * face_normal[selected][:, None, :]).sum(2) > 0.3) & \
                (distance < 2.0 * h_mm)
            first = np.argmax(fits, axis=1)
            rows = np.arange(len(selected))
            found = fits[rows, first]
            mesh_face[selected[found]] = tri_face[source[nearest[rows, first]]][found]
        for index, condition in enumerate(conditions):
            mask = np.zeros(n_faces, bool)
            for b, faces in condition["targets"].items():
                mask |= (face_body == b) & np.isin(mesh_face, faces)
            face_condition[mask] = index
        for index, condition in enumerate(conditions):
            if condition["targets"] and not (face_condition == index).any():
                messages.append(f"⚠ Face condition '{condition['name']}' reaches no outside "
                                "surface (its faces touch another body or are inside a housing).")

    def condition_mask(kind):
        return np.isin(face_condition, [i for i, c in enumerate(conditions) if c["kind"] == kind])

    insulated_faces = condition_mask("insulated")
    fixed_faces = condition_mask("fixed")
    own_convection = condition_mask("convection")
    no_exchange = insulated_faces | fixed_faces
    own_h = np.zeros(n_faces)
    face_ambient = np.full(n_faces, float(ambient))       # air temperature of every face
    load_flux = np.zeros(n_nodes)
    fixed_value = np.full(n_nodes, np.nan)
    fixed_owner = np.full(n_nodes, -1)
    flux_power = np.zeros(len(conditions))
    for index, condition in enumerate(conditions):
        mask = face_condition == index
        if not mask.any():
            continue
        if condition["kind"] == "convection":
            own_h[mask] = max(0.0, condition["h"])
            face_ambient[mask] = ambient if condition["fluid_ambient"] else condition["fluid_c"]
        elif condition["kind"] == "flux":
            area = float(face_area[mask].sum())
            density = condition["power_w"] / area if condition["flux_mode"] == "total" \
                else condition["flux_w_m2"]
            flux_power[index] = density * area
            load_flux += np.bincount(face_nodes[mask].ravel(),
                                     np.repeat(density * face_area[mask] / 4.0, 4), n_nodes)
        elif condition["kind"] == "fixed":
            nodes = face_nodes[mask].ravel()
            fixed_value[nodes] = condition["temperature_c"]
            fixed_owner[nodes] = index
    fixed_nodes = np.nonzero(np.isfinite(fixed_value))[0]
    total_in = total_power + float(flux_power.sum())

    def face_coefficients(T):
        """(convection h, radiation h) per exposed face for the nodal temperatures T."""
        face_t = T[face_nodes].mean(1)
        if settings["convection"] == "fixed":
            convection = np.full(len(face_t), max(0.0, settings["h"]))
        else:
            mean = np.bincount(face_body, face_t * face_area, air_index + 1) / \
                np.maximum(exposed_area, 1e-30)
            convection = _natural_h(mean[face_body] - ambient, face_length, orientation)
        radiation = np.zeros(len(face_t))
        if settings["radiation"]:
            tk = face_t + 273.15
            radiation = face_emissivity * STEFAN_BOLTZMANN * (tk ** 2 + ambient_k ** 2) * (tk + ambient_k)
        if conditions:
            convection = np.where(own_convection, own_h, convection)
            convection[no_exchange] = 0.0
            radiation[no_exchange] = 0.0
        return convection, radiation

    # 4. result surfaces -----------------------------------------------------------------
    # CAD meshes have large triangles, so the body surfaces are subdivided to about the cell
    # size; every new vertex is interpolated from (up to) 8 nodes with fixed weights, so each
    # stored temperature field (one, or one per frame of a transient run) gives its colours
    step(43, "Preparing the result surfaces...")
    node_index = np.empty((n_nodes, 3), np.int64)
    node_index[:, 0] = nodes_global // (ny1 * nz1)
    node_index[:, 1] = (nodes_global // nz1) % ny1
    node_index[:, 2] = nodes_global % nz1
    node_points = origin + node_index * h_mm
    total_area = sum(body["area"] for body in bodies)
    edge = max(h_mm, math.sqrt(2.0 * total_area / DISPLAY_TRIANGLE_LIMIT))
    surface, tree = {}, None
    for body in bodies:
        vertices, triangles = refine_triangles(body["V"], body["T"], edge)
        index, weights = _surface_weights(vertices, origin, h_mm, shape, nodes_global)
        lost = weights.sum(1) <= 0
        if lost.any():                       # no node around: take the nearest one
            if tree is None:
                tree = cKDTree(node_points)
            _distance, nearest = tree.query(vertices[lost])
            index[lost] = nearest[:, None]
            weights[lost] = 0.0
            weights[lost, 0] = 1.0
        surface[body["job"]["key"]] = {"vertices": vertices.astype(np.float32),
                                       "triangles": triangles.astype(np.int32),
                                       "index": index, "weights": weights}
    used = np.unique(np.concatenate([item["index"].ravel() for item in surface.values()]))
    for item in surface.values():          # index into the stored (surface) node temperatures
        item["index"] = np.searchsorted(used, item["index"]).astype(np.int32)
    body_nodes = [np.unique(conn[element_body == index]) if body["cells"] else None
                  for index, body in enumerate(bodies)]
    body_elements = [conn[element_body == index] if body["cells"] else None
                     for index, body in enumerate(bodies)]

    def body_stats(T):
        """(3, bodies): minimum, mean and maximum temperature of every body."""
        out = np.full((3, len(bodies)), np.nan)
        for index in range(len(bodies)):
            if body_nodes[index] is not None:
                values = T[body_nodes[index]]
                out[:, index] = (values.min(), T[body_elements[index]].mean(), values.max())
        return out

    # 5. solve ----------------------------------------------------------------------------
    cg_iterations = [0]

    def cg_callback(_x):
        cg_iterations[0] += 1
        if cg_iterations[0] % 25 == 0 and cancelled is not None and cancelled():
            raise ThermalCancelled()

    face_mass = _FACE_MASS.ravel()

    def system(convection_weight, radiation_weight):
        """Matrix data and right-hand side for the surface coefficients · areas."""
        weight = convection_weight + radiation_weight
        data = data_k + np.bincount(face_slots.ravel(), (weight[:, None] * face_mass[None, :]).ravel(), nnz)
        outside = (convection_weight * face_ambient + radiation_weight * ambient) / 4.0
        rhs = load_source + load_flux + np.bincount(face_nodes.ravel(), np.repeat(outside, 4), n_nodes)
        return data, rhs

    # fixed temperatures: their rows and columns leave the system (identity rows), their known
    # values move to the right-hand side, so the matrix stays symmetric for the CG solver
    if len(fixed_nodes):
        fixed_mask = np.zeros(n_nodes, bool)
        fixed_mask[fixed_nodes] = True
        entry_row = np.repeat(np.arange(n_nodes), np.diff(indptr))
        fixed_entries = fixed_mask[entry_row] | fixed_mask[indices]
        del entry_row
        fixed_vector = np.zeros(n_nodes)
        fixed_vector[fixed_nodes] = fixed_value[fixed_nodes]

    def constrain(data, rhs):
        if not len(fixed_nodes):
            return data, rhs
        matrix = sp.csr_matrix((data, indices, indptr), shape=(n_nodes, n_nodes))
        rhs = rhs - matrix @ fixed_vector
        rhs[fixed_nodes] = fixed_vector[fixed_nodes]
        data = data.copy()
        data[fixed_entries] = 0.0
        data[diagonal_slots[fixed_nodes]] = 1.0
        return data, rhs

    # two-level preconditioner: Jacobi plus a coarse correction on blocks of grid nodes (solved
    # directly). Jacobi alone handles local errors; the coarse part handles the smooth, global
    # ones, which dominate in well-conducting metal parts.
    prolongation = _coarse_space(node_index, COARSE_BLOCK) if COARSE_BLOCK > 1 else None
    coarse = {"lu": None, "calls": 0}

    def solve(data, rhs, start, tolerance, refresh=True):
        matrix = sp.csr_matrix((data, indices, indptr), shape=(n_nodes, n_nodes))
        inverse_diagonal = 1.0 / data[diagonal_slots]
        if prolongation is not None:
            if refresh or coarse["lu"] is None:
                coarse_matrix = (prolongation.T @ matrix @ prolongation).tocsc()
                coarse["lu"] = spla.splu(coarse_matrix)
            lu = coarse["lu"]

            def apply(v):
                return inverse_diagonal * v + prolongation @ lu.solve(prolongation.T @ v)
        else:
            def apply(v):
                return inverse_diagonal * v
        preconditioner = spla.LinearOperator(matrix.shape, matvec=apply, dtype=float)
        cg_iterations[0] = 0
        solution, info = spla.cg(matrix, rhs, x0=start, rtol=tolerance, atol=0.0,
                                 maxiter=max(3000, 4 * int(round(max(shape) * 10))),
                                 M=preconditioner, callback=cg_callback)
        if not np.all(np.isfinite(solution)):
            raise ThermalError("The solver did not converge (a body may have no way to lose heat).")
        return solution, info, cg_iterations[0]

    transient = settings["analysis"] == "transient"
    nonlinear = settings["convection"] == "natural" or settings["radiation"]
    cg_total, solver_short, q_stored = 0, False, 0.0
    if not transient:
        # Picard iteration over the temperature-dependent surface coefficients
        T = np.full(n_nodes, ambient + total_in / (10.0 * exposed_area.sum()))
        T[fixed_nodes] = fixed_value[fixed_nodes]
        convection_c = radiation_c = None
        change, relaxation, previous_change = float("inf"), 1.0, float("inf")
        iterations = 0
        for iteration in range(40 if nonlinear else 1):
            step(45 + 50 * min(1.0, iteration / 8.0),
                 f"Solving {n_nodes:,} nodes — iteration {iteration + 1}".replace(",", " ") +
                 ("" if not iteration else f", change {change:.3g} K"))
            convection, radiation = face_coefficients(T)
            if change > previous_change:          # oscillating: damp the coefficient update
                relaxation = max(0.3, relaxation * 0.6)
            previous_change = change
            if convection_c is None:
                convection_c, radiation_c = convection, radiation
            else:
                convection_c = (1.0 - relaxation) * convection_c + relaxation * convection
                radiation_c = (1.0 - relaxation) * radiation_c + relaxation * radiation
            last_system = system(convection_c * face_area, radiation_c * face_area)
            data, rhs = constrain(*last_system)
            # loose while the surface coefficients still change, tight near the end
            tolerance = 1e-9 if not nonlinear or change < 0.05 else 1e-6
            solution, info, steps_used = solve(data, rhs, T, tolerance)
            cg_total += steps_used
            solver_short |= info > 0
            change = float(np.abs(solution - T).max())
            T = solution
            iterations = iteration + 1
            if not nonlinear or (iteration >= 1 and change < 0.005 and tolerance < 1e-8):
                break
        if nonlinear and change >= 0.005:
            messages.append(f"⚠ The surface coefficients did not fully settle (last change "
                            f"{change:.3g} K).")
        frame_steps = np.array([0])
        frame_times = np.array([0.0])
        frames = T[used][None, :].astype(np.float32)
        hottest = int(np.argmax(T))
        hot_points = node_points[hottest][None, :]
        hot_values = np.array([T[hottest]])
        t_min, t_max = float(T.min()), float(T.max())
        series = None
    else:
        # implicit (backward) Euler: (C/Δt + K + H) T₁ = C/Δt T₀ + q; the surface coefficients
        # of a step are taken from the temperatures at its start
        duration, n_steps = settings["duration_s"], settings["time_steps"]
        dt = duration / n_steps
        capacity = np.array([body["props"]["density"] * body["props"]["cp"] for body in bodies] +
                            [AIR_DENSITY * AIR_HEAT_CAPACITY])                     # J/(m³·K)
        node_capacity = np.bincount(conn.ravel(), np.repeat(capacity[element_body] * h ** 3 / 8.0, 8),
                                    n_nodes) / dt                                    # W/K
        start = ambient if settings["initial_ambient"] else settings["initial_c"]
        T = np.full(n_nodes, float(start))
        T[fixed_nodes] = fixed_value[fixed_nodes]          # held from the start
        frame_steps = np.unique(np.round(np.linspace(0, n_steps, min(n_steps, MAX_FRAMES) + 1))
                                .astype(int))
        frame_of_step = {int(s): f for f, s in enumerate(frame_steps)}
        frame_times = frame_steps * dt
        frames = np.empty((len(frame_steps), len(used)), np.float32)
        hot_points = np.empty((len(frame_steps), 3))
        hot_values = np.empty(len(frame_steps))
        stats = np.empty((n_steps + 1, 3, len(bodies)))
        t_min, t_max = float(T.min()), float(T.max())

        def record(step_index, T):
            stats[step_index] = body_stats(T)
            frame = frame_of_step.get(step_index)
            if frame is not None:
                frames[frame] = T[used]
                hottest = int(np.argmax(T))
                hot_points[frame] = node_points[hottest]
                hot_values[frame] = T[hottest]

        record(0, T)
        previous = T
        for step_index in range(1, n_steps + 1):
            step(45 + 50 * (step_index - 1) / n_steps,
                 f"Time step {step_index} of {n_steps} — t = {format_time(step_index * dt)}")
            convection, radiation = face_coefficients(T)
            data, rhs = system(convection * face_area, radiation * face_area)
            data[diagonal_slots] += node_capacity
            rhs += node_capacity * T
            last_system = (data, rhs)
            data, rhs = constrain(data, rhs)
            # start from the linear extrapolation of the last two steps (far fewer iterations)
            guess = T if step_index == 1 else T + (T - previous)
            previous = T
            # the surface coefficients change slowly: the coarse factorisation is renewed now
            # and then only
            T, info, steps_used = solve(data, rhs, guess, TRANSIENT_TOLERANCE,
                                        refresh=step_index % 10 == 1)
            cg_total += steps_used
            solver_short |= info > 0
            t_min, t_max = min(t_min, float(T.min())), max(t_max, float(T.max()))
            record(step_index, T)
        q_stored = float((node_capacity * (T - previous)).sum())       # W, heat still stored
        iterations = n_steps
        series = {"times": np.arange(n_steps + 1) * dt, "min": stats[:, 0, :].T,
                  "mean": stats[:, 1, :].T, "max": stats[:, 2, :].T}
    if solver_short:
        messages.append("⚠ The linear solver stopped before full accuracy; the result may be "
                        "slightly off.")

    # 6. results --------------------------------------------------------------------------
    step(97, "Evaluating results...")
    convection, radiation = face_coefficients(T)
    face_t = T[face_nodes].mean(1)
    convection_flow = convection * face_area * (face_t - face_ambient)       # W per face
    radiation_flow = radiation * face_area * (face_t - ambient)
    q_convection = float(convection_flow.sum())
    q_radiation = float(radiation_flow.sum())
    # heat put in at fixed-temperature nodes to hold them = residual of their rows in the full
    # system (negative: heat leaves the parts there)
    reaction = np.zeros(n_nodes)
    if len(fixed_nodes):
        full = sp.csr_matrix((last_system[0], indices, indptr), shape=(n_nodes, n_nodes))
        reaction[fixed_nodes] = (full @ T - last_system[1])[fixed_nodes]
    q_fixed = -float(reaction.sum())                    # leaving through fixed-temperature faces
    q_out = q_convection + q_radiation + q_fixed
    scale = max(total_in, abs(q_convection) + abs(q_radiation) + float(np.abs(reaction).sum()), 1e-9)
    balance_error = (q_out + q_stored - total_in) / scale
    if abs(balance_error) > (0.05 if transient else 0.02):
        messages.append(f"⚠ Heat balance off by {balance_error:+.1%}; the result is not converged"
                        + (" (use more time steps)." if transient else "."))
    condition_results = []
    for index, condition in enumerate(conditions):
        mask = face_condition == index
        heat = {"fixed": -float(reaction[fixed_owner == index].sum()),       # + = leaves
                "flux": float(flux_power[index]),
                "convection": float(convection_flow[mask].sum()),
                "insulated": 0.0}[condition["kind"]]
        condition_results.append({"name": condition["name"], "kind": condition["kind"],
                                  "cell_faces": int(mask.sum()), "heat": heat,
                                  "area_mm2": float(face_area[mask].sum() * 1e6)})
    final = body_stats(T)
    body_results = []
    for index, body in enumerate(bodies):
        entry = {"key": body["job"]["key"], "name": body["name"], "material": body["material"],
                 "power": body["power"], "cells": body["cells"], "max_c": body["props"].get("max_c"),
                 "volume_error": body.get("volume_error"), "min": None, "mean": None, "max": None,
                 "convection": float(convection_flow[face_body == index].sum()),
                 "radiation": float(radiation_flow[face_body == index].sum())}
        if body_nodes[index] is not None:
            entry.update(min=float(final[0, index]), mean=float(final[1, index]),
                         max=float(final[2, index]))
        body_results.append(entry)
    elapsed = time.perf_counter() - started
    summary = [f"Mesh: cell size {h_mm:g} mm, {n_elements:,} elements, {n_nodes:,} nodes, "
               f"{len(face_element):,} exposed faces".replace(",", " ")]
    fixed_in = float(reaction[reaction > 0].sum())
    fixed_out = -float(reaction[reaction < 0].sum())
    sources = " + ".join([f"sources {total_power:.4g} W"] +
                         ([f"flux {flux_power.sum():.4g} W"] if flux_power.any() else []) +
                         ([f"fixed-temperature faces {fixed_in:.4g} W"] if fixed_in > 1e-9 else []))
    fixed_text = f", fixed-temperature faces {fixed_out:.4g} W" if fixed_out > 1e-9 else ""
    if transient:
        summary += [f"Transient: {format_time(duration)} in {n_steps} steps of {format_time(dt)}, "
                    f"start at {start:g} °C",
                    f"Solved in {elapsed:.1f} s ({cg_total} solver steps)",
                    f"Heat balance at {format_time(duration)}: in {sources}; out convection "
                    f"{q_convection:.4g} W, radiation {q_radiation:.4g} W{fixed_text}, still "
                    f"stored {q_stored:.4g} W (difference {balance_error:+.2%})"]
        rates = (series["max"][:, -1] - series["max"][:, -2]) / dt if n_steps > 1 else \
            np.zeros(len(bodies))
        rising = [(rate, bodies[index]["name"]) for index, rate in enumerate(rates)
                  if np.isfinite(rate) and rate * 60.0 > 0.05]
        if rising:
            rate, name = max(rising)
            messages.append(f"Still heating up at the end: '{name}' rises {rate * 60.0:.2g} K/min. "
                            "A longer duration (or Steady state) shows the final temperature.")
        peak = np.nanmax(series["max"], axis=1)
    else:
        summary += [f"Solved in {elapsed:.1f} s ({iterations} surface iteration"
                    f"{'s' if iterations != 1 else ''}, {cg_total} solver steps)",
                    f"Heat balance: in {sources}; out convection {q_convection:.4g} W, "
                    f"radiation {q_radiation:.4g} W{fixed_text} (difference {balance_error:+.2%})"]
        peak = np.array([entry["max"] if entry["max"] is not None else np.nan
                         for entry in body_results])
    for entry in condition_results:
        title = CONDITION_KINDS[entry["kind"]][0]
        effect = {"fixed": f"{entry['heat']:.4g} W leave" if entry["heat"] >= 0 else
                           f"{-entry['heat']:.4g} W enter",
                  "flux": f"{entry['heat']:.4g} W enter",
                  "convection": f"{entry['heat']:.4g} W leave",
                  "insulated": "no heat crosses"}[entry["kind"]]
        summary.append(f"Face condition '{entry['name']}' ({title}, {entry['area_mm2']:.4g} mm²): "
                       f"{effect}")
    for entry, value in zip(body_results, peak):
        if np.isfinite(value) and entry["max_c"] is not None and value > entry["max_c"]:
            messages.append(f"⚠ '{entry['name']}' reaches {value:.1f} °C, above the "
                            f"{entry['max_c']:g} °C limit of {entry['material']}.")
    step(100, "Done")
    return {"analysis": "transient" if transient else "steady", "cell_mm": h_mm,
            "elements": n_elements, "nodes": n_nodes, "ambient": ambient,
            "t_min": t_min, "t_max": t_max,
            "hot_point": hot_points[-1], "hot_value": float(hot_values[-1]),
            "bodies": body_results, "surface": surface,
            "frames": frames, "frame_times": frame_times, "frame_steps": frame_steps,
            "hot_points": hot_points, "hot_values": hot_values, "series": series,
            "power": total_power, "q_flux": float(flux_power.sum()), "q_fixed": q_fixed,
            "q_convection": q_convection, "q_radiation": q_radiation,
            "q_stored": q_stored, "balance_error": balance_error, "iterations": iterations,
            "conditions": condition_results,
            "seconds": elapsed, "summary": summary, "messages": messages}


def surface_values(result, key, frame=-1):
    """Temperatures at the result-surface vertices of a body for one stored frame."""
    item = result["surface"][key]
    values = result["frames"][frame][item["index"]]
    return (values * item["weights"]).sum(1)


def frame_body_stats(result, frame):
    """[(min, mean, max)] of every result body at a stored frame."""
    if result["series"] is None:
        return [(entry["min"], entry["mean"], entry["max"]) for entry in result["bodies"]]
    step_index = int(result["frame_steps"][frame])
    series = result["series"]
    return [tuple(None if not np.isfinite(v) else float(v)
                  for v in (series["min"][i, step_index], series["mean"][i, step_index],
                            series["max"][i, step_index]))
            for i in range(len(result["bodies"]))]


DISPLAY_TRIANGLE_LIMIT = 1_000_000      # triangles of the refined result surfaces together


def refine_triangles(V, T, max_edge, with_source=False):
    """Split every triangle into n² equal triangles (n from its longest edge) so no edge is
    longer than about `max_edge`. Returns unshared (vertices, triangles), plus the original
    triangle of every vertex with `with_source`; the orientation of the triangles is kept."""
    V = np.asarray(V, float)
    T = np.asarray(T, np.int64).reshape(-1, 3)
    P = V[T]
    longest = np.max(np.linalg.norm(P[:, [1, 2, 0]] - P, axis=2), axis=1)
    counts = np.clip(np.ceil(longest / max(max_edge, 1e-9)), 1, 256).astype(np.int64)
    out_vertices, out_triangles, out_source, offset = [], [], [], 0
    for n in np.unique(counts):
        n = int(n)
        chosen = np.nonzero(counts == n)[0]
        selected = P[chosen]
        lattice = np.array([(i, j) for i in range(n + 1) for j in range(n + 1 - i)], float)
        index = {(int(i), int(j)): k for k, (i, j) in enumerate(lattice)}
        local = []
        for i in range(n):
            for j in range(n - i):
                local.append((index[i, j], index[i + 1, j], index[i, j + 1]))
                if i + j < n - 1:
                    local.append((index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]))
        local = np.array(local, np.int64)
        a, b = lattice[:, 0] / n, lattice[:, 1] / n
        points = selected[:, None, 0] + a[None, :, None] * (selected[:, None, 1] - selected[:, None, 0]) \
            + b[None, :, None] * (selected[:, None, 2] - selected[:, None, 0])
        k = len(lattice)
        out_vertices.append(points.reshape(-1, 3))
        out_triangles.append((local[None, :, :] + (offset + k * np.arange(len(selected)))[:, None, None])
                             .reshape(-1, 3))
        out_source.append(np.repeat(chosen, k))
        offset += k * len(selected)
    if with_source:
        return np.concatenate(out_vertices), np.concatenate(out_triangles), np.concatenate(out_source)
    return np.concatenate(out_vertices), np.concatenate(out_triangles)


def _surface_weights(points, origin, h_mm, shape, nodes_global):
    """(node indices (n, 8), weights (n, 8)) of the trilinear interpolation at points (mm),
    restricted to existing nodes and normalised; all weights are 0 where no node is near."""
    ny1, nz1 = shape[1] + 1, shape[2] + 1
    u = (np.asarray(points, float) - origin) / h_mm
    base = np.clip(np.floor(u).astype(np.int64), 0, np.array(shape) - 1)
    frac = np.clip(u - base, 0.0, 1.0)
    index = np.empty((len(u), 8), np.int64)
    weights = np.empty((len(u), 8))
    for c, corner in enumerate(_CORNERS):
        cell = base + corner
        gid = (cell[:, 0] * ny1 + cell[:, 1]) * nz1 + cell[:, 2]
        position = np.minimum(np.searchsorted(nodes_global, gid), len(nodes_global) - 1)
        index[:, c] = position
        weights[:, c] = np.prod(np.where(corner, frac, 1.0 - frac), axis=1) * \
            (nodes_global[position] == gid)
    total = weights.sum(1, keepdims=True)
    weights = np.where(total > 1e-9, weights / np.maximum(total, 1e-12), 0.0)
    return index, weights.astype(np.float32)


# ── colour map and legend ─────────────────────────────────────────────────────
_COLOR_STOPS = ((0.00, (0.13, 0.17, 0.60)), (0.16, (0.10, 0.42, 0.95)),
                (0.33, (0.10, 0.78, 0.95)), (0.50, (0.35, 0.86, 0.45)),
                (0.67, (0.98, 0.86, 0.25)), (0.83, (0.98, 0.52, 0.15)),
                (1.00, (0.86, 0.13, 0.11)))


def temperature_colors(values, low, high):
    """RGB (n, 3) float32 of temperatures on the blue → red scale between low and high."""
    span = max(float(high) - float(low), 1e-9)
    t = np.clip((np.asarray(values, float) - low) / span, 0.0, 1.0)
    positions = np.array([stop[0] for stop in _COLOR_STOPS])
    colors = np.array([stop[1] for stop in _COLOR_STOPS])
    return np.stack([np.interp(t, positions, colors[:, channel]) for channel in range(3)],
                    axis=1).astype(np.float32)


class TemperatureLegend(QWidget):
    """Colour bar beside the 3-D view while temperatures are shown."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(96)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._range = None
        self._note = ""

    def set_range(self, low, high, note=""):
        self._range = (float(low), float(high))
        self._note = note
        self.update()

    def clear(self):
        self._range = None
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1c1c21"))
        if self._range is None:
            return
        low, high = self._range
        top, bottom = 34.0, max(60.0, self.height() - 40.0)
        bar = QRectF(10.0, top, 18.0, bottom - top)
        gradient = QLinearGradient(0.0, bottom, 0.0, top)
        for position, color in _COLOR_STOPS:
            gradient.setColorAt(position, QColor.fromRgbF(*color))
        painter.fillRect(bar, gradient)
        painter.setPen(QColor("#5b6370"))
        painter.drawRect(bar)
        font = QFont(self.font()); font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#e2e5e9"))
        painter.drawText(QRectF(4.0, 6.0, self.width() - 8.0, 20.0), Qt.AlignLeft | Qt.AlignVCenter,
                         "T  [°C]")
        font.setBold(False)
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        steps = 6
        for i in range(steps + 1):
            value = low + (high - low) * i / steps
            y = bottom - (bottom - top) * i / steps
            painter.drawLine(QPoint(28, int(y)), QPoint(33, int(y)))
            painter.drawText(QRectF(36.0, y - metrics.height() / 2, self.width() - 38.0,
                                    metrics.height()), Qt.AlignLeft | Qt.AlignVCenter, f"{value:.1f}")
        if self._note:
            painter.setPen(QColor("#ffb454"))
            painter.drawText(QRectF(4.0, bottom + 8.0, self.width() - 8.0, 30.0),
                             Qt.AlignLeft | Qt.TextWordWrap, self._note)


_PLOT_COLORS = ("#ff6b6b", "#4fc3f7", "#ffd54f", "#81c784", "#ba68c8", "#ff8a65", "#4db6ac",
                "#f06292", "#aed581", "#90a4ae")
PLOT_MAX_CURVES = len(_PLOT_COLORS)


class TimePlot(QWidget):
    """Temperature-over-time chart of a transient result: one line per body, a cursor at the
    shown time and dashed lines at material limits. Clicking or dragging selects a time."""
    timeClicked = pyqtSignal(float)
    _MARGINS = (48, 10, 10, 40)          # left, top, right, bottom (time axis + legend)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(190)
        self.setCursor(Qt.CrossCursor)
        self._times = None
        self._curves = []                 # [(name, values, colour)]
        self._limits = []                 # [(value, colour)]
        self._cursor = None

    def set_data(self, times, curves, limits=()):
        """curves: [(name, values)] over `times`, or [(name, values, own times)]."""
        self._times = np.asarray(times, float)
        self._curves = []
        for i, curve in enumerate(curves):
            curve_times = np.asarray(curve[2], float) if len(curve) > 2 else self._times
            self._curves.append((curve[0], np.asarray(curve[1], float),
                                 _PLOT_COLORS[i % len(_PLOT_COLORS)], curve_times))
        colours = {curve[0]: curve[2] for curve in self._curves}
        self._limits = [(value, colours.get(name, "#aab2bd")) for name, value in limits]
        self.update()

    def set_cursor(self, time_value):
        self._cursor = time_value
        self.update()

    def clear(self):
        self._times, self._curves, self._limits, self._cursor = None, [], [], None
        self.update()

    def _frame(self):
        left, top, right, bottom = self._MARGINS
        return QRectF(left, top, max(10.0, self.width() - left - right),
                      max(10.0, self.height() - top - bottom))

    def _range(self):
        values = [curve[1][np.isfinite(curve[1])] for curve in self._curves]
        values = np.concatenate(values) if values else np.array([])
        if not len(values):
            return 0.0, 1.0
        low, high = float(values.min()), float(values.max())
        limits = [value for value, _c in self._limits if low <= value <= high * 1.15 + 1.0]
        high = max([high] + limits)
        pad = max(0.5, 0.06 * (high - low))
        return low - pad, high + pad

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#171a1f"))
        painter.setPen(QColor("#3b424d"))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if self._times is None or len(self._times) < 2 or not self._curves:
            return
        frame = self._frame()
        t0, t1 = float(self._times[0]), float(self._times[-1]) or 1.0
        low, high = self._range()
        font = QFont(self.font()); font.setPointSizeF(max(7.0, font.pointSizeF() - 1.0))
        painter.setFont(font)
        metrics = QFontMetricsF(font)

        def x_of(t):
            return frame.left() + (t - t0) / max(t1 - t0, 1e-12) * frame.width()

        def y_of(value):
            return frame.bottom() - (value - low) / max(high - low, 1e-12) * frame.height()

        painter.setPen(QPen(QColor("#2c323b"), 1.0))
        for i in range(5):                                   # grid and temperature axis
            value = low + (high - low) * i / 4
            y = y_of(value)
            painter.drawLine(QPointF(frame.left(), y), QPointF(frame.right(), y))
            painter.setPen(QColor("#aab2bd"))
            painter.drawText(QRectF(0, y - 8, frame.left() - 4, 16), Qt.AlignRight | Qt.AlignVCenter,
                             f"{value:.0f}" if high - low > 8 else f"{value:.1f}")
            painter.setPen(QPen(QColor("#2c323b"), 1.0))
        painter.setPen(QColor("#aab2bd"))
        for fraction in (0.0, 0.5, 1.0):                     # time axis
            t = t0 + (t1 - t0) * fraction
            text = format_time(t) if t > 0 else "0"
            width = metrics.horizontalAdvance(text)
            x = min(max(x_of(t) - width / 2, frame.left() - 4), self.width() - width - 4)
            painter.drawText(QPointF(x, frame.bottom() + metrics.ascent() + 3), text)
        for value, colour in self._limits:                   # material limits
            if low <= value <= high:
                pen = QPen(QColor(colour), 1.0, Qt.DashLine)
                painter.setPen(pen)
                painter.drawLine(QPointF(frame.left(), y_of(value)), QPointF(frame.right(), y_of(value)))
        for name, values, colour, times in self._curves:     # curves (probes dashed)
            painter.setPen(QPen(QColor(colour), 1.8, Qt.DashLine if name.startswith("◉") else Qt.SolidLine))
            points = [QPointF(x_of(t), y_of(v)) for t, v in zip(times, values) if np.isfinite(v)]
            if len(points) > 1:
                painter.drawPolyline(QPolygonF(points))
        if self._cursor is not None:                         # shown time
            painter.setPen(QPen(QColor("#e2e5e9"), 1.0, Qt.DashLine))
            x = x_of(self._cursor)
            painter.drawLine(QPointF(x, frame.top()), QPointF(x, frame.bottom()))
        x, y = 6.0, self.height() - 6.0                      # legend
        for name, _values, colour, _times in self._curves:
            width = metrics.horizontalAdvance(name) + 22
            if x + width > self.width() - 4:
                break
            painter.fillRect(QRectF(x, y - 7, 10, 3), QColor(colour))
            painter.setPen(QColor("#e2e5e9"))
            painter.drawText(QPointF(x + 13, y - 2), name)
            x += width

    def _emit_time(self, pos):
        if self._times is None or len(self._times) < 2:
            return
        frame = self._frame()
        fraction = min(1.0, max(0.0, (pos.x() - frame.left()) / frame.width()))
        self.timeClicked.emit(float(self._times[0] + fraction * (self._times[-1] - self._times[0])))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._emit_time(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._emit_time(event.pos())


# ── 3-D view ──────────────────────────────────────────────────────────────────
class ThermalView3D(cad.AssemblyView3D):
    """Assembly view (picking, move / rotate arrows) that can also show temperatures: every
    body with a result is drawn with per-vertex colours from the colour map."""
    selection_color = (1.0, 0.8, 0.2)
    facePicked = pyqtSignal(object, int)        # render entry, face index (face picking mode)
    pointPicked = pyqtSignal(object, object)    # render entry, world point (probe mode)

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.result_mode = False
        self.hot_marker = None          # (world point, label) of the hottest node
        self.pick_mode = None           # None, "faces" or "probe"
        self.face_overlays = []         # [(entry, face index, rgb, strong)] of face conditions
        self.probe_markers = []         # [(world point, label, selected)]
        self._face_hover = None         # (entry, face index) under the cursor while picking

    def set_pick_mode(self, mode):
        self.pick_mode = mode
        self._face_hover = None
        self.setCursor(Qt.PointingHandCursor if mode else self._default_cursor())
        self.update()

    def _surface_hit(self, pos):
        """(entry, face index, world point) under the cursor, or None."""
        if self._mvp is None:
            return None
        origin, direction = self._ray(pos)
        hit = self._pick_face(origin, direction)
        if hit is None:
            return None
        t, entry, face = hit
        return entry, face, origin + t * direction

    def _click_select3d(self, pos, ctrl=False):
        if self.pick_mode == "faces":
            hit = self._surface_hit(pos)
            if hit is not None and hit[1] >= 0:
                self.facePicked.emit(hit[0], int(hit[1]))
            return
        if self.pick_mode == "probe":
            hit = self._surface_hit(pos)
            if hit is not None:
                self.pointPicked.emit(hit[0], np.asarray(hit[2], float))
            return
        super()._click_select3d(pos, ctrl)

    def mouseMoveEvent(self, e):
        if self.pick_mode == "faces" and not e.buttons():
            hit = self._surface_hit(e.pos())
            hover = (hit[0], hit[1]) if hit is not None and hit[1] >= 0 else None
            if (hover is None) != (self._face_hover is None) or (
                    hover is not None and (hover[0] is not self._face_hover[0] or
                                           hover[1] != self._face_hover[1])):
                self._face_hover = hover
                self.update()
        super().mouseMoveEvent(e)

    def _default_cursor(self):
        if getattr(self, "pick_mode", None):
            return QCursor(Qt.PointingHandCursor)
        return super()._default_cursor()

    def _draw_face_overlays(self):
        items = list(self.face_overlays)
        if self._face_hover is not None:
            items.append((self._face_hover[0], self._face_hover[1], (0.55, 1.0, 0.55), True))
        if not items:
            return
        gl.glPushAttrib(gl.GL_ENABLE_BIT | gl.GL_CURRENT_BIT | gl.GL_LINE_BIT |
                        gl.GL_COLOR_BUFFER_BIT | gl.GL_POLYGON_BIT | gl.GL_DEPTH_BUFFER_BIT)
        try:
            gl.glDisable(gl.GL_LIGHTING)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glDepthMask(gl.GL_FALSE)
            gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
            gl.glPolygonOffset(-2.0, -2.0)
            for entry, face, color, strong in items:
                if not cad.entry_visible(entry) or "solid" not in entry or \
                        not 0 <= face < len(entry.get("faces", ())):
                    continue
                if not self.result_mode:            # fill (kept off the temperature colours)
                    gl.glColor4f(*color, 0.62 if strong else 0.42)
                    self._draw_face_triangles(entry, face)
                gl.glColor4f(*color, 1.0)
                gl.glLineWidth(3.0 if strong else 1.8)
                self._draw_face_outline(entry, face)
        finally:
            gl.glPopAttrib()

    def _draw_probe_markers(self):
        if not self.probe_markers:
            return
        font = QFont("Arial", 9, QFont.Bold)
        gl.glPushAttrib(gl.GL_ENABLE_BIT | gl.GL_POINT_BIT | gl.GL_CURRENT_BIT)
        try:
            gl.glDisable(gl.GL_LIGHTING)
            gl.glDisable(gl.GL_DEPTH_TEST)
            for point, label, selected in self.probe_markers:
                for size, color in ((12.0, (0.05, 0.05, 0.05)),
                                    (8.0, (1.0, 0.85, 0.2) if selected else (0.85, 0.95, 1.0))):
                    gl.glPointSize(size)
                    gl.glColor3f(*color)
                    gl.glBegin(gl.GL_POINTS)
                    gl.glVertex3f(*[float(v) for v in point])
                    gl.glEnd()
                gl.glColor3f(*((1.0, 0.85, 0.2) if selected else (0.92, 0.95, 1.0)))
                self.renderText(float(point[0]), float(point[1]), float(point[2]), "  " + label, font)
        finally:
            gl.glPopAttrib()

    def release_buffers(self, entry):
        super().release_buffers(entry)
        self.drop_color_buffer(entry)

    def _drop(self, entry, key):
        buffer = entry.pop(key, None)
        if buffer is not None:
            self._pending_buffer_deletes.append(buffer)

    def drop_color_buffer(self, entry):
        """Forget the GPU buffers of the result surface (geometry and colours)."""
        self._drop(entry, "_thermal_vbo")
        self._drop(entry, "_thermal_color_vbo")

    def set_entry_colors(self, entry, colors, mesh=None):
        """Per-vertex colours of the result surface `mesh` (vertices, triangles) drawn in place
        of the body while temperatures are shown; None removes them. A new colouring of the same
        mesh (another time of a transient run) only replaces the colour buffer."""
        if colors is None or mesh is not entry.get("_thermal_mesh"):
            self._drop(entry, "_thermal_vbo")
        self._drop(entry, "_thermal_color_vbo")
        if colors is None:
            entry.pop("_thermal_colors", None)
            entry.pop("_thermal_mesh", None)
        else:
            entry["_thermal_colors"] = np.asarray(colors, np.float32)
            entry["_thermal_mesh"] = mesh

    @staticmethod
    def _upload(data):
        buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        return buffer

    def _color_buffer(self, entry):
        """(geometry buffer: normal + position, colour buffer) of the result surface, or None."""
        mesh = entry.get("_thermal_mesh")
        if mesh is None or len(entry["_thermal_colors"]) != len(mesh[0]):
            return None
        T = np.asarray(mesh[1], np.int64).reshape(-1, 3)
        if entry.get("_thermal_vbo") is None:
            P = np.asarray(mesh[0], np.float32).reshape(-1, 3)[T]
            normal = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
            length = np.linalg.norm(normal, axis=1, keepdims=True)
            normal = normal / np.where(length > 0, length, 1.0)
            normals = np.repeat(normal[:, None, :], 3, axis=1)
            entry["_thermal_vbo"] = self._upload(np.ascontiguousarray(
                np.concatenate([normals, P], axis=2), dtype=np.float32).ravel())
            entry["_thermal_count"] = 3 * len(T)
        if entry.get("_thermal_color_vbo") is None:
            entry["_thermal_color_vbo"] = self._upload(np.ascontiguousarray(
                entry["_thermal_colors"][T], dtype=np.float32).ravel())
        return entry["_thermal_vbo"], entry["_thermal_color_vbo"]

    def _draw_solids_inner(self):
        if not self.result_mode:
            return super()._draw_solids_inner()
        objects = [entry for entry in self.state.get("objects", [])
                   if cad.entry_visible(entry) and "solid" in entry]
        selected_ids = set(self.state.get("selected_entry_ids", set()))
        if self.state.get("selected") is not None:
            selected_ids.add(id(self.state["selected"]))
        gl.glPushAttrib(gl.GL_ENABLE_BIT | gl.GL_DEPTH_BUFFER_BIT | gl.GL_CURRENT_BIT |
                        gl.GL_LINE_BIT | gl.GL_COLOR_BUFFER_BIT)
        try:
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glDepthMask(gl.GL_TRUE)
            without = []
            for entry in objects:
                buffers = self._color_buffer(entry) if entry.get("_thermal_colors") is not None else None
                if buffers is None:
                    without.append(entry)
                    continue
                gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
                gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                gl.glEnableClientState(gl.GL_COLOR_ARRAY)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffers[0])
                gl.glNormalPointer(gl.GL_FLOAT, 24, ctypes.c_void_p(0))
                gl.glVertexPointer(3, gl.GL_FLOAT, 24, ctypes.c_void_p(12))
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffers[1])
                gl.glColorPointer(3, gl.GL_FLOAT, 0, ctypes.c_void_p(0))
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, entry["_thermal_count"])
                gl.glDisableClientState(gl.GL_COLOR_ARRAY)
                gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
                gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            # bodies that were not simulated (hidden at the time of the run): faint grey
            gl.glDepthMask(gl.GL_FALSE)
            for entry in without:
                triangles_buffer, _edges = cad.MainWindow._upload_vbo(entry)
                if not triangles_buffer:
                    continue
                gl.glColor4f(0.62, 0.64, 0.68, 0.18)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, triangles_buffer)
                gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
                gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                gl.glNormalPointer(gl.GL_FLOAT, 24, ctypes.c_void_p(0))
                gl.glVertexPointer(3, gl.GL_FLOAT, 24, ctypes.c_void_p(12))
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(entry["solid"][1]) * 3)
                gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
                gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glDepthMask(gl.GL_TRUE)
            gl.glDisable(gl.GL_LIGHTING)
            for entry in objects:
                _triangles, edges = cad.MainWindow._upload_vbo(entry)
                if not edges:
                    continue
                selected = id(entry) in selected_ids
                gl.glColor4f(*((1.0, 0.85, 0.2, 1.0) if selected else (0.05, 0.05, 0.07, 0.5)))
                gl.glLineWidth(2.5 if selected else 1.2)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, edges)
                gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                gl.glVertexPointer(3, gl.GL_FLOAT, 0, ctypes.c_void_p(0))
                gl.glDrawArrays(gl.GL_LINES, 0, len(entry["_edge_data"]) // 3)
                gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glLineWidth(1.0)
        finally:
            gl.glPopAttrib()

    def paintGL(self):
        super().paintGL()
        if self._mvp is not None:
            self._draw_face_overlays()
            self._draw_probe_markers()
        if not self.result_mode or self.hot_marker is None or self._mvp is None:
            return
        point, label = self.hot_marker
        gl.glPushAttrib(gl.GL_ENABLE_BIT | gl.GL_POINT_BIT | gl.GL_CURRENT_BIT)
        gl.glDisable(gl.GL_LIGHTING); gl.glDisable(gl.GL_DEPTH_TEST)
        for size, color in ((13.0, (0.05, 0.05, 0.05)), (9.0, (1.0, 1.0, 1.0))):
            gl.glPointSize(size)
            gl.glColor3f(*color)
            gl.glBegin(gl.GL_POINTS); gl.glVertex3f(*[float(v) for v in point]); gl.glEnd()
        font = QFont("Arial", 9, QFont.Bold)
        gl.glColor3f(1.0, 1.0, 1.0)
        self.renderText(float(point[0]), float(point[1]), float(point[2]), "   " + label, font)
        gl.glPopAttrib()


# ── background solver ─────────────────────────────────────────────────────────
class ThermalWorker(QThread):
    progressed = pyqtSignal(int, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)            # "" = cancelled

    def __init__(self, jobs, settings, conditions=(), parent=None):
        super().__init__(parent)
        self._jobs, self._settings, self._conditions = jobs, settings, list(conditions)
        self._cancel = False
        self._last_emit = 0.0

    def cancel(self):
        self._cancel = True

    def _progress(self, percent, text):
        now = time.perf_counter()
        if now - self._last_emit > 0.05 or percent >= 100:
            self._last_emit = now
            self.progressed.emit(percent, text)

    def run(self):
        try:
            result = solve_thermal(self._jobs, self._settings, self._progress, lambda: self._cancel,
                                   self._conditions)
        except ThermalCancelled:
            self.failed.emit("")
        except ThermalError as ex:
            self.failed.emit(str(ex))
        except MemoryError:
            self.failed.emit("Not enough memory for this mesh. Choose a larger cell size "
                             "or fewer Max cells.")
        except Exception as ex:          # report, never crash the application
            cad._trace("Thermal solver error:\n" + traceback.format_exc())
            self.failed.emit(f"{type(ex).__name__}: {ex}")
        else:
            self.succeeded.emit(result)


# ── small helpers ─────────────────────────────────────────────────────────────
def _icon(name):
    return QIcon(os.path.join(cad._LOG_DIR, "icons", name))


def _spin(minimum, maximum, suffix="", step=1.0, decimals=3, prefix="", width=None):
    spin = QDoubleSpinBox()
    spin.setLocale(cad.QLocale.c())
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setSuffix(suffix)
    spin.setPrefix(prefix)
    spin.setKeyboardTracking(False)
    if width:
        spin.setFixedWidth(width)
    else:
        text = prefix + "-99999.999" + suffix
        spin.setMinimumWidth(max(120, int(spin.fontMetrics().horizontalAdvance(text) + 44)))
    return spin


def _material_combo(current=None):
    combo = QComboBox()
    combo.addItem("— select material —", None)
    for name in material_names():
        combo.addItem(name, name)
        combo.setItemData(combo.count() - 1, material_summary(name) + "\n" +
                          MATERIAL_THERMAL[name].get("note", ""), Qt.ToolTipRole)
    index = combo.findData(current) if current else 0
    combo.setCurrentIndex(max(0, index))
    return combo


_LIST_INDICATOR_STYLE = """
    QListWidget::indicator {
        width: 15px; height: 15px; border: 2px solid rgb(28, 198, 251);
        border-radius: 3px; background-color: #171a1f;
    }
    QListWidget::indicator:checked { background-color: rgb(28, 198, 251); image: none; }
    QListWidget::indicator:unchecked { image: none; }
    QListWidget::indicator:unchecked:hover { background-color: #263744; }
"""


def _header(text):
    label = QLabel(f"<b>{text}</b>")
    label.setStyleSheet("color:#e2e5e9; padding-top:6px;")
    return label


def _hint(text):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color:#aab2bd; font-size:9px;")
    return label


def _collapsible_panel(title_text, content):
    """Panel with a title bar and a collapse arrow, like the Assembly window's side panels."""
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    title_bar = QWidget(panel)
    title_bar.setObjectName("thermalPanelTitleBar")
    title_bar.setStyleSheet(
        "QWidget#thermalPanelTitleBar { background:#292e36; border-bottom:1px solid #3b424d; }"
        "QLabel { color:#e2e5e9; font-weight:bold; }"
        "QToolButton { padding:0; margin:0; min-width:16px; max-width:16px; "
        "min-height:16px; max-height:16px; }")
    title_bar.setFixedHeight(24)
    title_layout = QHBoxLayout(title_bar)
    title_layout.setContentsMargins(6, 1, 4, 1)
    title_layout.setSpacing(4)
    title = QLabel(title_text, title_bar)
    title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    title_layout.addWidget(title)
    button = QToolButton(title_bar)
    button.setFixedSize(18, 18)
    button.setAutoRaise(True)
    button.setArrowType(Qt.DownArrow)
    button.setToolTip("Collapse panel")
    title_layout.addWidget(button)
    layout.addWidget(title_bar, 0, Qt.AlignTop)
    layout.addWidget(content, 1)
    state = {"panel": panel, "content": content, "title": title, "title_bar": title_bar,
             "button": button, "collapsed": False, "expanded_width": None,
             "minimum_width": panel.minimumWidth(), "maximum_width": panel.maximumWidth(),
             "collapsed_width": title_layout.contentsMargins().left() + button.width() +
                                title_layout.contentsMargins().right()}
    return panel, state


# ── heat source placement ─────────────────────────────────────────────────────
class HeatSourceDialog(QDialog):
    """Block or cylinder heat source: size, centre position, power and material."""
    _MATERIAL_HINT = ("Choose the material of the component (e.g. FR-4 for a board-like part, "
                      "Aluminium for a metal block).")

    def __init__(self, parent, kind, name, anchor=None):
        super().__init__(parent)
        self.setWindowTitle("Heat Source")
        self.setWindowIcon(_icon("heat_source_extrusion.png"))
        self._anchor = anchor                      # (part name, (low xyz, high xyz)) or None
        form = QFormLayout(self)
        self.ed_name = QLineEdit(name)
        form.addRow("Name:", self.ed_name)
        self.cb_kind = QComboBox()
        self.cb_kind.addItem("Block", "block")
        self.cb_kind.addItem("Cylinder", "cylinder")
        self.cb_kind.setCurrentIndex(0 if kind == "block" else 1)
        form.addRow("Shape:", self.cb_kind)
        self.sp_size = [_spin(0.01, 10000.0, " mm", 1.0, prefix=f"{axis} ") for axis in "XYZ"]
        for spin, value in zip(self.sp_size, (10.0, 10.0, 2.0)):
            spin.setValue(value)
        block = QWidget()
        block_layout = QVBoxLayout(block)
        block_layout.setContentsMargins(0, 0, 0, 0)
        for spin in self.sp_size:
            block_layout.addWidget(spin)
        self.sp_diameter = _spin(0.01, 10000.0, " mm", 1.0, prefix="Ø ")
        self.sp_diameter.setValue(8.0)
        self.sp_height = _spin(0.01, 10000.0, " mm", 1.0, prefix="H ")
        self.sp_height.setValue(12.0)
        self.cb_axis = QComboBox()
        for axis in "ZXY":
            self.cb_axis.addItem(f"Axis {axis}", axis)
        cylinder = QWidget()
        cylinder_layout = QVBoxLayout(cylinder)
        cylinder_layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self.sp_diameter, self.sp_height, self.cb_axis):
            cylinder_layout.addWidget(widget)
        self.sizes = QStackedWidget()
        self.sizes.addWidget(block)
        self.sizes.addWidget(cylinder)
        form.addRow("Size:", self.sizes)
        self.chk_on_top = QCheckBox(f"On top of {anchor[0]}" if anchor else "On top")
        self.chk_on_top.setVisible(anchor is not None)
        self.chk_on_top.setChecked(anchor is not None)
        form.addRow("", self.chk_on_top)
        self.sp_position = [_spin(-100000.0, 100000.0, " mm", 1.0, prefix=f"{axis} ")
                            for axis in "XYZ"]
        position = QWidget()
        position_layout = QVBoxLayout(position)
        position_layout.setContentsMargins(0, 0, 0, 0)
        for spin in self.sp_position:
            position_layout.addWidget(spin)
        form.addRow("Centre:", position)
        self.sp_power = _spin(0.0, 100000.0, " W", 0.1)
        self.sp_power.setValue(1.0)
        form.addRow("Power:", self.sp_power)
        self.cb_material = _material_combo()
        form.addRow("Material:", self.cb_material)
        self.lbl_material = _hint(self._MATERIAL_HINT)
        form.addRow("", self.lbl_material)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)
        self.cb_kind.currentIndexChanged.connect(self._update)
        for widget in self.sp_size + [self.sp_diameter, self.sp_height]:
            widget.valueChanged.connect(self._update)
        self.cb_axis.currentIndexChanged.connect(self._update)
        self.chk_on_top.toggled.connect(self._update)
        self.cb_material.currentIndexChanged.connect(self._update)
        cad._style_checkbox_like_explorer(self.chk_on_top)          # as the object explorer
        self._update()

    def _shape(self):
        if self.cb_kind.currentData() == "block":
            return {"size": [spin.value() for spin in self.sp_size]}
        return {"diameter": self.sp_diameter.value(), "height": self.sp_height.value(),
                "axis": self.cb_axis.currentData()}

    def _update(self, *_args):
        kind = self.cb_kind.currentData()
        self.sizes.setCurrentIndex(0 if kind == "block" else 1)
        on_top = self._anchor is not None and self.chk_on_top.isChecked()
        if on_top:
            low, high = (np.asarray(v, float) for v in self._anchor[1])
            extent = primitive_extent(kind, self._shape())
            centre = [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, high[2] + extent[2] / 2]
            for spin, value in zip(self.sp_position, centre):
                spin.blockSignals(True); spin.setValue(value); spin.blockSignals(False)
        for spin in self.sp_position:
            spin.setEnabled(not on_top)
        material = self.cb_material.currentData()
        self.lbl_material.setText(material_summary(material) if material else self._MATERIAL_HINT)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(material is not None)

    def values(self):
        return (self.cb_kind.currentData(), self._shape(),
                self.ed_name.text().strip() or "Heat source",
                [spin.value() for spin in self.sp_position], self.sp_power.value(),
                self.cb_material.currentData())


# ── Thermal window ──────────────────────────────────────────────────────
_VIEWS = (("view_isometric.png", "Isometric", -54.735610317245346, -135.0),
          ("view_top.png", "Top (XY)", 0.0, 0.0),
          ("view_front.png", "Front (XZ)", -90.0, 0.0),
          ("view_right.png", "Right (YZ)", -90.0, -90.0),
          ("view_left.png", "Left (YZ)", -90.0, 90.0))


def _fmt_c(value):
    return "—" if value is None else f"{value:.1f}"


class ThermalDialog(QDialog):
    """Thermal window (a window of its own, like the Assembly window): study explorer and
    properties on the left, the 3-D view with the temperature legend in the middle, simulation
    settings and results on the right."""

    def __init__(self, window):
        super().__init__(None)
        self._window = window
        if getattr(window, "_thermal_study", None) is None:
            window._thermal_study = study_new()
            window._thermal_file = None
            window._thermal_saved = study_state(window._thermal_study)
        self.study = window._thermal_study
        self.study.setdefault("conditions", [])
        self.study.setdefault("probes", [])
        self._selected = None
        self._selected_body = None
        self._selected_condition = None      # index into study["conditions"]
        self._selected_probe = None          # index into study["probes"]
        self._probe_cache = {}
        self._loading = False
        self._ready = False
        self._scene = []
        self._drag_base = None
        self._history = []
        self._history_index = -1
        self._restoring = False
        self._result = None
        self._worker = None
        self._status = {}                  # id(component) → "ok" / "outdated" / "missing"
        self.setWindowIcon(_icon("thermal_sim.png"))
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint |
                            Qt.WindowCloseButtonHint | Qt.WindowSystemMenuHint)
        self.resize(1480, 900)
        self._build_ui()
        # Enter in a number field only confirms it: no button is the dialog's default button
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False); button.setDefault(False)
        self._refresh_status()
        self._rebuild_scene()
        self._rebuild_tree()
        self._show_properties()
        self._show_settings()
        self._discard_results()
        self.view.fit_all()
        self._ready = True
        self._reset_history()
        self._update_title()
        self._update_estimate()
        self._set_status("Import parts or place a heat source, set the materials and powers, "
                         "then Start Simulation.")

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        def action(icon, text, tip, slot, checkable=False, shortcut=None):
            act = QAction(_icon(icon), text, self, toolTip=tip, checkable=checkable)
            if checkable:
                act.toggled.connect(slot)
            else:
                act.triggered.connect(lambda _checked=False: slot())
            if shortcut is not None:
                act.setShortcut(QKeySequence(shortcut))
                act.setShortcutContext(Qt.WindowShortcut)
                self.addAction(act)
            return act

        self.act_new = action("new_file.png", "New study", "New — start an empty thermal study",
                              self._new_file, shortcut=QKeySequence.New)
        self.act_open = action("open.png", "Open study…", f"Open a thermal study (*{THERMAL_EXTENSION})",
                               self._open_file, shortcut=QKeySequence.Open)
        self.act_save = action("save.png", "Save study", "Save the thermal study", self._save,
                               shortcut=QKeySequence.Save)
        self.act_save_as = action("save_as.png", "Save study as…",
                                  "Save the thermal study under a new name",
                                  lambda: self._save(True), shortcut="Ctrl+Shift+S")
        self.import_menu = QMenu("&Import", self)
        self.import_menu.setIcon(_icon("import.png"))
        self.import_menu.addAction("STEP file…", lambda: self._import_files("step"))
        self.import_menu.addAction("STL file…", lambda: self._import_files("stl"))
        self.import_menu.addAction("OBJ file / SimpleCAD project…", lambda: self._import_files("obj"))
        self.import_menu.addAction("Assembly file (.asby)…", self._import_assembly_file)
        self.import_menu.addSeparator()
        self.import_menu.addAction("Assembly of the open project", self._import_project_assembly)
        self.import_menu.addAction("Parts of the open project…", self._import_project_parts)
        self.act_import = action(
            "import.png", "Import", "Import — STEP, STL or OBJ files, a SimpleCAD project, an "
            "assembly, or parts of the open project", lambda: None)
        self.act_import.setMenu(self.import_menu)
        self.act_export = action("export.png", "Export",
                                 "Export — the export formats will be defined later", lambda: None)
        self.act_export.setEnabled(False)
        self.act_undo = action("undo.png", "Undo", "Undo the last study change", self._undo,
                               shortcut=QKeySequence.Undo)
        self.act_redo = action("redo.png", "Redo", "Redo the last undone study change", self._redo,
                               shortcut="Ctrl+Y")
        self.act_remove = action("delete_selected.png", "Remove",
                                 "Remove the selected part or heat source from the study",
                                 self._remove_selected)
        self.heat_menu = QMenu("&Heat Source", self)
        self.heat_menu.setIcon(_icon("heat_source_extrusion.png"))
        self.heat_menu.addAction("Block…", lambda: self._add_heat_source("block"))
        self.heat_menu.addAction("Cylinder…", lambda: self._add_heat_source("cylinder"))
        self.heat_menu.addSeparator()
        self.heat_menu.addAction("Power of the selected body…", self._focus_power)
        self.act_heat = action(
            "heat_source_extrusion.png", "Heat Source",
            "Heat Source — place a block or cylinder heat source (on top of the selected part), "
            "or give the selected body a power", lambda: None)
        self.act_heat.setMenu(self.heat_menu)
        self.condition_menu = QMenu("&Face Condition", self)
        self.condition_menu.setIcon(_icon("heat_convection.png"))
        for kind, (title, _color) in CONDITION_KINDS.items():
            self.condition_menu.addAction(f"{title}…", lambda k=kind: self._add_condition(k))
        self.act_condition = action(
            "heat_convection.png", "Face Condition",
            "Face Condition — fixed temperature, heat flux, own convection or insulation on faces "
            "you click in the view", lambda: None)
        self.act_condition.setMenu(self.condition_menu)
        self.act_probe = action(
            "measurement.png", "Probe",
            "Probe — click on a part to place a temperature probe (Esc ends)", self._on_probe_toggled,
            checkable=True)
        self.act_start = action(
            "start_thermal_simulation.png", "Start Simulation",
            "Start Simulation (F5) — mesh the visible bodies and compute the steady-state "
            "temperatures", self._start_simulation, shortcut="F5")
        self.act_results = action(
            "thermal_sim.png", "Show temperatures",
            "Show temperatures — switch between the temperature map and the normal view",
            self._on_results_toggled, checkable=True)
        self.act_results.setEnabled(False)
        self.act_cancel = QAction("Cancel simulation", self, toolTip="Stop the running simulation")
        self.act_cancel.triggered.connect(lambda _checked=False: self._cancel_simulation())
        self.act_cancel.setEnabled(False)
        self._view_actions = [
            action(icon, text, f"{text} view", lambda rx=rot_x, rz=rot_z: self.view.set_camera(rx, rz))
            for icon, text, rot_x, rot_z in _VIEWS]
        self.act_origin = cad.origin_toggle_action(self, lambda: getattr(self, "view", None))
        self.act_show_all = action("show_all_objects.png", "Show all",
                                   "Show all parts and fit them into the view", self._show_all)
        self.act_update_selected = action(
            "update_selected.png", "Update selected",
            "Update Selected — re-read the source of the selected part (project object or file); "
            "placement, materials chosen here and powers stay.",
            lambda: self._update_components([self._selected] if self._selected else []))
        self.act_update_all = action(
            "update_all.png", "Update all",
            "Update All — re-read the sources of all parts; placement, materials chosen here and "
            "powers stay.",
            lambda: self._update_components([c for c in self.study["components"]
                                             if c.get("kind") in SOURCE_KINDS], every=True))
        self.act_close = QAction("Close", self, toolTip="Close the Thermal window")
        self.act_close.triggered.connect(lambda _checked=False: self.close())
        self.act_close.setShortcut(QKeySequence("Ctrl+W"))
        self.act_close.setShortcutContext(Qt.WindowShortcut)
        self.addAction(self.act_close)

        toolbar = QToolBar(self)
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(32, 32))
        toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        toolbar.setStyleSheet("QToolButton:checked{background:#315f94;border:1px solid #548ac2;"
                              "border-radius:3px;}")
        for act in (self.act_new, self.act_open, self.act_save, self.act_save_as, None,
                    self.act_import, self.act_export, None, self.act_undo, self.act_redo, None,
                    self.act_probe, self.act_remove, None, self.act_heat, self.act_condition, None,
                    self.act_start, self.act_results, None,
                    *self._view_actions, self.act_origin, self.act_show_all, None,
                    self.act_update_selected, self.act_update_all):
            toolbar.addSeparator() if act is None else toolbar.addAction(act)

        menu_bar = QMenuBar(self)
        file_menu = menu_bar.addMenu("&File")
        for item in (self.act_new, self.act_open, self.act_save, self.act_save_as, None,
                     self.import_menu, self.act_export, None, self.act_close):
            if item is None:
                file_menu.addSeparator()
            elif isinstance(item, QMenu):
                file_menu.addMenu(item)
            else:
                file_menu.addAction(item)
        edit_menu = menu_bar.addMenu("&Edit")
        for item in (self.act_undo, self.act_redo, None, self.act_remove):
            edit_menu.addSeparator() if item is None else edit_menu.addAction(item)
        edit_menu.addAction("Rename…", self._rename_selected)
        edit_menu.addSeparator()
        edit_menu.addAction(self.act_update_selected)
        edit_menu.addAction(self.act_update_all)
        view_menu = menu_bar.addMenu("&View")
        for item in self._view_actions + [None, self.act_origin, self.act_show_all, self.act_results]:
            view_menu.addSeparator() if item is None else view_menu.addAction(item)
        simulation_menu = menu_bar.addMenu("&Simulation")
        simulation_menu.addMenu(self.heat_menu)
        simulation_menu.addMenu(self.condition_menu)
        simulation_menu.addAction(self.act_probe)
        simulation_menu.addSeparator()
        for item in (self.act_start, self.act_cancel, self.act_results):
            simulation_menu.addAction(item)

        # left: study explorer and properties --------------------------------------
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Study  (☑ = visible)")
        self.tree.itemSelectionChanged.connect(self._on_tree_selection)
        self.tree.itemChanged.connect(self._on_tree_item_changed)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_rclick)
        QShortcut(QKeySequence(Qt.Key_Delete), self.tree, activated=self._remove_selected,
                  context=Qt.WidgetShortcut)

        props = QWidget()
        grid = QGridLayout(props)
        grid.setContentsMargins(6, 6, 6, 6)
        # fixed label and value columns; the empty third column takes any extra panel width, so
        # nothing moves when the panel is resized
        label_width, value_width = 70, 200
        grid.setColumnMinimumWidth(0, label_width)
        grid.setColumnMinimumWidth(1, value_width)
        grid.setColumnStretch(0, 0); grid.setColumnStretch(1, 0); grid.setColumnStretch(2, 1)
        span_width = label_width + grid.horizontalSpacing() + value_width
        row = 0

        def add(widget, label=None, span=False):
            nonlocal row
            if span:
                grid.addWidget(widget, row, 0, 1, 2, Qt.AlignLeft)
            else:
                if label is not None:
                    caption = QLabel(label)
                    caption.setFixedWidth(label_width)
                    grid.addWidget(caption, row, 0)
                grid.addWidget(widget, row, 1, Qt.AlignLeft)
            row += 1

        def value_label():
            label = QLabel("—")
            label.setWordWrap(True)
            label.setFixedWidth(value_width)
            return label

        add(_header("Selected part"), span=True)
        self.lbl_source = value_label()
        add(self.lbl_source, "Source:")
        self.lbl_state = value_label()
        add(self.lbl_state, "State:")
        self.sp_pos = [_spin(-100000.0, 100000.0, " mm", 1.0, prefix=f"{axis} ") for axis in "XYZ"]
        self.sp_rot = [_spin(-360.0, 360.0, "°", 1.0, prefix=f"{axis} ") for axis in "XYZ"]
        add(QLabel("Position (centre of the part):"), span=True)
        for spin in self.sp_pos:
            add(spin, span=True)
        add(QLabel("Rotation (about the part centre):"), span=True)
        for spin in self.sp_rot:
            add(spin, span=True)
        for spin in self.sp_pos + self.sp_rot:
            spin.valueChanged.connect(self._on_placement_edited)

        self.w_shape = QWidget()
        shape_layout = QVBoxLayout(self.w_shape)
        shape_layout.setContentsMargins(0, 0, 0, 0)
        shape_layout.addWidget(_header("Heat source size"))
        self.sp_shape_size = [_spin(0.01, 10000.0, " mm", 1.0, prefix=f"{axis} ") for axis in "XYZ"]
        block = QWidget(); block_layout = QVBoxLayout(block); block_layout.setContentsMargins(0, 0, 0, 0)
        for spin in self.sp_shape_size:
            block_layout.addWidget(spin)
        self.sp_shape_diameter = _spin(0.01, 10000.0, " mm", 1.0, prefix="Ø ")
        self.sp_shape_height = _spin(0.01, 10000.0, " mm", 1.0, prefix="H ")
        self.cb_shape_axis = QComboBox()
        self.cb_shape_axis.setFixedWidth(value_width)
        for axis in "ZXY":
            self.cb_shape_axis.addItem(f"Axis {axis}", axis)
        cylinder = QWidget(); cylinder_layout = QVBoxLayout(cylinder)
        cylinder_layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self.sp_shape_diameter, self.sp_shape_height, self.cb_shape_axis):
            cylinder_layout.addWidget(widget)
        self.st_shape = QStackedWidget()
        self.st_shape.addWidget(block); self.st_shape.addWidget(cylinder)
        shape_layout.addWidget(self.st_shape)
        for widget in self.sp_shape_size + [self.sp_shape_diameter, self.sp_shape_height]:
            widget.valueChanged.connect(self._on_shape_edited)
        self.cb_shape_axis.currentIndexChanged.connect(self._on_shape_edited)
        add(self.w_shape, span=True)

        add(_header("Body"), span=True)
        self.lbl_body = value_label()
        add(self.lbl_body, "Name:")
        self.cb_material = _material_combo()
        self.cb_material.setFixedWidth(value_width)
        self.cb_material.currentIndexChanged.connect(self._on_material_changed)
        add(self.cb_material, "Material:")
        self.lbl_material = _hint("")
        self.lbl_material.setFixedWidth(span_width)
        add(self.lbl_material, span=True)
        self.sp_power = _spin(0.0, 100000.0, " W", 0.1)
        self.sp_power.setToolTip("Heat produced inside this body (spread evenly over its volume)")
        self.sp_power.valueChanged.connect(self._on_power_changed)
        add(self.sp_power, "Power:")
        self.lbl_power = _hint("")
        self.lbl_power.setFixedWidth(span_width)
        add(self.lbl_power, span=True)
        self.chk_emissivity = QCheckBox("Own surface emissivity")
        self.chk_emissivity.setToolTip("e.g. 0.8 for anodised or painted metal; unticked = the "
                                       "material's value")
        self.chk_emissivity.toggled.connect(self._on_emissivity_changed)
        add(self.chk_emissivity, span=True)
        self.sp_emissivity = _spin(0.0, 1.0, "", 0.05, decimals=2, prefix="ε ")
        self.sp_emissivity.valueChanged.connect(self._on_emissivity_changed)
        add(self.sp_emissivity, span=True)
        grid.setRowStretch(row, 1)

        def page_form():
            """(page, add) — a properties page with the same fixed columns as above."""
            page = QWidget()
            page_grid = QGridLayout(page)
            page_grid.setContentsMargins(6, 6, 6, 6)
            page_grid.setColumnMinimumWidth(0, label_width)
            page_grid.setColumnMinimumWidth(1, value_width)
            page_grid.setColumnStretch(2, 1)
            rows = [0]

            def put_row(widget, label=None, span=False):
                caption = None
                if span:
                    page_grid.addWidget(widget, rows[0], 0, 1, 2, Qt.AlignLeft)
                else:
                    if label is not None:
                        caption = QLabel(label)
                        caption.setFixedWidth(label_width)
                        page_grid.addWidget(caption, rows[0], 0)
                    page_grid.addWidget(widget, rows[0], 1, Qt.AlignLeft)
                rows[0] += 1
                return caption

            def finish():
                page_grid.setRowStretch(rows[0], 1)
            return page, put_row, finish

        # face condition page
        condition_page, put_row, finish = page_form()
        put_row(_header("Face condition"), span=True)
        self.lbl_condition_kind = value_label()
        put_row(self.lbl_condition_kind, "Type:")
        self.lbl_condition_name = value_label()
        put_row(self.lbl_condition_name, "Name:")
        self.chk_condition_enabled = QCheckBox("Active (untick to leave it out of the simulation)")
        put_row(self.chk_condition_enabled, span=True)
        self.sp_condition_temperature = _spin(-273.0, 5000.0, " °C", 1.0, decimals=2)
        self.cb_flux_mode = QComboBox()
        self.cb_flux_mode.addItem("Total power", "total")
        self.cb_flux_mode.addItem("Flux density", "density")
        self.cb_flux_mode.setFixedWidth(value_width)
        self.sp_condition_power = _spin(0.0, 1e6, " W", 0.1)
        self.sp_condition_flux = _spin(0.0, 1e8, " W/m²", 100.0, decimals=1)
        self.sp_condition_h = _spin(0.0, 1e6, " W/m²K", 1.0, decimals=2)
        self.chk_condition_fluid = QCheckBox("Air at the ambient temperature")
        self.sp_condition_fluid = _spin(-273.0, 5000.0, " °C", 1.0, decimals=1)
        self._condition_rows = {
            "fixed": [self.sp_condition_temperature,
                      put_row(self.sp_condition_temperature, "Temperature:")],
            "flux": [self.cb_flux_mode, put_row(self.cb_flux_mode, "Given as:"),
                     self.sp_condition_power, put_row(self.sp_condition_power, "Power:"),
                     self.sp_condition_flux, put_row(self.sp_condition_flux, "Flux:")],
            "convection": [self.sp_condition_h, put_row(self.sp_condition_h, "h:")]}
        put_row(self.chk_condition_fluid, span=True)
        self._condition_rows["convection"] += [self.chk_condition_fluid, self.sp_condition_fluid,
                                               put_row(self.sp_condition_fluid, "Air:")]
        self.lbl_condition_hint = _hint("")
        self.lbl_condition_hint.setFixedWidth(span_width)
        put_row(self.lbl_condition_hint, span=True)
        put_row(_header("Faces"), span=True)
        self.lbl_condition_faces = value_label()
        put_row(self.lbl_condition_faces, "Selected:")
        self.btn_pick_faces = QPushButton("Pick faces in the view")
        self.btn_pick_faces.setCheckable(True)
        self.btn_pick_faces.setFixedWidth(value_width)
        self.btn_pick_faces.toggled.connect(
            lambda on: self._set_pick_mode("faces" if on else None))
        put_row(self.btn_pick_faces, "")
        self.btn_clear_faces = QPushButton("Remove all faces")
        self.btn_clear_faces.setFixedWidth(value_width)
        self.btn_clear_faces.clicked.connect(self._clear_condition_faces)
        put_row(self.btn_clear_faces, "")
        pick_hint = _hint("Click faces to add them; click a chosen face again to remove it. "
                          "Esc or the button ends picking. Faces without a condition exchange heat "
                          "with the environment (Simulation panel).")
        pick_hint.setFixedWidth(span_width)
        put_row(pick_hint, span=True)
        finish()
        self.chk_condition_enabled.toggled.connect(self._on_condition_edited)
        self.cb_flux_mode.currentIndexChanged.connect(self._on_condition_edited)
        self.chk_condition_fluid.toggled.connect(self._on_condition_edited)
        for spin in (self.sp_condition_temperature, self.sp_condition_power, self.sp_condition_flux,
                     self.sp_condition_h, self.sp_condition_fluid):
            spin.valueChanged.connect(self._on_condition_edited)

        # probe page
        probe_page, put_row, finish = page_form()
        put_row(_header("Probe"), span=True)
        self.lbl_probe_name = value_label()
        put_row(self.lbl_probe_name, "Name:")
        self.lbl_probe_part = value_label()
        put_row(self.lbl_probe_part, "On part:")
        self.lbl_probe_point = value_label()
        put_row(self.lbl_probe_point, "Position:")
        self.lbl_probe_value = value_label()
        put_row(self.lbl_probe_value, "Temperature:")
        probe_hint = _hint("A probe stays on its part when the part moves. Its temperature is "
                           "shown in the results and in the 3-D view; in a transient run also "
                           "over time in the plot.")
        probe_hint.setFixedWidth(span_width)
        put_row(probe_hint, span=True)
        finish()

        self._props_stack = QStackedWidget()
        for page in (props, condition_page, probe_page):
            self._props_stack.addWidget(page)
        props_scroll = QScrollArea()
        props_scroll.setWidgetResizable(True)
        props_scroll.setFrameShape(QFrame.NoFrame)
        props_scroll.setWidget(self._props_stack)
        left = QSplitter(Qt.Vertical)
        left.addWidget(self.tree)
        left.addWidget(props_scroll)
        left.setStretchFactor(0, 2); left.setStretchFactor(1, 3)
        self._left_content = left

        # middle: 3-D view and legend -----------------------------------------------
        self._view_state = {"objects": [], "selected": None, "mode": cad.MODE_NORMAL,
                            "active_sketch": None, "sketch_view_state": {}}
        self.view = ThermalView3D(self._view_state)
        self.view.setMinimumSize(480, 360)
        self.view.bodyClicked.connect(self._on_body_clicked)
        self.view.gizmoStarted.connect(self._on_gizmo_started)
        self.view.gizmoMoved.connect(self._on_gizmo_moved)
        self.view.gizmoFinished.connect(self._on_gizmo_finished)
        self.view.facePicked.connect(self._on_face_picked)
        self.view.pointPicked.connect(self._on_point_picked)
        self.legend = TemperatureLegend()
        view_area = QWidget()
        view_outer = QVBoxLayout(view_area)
        view_outer.setContentsMargins(0, 0, 0, 0)
        view_outer.setSpacing(0)
        view_layout = QHBoxLayout()
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)
        view_layout.addWidget(self.view, 1)
        view_layout.addWidget(self.legend)
        view_outer.addLayout(view_layout, 1)
        # time slider of transient results
        self.w_playback = QWidget()
        playback = QHBoxLayout(self.w_playback)
        playback.setContentsMargins(6, 4, 6, 4)
        self.btn_play = QToolButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setToolTip("Play / pause the temperatures over time")
        self.btn_play.clicked.connect(self._toggle_playback)
        self.sl_time = QSlider(Qt.Horizontal)
        self.sl_time.setToolTip("Time after switching the heat sources on")
        self.sl_time.valueChanged.connect(self._on_frame_changed)
        self.lbl_time = QLabel("")
        self.lbl_time.setMinimumWidth(190)
        playback.addWidget(self.btn_play)
        playback.addWidget(self.sl_time, 1)
        playback.addWidget(self.lbl_time)
        self.w_playback.hide()
        view_outer.addWidget(self.w_playback)
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(90)
        self._play_timer.timeout.connect(self._play_step)

        # right: simulation settings and results -------------------------------------
        settings = QWidget()
        form = QGridLayout(settings)
        form.setContentsMargins(6, 6, 6, 6)
        srow = 0

        def put(widget, label=None, span=False):
            """Add a row; returns the row's caption label (None without one)."""
            nonlocal srow
            caption = None
            if span:
                form.addWidget(widget, srow, 0, 1, 2)
            else:
                if label is not None:
                    caption = QLabel(label)
                    form.addWidget(caption, srow, 0)
                form.addWidget(widget, srow, 1)
            srow += 1
            return caption

        put(_header("Solver"), span=True)
        put(_hint("Conduction (FEM): heat conduction in the bodies; air cooling and radiation "
                  "act on their surfaces."), span=True)
        self.cb_analysis = QComboBox()
        self.cb_analysis.addItem("Steady state", "steady")
        self.cb_analysis.addItem("Transient (over time)", "transient")
        self.cb_analysis.setToolTip(
            "Steady state: the final temperatures after a long time.\n"
            "Transient: how the temperatures develop after the heat sources are switched on.")
        put(self.cb_analysis, "Analysis:")
        self.sp_duration = _spin(0.001, 1000000.0, "", 1.0, decimals=3, width=110)
        self.sp_duration.setToolTip("Simulated time after switching the heat sources on")
        self.cb_duration_unit = QComboBox()
        for unit in TIME_UNITS:
            self.cb_duration_unit.addItem(unit, unit)
        duration = QWidget()
        duration_layout = QHBoxLayout(duration)
        duration_layout.setContentsMargins(0, 0, 0, 0)
        duration_layout.addWidget(self.sp_duration)
        duration_layout.addWidget(self.cb_duration_unit)
        duration_layout.addStretch(1)
        self.sp_time_steps = QSpinBox()
        self.sp_time_steps.setRange(1, 10000)
        self.sp_time_steps.setSingleStep(10)
        self.sp_time_steps.setKeyboardTracking(False)
        self.sp_time_steps.setMinimumWidth(110)
        self.sp_time_steps.setToolTip("Number of time steps; more steps = more accurate and slower. "
                                      "Up to 100 of them can be shown with the time slider.")
        self.chk_initial_ambient = QCheckBox("Start at the ambient temperature")
        self.sp_initial = _spin(-100.0, 1000.0, " °C", 1.0, decimals=1)
        self.sp_initial.setToolTip("Temperature of all bodies at the start")
        self.lbl_time_constant = _hint("")
        rows = [duration, put(duration, "Duration:")]
        rows += [self.sp_time_steps, put(self.sp_time_steps, "Time steps:")]
        put(self.chk_initial_ambient, span=True)
        rows += [self.chk_initial_ambient, self.sp_initial, put(self.sp_initial, "Start at:")]
        put(self.lbl_time_constant, span=True)
        self._transient_rows = rows + [self.lbl_time_constant]   # shown for Transient only
        put(_header("Environment"), span=True)
        self.sp_ambient = _spin(-100.0, 500.0, " °C", 1.0, decimals=1)
        self.sp_ambient.setToolTip("Temperature of the surrounding air and surfaces")
        put(self.sp_ambient, "Ambient:")
        self.cb_convection = QComboBox()
        self.cb_convection.addItem("Natural convection (auto h)", "natural")
        self.cb_convection.addItem("Fixed coefficient h", "fixed")
        self.cb_convection.setToolTip(
            "Natural convection: h from correlations for vertical, upward and downward facing "
            "surfaces (gravity along −Z).\nFixed: one heat transfer coefficient for all surfaces "
            "(≈ 5–10 still air, 20–100 with a fan).")
        put(self.cb_convection, "Convection:")
        self.sp_h = _spin(0.0, 100000.0, " W/m²K", 1.0, decimals=2)
        put(self.sp_h, "h:")
        self.chk_radiation = QCheckBox("Radiation to the surroundings")
        put(self.chk_radiation, span=True)
        put(_hint("Gravity points along −Z. Surfaces that do not touch another body exchange "
                  "heat with the ambient air; air enclosed in a housing conducts as still air."),
            span=True)
        put(_header("Mesh"), span=True)
        self.chk_auto_cell = QCheckBox("Automatic cell size")
        put(self.chk_auto_cell, span=True)
        self.sp_cell = _spin(0.01, 1000.0, " mm", 0.1, decimals=3)
        self.sp_cell.setToolTip("Edge length of the cubic cells. Thin walls need at least two "
                                "cells across.")
        put(self.sp_cell, "Cell size:")
        self.sp_max_cells = QSpinBox()
        self.sp_max_cells.setRange(10000, 5000000)
        self.sp_max_cells.setSingleStep(50000)
        self.sp_max_cells.setKeyboardTracking(False)
        self.sp_max_cells.setMinimumWidth(130)
        self.sp_max_cells.setToolTip("Upper limit of solid cells; more cells = finer result, "
                                     "more time and memory")
        put(self.sp_max_cells, "Max cells:")
        self.lbl_estimate = _hint("")
        put(self.lbl_estimate, span=True)
        for widget in (self.sp_ambient, self.sp_h, self.sp_cell):
            widget.valueChanged.connect(self._on_settings_edited)
        self.sp_max_cells.valueChanged.connect(self._on_settings_edited)
        self.cb_convection.currentIndexChanged.connect(self._on_settings_edited)
        self.chk_radiation.toggled.connect(self._on_settings_edited)
        self.chk_auto_cell.toggled.connect(self._on_settings_edited)
        for widget in (self.cb_analysis, self.cb_duration_unit):
            widget.currentIndexChanged.connect(self._on_settings_edited)
        for widget in (self.sp_duration, self.sp_time_steps, self.sp_initial):
            widget.valueChanged.connect(self._on_settings_edited)
        self.chk_initial_ambient.toggled.connect(self._on_settings_edited)
        self._setting_inputs = (self.sp_ambient, self.cb_convection, self.sp_h, self.chk_radiation,
                                self.chk_auto_cell, self.sp_cell, self.sp_max_cells,
                                self.cb_analysis, self.sp_duration, self.cb_duration_unit,
                                self.sp_time_steps, self.chk_initial_ambient, self.sp_initial)

        results = QWidget()
        results_layout = QVBoxLayout(results)
        results_layout.setContentsMargins(0, 6, 0, 0)
        results_layout.addWidget(_header("Results"))
        self.lbl_result = QLabel("")
        self.lbl_result.setWordWrap(True)
        results_layout.addWidget(self.lbl_result)
        self.tbl_results = QTableWidget(0, 6)
        self.tbl_results.setHorizontalHeaderLabels(["Body", "Material", "W", "Min °C", "Mean °C",
                                                    "Max °C"])
        self.tbl_results.verticalHeader().hide()
        self.tbl_results.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_results.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_results.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self.tbl_results.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.tbl_results.setFixedHeight(170)
        self.tbl_results.itemSelectionChanged.connect(self._on_result_row)
        results_layout.addWidget(self.tbl_results)
        self.lbl_probes = _header("Probes")
        self.tbl_probes = QTableWidget(0, 3)
        self.tbl_probes.setHorizontalHeaderLabels(["Probe", "On part", "°C"])
        self.tbl_probes.verticalHeader().hide()
        self.tbl_probes.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_probes.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_probes.setSelectionMode(QAbstractItemView.SingleSelection)
        probe_header = self.tbl_probes.horizontalHeader()
        probe_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        probe_header.setSectionResizeMode(1, QHeaderView.Stretch)
        probe_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tbl_probes.setFixedHeight(120)
        self.tbl_probes.itemSelectionChanged.connect(self._on_probe_row)
        results_layout.addWidget(self.lbl_probes)
        results_layout.addWidget(self.tbl_probes)
        self.lbl_plot = _header("Maximum temperature over time")
        self.plot = TimePlot()
        self.plot.timeClicked.connect(self._on_plot_clicked)
        results_layout.addWidget(self.lbl_plot)
        results_layout.addWidget(self.plot)
        range_row = QHBoxLayout()
        self.chk_auto_range = QCheckBox("Auto colour range")
        self.chk_auto_range.setChecked(True)
        self.sp_range_low = _spin(-273.0, 5000.0, " °C", 1.0, decimals=1, width=92)
        self.sp_range_high = _spin(-273.0, 5000.0, " °C", 1.0, decimals=1, width=92)
        for widget in (self.chk_auto_range, self.sp_range_low, self.sp_range_high):
            range_row.addWidget(widget)
        range_row.addStretch(1)
        results_layout.addLayout(range_row)
        self.chk_auto_range.toggled.connect(self._on_range_edited)
        self.sp_range_low.valueChanged.connect(self._on_range_edited)
        self.sp_range_high.valueChanged.connect(self._on_range_edited)
        results_layout.addWidget(_header("Messages"))
        self.txt_messages = QPlainTextEdit()
        self.txt_messages.setReadOnly(True)
        self.txt_messages.setFont(QFont("Consolas", 9))
        self.txt_messages.setMinimumHeight(150)
        results_layout.addWidget(self.txt_messages, 1)
        put(results, span=True)
        form.setRowStretch(srow, 1)

        right = QScrollArea()                  # one column: settings, Start, results, messages
        right.setWidgetResizable(True)
        right.setFrameShape(QFrame.NoFrame)
        right.setWidget(settings)

        self._study_panel, self._study_panel_state = _collapsible_panel("Study", left)
        self._simulation_panel, self._simulation_panel_state = _collapsible_panel("Simulation", right)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._study_panel)
        split.addWidget(view_area)
        split.addWidget(self._simulation_panel)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 1); split.setStretchFactor(2, 0)
        split.setSizes([330, 790, 360])
        self._panel_splitter = split
        self._panel_view = view_area
        for state in (self._study_panel_state, self._simulation_panel_state):
            state["button"].clicked.connect(
                lambda _checked=False, s=state: cad.AssemblyDialog._toggle_panel(self, s))

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#80c8ff;font-size:10px;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedWidth(220)
        self.progress.hide()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel_simulation)
        self.btn_cancel.hide()
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.progress)
        bottom.addWidget(self.btn_cancel)
        bottom.addWidget(close)

        root = QVBoxLayout(self)
        root.setMenuBar(menu_bar)
        cad.enlarge_toolbar_icons(toolbar)
        for button in toolbar.findChildren(QToolButton):   # Import / Heat Source open their menu
            if button.defaultAction() in (self.act_import, self.act_heat, self.act_condition):
                button.setPopupMode(QToolButton.InstantPopup)
        root.addWidget(toolbar)
        root.addWidget(split, 1)
        root.addLayout(bottom)
        for checkbox in self.findChildren(QCheckBox):       # as the object explorer
            cad._style_checkbox_like_explorer(checkbox)

    # ── status, title, saved state ────────────────────────────────────────────
    def event(self, e):
        # parts may have been edited in the main window meanwhile: refresh the Outdated markers
        if e.type() == QEvent.WindowActivate and self._ready and self._worker is None:
            before = dict(self._status)
            self._refresh_status()
            if self._status != before:
                self._rebuild_tree()
                self._show_properties()
        return super().event(e)

    def _refresh_status(self):
        objects = self._window.state["objects"]
        status = {}
        for component in self.study["components"]:
            try:
                status[id(component)] = component_status(component, objects)
            except Exception:
                status[id(component)] = "ok"
        self._status = status

    def _set_status(self, text):
        self.status.setText(text)
        if text:
            cad.log_action(f"Thermal: {text}")

    def _update_title(self):
        path = self._window._thermal_file
        name = os.path.basename(path) if path else THERMAL_DEFAULT_NAME
        self.setWindowTitle(f"Thermal — {name}" + (" *" if self.has_unsaved_changes() else ""))

    def has_unsaved_changes(self):
        return state_key(study_state(self.study)) != state_key(self._window._thermal_saved)

    def _mark_saved(self):
        self._window._thermal_saved = study_state(self.study)
        self._update_title()

    def _confirm_close(self):
        if getattr(self, "_force_close", False) or not self.has_unsaved_changes():
            return True
        answer = QMessageBox.question(
            self, "Unsaved thermal study",
            f"The thermal study '{self.study['name']}' has unsaved changes. Save them?\n\n"
            "Discard returns the study to its last saved state.",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if answer == QMessageBox.Save:
            return self._save()
        if answer == QMessageBox.Discard:
            restored = study_from_state(self._window._thermal_saved)
            self.study.clear()
            self.study.update(restored)
            return True
        return False

    def reject(self):
        # Esc first ends face / probe picking; otherwise Esc and Close go through close(), so
        # every way of closing asks about unsaved changes
        if getattr(self, "_in_close", False):
            super().reject()
        elif self.view.pick_mode is not None:
            self._set_pick_mode(None)
        else:
            self.close()

    def closeEvent(self, e):
        if not self._confirm_close():
            e.ignore()
            return
        if self._worker is not None:
            self._worker.blockSignals(True)
            self._worker.cancel()
            self._worker.wait()
            self._worker = None
            self._pause_main_redraw(False)
        for component in self.study["components"]:     # GL buffers belong to this window's view
            for body in component["bodies"]:
                body.pop("_render", None)
        self._in_close = True
        try:
            super().closeEvent(e)
        finally:
            self._in_close = False

    # ── history (undo / redo) ─────────────────────────────────────────────────
    def _reset_history(self):
        state = study_state(self.study)
        self._history = [(state, state_key(state))]
        self._history_index = 0
        self._update_history_actions()

    def _update_history_actions(self):
        idle = self._worker is None
        self.act_undo.setEnabled(idle and self._history_index > 0)
        self.act_redo.setEnabled(idle and 0 <= self._history_index < len(self._history) - 1)

    def _changed(self, results=True):
        """Record a change of the study (undo step, title). `results`: the change affects the
        simulation, so the shown temperatures no longer belong to the setup."""
        if results and self._result is not None:
            self._discard_results(outdated=True)
        if not self._restoring and self._ready:
            state = study_state(self.study)
            key = state_key(state)
            if key != self._history[self._history_index][1]:
                del self._history[self._history_index + 1:]
                self._history.append((state, key))
                self._history_index += 1
                if len(self._history) > 50:
                    del self._history[0]
                    self._history_index -= 1
            self._update_history_actions()
        self._update_title()
        self._update_estimate()

    def _restore_history(self, index, verb):
        selected_name = self._selected["name"] if self._selected is not None else None
        selected_body = self._selected_body
        restored = study_from_state(self._history[index][0])
        self._restoring = True
        try:
            self._history_index = index
            self.study.clear()
            self.study.update(restored)
            self._selected = next((c for c in self.study["components"]
                                   if c["name"] == selected_name), None)
            self._selected_body = selected_body if self._selected is not None and \
                selected_body is not None and selected_body < len(self._selected["bodies"]) else \
                (0 if self._selected is not None and len(self._selected["bodies"]) == 1 else None)
            if self._selected_condition is not None and \
                    self._selected_condition >= len(self.study["conditions"]):
                self._selected_condition = None
            if self._selected_probe is not None and self._selected_probe >= len(self.study["probes"]):
                self._selected_probe = None
            if self._selected_condition is None and self.view.pick_mode == "faces":
                self._set_pick_mode(None)
            self._discard_results()
            self._refresh_status()
            self._rebuild_scene()
            self._rebuild_tree()
            self._show_properties()
            self._show_settings()
            self._update_title()
            self._update_estimate()
            self._set_status(f"{verb} study change")
        finally:
            self._restoring = False
        self._update_history_actions()

    def _undo(self):
        if self._history_index > 0 and self._worker is None:
            self._restore_history(self._history_index - 1, "Undid")

    def _redo(self):
        if self._history_index < len(self._history) - 1 and self._worker is None:
            self._restore_history(self._history_index + 1, "Redid")

    # ── scene (render entries of the 3-D view) ────────────────────────────────
    def _rebuild_scene(self):
        keep = []
        for component in self.study["components"]:
            for index, body in enumerate(component["bodies"]):
                entry = body.get("_render")
                if entry is None:
                    entry = {"type": "extrusion", "parent": None}
                    body["_render"] = entry
                entry.update(name=body["name"], opacity=body["opacity"], color=tuple(body["color"]),
                             faces=body["faces"], visible=component["visible"])
                entry["_thermal_component"] = component
                entry["_body_index"] = index
                keep.append(entry)
            self._apply_transform(component)
        kept = {id(entry) for entry in keep}
        for entry in self._scene:
            if id(entry) not in kept:
                self.view.release_buffers(entry)
        self._scene = keep
        self._view_state["objects"][:] = keep
        if self._result is not None:
            self._apply_result_colors()
        self._sync_selection()
        self._refresh_overlays()
        self.view.update()

    def _refresh_overlays(self):
        """Face-condition faces (coloured by type, the selected condition stronger) and probe
        markers (with their temperature while results are shown) in the 3-D view."""
        components = {c["name"]: c for c in self.study["components"]}
        overlays = []
        for index, condition in enumerate(self.study["conditions"]):
            strong = index == self._selected_condition
            if not strong and (not condition.get("enabled", True) or self.view.result_mode):
                continue
            color = CONDITION_KINDS[condition["kind"]][1]
            for ref in condition["faces"]:
                component = components.get(ref["component"])
                if component is None or not 0 <= ref["body"] < len(component["bodies"]):
                    continue
                entry = component["bodies"][ref["body"]].get("_render")
                if entry is not None:
                    overlays.append((entry, ref["face"], color, strong))
        self.view.face_overlays = overlays
        markers = []
        values = self._probe_frame_values()
        for index, probe in enumerate(self.study["probes"]):
            point = probe_world_point(self.study, probe)
            if point is None:
                continue
            component = components.get(probe["component"])
            if component is not None and not component.get("visible", True):
                continue
            label = probe["name"]
            if self.view.result_mode and values[index] is not None:
                label += f"  {values[index]:.1f} °C"
            markers.append((point, label, index == self._selected_probe))
        self.view.probe_markers = markers
        self.view.update()

    def _apply_transform(self, component):
        for body in component["bodies"]:
            entry = body.get("_render")
            if entry is None:
                continue
            world = cad.assembly_world_vertices(component, body["vertices"])
            entry["solid"] = ([tuple(point) for point in world.tolist()], body["triangles"])
            for key in cad._ASSEMBLY_CACHE_KEYS:
                entry.pop(key, None)
            self.view.drop_color_buffer(entry)
            entry["_centre"] = (world.min(0) + world.max(0)) / 2 if len(world) else None
        if component is self._selected:
            self._update_gizmo()

    def _sync_selection(self):
        component = self._selected
        entry, ids = None, set()
        if component is not None:
            body = self._body()
            if body is not None:
                entry = body.get("_render")
            else:
                ids = {id(b["_render"]) for b in component["bodies"] if b.get("_render") is not None}
        self._view_state["selected"] = entry
        self._view_state["selected_entry_ids"] = ids
        self._update_gizmo()

    def _update_gizmo(self):
        component = self._selected
        centre = None
        if component is not None and component["visible"] and not self.view.result_mode and \
                self._worker is None and self.view.pick_mode is None:
            box = cad.assembly_bounds(component)
            if box is not None:
                centre = (box[0] + box[1]) / 2.0
        self.view.gizmo_standard_handles = True
        self.view.gizmo_twist = None
        self.view.explosion_arrow = None
        self.view.gizmo_center = None if centre is None else np.array(centre, float)

    # ── explorer ──────────────────────────────────────────────────────────────
    @staticmethod
    def _body_text(body, with_name=True):
        parts = [body["name"]] if with_name else []
        parts.append(f"[{body['material']}]" if body.get("material") else "[⚠ no material]")
        if body.get("power_w", 0.0) > 0:
            parts.append(f"{body['power_w']:g} W")
        return "   ".join(parts)

    def _rebuild_tree(self):
        self._loading = True
        try:
            self.tree.clear()
            root_item = QTreeWidgetItem([self.study["name"]])
            font = root_item.font(0); font.setBold(True); root_item.setFont(0, font)
            root_item.setIcon(0, _icon("thermal_sim.png"))
            self.tree.addTopLevelItem(root_item)
            selected_item = None
            heat_icon = _icon("heat_source_extrusion.png")
            for ci, component in enumerate(self.study["components"]):
                bodies = component["bodies"]
                single = len(bodies) == 1
                if single:
                    text = f"{component['name']}   " + self._body_text(bodies[0], with_name=False)
                else:
                    missing = sum(1 for b in bodies if not b.get("material") and not b.get("fused"))
                    power = sum(b.get("power_w", 0.0) for b in bodies)
                    text = f"{component['name']}   ({len(bodies)} bodies" + \
                        (f", {power:g} W" if power > 0 else "") + ")" + \
                        (f"   ⚠ {missing} without material" if missing else "")
                status = self._status.get(id(component), "ok")
                text += {"ok": "", "outdated": "   (outdated)",
                         "missing": "   (source missing)"}[status]
                item = QTreeWidgetItem([text])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked if component["visible"] else Qt.Unchecked)
                item.setData(0, Qt.UserRole, ci)
                item.setData(0, Qt.UserRole + 1, -1)
                if status == "missing":
                    item.setForeground(0, QColor("#ff6b6b"))
                elif status == "outdated" or \
                        any(not b.get("material") and not b.get("fused") for b in bodies):
                    item.setForeground(0, QColor("#ffb454"))
                if component["kind"] in PRIMITIVE_KINDS or \
                        any(b.get("power_w", 0.0) > 0 for b in bodies):
                    item.setIcon(0, heat_icon)
                if not single:
                    for bi, body in enumerate(bodies):
                        child = QTreeWidgetItem([self._body_text(body)])
                        child.setData(0, Qt.UserRole, ci)
                        child.setData(0, Qt.UserRole + 1, bi)
                        if not body.get("material") and not body.get("fused"):
                            child.setForeground(0, QColor("#ffb454"))
                        if body.get("power_w", 0.0) > 0:
                            child.setIcon(0, heat_icon)
                        item.addChild(child)
                        if component is self._selected and self._selected_body == bi:
                            selected_item = child
                root_item.addChild(item)
                if component is self._selected and (single or self._selected_body is None) and \
                        self._selected_condition is None and self._selected_probe is None:
                    selected_item = item
            group_font = QFont(self.tree.font()); group_font.setItalic(True)
            if self.study["conditions"]:
                group = QTreeWidgetItem(["Face conditions"])
                group.setFont(0, group_font)
                group.setIcon(0, _icon("heat_convection.png"))
                root_item.addChild(group)
                for index, condition in enumerate(self.study["conditions"]):
                    item = QTreeWidgetItem([self._condition_text(condition)])
                    item.setToolTip(0, self._condition_text(condition))
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(0, Qt.Checked if condition.get("enabled", True) else Qt.Unchecked)
                    item.setData(0, Qt.UserRole + 2, f"condition:{index}")
                    item.setForeground(0, QColor.fromRgbF(*CONDITION_KINDS[condition["kind"]][1]))
                    group.addChild(item)
                    if index == self._selected_condition:
                        selected_item = item
            if self.study["probes"]:
                group = QTreeWidgetItem(["Probes"])
                group.setFont(0, group_font)
                group.setIcon(0, _icon("measurement.png"))
                root_item.addChild(group)
                for index, probe in enumerate(self.study["probes"]):
                    item = QTreeWidgetItem([f"{probe['name']}   on {probe['component']}"])
                    item.setData(0, Qt.UserRole + 2, f"probe:{index}")
                    group.addChild(item)
                    if index == self._selected_probe:
                        selected_item = item
            self.tree.expandAll()
            if selected_item is not None:
                self.tree.setCurrentItem(selected_item)
        finally:
            self._loading = False

    @staticmethod
    def _condition_text(condition):
        kind = condition["kind"]
        value = {"fixed": lambda c: f"{c['temperature_c']:g} °C",
                 "flux": lambda c: f"{c['power_w']:g} W" if c["flux_mode"] == "total"
                 else f"{c['flux_w_m2']:g} W/m²",
                 "convection": lambda c: f"h {c['h']:g}" + ("" if c["fluid_ambient"]
                                                           else f", air {c['fluid_c']:g} °C"),
                 "insulated": lambda c: ""}[kind](condition)
        count = len(condition["faces"])
        return (f"{condition['name']}   [{CONDITION_KINDS[kind][0]}" + (f", {value}" if value else "")
                + f"]   {count} face{'s' if count != 1 else ''}" + ("" if count else "  ⚠"))

    @staticmethod
    def _item_extra(item):
        """("condition" | "probe", index) of a face-condition or probe item, else (None, None)."""
        data = item.data(0, Qt.UserRole + 2) if item is not None else None
        if not isinstance(data, str) or ":" not in data:
            return None, None
        kind, index = data.split(":", 1)
        return kind, int(index)

    def _item_target(self, item):
        if item is None:
            return None, None
        ci, bi = item.data(0, Qt.UserRole), item.data(0, Qt.UserRole + 1)
        if not isinstance(ci, int) or not 0 <= ci < len(self.study["components"]):
            return None, None
        return self.study["components"][ci], (bi if isinstance(bi, int) and bi >= 0 else None)

    def _on_tree_selection(self):
        if self._loading:
            return
        kind, index = self._item_extra(self.tree.currentItem())
        if kind == "condition":
            self._select_condition(index, update_tree=False)
            return
        if kind == "probe":
            self._select_probe(index, update_tree=False)
            return
        component, body_index = self._item_target(self.tree.currentItem())
        self._select(component, body_index, update_tree=False)

    def _on_tree_item_changed(self, item, _column):
        if self._loading:
            return
        kind, index = self._item_extra(item)
        if kind == "condition" and 0 <= index < len(self.study["conditions"]):
            self.study["conditions"][index]["enabled"] = item.checkState(0) == Qt.Checked
            self._changed()
            self._refresh_overlays()
            if index == self._selected_condition:
                self._show_properties()
            return
        component, body_index = self._item_target(item)
        if component is None or body_index is not None:
            return
        component["visible"] = item.checkState(0) == Qt.Checked
        for body in component["bodies"]:
            if body.get("_render") is not None:
                body["_render"]["visible"] = component["visible"]
        self._update_gizmo()
        self.view.update()
        self._changed(results=False)

    def _on_tree_rclick(self, pos):
        item = self.tree.itemAt(pos)
        kind, index = self._item_extra(item)
        if kind is not None:
            if kind == "condition":
                self._select_condition(index)
            else:
                self._select_probe(index)
            menu = QMenu(self)
            pick = menu.addAction("Pick faces") if kind == "condition" else None
            rename = menu.addAction("Rename…")
            remove = menu.addAction("Remove")
            chosen = menu.exec_(self.tree.viewport().mapToGlobal(pos))
            if chosen is not None and chosen is pick:
                self._set_pick_mode("faces")
            elif chosen is not None and chosen is rename:
                self._rename_selected()
            elif chosen is not None and chosen is remove:
                self._remove_selected()
            return
        component, body_index = self._item_target(item)
        if component is None:
            return
        self._select(component, body_index)
        menu = QMenu(self)
        update = menu.addAction("Update") if component.get("kind") in SOURCE_KINDS else None
        power = menu.addAction("Set power…") if self._body() is not None else None
        rename = menu.addAction("Rename…")
        remove = menu.addAction("Remove")
        chosen = menu.exec_(self.tree.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is update:
            self._update_components([component])
        elif chosen is power:
            self._focus_power()
        elif chosen is rename:
            self._rename(component)
        elif chosen is remove:
            self._remove_selected()

    def _on_body_clicked(self, entry):
        if entry is None:
            self._select(None)
            return
        component = entry.get("_thermal_component")
        self._select(component, entry.get("_body_index"))

    def _body(self):
        component, index = self._selected, self._selected_body
        if component is None or index is None or not 0 <= index < len(component["bodies"]):
            return None
        return component["bodies"][index]

    def _select(self, component, body_index=None, update_tree=True):
        self._selected_condition = None
        self._selected_probe = None
        if self.view.pick_mode == "faces":
            self._set_pick_mode(None)
        self._selected = component
        if component is None:
            body_index = None
        elif len(component["bodies"]) == 1:
            body_index = 0
        elif body_index is not None and not 0 <= body_index < len(component["bodies"]):
            body_index = None
        self._selected_body = body_index
        if update_tree:
            self._loading = True
            try:
                self.tree.clearSelection()

                def find(parent):
                    for row in range(parent.childCount()):
                        item = parent.child(row)
                        target, index = self._item_target(item)
                        if target is component and (index == body_index or
                                                    (index is None and len(component["bodies"]) == 1)):
                            return item
                        found = find(item)
                        if found is not None:
                            return found
                    return None
                root_item = self.tree.topLevelItem(0)
                item = find(root_item) if root_item is not None and component is not None else None
                if item is not None:
                    self.tree.setCurrentItem(item)
            finally:
                self._loading = False
        self._sync_selection()
        self._show_properties()
        self._select_result_row()
        self._refresh_overlays()
        self.view.update()

    def _select_extra(self, kind, index, update_tree):
        self._selected, self._selected_body = None, None
        self._selected_condition = index if kind == "condition" else None
        self._selected_probe = index if kind == "probe" else None
        if kind != "condition" and self.view.pick_mode == "faces":
            self._set_pick_mode(None)
        if update_tree:
            self._rebuild_tree()
        self._sync_selection()
        self._show_properties()
        self._refresh_overlays()

    def _select_condition(self, index, update_tree=True):
        if 0 <= index < len(self.study["conditions"]):
            self._select_extra("condition", index, update_tree)

    def _select_probe(self, index, update_tree=True):
        if 0 <= index < len(self.study["probes"]):
            self._select_extra("probe", index, update_tree)

    # ── properties ────────────────────────────────────────────────────────────
    def _show_properties(self):
        if self._selected_condition is not None:
            self._props_stack.setCurrentIndex(1)
            self._show_condition()
            return
        if self._selected_probe is not None:
            self._props_stack.setCurrentIndex(2)
            self._show_probe()
            return
        self._props_stack.setCurrentIndex(0)
        component, body = self._selected, self._body()
        self._loading = True
        try:
            for spin in self.sp_pos + self.sp_rot:
                spin.setEnabled(component is not None)
            if component is None:
                self.lbl_source.setText("—")
                self.lbl_state.setText("—")
                for spin in self.sp_pos + self.sp_rot:
                    spin.setValue(0.0)
            else:
                for spin, value in zip(self.sp_pos, component["position"]):
                    spin.setValue(value)
                for spin, value in zip(self.sp_rot, component["rotation"]):
                    spin.setValue(value)
                self.lbl_source.setText(component.get("source") or component.get("path") or
                                        component["kind"])
                self.lbl_state.setText(
                    "Placed in the study (no source)" if component.get("kind") not in SOURCE_KINDS
                    else {"ok": "Up to date",
                          "outdated": "<span style='color:#ffb454'>Source changed — use Update</span>",
                          "missing": "<span style='color:#ff6b6b'>Source missing — the last "
                                     "snapshot is kept</span>"}[self._status.get(id(component), "ok")])
            primitive = component is not None and component["kind"] in PRIMITIVE_KINDS
            self.w_shape.setVisible(primitive)
            if primitive:
                shape = component.get("shape", {})
                self.st_shape.setCurrentIndex(0 if component["kind"] == "block" else 1)
                for spin, value in zip(self.sp_shape_size, shape.get("size", (10.0, 10.0, 2.0))):
                    spin.setValue(value)
                self.sp_shape_diameter.setValue(shape.get("diameter", 8.0))
                self.sp_shape_height.setValue(shape.get("height", 12.0))
                self.cb_shape_axis.setCurrentIndex(max(0, self.cb_shape_axis.findData(
                    shape.get("axis", "Z"))))
            for widget in (self.cb_material, self.sp_power, self.chk_emissivity):
                widget.setEnabled(body is not None)
            if body is None:
                self.lbl_body.setText("Select a body of the part" if component is not None else "—")
                self.cb_material.setCurrentIndex(0)
                self.lbl_material.setText("")
                self.sp_power.setValue(0.0)
                self.lbl_power.setText("")
                self.chk_emissivity.setChecked(False)
                self.sp_emissivity.setEnabled(False)
                return
            self.lbl_body.setText(body["name"] + ("  (inside a Bridge, not simulated)"
                                                  if body.get("fused") else ""))
            material = body.get("material")
            self.cb_material.setCurrentIndex(max(0, self.cb_material.findData(material))
                                             if material else 0)
            self.lbl_material.setText(material_summary(material) +
                                      (("\n" + MATERIAL_THERMAL[material].get("note", ""))
                                       if material in MATERIAL_THERMAL else ""))
            self.lbl_material.setStyleSheet("color:#aab2bd; font-size:9px;" if material else
                                            "color:#ffb454; font-size:9px;")
            power = float(body.get("power_w", 0.0))
            self.sp_power.setValue(power)
            volume = body_volume_mm3(body)
            if power > 0 and volume > 0:
                self.lbl_power.setText(f"= {power / (volume / 1000.0):.4g} W/cm³ over "
                                       f"{volume / 1000.0:.4g} cm³")
            else:
                self.lbl_power.setText("No heat source" if power <= 0 else "")
            own = body.get("emissivity") is not None
            self.chk_emissivity.setChecked(own)
            self.sp_emissivity.setEnabled(own)
            props = material_properties(material)
            self.sp_emissivity.setValue(body["emissivity"] if own else
                                        props["emissivity"] if props else 0.9)
        finally:
            self._loading = False

    def _on_placement_edited(self, _value=None):
        component = self._selected
        if self._loading or component is None:
            return
        component["position"] = [spin.value() for spin in self.sp_pos]
        component["rotation"] = [spin.value() for spin in self.sp_rot]
        self._apply_transform(component)
        self.view.update()
        self._changed()

    def _on_shape_edited(self, *_args):
        component = self._selected
        if self._loading or component is None or component["kind"] not in PRIMITIVE_KINDS:
            return
        if component["kind"] == "block":
            shape = {"size": [spin.value() for spin in self.sp_shape_size]}
        else:
            shape = {"diameter": self.sp_shape_diameter.value(),
                     "height": self.sp_shape_height.value(),
                     "axis": self.cb_shape_axis.currentData()}
        primitive_set_shape(component, shape)
        refresh_signatures(self.study, component)      # same faces, new sizes
        self._selected_body = 0
        self._rebuild_scene()
        self._show_properties()
        self._changed()

    def _on_material_changed(self, _index):
        body = self._body()
        if self._loading or body is None:
            return
        body["material"] = self.cb_material.currentData()
        body["material_auto"] = False              # chosen here: Update keeps it
        self._show_properties()
        self._rebuild_tree()
        self._changed()

    def _on_power_changed(self, value):
        body = self._body()
        if self._loading or body is None:
            return
        body["power_w"] = max(0.0, float(value))
        self._show_properties()
        self._rebuild_tree()
        self._changed()

    def _on_emissivity_changed(self, *_args):
        body = self._body()
        if self._loading or body is None:
            return
        body["emissivity"] = float(self.sp_emissivity.value()) \
            if self.chk_emissivity.isChecked() else None
        self._show_properties()
        self._changed()

    def _focus_power(self):
        if self._body() is None:
            self._set_status("Select a body first (click it in the view or in the study list)")
            return
        self.sp_power.setFocus()
        self.sp_power.selectAll()
        self._set_status(f"Enter the power of {self._body()['name']} in watts")

    # ── face conditions ───────────────────────────────────────────────────────
    def _condition(self):
        index = self._selected_condition
        if index is None or not 0 <= index < len(self.study["conditions"]):
            return None
        return self.study["conditions"][index]

    def _add_condition(self, kind):
        if self._worker is not None:
            return
        title = CONDITION_KINDS[kind][0]
        names = {c["name"] for c in self.study["conditions"]}
        number = 1
        while f"{title} {number}" in names:
            number += 1
        condition = condition_new(kind, f"{title} {number}")
        condition["temperature_c"] = float(self.study["settings"]["ambient_c"])
        condition["fluid_c"] = float(self.study["settings"]["ambient_c"])
        self.study["conditions"].append(condition)
        self._select_condition(len(self.study["conditions"]) - 1)
        self._changed()
        self._set_pick_mode("faces")

    def _show_condition(self):
        condition = self._condition()
        if condition is None:
            return
        kind = condition["kind"]
        self._loading = True
        try:
            self.lbl_condition_kind.setText(CONDITION_KINDS[kind][0])
            self.lbl_condition_name.setText(condition["name"])
            self.chk_condition_enabled.setChecked(condition.get("enabled", True))
            self.sp_condition_temperature.setValue(condition["temperature_c"])
            self.cb_flux_mode.setCurrentIndex(0 if condition["flux_mode"] == "total" else 1)
            self.sp_condition_power.setValue(condition["power_w"])
            self.sp_condition_flux.setValue(condition["flux_w_m2"])
            self.sp_condition_h.setValue(condition["h"])
            self.chk_condition_fluid.setChecked(condition["fluid_ambient"])
            self.sp_condition_fluid.setValue(self.study["settings"]["ambient_c"]
                                             if condition["fluid_ambient"] else condition["fluid_c"])
            self.sp_condition_fluid.setEnabled(not condition["fluid_ambient"])
            for rows_kind, widgets in self._condition_rows.items():
                for widget in widgets:
                    if widget is not None:
                        widget.setVisible(rows_kind == kind)
            if kind == "flux":
                total = condition["flux_mode"] == "total"
                for widget in (self.sp_condition_power,
                               self._condition_rows["flux"][3]):
                    widget.setVisible(total)
                for widget in (self.sp_condition_flux, self._condition_rows["flux"][5]):
                    widget.setVisible(not total)
            self.lbl_condition_hint.setText(CONDITION_HINTS[kind])
            faces = condition["faces"]
            parts = {ref["component"] for ref in faces}
            area = 0.0
            components = {c["name"]: c for c in self.study["components"]}
            for ref in faces:
                component = components.get(ref["component"])
                if component is not None and 0 <= ref["body"] < len(component["bodies"]):
                    body = component["bodies"][ref["body"]]
                    if 0 <= ref["face"] < len(body["faces"]):
                        area += face_signature(body, ref["face"])[6]
            self.lbl_condition_faces.setText(
                "none — pick faces in the view" if not faces else
                f"{len(faces)} face{'s' if len(faces) != 1 else ''} on {len(parts)} "
                f"part{'s' if len(parts) != 1 else ''}, {area:.4g} mm²")
            self.btn_pick_faces.setChecked(self.view.pick_mode == "faces")
            self.btn_clear_faces.setEnabled(bool(faces))
        finally:
            self._loading = False

    def _on_condition_edited(self, *_args):
        condition = self._condition()
        if self._loading or condition is None:
            return
        condition["enabled"] = self.chk_condition_enabled.isChecked()
        condition["temperature_c"] = float(self.sp_condition_temperature.value())
        condition["flux_mode"] = self.cb_flux_mode.currentData()
        condition["power_w"] = float(self.sp_condition_power.value())
        condition["flux_w_m2"] = float(self.sp_condition_flux.value())
        condition["h"] = float(self.sp_condition_h.value())
        condition["fluid_ambient"] = self.chk_condition_fluid.isChecked()
        if not condition["fluid_ambient"]:
            condition["fluid_c"] = float(self.sp_condition_fluid.value())
        self._show_condition()
        self._rebuild_tree()
        self._refresh_overlays()
        self._changed()

    def _clear_condition_faces(self):
        condition = self._condition()
        if condition is None or not condition["faces"]:
            return
        condition["faces"] = []
        self._show_condition()
        self._rebuild_tree()
        self._refresh_overlays()
        self._changed()

    def _set_pick_mode(self, mode):
        """None, "faces" (add / remove faces of the selected condition) or "probe"."""
        if mode == "faces" and self._condition() is None:
            mode = None
        self.view.set_pick_mode(mode)
        for widget, on in ((self.btn_pick_faces, mode == "faces"), (self.act_probe, mode == "probe")):
            widget.blockSignals(True)
            widget.setChecked(on)
            widget.blockSignals(False)
        self._update_gizmo()
        if mode == "faces":
            self._set_status(f"Click faces to add them to '{self._condition()['name']}' (click again "
                             "to remove; Esc ends)")
        elif mode == "probe":
            self._set_status("Click on a part to place a probe (Esc ends)")
        self.view.update()

    def _on_face_picked(self, entry, face):
        condition = self._condition()
        component = entry.get("_thermal_component")
        body_index = entry.get("_body_index")
        if condition is None or component is None or body_index is None:
            return
        added = condition_toggle_face(condition, component, body_index, face)
        self._show_condition()
        self._rebuild_tree()
        self._refresh_overlays()
        self._changed()
        self._set_status(f"{'Added' if added else 'Removed'} a face of {component['name']} "
                         f"{'to' if added else 'from'} '{condition['name']}'")

    # ── probes ────────────────────────────────────────────────────────────────
    def _on_probe_toggled(self, on):
        if on:
            self._set_pick_mode("probe")
        elif self.view.pick_mode == "probe":
            self._set_pick_mode(None)

    def _on_point_picked(self, entry, point):
        component = entry.get("_thermal_component")
        body_index = entry.get("_body_index")
        if component is None or body_index is None:
            return
        names = {p["name"] for p in self.study["probes"]}
        number = 1
        while f"P{number}" in names:
            number += 1
        local = cad.assembly_local_point(component, point)
        self.study["probes"].append({"name": f"P{number}", "component": component["name"],
                                     "body": int(body_index), "point": [float(x) for x in local]})
        self._changed(results=False)          # probes read the existing results
        self._rebuild_tree()
        self._refresh_overlays()
        self._fill_probe_table()
        self._update_plot()
        self._set_status(f"Probe P{number} placed on {component['name']} — click for more, Esc ends")

    def _probe_series(self, probe):
        """Temperatures of a probe for every stored frame of the result, or None. The probe
        takes the result-surface vertex next to it (surfaces are refined to the cell size)."""
        result = self._result
        if result is None:
            return None
        index = next((i for i, c in enumerate(self.study["components"])
                      if c["name"] == probe["component"]), None)
        surface = result["surface"].get((index, probe["body"])) if index is not None else None
        point = probe_world_point(self.study, probe)
        if surface is None or point is None:
            return None
        key = (id(result), probe["component"], probe["body"], tuple(probe["point"]))
        if key not in self._probe_cache:
            vertex = int(np.argmin(((surface["vertices"] - point) ** 2).sum(1)))
            columns = surface["index"][vertex]
            self._probe_cache[key] = (result["frames"][:, columns] * surface["weights"][vertex]).sum(1)
        return self._probe_cache[key]

    def _probe_frame_values(self):
        """Temperature of every probe at the shown frame (None without a result)."""
        out = []
        for probe in self.study["probes"]:
            series = self._probe_series(probe)
            out.append(None if series is None else float(series[min(self._frame, len(series) - 1)]))
        return out

    def _fill_probe_table(self):
        probes = self.study["probes"]
        self.lbl_probes.setVisible(bool(probes))
        self.tbl_probes.setVisible(bool(probes))
        values = self._probe_frame_values()
        self.tbl_probes.blockSignals(True)
        self.tbl_probes.setRowCount(len(probes))
        for row, (probe, value) in enumerate(zip(probes, values)):
            for column, text in enumerate((probe["name"], probe["component"],
                                           "—" if value is None else f"{value:.1f}")):
                item = QTableWidgetItem(text)
                if column == 2:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl_probes.setItem(row, column, item)
            if row == self._selected_probe:
                self.tbl_probes.selectRow(row)
        self.tbl_probes.blockSignals(False)

    def _on_probe_row(self):
        rows = self.tbl_probes.selectionModel().selectedRows()
        if rows:
            self._select_probe(rows[0].row())

    def _show_probe(self):
        index = self._selected_probe
        if index is None or not 0 <= index < len(self.study["probes"]):
            return
        probe = self.study["probes"][index]
        point = probe_world_point(self.study, probe)
        self.lbl_probe_name.setText(probe["name"])
        self.lbl_probe_part.setText(probe["component"])
        self.lbl_probe_point.setText("—" if point is None else
                                     "  ".join(f"{axis} {value:.2f}" for axis, value in zip("XYZ", point)))
        value = self._probe_frame_values()[index]
        if value is None:
            self.lbl_probe_value.setText("— (start the simulation)" if self._result is None else "—")
        else:
            when = "" if self._result["analysis"] != "transient" else \
                f" at t = {format_time(self._result['frame_times'][self._frame])}"
            self.lbl_probe_value.setText(f"{value:.2f} °C{when}")

    def _rename_selected(self):
        condition = self._condition()
        if condition is not None or self._selected_probe is not None:
            item = condition if condition is not None else self.study["probes"][self._selected_probe]
            items = self.study["conditions"] if condition is not None else self.study["probes"]
            name, ok = QInputDialog.getText(self, "Rename", "New name:", text=item["name"])
            name = name.strip()
            if not ok or not name or name == item["name"]:
                return
            if any(other["name"] == name for other in items):
                self._set_status(f"'{name}' is already used")
                return
            item["name"] = name
            self._rebuild_tree()
            self._show_properties()
            self._refresh_overlays()
            self._fill_probe_table()
            self._update_plot()
            self._changed(results=False)
            return
        self._rename(self._selected)

    # ── settings ──────────────────────────────────────────────────────────────
    def _show_settings(self):
        settings = self.study["settings"]
        self._loading = True
        try:
            self.sp_ambient.setValue(settings["ambient_c"])
            self.cb_convection.setCurrentIndex(max(0, self.cb_convection.findData(
                settings["convection"])))
            self.sp_h.setValue(settings["h"])
            self.sp_h.setEnabled(settings["convection"] == "fixed")
            self.chk_radiation.setChecked(settings["radiation"])
            automatic = settings["cell_mm"] <= 0
            self.chk_auto_cell.setChecked(automatic)
            self.sp_cell.setEnabled(not automatic)
            if not automatic:
                self.sp_cell.setValue(settings["cell_mm"])
            self.sp_max_cells.setValue(int(settings["max_cells"]))
            transient = settings["analysis"] == "transient"
            self.cb_analysis.setCurrentIndex(max(0, self.cb_analysis.findData(settings["analysis"])))
            unit = settings["duration_unit"]
            self.cb_duration_unit.setCurrentIndex(max(0, self.cb_duration_unit.findData(unit)))
            self.sp_duration.setValue(settings["duration_s"] / TIME_UNITS[unit])
            self.sp_time_steps.setValue(int(settings["time_steps"]))
            self.chk_initial_ambient.setChecked(settings["initial_ambient"])
            self.sp_initial.setValue(settings["ambient_c"] if settings["initial_ambient"]
                                     else settings["initial_c"])
            self.sp_initial.setEnabled(not settings["initial_ambient"])
            for widget in self._transient_rows:
                if widget is not None:
                    widget.setVisible(transient)
        finally:
            self._loading = False

    def _on_settings_edited(self, *_args):
        if self._loading:
            return
        settings = self.study["settings"]
        settings["ambient_c"] = float(self.sp_ambient.value())
        settings["convection"] = self.cb_convection.currentData()
        settings["h"] = float(self.sp_h.value())
        settings["radiation"] = self.chk_radiation.isChecked()
        settings["cell_mm"] = 0.0 if self.chk_auto_cell.isChecked() else float(self.sp_cell.value())
        settings["max_cells"] = int(self.sp_max_cells.value())
        settings["analysis"] = self.cb_analysis.currentData()
        unit = self.cb_duration_unit.currentData()
        if unit != settings["duration_unit"]:
            settings["duration_unit"] = unit           # same duration, shown in the new unit
        else:
            settings["duration_s"] = max(0.001, self.sp_duration.value() * TIME_UNITS[unit])
        settings["time_steps"] = int(self.sp_time_steps.value())
        settings["initial_ambient"] = self.chk_initial_ambient.isChecked()
        if not settings["initial_ambient"]:
            settings["initial_c"] = float(self.sp_initial.value())
        self._show_settings()
        self._changed()

    def _update_estimate(self):
        settings = self.study["settings"]
        if settings["analysis"] == "transient":
            try:
                tau = estimate_time_constant(self.study)
            except Exception:
                tau = None
            dt = settings["duration_s"] / settings["time_steps"]
            text = f"Time step {format_time(dt)}. "
            if tau is None:
                text += "The time constant is shown once all bodies have a material."
            else:
                text += (f"Estimated time constant ≈ {format_time(tau)}: after about "
                         f"{format_time(5 * tau)} the temperatures are close to steady state.")
            self.lbl_time_constant.setText(text)
        try:
            estimate = estimate_cells(self.study)
        except Exception:
            estimate = None
        if estimate is None:
            self.lbl_estimate.setText("No visible bodies yet.")
            return
        h, cells = estimate
        self.lbl_estimate.setText(f"≈ {cells:,} solid cells of {h:g} mm".replace(",", " ") +
                                  (" (automatic)" if self.study["settings"]["cell_mm"] <= 0 else ""))
        if self.study["settings"]["cell_mm"] <= 0:
            self.sp_cell.blockSignals(True)
            self.sp_cell.setValue(h)
            self.sp_cell.blockSignals(False)

    # ── adding and removing parts ─────────────────────────────────────────────
    def _start_directory(self):
        for path in (self._window._thermal_file, getattr(self._window, "_file", None)):
            if path:
                return os.path.dirname(path)
        return ""

    def _after_components_changed(self, select=None):
        if select is not None:
            self._selected = select
            self._selected_body = 0 if len(select["bodies"]) == 1 else None
        if self._selected is not None and \
                not any(self._selected is c for c in self.study["components"]):
            self._selected, self._selected_body = None, None
        self._refresh_status()
        self._rebuild_scene()
        self._rebuild_tree()
        self._show_properties()
        self._changed()

    def _added(self, components, text):
        if not components:
            return
        self._after_components_changed(select=components[-1])
        self.view.fit_all()
        missing = sum(1 for component in components for body in component["bodies"]
                      if not body.get("material") and not body.get("fused"))
        if missing:
            text += f" — {missing} bod{'y needs' if missing == 1 else 'ies need'} a material (⚠)"
        self._set_status(text)

    @staticmethod
    def _to_thermal(component, source):
        component["bodies"] = [thermal_body_from_assembly(body) for body in component["bodies"]]
        component["source"] = source
        return component

    def _import_files(self, kind):
        patterns = {"step": "STEP files (*.step *.stp)", "stl": "STL files (*.stl)",
                    "obj": "OBJ files / SimpleCAD projects (*.obj)"}
        paths, _ = QFileDialog.getOpenFileNames(self, "Import into the thermal study",
                                                self._start_directory(), patterns[kind])
        if not paths:
            return
        added, failed = [], []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for path in paths:
                try:
                    for component in cad.assembly_add_file_components(self.study, path, self._window):
                        root = component.get("file_root")
                        added.append(self._to_thermal(
                            component, path + (f" — {root}" if root else "")))
                except Exception as ex:
                    failed.append(f"{os.path.basename(path)}: {ex}")
        finally:
            QApplication.restoreOverrideCursor()
        if failed:
            QMessageBox.critical(self, "Import failed", "\n".join(failed))
        self._added(added, f"Imported {', '.join(c['name'] for c in added)}")

    def _choose(self, title, text, labels):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(text))
        listing = QListWidget()
        listing.setStyleSheet(_LIST_INDICATOR_STYLE)        # checkboxes as in the object explorer
        for label in labels:
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            listing.addItem(item)
        layout.addWidget(listing)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(380, 360)
        if dialog.exec_() != QDialog.Accepted:
            return []
        return [row for row in range(listing.count())
                if listing.item(row).checkState() == Qt.Checked]

    def _import_project_parts(self):
        objects = self._window.state["objects"]
        roots = cad.assembly_root_entries(objects)
        if not roots:
            QMessageBox.information(self, "Parts of the open project",
                                    "The open project has no solids yet.")
            return
        labels = []
        for root in roots:
            extra = len(cad.assembly_part_entries(root, objects)) - 1
            labels.append(root["name"] + (f"   (+{extra} built on it)" if extra else ""))
        rows = self._choose("Parts of the open project",
                            "Choose the parent objects. Everything built on a parent is part of it.",
                            labels)
        added = [self._to_thermal(cad.assembly_add_project_component(self.study, roots[row], objects),
                                  f"Project object {roots[row]['name']}") for row in rows]
        self._added(added, f"Added {', '.join(c['name'] for c in added)}")

    def _import_project_assembly(self):
        assembly = getattr(self._window, "_assembly", None)
        if not assembly or not assembly.get("components"):
            QMessageBox.information(self, "Assembly of the open project",
                                    "The open project has no assembly yet (Assembly window).")
            return
        self._import_assembly(assembly, "of the project", objects=self._window.state["objects"])

    def _import_assembly_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import an assembly", self._start_directory(),
            f"SimpleCAD assembly (*{cad.ASSEMBLY_EXTENSION})")
        if not path:
            return
        try:
            assembly = cad.assembly_load_file(path)
        except (OSError, ValueError, KeyError, TypeError) as ex:
            QMessageBox.critical(self, "Import failed", f"Could not open\n{path}\n\n{ex}")
            return
        if not assembly["components"]:
            QMessageBox.information(self, "Import", "The assembly has no components.")
            return
        self._import_assembly(assembly, os.path.basename(path), objects=self._window.state["objects"])

    def _import_assembly(self, assembly, label, objects=None):
        """Every assembly component becomes a study part at its assembly placement, with the
        component's own source (project object or file). Bodies without a stored material take
        it from the project object of the same name."""
        sources = {o.get("name"): o for o in (objects or [])
                   if o.get("type") in ("extrusion", "imported")}
        names = {component["name"] for component in self.study["components"]}
        added = []
        for item in assembly["components"]:
            bodies = []
            for body in item["bodies"]:
                material = body.get("material")
                if material is None and item.get("kind") == "project" and body["name"] in sources:
                    material = sources[body["name"]].get("material")
                bodies.append(thermal_body_from_assembly(body, material))
            name, number = item["name"], 2
            while name in names:
                name = f"{item['name']} ({number})"; number += 1
            names.add(name)
            # the part keeps the source of the assembly component (project object or file), so
            # Update re-reads it like the Assembly window does
            component = {"name": name, "visible": bool(item.get("visible", True)),
                         "kind": item.get("kind") if item.get("kind") in SOURCE_KINDS else "assembly",
                         "position": [float(x) for x in item["position"]],
                         "rotation": [float(x) for x in item["rotation"]],
                         "pivot": [float(x) for x in item["pivot"]], "bodies": bodies,
                         "source": f"Assembly {label}: {item['name']}"}
            for key in ("source_name", "signature", "path", "file_stamp", "file_root"):
                if key in item:
                    component[key] = copy.deepcopy(item[key])
            self.study["components"].append(component)
            added.append(component)
        self._added(added, f"Imported {len(added)} assembly part{'s' if len(added) != 1 else ''}")

    def _add_heat_source(self, kind):
        anchor = None
        if self._selected is not None:
            box = cad.assembly_bounds(self._selected)
            if box is not None:
                anchor = (self._selected["name"], box)
        number = 1 + sum(1 for c in self.study["components"] if c["kind"] in PRIMITIVE_KINDS)
        dialog = HeatSourceDialog(self, kind, f"Heat source {number}", anchor)
        if dialog.exec_() != QDialog.Accepted:
            return
        kind, shape, name, position, power, material = dialog.values()
        component = study_add_primitive(self.study, kind, shape, name, position, power, material)
        self._after_components_changed(select=component)
        self._set_status(f"Placed {component['name']} ({power:g} W, {material}) — move it with the "
                         "arrows or the position fields")

    def _remove_selected(self):
        if self._worker is not None:
            return
        condition = self._condition()
        if condition is not None or self._selected_probe is not None:
            items = self.study["conditions"] if condition is not None else self.study["probes"]
            index = self._selected_condition if condition is not None else self._selected_probe
            name = items[index]["name"]
            if self.view.pick_mode is not None:
                self._set_pick_mode(None)
            del items[index]
            self._selected_condition = self._selected_probe = None
            self._rebuild_tree()
            self._show_properties()
            self._refresh_overlays()
            self._fill_probe_table()
            self._changed(results=condition is not None)
            self._update_plot()
            self._set_status(f"Removed {name}")
            return
        component = self._selected
        if component is None:
            self._set_status("Select a part, face condition or probe first")
            return
        if QMessageBox.question(self, "Remove", f"Remove '{component['name']}' from the study?\n"
                                "Project objects and files are not changed. Its faces leave the "
                                "face conditions and its probes are removed.",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.study["components"] = [c for c in self.study["components"] if c is not component]
        drop_component_refs(self.study, component["name"])
        self._selected, self._selected_body = None, None
        self._after_components_changed()
        self._fill_probe_table()
        self._set_status(f"Removed {component['name']}")

    def _rename(self, component):
        if component is None:
            self._set_status("Select a part first")
            return
        name, ok = QInputDialog.getText(self, "Rename", "New name:", text=component["name"])
        name = name.strip()
        if not ok or not name or name == component["name"]:
            return
        if any(c["name"] == name for c in self.study["components"]):
            self._set_status(f"'{name}' is already used in this study")
            return
        rename_component_refs(self.study, component["name"], name)
        component["name"] = name
        if component["kind"] in PRIMITIVE_KINDS and component["bodies"]:
            component["bodies"][0]["name"] = name
        self._probe_cache.clear()
        self._rebuild_scene()
        self._rebuild_tree()
        self._show_properties()
        self._fill_probe_table()
        self._changed(results=False)

    def _update_components(self, components, every=False):
        """Re-read the sources of parts, like Update Selected / Update All in the Assembly."""
        if not components:
            self._set_status("No part with a source to update (heat sources placed in the study "
                             "have none)" if every else "Select a part first")
            return
        objects = self._window.state["objects"]
        updated, failed, lost = [], [], 0
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for component in components:
                try:
                    study_refresh_component(component, objects, self._window)
                    lost += remap_condition_faces(self.study, component)
                    updated.append(component["name"])
                except Exception as ex:
                    failed.append(f"{component['name']}: {ex}")
        finally:
            QApplication.restoreOverrideCursor()
        if lost:
            failed.append(f"{lost} face{'s' if lost != 1 else ''} of face conditions no longer "
                          "exist and were removed")
        if updated:
            if self._selected is not None and self._selected_body is not None and \
                    self._selected_body >= len(self._selected["bodies"]):
                self._selected_body = None
            self._after_components_changed()
        else:
            self._refresh_status(); self._rebuild_tree(); self._show_properties()
        text = f"Updated {', '.join(updated)}" if updated else ""
        if failed:
            text += ("   " if text else "") + "Not updated — " + "; ".join(failed)
        self._set_status(text)

    def _show_all(self):
        for component in self.study["components"]:
            component["visible"] = True
        self._rebuild_scene()
        self._rebuild_tree()
        self.view.fit_all()
        self._changed(results=False)

    # ── move / rotate arrows ──────────────────────────────────────────────────
    def _on_gizmo_started(self):
        component = self._selected
        box = cad.assembly_bounds(component) if component is not None else None
        self._drag_base = None if box is None else {
            "component": component, "position": list(component["position"]),
            "rotation": list(component["rotation"]), "centre": (box[0] + box[1]) / 2.0}

    def _on_gizmo_moved(self, kind, axis, value):
        base = self._drag_base
        if base is None or kind not in ("move", "rotate"):
            return
        component = base["component"]
        if kind == "move":
            component["position"] = [base["position"][i] + (value if i == axis else 0.0)
                                     for i in range(3)]
            self._set_status(f"Move {component['name']} along {'XYZ'[axis]}: {value:+.2f} mm")
        else:
            cad.assembly_rotate_about(component, base, base["centre"], axis, value)
            self._set_status(f"Rotate {component['name']} about {'XYZ'[axis]}: {value:+.1f}°")
        self._apply_transform(component)
        self._show_properties()
        self.view.update()

    def _on_gizmo_finished(self):
        if self._drag_base is not None:
            self._drag_base = None
            self._changed()

    # ── files ─────────────────────────────────────────────────────────────────
    def _save(self, save_as=False):
        path = None if save_as else self._window._thermal_file
        if not path:
            start = self._window._thermal_file or os.path.join(
                self._start_directory(), (self.study["name"] or THERMAL_DEFAULT_NAME) + THERMAL_EXTENSION)
            path, _ = QFileDialog.getSaveFileName(self, "Save thermal study", start,
                                                  f"SimpleCAD thermal study (*{THERMAL_EXTENSION})")
            if not path:
                return False
        if not path.lower().endswith(THERMAL_EXTENSION):
            path = os.path.splitext(path)[0] + THERMAL_EXTENSION
        self.study["name"] = os.path.splitext(os.path.basename(path))[0]
        try:
            study_save_file(self.study, path)
        except (OSError, ValueError) as ex:
            QMessageBox.critical(self, "Save failed", f"Could not save\n{path}\n\n{ex}")
            return False
        self._window._thermal_file = path
        self._rebuild_tree()
        self._mark_saved()
        self._set_status(f"Thermal study saved to {path}")
        return True

    def _new_file(self):
        if self._worker is not None:
            return
        if self.has_unsaved_changes():
            answer = QMessageBox.question(
                self, "New thermal study",
                f"The thermal study '{self.study['name']}' has unsaved changes. Save them first?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if answer == QMessageBox.Cancel or (answer == QMessageBox.Save and not self._save()):
                return
        self.study.clear()
        self.study.update(study_new())
        self._window._thermal_file = None
        self._selected, self._selected_body = None, None
        self._selected_condition = self._selected_probe = None
        self._set_pick_mode(None)
        self._discard_results()
        self.txt_messages.clear()
        self._refresh_status()
        self._rebuild_scene()
        self._rebuild_tree()
        self._show_properties()
        self._show_settings()
        self._reset_history()
        self._mark_saved()
        self._update_estimate()
        self.view.fit_all()
        self._set_status("New thermal study — import parts or place a heat source")

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open thermal study", self._start_directory(),
                                              f"SimpleCAD thermal study (*{THERMAL_EXTENSION})")
        if not path:
            return
        if self.has_unsaved_changes() and QMessageBox.question(
                self, "Open thermal study", "Discard the unsaved changes of the current study?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            loaded = study_load_file(path)
        except (OSError, ValueError, KeyError, TypeError) as ex:
            QMessageBox.critical(self, "Open failed", f"Could not open\n{path}\n\n{ex}")
            return
        self.study.clear()
        self.study.update(loaded)
        self._window._thermal_file = path
        self._selected, self._selected_body = None, None
        self._selected_condition = self._selected_probe = None
        self._set_pick_mode(None)
        self._discard_results()
        self._refresh_status()
        self._rebuild_scene()
        self._rebuild_tree()
        self._show_properties()
        self._show_settings()
        self._reset_history()
        self._mark_saved()
        self._update_estimate()
        self.view.fit_all()
        self._set_status(f"Opened {path}")

    # ── simulation ────────────────────────────────────────────────────────────
    def _start_simulation(self):
        if self._worker is not None:
            return
        jobs = study_jobs(self.study)
        if not jobs:
            QMessageBox.information(self, ADDON_TITLE, "There is nothing to simulate. Import parts "
                                    "or place a heat source first (hidden parts are left out).")
            return
        missing = [job for job in jobs if job["material"] not in MATERIAL_THERMAL]
        if missing:
            ci, bi = missing[0]["key"]
            self._select(self.study["components"][ci], bi)
            names = "\n".join("•  " + job["name"] for job in missing[:10]) + \
                ("\n…" if len(missing) > 10 else "")
            QMessageBox.warning(self, "Material missing",
                                f"These bodies have no material yet:\n\n{names}\n\n"
                                "Choose a material in the properties (Body → Material).")
            return
        conditions = study_conditions(self.study, jobs)
        drives = any(c["kind"] in ("flux", "fixed") and c["targets"] for c in conditions)
        if sum(job["power"] for job in jobs) <= 0 and not drives:
            QMessageBox.warning(self, "No heat source",
                                "No body produces heat. Place a heat source (Heat Source → Block / "
                                "Cylinder), give a body a power in the properties, or put a heat "
                                "flux or fixed temperature on faces (Face Condition).")
            return
        empty = [c["name"] for c in self.study["conditions"] if c.get("enabled", True) and not c["faces"]]
        if empty:
            self.txt_messages.setPlainText("Face conditions without faces are left out: " +
                                           ", ".join(empty))
        if self.view.pick_mode is not None:
            self._set_pick_mode(None)
        self._discard_results()
        worker = ThermalWorker(jobs, dict(self.study["settings"]), conditions)
        worker.progressed.connect(self._on_progress)
        worker.succeeded.connect(self._on_simulation_done)
        worker.failed.connect(self._on_simulation_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        self._set_running(True)
        self._set_status("Simulation started...")
        worker.start()

    def _cancel_simulation(self):
        if self._worker is not None:
            self._worker.cancel()
            self._set_status("Stopping the simulation...")

    def _pause_main_redraw(self, pause):
        """The main window's continuous FPS redraw holds the Python interpreter lock so often
        that it slows the solver thread several times; it pauses while a simulation runs."""
        view = getattr(self._window, "view3d", None)
        timer = getattr(view, "_fps_timer", None)
        if timer is None:
            return
        if pause:
            self._fps_paused = timer.isActive()
            if self._fps_paused:
                timer.stop()
        elif getattr(self, "_fps_paused", False):
            self._fps_paused = False
            if view.show_fps:
                timer.start()

    def _set_running(self, running):
        self._pause_main_redraw(running)
        for act in (self.act_new, self.act_open, self.act_save, self.act_save_as, self.act_import,
                    self.act_remove, self.act_heat, self.act_start, self.act_update_selected,
                    self.act_update_all, self.act_condition, self.act_probe):
            act.setEnabled(not running)
        self.act_cancel.setEnabled(running)
        self.act_results.setEnabled(not running and self._result is not None)
        self._update_history_actions()
        self._left_content.setEnabled(not running)
        if running:
            for widget in self._setting_inputs:
                widget.setEnabled(False)
        else:
            for widget in self._setting_inputs:
                widget.setEnabled(True)
            self._show_settings()               # h / cell size follow their mode again
        self.progress.setVisible(running)
        self.btn_cancel.setVisible(running)
        if running:
            self.progress.setValue(0)
        self._update_gizmo()
        self.view.update()

    def _on_progress(self, percent, text):
        self.progress.setValue(percent)
        self.status.setText(text)

    def _on_simulation_done(self, result):
        self._result = result
        self._show_result()
        hottest = max((b for b in result["bodies"] if b["max"] is not None),
                      key=lambda b: b["max"], default=None)
        self._set_status(f"Simulation finished in {result['seconds']:.1f} s — maximum "
                         f"{result['hot_value']:.1f} °C" +
                         (f" in {hottest['name']}" if hottest else ""))

    def _on_simulation_failed(self, text):
        if text:
            self.txt_messages.setPlainText(text)
            QMessageBox.warning(self, "Simulation not possible", text)
            self._set_status("Simulation stopped: " + text)
        else:
            self._set_status("Simulation cancelled")

    def _on_worker_finished(self):
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()
        self._set_running(False)

    # ── results ───────────────────────────────────────────────────────────────
    def _discard_results(self, outdated=False):
        self._result = None
        self._frame = 0
        self._play_timer.stop()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.w_playback.hide()
        self.plot.clear()
        self.plot.hide()
        self.lbl_plot.hide()
        self.view.result_mode = False
        self.view.hot_marker = None
        for entry in self._scene:
            self.view.set_entry_colors(entry, None)
        self.act_results.blockSignals(True)
        self.act_results.setChecked(False)
        self.act_results.blockSignals(False)
        self.act_results.setEnabled(False)
        self.legend.clear()
        self.legend.hide()
        self.tbl_results.setRowCount(0)
        self.lbl_result.setText("The setup changed — start the simulation again." if outdated else
                                "No results yet — Start Simulation.")
        if outdated:
            self.txt_messages.clear()
        self._probe_cache.clear()
        self._fill_probe_table()
        self._refresh_overlays()
        if self._selected_probe is not None:
            self._show_probe()
        self._update_gizmo()
        self.view.update()

    def _color_range(self):
        result = self._result
        if self.chk_auto_range.isChecked():
            low, high = result["t_min"], result["t_max"]
            if high - low < 0.1:
                high = low + 0.1
            for spin, value in ((self.sp_range_low, low), (self.sp_range_high, high)):
                spin.blockSignals(True); spin.setValue(value); spin.blockSignals(False)
        else:
            low, high = self.sp_range_low.value(), self.sp_range_high.value()
            if high <= low:
                high = low + 0.1
        self.sp_range_low.setEnabled(not self.chk_auto_range.isChecked())
        self.sp_range_high.setEnabled(not self.chk_auto_range.isChecked())
        return low, high

    def _apply_result_colors(self):
        """Colour the bodies with the temperatures of the shown frame (the colour range of a
        transient run covers the whole run, so the frames can be compared)."""
        result = self._result
        if result is None:
            return
        low, high = self._color_range()
        for ci, component in enumerate(self.study["components"]):
            for bi, body in enumerate(component["bodies"]):
                entry = body.get("_render")
                if entry is None:
                    continue
                surface = result["surface"].get((ci, bi))
                if surface is not None:
                    mesh = surface.setdefault("mesh", (surface["vertices"], surface["triangles"]))
                    values = surface_values(result, (ci, bi), self._frame)
                    self.view.set_entry_colors(entry, temperature_colors(values, low, high), mesh)
                else:
                    self.view.set_entry_colors(entry, None)
        self.legend.set_range(low, high)
        self.view.update()

    def _on_range_edited(self, *_args):
        if self._result is not None:
            self._apply_result_colors()
        else:
            self._color_range()

    def _show_result(self):
        result = self._result
        transient = result["analysis"] == "transient"
        last = len(result["frame_times"]) - 1
        self._frame = last                          # the state at the end of the run
        self.tbl_results.blockSignals(True)
        self.tbl_results.setRowCount(len(result["bodies"]))
        for row, body in enumerate(result["bodies"]):
            for column in range(6):
                item = QTableWidgetItem("")
                item.setData(Qt.UserRole, body["key"])
                if column >= 2:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl_results.setItem(row, column, item)
            self.tbl_results.item(row, 0).setText(body["name"])
            self.tbl_results.item(row, 1).setText(body["material"] or "")
            self.tbl_results.item(row, 2).setText(f"{body['power']:g}")
        self.tbl_results.blockSignals(False)
        self.w_playback.setVisible(transient)
        self.lbl_plot.setVisible(transient)
        self.plot.setVisible(transient)
        if transient:
            self._update_plot()
            self.sl_time.blockSignals(True)
            self.sl_time.setRange(0, last)
            self.sl_time.setValue(last)
            self.sl_time.blockSignals(False)
        self.txt_messages.setPlainText("\n".join(result["summary"] + [""] + result["messages"]))
        self._set_frame(last)
        self.act_results.setEnabled(True)
        self.act_results.setChecked(True)          # → _on_results_toggled
        self._on_results_toggled(True)

    def _update_plot(self):
        """Temperature-over-time chart of a transient result: probes (dashed, over the stored
        frames) first, then the maxima of the hottest bodies."""
        result = self._result
        if result is None or result["analysis"] != "transient":
            return
        series = result["series"]
        probe_curves = []
        for probe in self.study["probes"][:PLOT_MAX_CURVES // 2]:
            values = self._probe_series(probe)
            if values is not None:
                probe_curves.append((f"◉ {probe['name']}", values, result["frame_times"]))
        peaks = np.nan_to_num(np.nanmax(series["max"], axis=1), nan=-np.inf)
        order = sorted(int(i) for i in np.argsort(-peaks)[:PLOT_MAX_CURVES - len(probe_curves)])
        self.plot.set_data(series["times"],
                           probe_curves +
                           [(result["bodies"][i]["name"], series["max"][i]) for i in order],
                           [(result["bodies"][i]["name"], result["bodies"][i]["max_c"])
                            for i in order if result["bodies"][i]["max_c"] is not None])
        self.plot.set_cursor(float(result["frame_times"][self._frame]))
        self.lbl_plot.setText("<b>Temperature over time</b> (bodies: maximum, ◉ probes)"
                              if probe_curves else "<b>Maximum temperature over time</b>")

    def _set_frame(self, frame):
        """Show the temperatures of one stored frame (the only one of a steady-state run)."""
        result = self._result
        if result is None:
            return
        frame = int(min(max(frame, 0), len(result["frame_times"]) - 1))
        self._frame = frame
        self._apply_result_colors()
        value = float(result["hot_values"][frame])
        self.view.hot_marker = (result["hot_points"][frame], f"max {value:.1f} °C")
        transient = result["analysis"] == "transient"
        self.tbl_results.blockSignals(True)
        for row, (body, stats) in enumerate(zip(result["bodies"], frame_body_stats(result, frame))):
            for column, number in zip((3, 4, 5), stats):
                self.tbl_results.item(row, column).setText(_fmt_c(number))
            item = self.tbl_results.item(row, 5)
            over = stats[2] is not None and body["max_c"] is not None and stats[2] > body["max_c"]
            item.setForeground(QColor("#ff6b6b" if over else "#e2e5e9"))
            item.setToolTip(f"Above the {body['max_c']:g} °C limit of {body['material']}" if over else "")
        self.tbl_results.blockSignals(False)
        if transient:
            time_value = float(result["frame_times"][frame])
            end = float(result["frame_times"][-1])
            self.lbl_time.setText(f"t = {format_time(time_value) if time_value else '0 s'}  /  "
                                  f"{format_time(end)}")
            self.plot.set_cursor(time_value)
            peak = float(np.nanmax(result["series"]["max"]))
            self.lbl_result.setText(
                f"At t = {format_time(time_value) if time_value else '0 s'}: maximum <b>{value:.1f} °C"
                f"</b> · highest in the run {peak:.1f} °C · ambient {result['ambient']:g} °C · "
                f"{result['power']:g} W<br>At the end: convection {result['q_convection']:.3g} W, "
                f"radiation {result['q_radiation']:.3g} W, still stored {result['q_stored']:.3g} W")
        else:
            self.lbl_result.setText(
                f"Maximum <b>{value:.1f} °C</b> · ambient {result['ambient']:g} °C · "
                f"{result['power']:g} W<br>Heat balance: convection {result['q_convection']:.3g} W, "
                f"radiation {result['q_radiation']:.3g} W ({result['balance_error']:+.2%})")
        lines = []
        for entry in result.get("conditions", [])[:4]:        # heat through the face conditions
            if entry["kind"] == "insulated":
                continue
            heat = entry["heat"]
            lines.append(f"{entry['name']}: {abs(heat):.3g} W "
                         f"{'in' if (heat < 0) == (entry['kind'] != 'flux') else 'out'}")
        if lines:
            self.lbl_result.setText(self.lbl_result.text() + "<br>" + " · ".join(lines) +
                                    (" (at the end)" if transient else ""))
        self._fill_probe_table()
        self._refresh_overlays()
        if self._selected_probe is not None:
            self._show_probe()
        self.view.update()

    def _on_frame_changed(self, value):
        self._set_frame(value)

    def _toggle_playback(self):
        if self._result is None or self._result["analysis"] != "transient":
            return
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            return
        if self.sl_time.value() >= self.sl_time.maximum():
            self.sl_time.setValue(0)
        if not self.act_results.isChecked():
            self.act_results.setChecked(True)
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self._play_timer.start()

    def _play_step(self):
        if self.sl_time.value() >= self.sl_time.maximum():
            self._play_timer.stop()
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            return
        self.sl_time.setValue(self.sl_time.value() + 1)

    def _on_plot_clicked(self, time_value):
        if self._result is None or self._result["analysis"] != "transient":
            return
        frame = int(np.argmin(np.abs(self._result["frame_times"] - time_value)))
        self.sl_time.setValue(frame)

    def _on_results_toggled(self, on):
        self.view.result_mode = bool(on) and self._result is not None
        self.legend.setVisible(self.view.result_mode)
        transient = self._result is not None and self._result["analysis"] == "transient"
        self.w_playback.setVisible(self.view.result_mode and transient)
        if not self.view.result_mode and self._play_timer.isActive():
            self._toggle_playback()                     # pause
        self._refresh_overlays()
        self._update_gizmo()
        self.view.update()

    def _on_result_row(self):
        items = self.tbl_results.selectedItems()
        if not items or self._result is None:
            return
        key = items[0].data(Qt.UserRole)
        if not key:
            return
        ci, bi = key
        if 0 <= ci < len(self.study["components"]):
            self._select(self.study["components"][ci], bi)

    def _select_result_row(self):
        if self._result is None:
            return
        component = self._selected
        key = None
        if component is not None and self._selected_body is not None:
            index = next((i for i, c in enumerate(self.study["components"]) if c is component), None)
            key = None if index is None else (index, self._selected_body)
        self.tbl_results.blockSignals(True)
        self.tbl_results.clearSelection()
        for row in range(self.tbl_results.rowCount()):
            item = self.tbl_results.item(row, 0)
            if key is not None and tuple(item.data(Qt.UserRole)) == key:
                self.tbl_results.selectRow(row)
                body = self._result["bodies"][row]
                low, mean, high = frame_body_stats(self._result, self._frame)[row]
                when = "" if self._result["analysis"] != "transient" else \
                    f" at t = {format_time(self._result['frame_times'][self._frame])}"
                self._set_status(f"{body['name']}{when}: min {_fmt_c(low)} °C, mean "
                                 f"{_fmt_c(mean)} °C, max {_fmt_c(high)} °C")
                break
        self.tbl_results.blockSignals(False)


def open_window(window):
    """Show the Thermal window of the main window (created on first use)."""
    dialog = getattr(window, "_thermal_dialog", None)
    if dialog is None:
        dialog = ThermalDialog(window)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.finished.connect(lambda _result=0, closed=dialog: _window_closed(window, closed))
        window._thermal_dialog = dialog
        dialog.move(window.frameGeometry().topLeft() + QPoint(80, 80))
        cad._set_dark_title_bar(dialog)    # before show(), so the frame is never drawn light
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog


def _window_closed(window, dialog):
    if getattr(window, "_thermal_dialog", None) is dialog:
        window._thermal_dialog = None
