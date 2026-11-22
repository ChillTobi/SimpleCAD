"""
SimpleCAD Thermal — optional add-on: heat conduction (FEM) and airflow + heat (CHT).

SimpleCAD shows the "Thermal" toolbar button only while this file sits next to
SimpleCAD.py (or is bundled into the executable). The window works like the Assembly window:
parts come from the open project, from STEP / STL / OBJ files or from an assembly; block and
cylinder heat sources are placed with XYZ fields or the move / rotate arrows, or extruded from a
picked face (Heat Source Extrusion; extruded into the body it is a heat zone in the body's
material). Every body needs a material, and any body can carry a heat power in watts, constant
or following a time profile (on / off cycle, ramp, CSV table).

Solver "Conduction (FEM)":
  * all visible bodies are voxelised on one Cartesian grid (cell-centre inside test, majority of
    three ray directions, so slightly open meshes still work). Bodies that touch share grid
    nodes, i.e. their contact is bonded; a contact item adds a contact resistance or an
    interface layer (thermal paste, pad, film) as a lower conductivity of the two touching cell
    layers. Later bodies win where bodies overlap; placed heat sources always win.
  * trilinear 8-node hexahedral finite elements; laminates (FR-4, G10, CFK) are orthotropic:
    the through-plane axis is the thinnest extent of the body.
  * exposed faces lose heat by convection (natural-convection correlations for vertical,
    upward- and downward-facing surfaces with gravity along -Z, forced air from an air speed,
    or a fixed h) and by radiation to the ambient. Air enclosed in a housing conducts heat as
    still air. Face conditions: fixed temperature, heat flux, own convection, insulated.
  * steady state by Picard iteration over the non-linear surface terms, transient by implicit
    Euler; each linear system is solved by conjugate gradients with a two-level preconditioner.

Solver "Airflow + heat (CHT)":
  * air and solids on one rectilinear grid that is fine at the parts and grows towards the sides
    of an air box (open, wall, fixed-temperature wall, inlet or symmetry plane per side).
  * steady incompressible Navier–Stokes with Boussinesq buoyancy, finite volumes on a staggered
    grid, SIMPLEC with a pseudo time step; one energy equation for air and solids; laminar or a
    mixing-length eddy viscosity; a thermal wall function for coarse cells. Fans (fixed flow or
    fan curve) and vents (pressure loss from the free area) open the walls they overlap.
  * checked against the de Vahl Davis heated cavity (Ra 1e3 … 1e6, Nu within 2 %).

  * steady state, or over time: implicit time steps, each converged by inner iterations.

Both solvers: user materials (library thermal_materials.json, copied into each study) with
optional k(T) and cp(T) tables; thermostats (power off / on by a probe or the body's own
temperature, with hysteresis; in steady state the share of the power that holds the setpoint).

Results: temperature map, section view (movable plane; cut faces in temperature colours, in
airflow results the air temperature or speed with velocity arrows), probes, streamlines, and
the heat balance of every run. Parameter sweeps run the study for several values of one
parameter and compare the temperatures. Export: PDF report, CSV tables, image of the view, and
the whole result for ParaView (VTU; a transient run as a .pvd time series).
Units in the window: mm, W, degrees Celsius; the solvers work in SI units.
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
ADDON_VERSION = "2.1"         # add-on release, shown in SimpleCAD's Help → About
THERMAL_FORMAT = "simplecad-thermal-study"
THERMAL_EXTENSION = ".thml"
THERMAL_DEFAULT_NAME = "Untitled"
HEAT_SOURCE_COLOR = (1.0, 0.45, 0.15)
PRIMITIVE_KINDS = ("block", "cylinder")
COMPONENT_KINDS = ("project", "file", "assembly") + PRIMITIVE_KINDS + ("fan", "vent")
FLOW_COLOR = (0.3, 0.85, 1.0)

# ── materials ─────────────────────────────────────────────────────────────────
# Thermal data at about 20 °C, typical handbook values (actual grades vary). Densities come from
# SimpleCAD's MATERIAL_DENSITIES_G_CM3, so both lists use the same material names.
#   k           thermal conductivity, W/(m·K); through-plane value for laminates
#   k_plane     in-plane conductivity of laminates (optional, makes the body orthotropic)
#   cp          specific heat capacity, J/(kg·K)  (transient runs)
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
AIR_VISCOSITY = 1.85e-5             # Pa·s
AIR_KINEMATIC_VISCOSITY = AIR_VISCOSITY / AIR_DENSITY   # m²/s
AIR_PRANDTL = 0.71
GRAVITY = 9.81                      # m/s², along -Z


# User materials: the user's own library (thermal_materials.json next to SimpleCAD) plus the
# materials of the open study. A study keeps a copy of every user material its bodies use, so it
# opens with the same data elsewhere; that copy wins over a library material of the same name.
# A user material can carry tables k(T) and cp(T) ([[°C, value], ...], linear in between, the
# end values beyond); the plain k is then the table value at 20 °C, and an in-plane k of a
# laminate scales with the table.
MATERIAL_LIBRARY_FILE = "thermal_materials.json"
MATERIAL_LIBRARY_FORMAT = "simplecad-thermal-materials"
REFERENCE_C = 20.0                  # temperature of the plain k / cp of a material with tables
USER_MATERIALS = {}                 # name → material (the user's library)
_ACTIVE_STUDY = {"materials": {}}   # materials of the study shown in the Thermal window


def material_library_path():
    return os.path.join(cad._LOG_DIR, MATERIAL_LIBRARY_FILE)


def _property_table(points, positive=True):
    """[[°C, value], …] sorted by temperature (one value per temperature); [] for fewer than two
    points."""
    rows = {}
    for point in points or ():
        try:
            t, value = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(t) and math.isfinite(value) and (value > 0 or not positive):
            rows[round(t, 6)] = value
    table = [[t, rows[t]] for t in sorted(rows)]
    return table if len(table) >= 2 else []


def table_value(table, t):
    """Value of a property table at the temperature(s) t (°C)."""
    points = np.asarray(table, float).reshape(-1, 2)
    return np.interp(t, points[:, 0], points[:, 1])


def material_settings(data):
    """A validated user material."""
    data = data if isinstance(data, dict) else {}

    def number(key, default, low=None, high=None):
        try:
            value = float(data.get(key, default))
        except (TypeError, ValueError):
            value = default
        if not math.isfinite(value):
            value = default
        if low is not None:
            value = max(low, value)
        if high is not None:
            value = min(high, value)
        return value

    material = {"k": number("k", 1.0, 1e-6), "cp": number("cp", 1000.0, 1e-3),
                "density": number("density", 1000.0, 1e-3),
                "emissivity": number("emissivity", 0.9, 0.0, 1.0),
                "k_plane": None, "max_c": None, "note": str(data.get("note", "") or ""),
                "k_table": _property_table(data.get("k_table")),
                "cp_table": _property_table(data.get("cp_table")), "user": True}
    if data.get("k_plane") not in (None, "", 0, 0.0):
        material["k_plane"] = number("k_plane", material["k"], 1e-6)
    if data.get("max_c") not in (None, ""):
        material["max_c"] = number("max_c", 100.0, -273.0)
    if material["k_table"]:
        material["k"] = float(table_value(material["k_table"], REFERENCE_C))
    if material["cp_table"]:
        material["cp"] = float(table_value(material["cp_table"], REFERENCE_C))
    return material


def load_material_library(path=None):
    """Read the user's material library into USER_MATERIALS; returns an error text or None."""
    path = path or material_library_path()
    USER_MATERIALS.clear()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or data.get("format") != MATERIAL_LIBRARY_FORMAT:
            raise ValueError("not a SimpleCAD thermal material library")
        for name, material in (data.get("materials") or {}).items():
            name = str(name).strip()
            if name and name not in MATERIAL_THERMAL:
                USER_MATERIALS[name] = material_settings(material)
    except (OSError, ValueError, TypeError, AttributeError) as ex:
        return f"Could not read the material library {path}: {ex}"
    return None


def save_material_library(path=None):
    path = path or material_library_path()
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump({"format": MATERIAL_LIBRARY_FORMAT, "version": 1,
                   "materials": {name: USER_MATERIALS[name] for name in sorted(USER_MATERIALS)}},
                  stream, indent=1)
    os.replace(temporary, path)


def set_active_study(study):
    """The study whose own materials material_properties() sees (the Thermal window's)."""
    _ACTIVE_STUDY["materials"] = study.setdefault("materials", {})


def user_material(name, materials=None):
    """A user material of the study (`materials`, default: the active study) or the library."""
    materials = _ACTIVE_STUDY["materials"] if materials is None else materials
    return materials.get(name) or USER_MATERIALS.get(name)


def material_known(name, materials=None):
    return bool(name) and (name in MATERIAL_THERMAL or user_material(name, materials) is not None)


def material_names(materials=None):
    """Materials that have thermal data: the built-in ones in the order of SimpleCAD's material
    list, then the user's (study and library)."""
    names = [name for name in cad.MATERIAL_DENSITIES_G_CM3 if name in MATERIAL_THERMAL]
    names += [name for name in MATERIAL_THERMAL if name not in names]
    materials = _ACTIVE_STUDY["materials"] if materials is None else materials
    return names + sorted(set(materials) | set(USER_MATERIALS), key=str.lower)


def material_properties(name, materials=None):
    """Thermal data of a material incl. density in kg/m³, or None for an unknown material."""
    data = MATERIAL_THERMAL.get(name)
    if data is not None:
        props = dict(data)
        density = cad.MATERIAL_DENSITIES_G_CM3.get(name)
        props["density"] = None if density is None else density * 1000.0
        return props
    data = user_material(name, materials) if name else None
    return None if data is None else copy.deepcopy(data)


def material_summary(name, materials=None):
    props = material_properties(name, materials)
    if props is None:
        return "No material set — choose one before the simulation." if not name else \
            f"Unknown material '{name}' — choose one before the simulation."
    k = f"k {props['k']:.4g} W/m·K"
    if props.get("k_plane"):
        k = f"k {props['k_plane']:.4g} W/m·K in-plane, {props['k']:.4g} through-plane"
    if props.get("k_table"):
        k += f" (k(T) table, {len(props['k_table'])} points)"
    parts = [k, f"cp {props['cp']:.4g} J/kg·K" +
             (f" (cp(T) table, {len(props['cp_table'])} points)" if props.get("cp_table") else "")]
    if props["density"]:
        parts.append(f"ρ {props['density']:g} kg/m³")
    parts.append(f"ε {props['emissivity']:g}")
    if props.get("max_c") is not None:
        parts.append(f"max {props['max_c']:g} °C")
    if props.get("user"):
        parts.append("user material")
    return " · ".join(parts)


def material_note(name, materials=None):
    props = material_properties(name, materials)
    return (props or {}).get("note", "") or ""


_LIBRARY_ERROR = load_material_library()


# ── study data model ──────────────────────────────────────────────────────────
# A study holds components like an assembly (cad.assembly_* helpers work on it): a placed snapshot
# of project parts or of a file (also when it came in through an assembly), or a block / cylinder
# heat source. Each body additionally carries "material" (name or None), "material_auto" (True =
# taken from the source, so Update brings a changed source material along; False = chosen in the
# study and kept), "power_w" and "emissivity" (None = the material's value).
FLOW_SIDES = ("-X", "+X", "-Y", "+Y", "-Z", "+Z")       # sides of the airflow domain
FLOW_SIDE_TITLES = {"-X": "−X", "+X": "+X", "-Y": "−Y", "+Y": "+Y", "-Z": "−Z (floor)",
                    "+Z": "+Z (top)"}
SIDE_KINDS = {"open": "Open", "wall": "Wall", "wall_fixed": "Wall, fixed temperature",
              "inlet": "Inlet", "symmetry": "Symmetry"}
SIDE_HINTS = ("Open: air can flow in (at the ambient temperature) and out freely.\n"
              "Wall: closed, no slip, no heat crosses (e.g. a table top or a cabinet wall).\n"
              "Wall, fixed temperature: closed wall held at 'Wall at'.\n"
              "Inlet: air blows in at the inlet speed and the ambient temperature.\n"
              "Symmetry: mirror plane (free slip, no heat crosses); simulates half a model.")
FLOW_KINDS = ("fan", "vent")    # airflow devices placed in the study (not solids)

DEFAULT_SETTINGS = {
    "mode": "conduction",      # "conduction" (FEM, air on the surfaces) or "airflow" (CHT)
    "ambient_c": 25.0,         # ambient air and surroundings
    "convection": "natural",   # "natural" (correlations, gravity -Z), "forced" (air speed) or "fixed"
    "h": 10.0,                 # W/(m²·K) for "fixed"
    "air_speed": 1.0,          # m/s for "forced" (flow along the surfaces, e.g. from a fan)
    "radiation": True,
    "cell_mm": 0.0,            # 0 = automatic
    "max_cells": 400000,
    "analysis": "steady",      # "steady" (final temperatures) or "transient" (over time)
    "duration_s": 600.0,       # transient: simulated time from switching on
    "duration_unit": "min",    # unit the duration is shown in: "s", "min" or "h"
    "time_steps": 100,         # transient: number of implicit time steps
    "initial_ambient": True,   # transient: everything starts at the ambient temperature
    "initial_c": 25.0,         # transient: start temperature when not at ambient
    # airflow (CHT): air domain around the parts, its sides, grid and iterations
    "flow_space": 0.0,         # mm of air beside the parts (X, Y); 0 = automatic
    "flow_below": 0.0,         # mm of air below the parts; 0 = automatic
    "flow_above": 0.0,         # mm of air above the parts; 0 = automatic
    "flow_sides": {side: "open" for side in FLOW_SIDES},
    "inlet_speed": 1.0,        # m/s through "inlet" sides
    "wall_c": 25.0,            # °C of "wall_fixed" sides
    "turbulence": "auto",      # "auto" (mixing length with fans / inlets), "laminar" or "mixing"
    "flow_cell_mm": 0.0,       # cell size at the parts; 0 = automatic
    "flow_max_cells": 250000,
    "flow_iterations": 800,
    "flow_step_iterations": 30,  # transient airflow: most iterations per time step
}
TIME_UNITS = {"s": 1.0, "min": 60.0, "h": 3600.0}
MAX_FRAMES = 100               # transient: stored temperature fields for the time slider
VOLUME_FRAME_BUDGET = 25_000_000   # node temperatures kept for section views (about 100 MB)
VELOCITY_FRAME_BUDGET = 15_000_000  # velocity components kept of a transient airflow run
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


# ── power profiles (transient runs) ───────────────────────────────────────────
# The power of a body or of a heat-flux condition can follow a time profile in transient runs.
# A steady-state run uses the long-term value: the average of an on / off cycle, the full power
# of a ramp, the last value of a table. Tables hold absolute powers in W.
PROFILE_KINDS = {"constant": "Constant", "duty": "On / off cycle", "ramp": "Ramp up",
                 "table": "Table (CSV)"}


def profile_new(kind="duty"):
    return {"kind": kind, "period_s": 60.0, "duty_pct": 50.0, "ramp_s": 60.0, "points": []}


def profile_settings(data):
    """A validated profile, or None (= constant power)."""
    if not isinstance(data, dict) or data.get("kind") not in PROFILE_KINDS or \
            data.get("kind") == "constant":
        return None
    profile = profile_new(data["kind"])
    try:
        profile["period_s"] = max(1e-3, float(data.get("period_s", 60.0)))
        profile["duty_pct"] = min(100.0, max(0.0, float(data.get("duty_pct", 50.0))))
        profile["ramp_s"] = max(1e-3, float(data.get("ramp_s", 60.0)))
        points = sorted((max(0.0, float(t)), max(0.0, float(p))) for t, p in data.get("points", []))
    except (TypeError, ValueError):
        return None
    profile["points"] = [[t, p] for t, p in points]
    return profile


def _table(profile):
    points = np.asarray(profile.get("points", ()), float).reshape(-1, 2)
    return points[:, 0], points[:, 1]


def profile_energy(profile, power, t):
    """Heat in J a source of nominal `power` W puts in from 0 to t (scalar or array)."""
    t = np.maximum(np.asarray(t, float), 0.0)
    if profile is None:
        return power * t
    kind = profile["kind"]
    if kind == "duty":
        period = profile["period_s"]
        on = period * profile["duty_pct"] / 100.0
        cycles = np.floor(t / period)
        return power * (cycles * on + np.minimum(t - cycles * period, on))
    if kind == "ramp":
        ramp = profile["ramp_s"]
        return power * np.where(t < ramp, t * t / (2.0 * ramp), ramp / 2.0 + (t - ramp))
    times, values = _table(profile)
    if not len(times):
        return 0.0 * t
    # cumulative energy at the table times (the first value holds before the first time, the
    # last one after the last time)
    knots = np.concatenate([[values[0] * times[0]],
                            values[0] * times[0] + np.cumsum((values[1:] + values[:-1]) / 2.0 *
                                                             np.diff(times))])
    segment = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 1)
    after = segment + 1 < len(times)
    nxt = np.minimum(segment + 1, len(times) - 1)
    span = np.where(after, times[nxt] - times[segment], 1.0)
    fraction = np.where(after, (t - times[segment]) / np.maximum(span, 1e-12), 0.0)
    value_t = values[segment] + fraction * (values[nxt] - values[segment])
    inside = knots[segment] + (values[segment] + value_t) / 2.0 * (t - times[segment])
    return np.where(t <= times[0], values[0] * t,
                    np.where(after, inside, knots[-1] + values[-1] * (t - times[-1])))


def profile_mean_power(profile, power, t0, t1):
    """Average power in W between t0 and t1 (exact, also for cycles shorter than the step)."""
    if t1 <= t0:
        return profile_power(profile, power, t0)
    return float((profile_energy(profile, power, t1) - profile_energy(profile, power, t0)) /
                 (t1 - t0))


def profile_power(profile, power, t):
    """Power in W at the time t (scalar or array)."""
    t = np.asarray(t, float)
    if profile is None:
        return power + 0.0 * t
    kind = profile["kind"]
    if kind == "duty":
        period = profile["period_s"]
        on = period * profile["duty_pct"] / 100.0
        return np.where(np.mod(t, period) < on, power, 0.0)
    if kind == "ramp":
        return power * np.clip(t / profile["ramp_s"], 0.0, 1.0)
    times, values = _table(profile)
    if not len(times):
        return 0.0 * t
    return np.interp(t, times, values)


def profile_steady_power(profile, power):
    """The long-term power a steady-state run uses."""
    if profile is None or profile["kind"] == "ramp":
        return float(power)
    if profile["kind"] == "duty":
        return float(power) * profile["duty_pct"] / 100.0
    _times, values = _table(profile)
    return float(values[-1]) if len(values) else 0.0


def profile_peak_power(profile, power):
    if profile is not None and profile["kind"] == "table":
        _times, values = _table(profile)
        return float(values.max()) if len(values) else 0.0
    return float(power)


def profile_text(profile, power):
    """Short description, e.g. "On / off: 60 s period, 25 % on (mean 0.5 W)"."""
    if profile is None:
        return "constant"
    kind = profile["kind"]
    if kind == "duty":
        return (f"on / off: {format_time(profile['period_s'])} period, {profile['duty_pct']:g} % "
                f"on (mean {profile_steady_power(profile, power):.3g} W)")
    if kind == "ramp":
        return f"ramp to {power:g} W in {format_time(profile['ramp_s'])}"
    times, values = _table(profile)
    if not len(times):
        return "table: no points yet"
    return (f"table: {len(times)} points, {format_time(times[0]) if times[0] else '0 s'} … "
            f"{format_time(times[-1])}, {values.min():.3g} … {values.max():.3g} W")


def read_profile_csv(path):
    """(time s, power W) rows of a CSV / text file. Separators: comma, semicolon (then a decimal
    comma is allowed), tab or spaces; header and comment lines are skipped."""
    points = _read_two_columns(path)
    if not points:
        raise ValueError("no rows with a time (s) and a power (W) were found")
    if any(p < 0 for _t, p in points) or any(t < 0 for t, _p in points):
        raise ValueError("times and powers must not be negative")
    return [[t, p] for t, p in sorted(points)]


def read_table_csv(path):
    """(temperature °C, value) rows of a property table (k(T), cp(T)); values must be > 0."""
    points = _read_two_columns(path)
    if len(points) < 2:
        raise ValueError("at least two rows with a temperature (°C) and a value are needed")
    if any(value <= 0 for _t, value in points):
        raise ValueError("the values must be positive")
    return _property_table(points)


def _read_two_columns(path):
    points = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ";" in line or "\t" in line:
                fields = [field.strip().replace(",", ".")
                          for field in line.replace("\t", ";").split(";")]
            elif "," in line:
                fields = [field.strip() for field in line.split(",")]
            else:
                fields = line.split()
            try:
                points.append((float(fields[0]), float(fields[1])))
            except (ValueError, IndexError):
                continue
    return points


# ── thermostats ───────────────────────────────────────────────────────────────
# A thermostat switches the power of a body (or of a heat-flux condition) off when its sensor
# rises above the setpoint plus half the hysteresis and on again below the setpoint minus half
# the hysteresis, like a heated bed with its thermistor. Sensor: a probe, or the hottest point or
# the mean temperature of the heated body itself. A transient run switches at the start of every
# time step; a steady-state run finds the share of the power that holds the sensor at the
# setpoint (the mean of the switching cycle). A time profile still applies while it is on.
THERMOSTAT_SENSORS = {"max": "Hottest point of the body", "mean": "Mean of the body",
                      "probe": "A probe"}


def thermostat_new():
    return {"sensor": "max", "probe": "", "setpoint_c": 60.0, "hysteresis_k": 2.0}


def thermostat_settings(data):
    """A validated thermostat, or None (no thermostat)."""
    if not isinstance(data, dict):
        return None
    thermostat = thermostat_new()
    thermostat["sensor"] = data.get("sensor") if data.get("sensor") in THERMOSTAT_SENSORS else "max"
    thermostat["probe"] = str(data.get("probe", "") or "")
    try:
        thermostat["setpoint_c"] = float(data.get("setpoint_c", 60.0))
        thermostat["hysteresis_k"] = max(0.0, float(data.get("hysteresis_k", 2.0)))
    except (TypeError, ValueError):
        pass
    return thermostat


def thermostat_text(thermostat, short=False):
    """'thermostat 60 °C ± 1 K at probe P1'."""
    if thermostat is None:
        return ""
    sensor = {"max": "body maximum", "mean": "body mean"}.get(thermostat["sensor"]) or \
        f"probe {thermostat['probe'] or '?'}"
    band = f" ± {thermostat['hysteresis_k'] / 2.0:g} K" if thermostat["hysteresis_k"] > 0 else ""
    if short:
        return f"thermostat {thermostat['setpoint_c']:g} °C"
    return f"thermostat {thermostat['setpoint_c']:g} °C{band} at the {sensor}"


def resolve_thermostat(study, thermostat):
    """Solver copy of a thermostat: a probe sensor gets the probe's world point ("point"; None
    when the probe is missing)."""
    thermostat = thermostat_settings(thermostat)
    if thermostat is None:
        return None
    if thermostat["sensor"] == "probe":
        probe = next((p for p in study.get("probes", []) if p["name"] == thermostat["probe"]), None)
        point = None if probe is None else probe_world_point(study, probe)
        thermostat["point"] = None if point is None else [float(x) for x in point]
    return thermostat


# ── contacts ──────────────────────────────────────────────────────────────────
# Touching bodies share grid nodes, so their contact is bonded (perfect). A contact item between
# two bodies adds a thermal resistance where they touch: a contact resistance (dry joints) or an
# interface layer of given thickness and conductivity (thermal paste, pad, insulating film).
CONTACT_KINDS = {"bonded": "Bonded", "resistance": "Contact resistance",
                 "layer": "Interface layer"}
TIM_PRESETS = (                       # (name, thickness mm, k W/(m·K)), typical datasheet values
    ("Thermal paste", 0.05, 1.0),
    ("Thermal pad", 0.5, 3.0),
    ("Soft silicone gap filler", 1.0, 1.5),
    ("Silicone insulator (Sil-Pad type)", 0.23, 0.9),
    ("Polyimide film (Kapton)", 0.05, 0.12),
    ("Mica washer", 0.1, 0.6),
    ("Air gap", 0.1, AIR_CONDUCTIVITY),
)


def contact_new(name, a, b):
    """Contact between bodies a and b, each given as (part name, body index)."""
    return {"name": name, "a": [str(a[0]), int(a[1])], "b": [str(b[0]), int(b[1])],
            "kind": "bonded", "enabled": True,
            "resistance": 1.0,           # K·cm²/W (1 K·cm²/W = 1e-4 m²K/W)
            "thickness_mm": 0.5, "k": 3.0}


def contact_settings(data):
    contact = contact_new(str(data.get("name", "Contact")), data["a"], data["b"])
    contact["kind"] = data.get("kind") if data.get("kind") in CONTACT_KINDS else "bonded"
    contact["enabled"] = bool(data.get("enabled", True))
    for key in ("resistance", "thickness_mm", "k"):
        if key in data:
            contact[key] = max(0.0, float(data[key]))
    contact["k"] = max(1e-4, contact["k"])
    return contact


def contact_resistance(contact):
    """Thermal resistance of the contact in m²K/W (0 = bonded)."""
    if contact["kind"] == "resistance":
        return max(0.0, contact["resistance"]) * 1e-4
    if contact["kind"] == "layer":
        return max(0.0, contact["thickness_mm"]) * 1e-3 / max(contact["k"], 1e-6)
    return 0.0


def contact_text(contact):
    kind = contact["kind"]
    if kind == "resistance":
        return f"{contact['resistance']:g} K·cm²/W"
    if kind == "layer":
        return f"{contact['thickness_mm']:g} mm, k {contact['k']:g} W/m·K"
    return "bonded"


_COMPONENT_KEYS = ("name", "kind", "visible", "position", "rotation", "pivot", "source",
                   "source_name", "signature", "path", "file_stamp", "file_root", "shape", "zone",
                   "flow")


def study_new(name=THERMAL_DEFAULT_NAME):
    return {"name": name, "components": [], "settings": copy.deepcopy(DEFAULT_SETTINGS),
            "conditions": [], "probes": [], "contacts": [], "section": None, "materials": {},
            "sweep": None}


def thermal_body(name, vertices, triangles, faces, color=cad.DEFAULT_SOLID_COLOR, opacity=1.0,
                 material=None, power_w=0.0, emissivity=None, fused=False, material_auto=False,
                 profile=None, thermostat=None, materials=None):
    body = cad.assembly_body(name, vertices, triangles, faces, color, opacity, fused=fused)
    body["material"] = material if material_known(material, materials) else None
    body["material_auto"] = bool(material_auto) and body["material"] is not None
    body["power_w"] = max(0.0, float(power_w or 0.0))
    body["emissivity"] = None if emissivity is None else min(1.0, max(0.0, float(emissivity)))
    body["profile"] = profile_settings(profile)
    body["thermostat"] = thermostat_settings(thermostat)
    return body


def body_has_power(body):
    return profile_peak_power(body.get("profile"), body.get("power_w", 0.0)) > 0


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
            body["profile"] = copy.deepcopy(old.get("profile"))
            body["thermostat"] = copy.deepcopy(old.get("thermostat"))
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
            "profile": None, "thermostat": None,                    # flux: time profile, control
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
    condition["profile"] = profile_settings(data.get("profile"))
    condition["thermostat"] = thermostat_settings(data.get("thermostat"))
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
    for contact in study.get("contacts", []):
        for side in ("a", "b"):
            if contact[side][0] == old:
                contact[side][0] = new


def drop_component_refs(study, name):
    for condition in study.get("conditions", []):
        condition["faces"] = [ref for ref in condition["faces"] if ref["component"] != name]
    study["probes"] = [probe for probe in study.get("probes", []) if probe["component"] != name]
    study["contacts"] = [contact for contact in study.get("contacts", [])
                         if name not in (contact["a"][0], contact["b"][0])]


def probe_settings(data):
    component = data.get("component")
    return {"name": str(data["name"]), "component": None if component is None else str(component),
            "body": int(data.get("body", 0)), "point": [float(x) for x in data["point"]]}


def probe_world_point(study, probe):
    """World position of a probe: on a part (it follows the part) or, without a part, a fixed
    point (a probe placed on a section through the air)."""
    if probe.get("component") is None:
        return np.asarray(probe["point"], float)
    component = next((c for c in study["components"] if c["name"] == probe["component"]), None)
    if component is None:
        return None
    return cad.assembly_world_point(component, probe["point"])


def study_contacts(study, jobs):
    """Enabled contacts between simulated bodies: [{name, kind, pair (job a, job b), R m²K/W}]."""
    job_of = {job["key"]: index for index, job in enumerate(jobs)}
    component_index = {c["name"]: i for i, c in enumerate(study["components"])}
    out = []
    for contact in study.get("contacts", []):
        if not contact.get("enabled", True):
            continue
        a = job_of.get((component_index.get(contact["a"][0]), contact["a"][1]))
        b = job_of.get((component_index.get(contact["b"][0]), contact["b"][1]))
        if a is None or b is None or a == b:
            continue
        out.append({"name": contact["name"], "kind": contact["kind"], "pair": (a, b),
                    "R": contact_resistance(contact), "text": contact_text(contact)})
    return out


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
        entry = {key: copy.deepcopy(value) for key, value in condition.items() if key != "faces"}
        entry["targets"] = dict(targets)
        entry["thermostat"] = resolve_thermostat(study, condition.get("thermostat")) \
            if condition["kind"] == "flux" else None
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
                        material_auto=old.get("material_auto", False), profile=old.get("profile"))
    component["bodies"][0] = body
    component["pivot"] = [0.0, 0.0, 0.0]
    return body


def study_add_primitive(study, kind, shape, name, position, power_w, material, rotation=None,
                        source=None, zone=False):
    """A block / cylinder placed in the study. `zone`: a heat zone inside another body (it takes
    over the cells it overlaps without an overlap warning)."""
    vertices, triangles, faces = primitive_mesh(kind, shape)
    body = thermal_body(name, vertices, triangles, faces, HEAT_SOURCE_COLOR,
                        material=material, power_w=power_w)
    component = cad.assembly_add_component(
        study, name, kind, [body], shape=copy.deepcopy(shape),
        source=source or f"{kind.capitalize()} placed in the study")
    body["name"] = component["name"]
    component["pivot"] = [0.0, 0.0, 0.0]
    component["position"] = [float(value) for value in position]
    if rotation is not None:
        component["rotation"] = [float(value) for value in rotation]
    if zone:
        component["zone"] = True
    return component


def face_frame(component, body, face_index):
    """(centre, u, v, outward normal) of a planar body face in world coordinates; u / v run
    along the face (u follows the world X axis where possible, else Y)."""
    signature = face_signature(body, face_index)
    centre = cad.assembly_world_point(component, signature[:3])
    normal = cad.assembly_world_direction(component, signature[3:6])
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    for axis in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        u = axis - normal * float(axis @ normal)
        if float(np.linalg.norm(u)) > 0.3:
            break
    u = u / float(np.linalg.norm(u))
    return centre, u, np.cross(normal, u), normal


# airflow devices --------------------------------------------------------------------
# Fans and vents are parts of the study (they can be moved and turned like heat sources) whose
# body only marks a region: the airflow solver turns solid cells inside it into air. A fan pushes
# the air along its local +Z axis (fixed volume flow, or a fan curve: pressure over flow); a vent
# (grille, perforated plate) lets air through with a pressure loss from its free area. Both act
# only in the Airflow + heat solver; their axis is snapped to the nearest grid axis.
def flow_new(kind):
    return {"form": "disc" if kind == "fan" else "rect", "width": 40.0, "height": 40.0,
            "diameter": 40.0, "thickness": 10.0 if kind == "fan" else 2.0,
            "flow_mode": "fixed",          # fan: "fixed" volume flow or "curve"
            "flow_m3h": 10.0,              # fan, fixed: m³/h
            "max_m3h": 15.0, "max_pa": 30.0,   # fan curve: free-blowing flow, stall pressure
            "curve": [],                   # fan curve table [[m³/h, Pa], …] (overrides max_*)
            "free_area": 50.0}             # vent: open area in %


def flow_settings(data, kind):
    flow = flow_new(kind)
    if not isinstance(data, dict):
        return flow
    flow["form"] = "rect" if data.get("form") == "rect" else "disc"
    for key in ("width", "height", "diameter", "thickness", "flow_m3h", "max_m3h", "max_pa"):
        if key in data:
            flow[key] = max(0.0, float(data[key]))
    for key in ("width", "height", "diameter", "thickness"):
        flow[key] = max(0.01, flow[key])
    flow["flow_mode"] = "curve" if data.get("flow_mode") == "curve" else "fixed"
    flow["free_area"] = min(100.0, max(1.0, float(data.get("free_area", 50.0))))
    flow["curve"] = sorted([max(0.0, float(q)), max(0.0, float(p))] for q, p in data.get("curve", []))
    return flow


def flow_mesh(flow):
    if flow["form"] == "rect":
        return _block_mesh((flow["width"], flow["height"], flow["thickness"]))
    return _cylinder_mesh(flow["diameter"], flow["thickness"], "Z")


def flow_curve(flow):
    """(flows m³/h, pressures Pa) of a fan curve, falling with the flow."""
    if flow.get("curve") and len(flow["curve"]) >= 2:
        points = np.asarray(flow["curve"], float)
        return points[:, 0], points[:, 1]
    return np.array([0.0, max(flow["max_m3h"], 1e-6)]), np.array([max(flow["max_pa"], 0.0), 0.0])


def vent_loss_coefficient(free_area_pct):
    """Pressure-loss coefficient ζ (Δp = ζ·ρv²/2, v = approach velocity) of a thin perforated
    plate or grille with the given free area (Idelchik)."""
    phi = min(1.0, max(0.01, free_area_pct / 100.0))
    return (0.707 * (1.0 - phi) ** 0.375 + 1.0 - phi) ** 2 / phi ** 2


def flow_text(kind, flow):
    size = (f"{flow['width']:g} × {flow['height']:g}" if flow["form"] == "rect"
            else f"Ø {flow['diameter']:g}") + " mm"
    if kind == "vent":
        return f"{size}, {flow['free_area']:g} % open"
    if flow["flow_mode"] == "fixed":
        return f"{size}, {flow['flow_m3h']:g} m³/h"
    flows, pressures = flow_curve(flow)
    return f"{size}, curve {pressures[0]:g} Pa … {flows[-1]:g} m³/h"


def study_add_flow_item(study, kind, flow, name, position, rotation=None):
    vertices, triangles, faces = flow_mesh(flow)
    body = thermal_body(name, vertices, triangles, faces, FLOW_COLOR, 0.45)
    component = cad.assembly_add_component(
        study, name, kind, [body], flow=copy.deepcopy(flow),
        source=f"{'Fan' if kind == 'fan' else 'Vent'} placed in the study")
    body["name"] = component["name"]
    component["pivot"] = [0.0, 0.0, 0.0]
    component["position"] = [float(value) for value in position]
    if rotation is not None:
        component["rotation"] = [float(value) for value in rotation]
    return component


def flow_set_shape(component, flow):
    component["flow"] = copy.deepcopy(flow)
    vertices, triangles, faces = flow_mesh(flow)
    old = component["bodies"][0]
    component["bodies"][0] = thermal_body(old["name"], vertices, triangles, faces, FLOW_COLOR,
                                          old.get("opacity", 0.45))
    component["pivot"] = [0.0, 0.0, 0.0]
    return component["bodies"][0]


def study_flow_items(study):
    """Visible fans and vents for the airflow solver: their world box, flow axis (index, sign,
    True when it is close to a grid axis) and settings."""
    items = []
    for component in study["components"]:
        if component.get("kind") not in FLOW_KINDS or not component.get("visible", True) or \
                not component["bodies"]:
            continue
        body = component["bodies"][0]
        world = cad.assembly_world_vertices(component, body["vertices"])
        direction = cad.assembly_world_direction(component, (0.0, 0.0, 1.0))
        axis = int(np.argmax(np.abs(direction)))
        items.append({"name": component["name"], "kind": component["kind"],
                      "low": world.min(0), "high": world.max(0), "axis": axis,
                      "sign": 1.0 if direction[axis] >= 0 else -1.0,
                      "aligned": abs(float(direction[axis])) > 0.996,
                      "flow": flow_settings(component.get("flow"), component["kind"])})
    return items


def extrusion_placement(frame, offset_u, offset_v, height, inward):
    """(position, rotation) of a block / cylinder of `height` standing on a face (frame from
    face_frame), its local Z along the face normal; `inward` sinks it into the body."""
    centre, u, v, normal = frame
    matrix = np.column_stack([u, v, normal])
    rotation = [float(x) for x in cad._import_euler_angles(matrix)]
    position = centre + offset_u * u + offset_v * v + (-1.0 if inward else 1.0) * normal * height / 2.0
    return [float(x) for x in position], rotation


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
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    for key, default in DEFAULT_SETTINGS.items():
        if key not in (data or {}):
            continue
        value = data[key]
        try:
            if isinstance(default, dict):
                if isinstance(value, dict):
                    settings[key] = {side: str(value.get(side, default[side])) for side in default}
            elif isinstance(default, bool):
                settings[key] = bool(value)
            elif isinstance(default, int):
                settings[key] = int(value)
            elif isinstance(default, float):
                settings[key] = float(value)
            else:
                settings[key] = str(value)
        except (TypeError, ValueError):
            pass
    if settings["convection"] not in ("natural", "forced", "fixed"):
        settings["convection"] = "natural"
    settings["air_speed"] = max(0.0, settings["air_speed"])
    if settings["mode"] not in ("conduction", "airflow"):
        settings["mode"] = "conduction"
    settings["flow_sides"] = {side: kind if kind in SIDE_KINDS else "open"
                              for side, kind in settings["flow_sides"].items()}
    if settings["turbulence"] not in ("auto", "laminar", "mixing"):
        settings["turbulence"] = "auto"
    settings["inlet_speed"] = max(0.0, settings["inlet_speed"])
    settings["flow_max_cells"] = min(5_000_000, max(5000, settings["flow_max_cells"]))
    settings["flow_iterations"] = min(20000, max(20, settings["flow_iterations"]))
    settings["flow_step_iterations"] = min(500, max(2, settings["flow_step_iterations"]))
    for key in ("flow_space", "flow_below", "flow_above", "flow_cell_mm"):
        settings[key] = max(0.0, settings[key])
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
        props = job.get("props")
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
    if settings["convection"] == "forced":
        h = max(h, float(_forced_h(settings["air_speed"], 0.05)))
    if settings["radiation"]:
        h += 6.0 * emissive / area            # ≈ 4 σ T³ ε near room temperature
    return capacity / max(h * area, 1e-12)


def study_used_materials(study):
    """Copies of the user materials the study's bodies use (the study's own copy first, else the
    library's)."""
    used = {}
    for component in study["components"]:
        for body in component["bodies"]:
            name = body.get("material")
            if name and name not in MATERIAL_THERMAL and name not in used:
                data = user_material(name, study.get("materials", {}))
                if data is not None:
                    used[name] = copy.deepcopy(data)
    return used


def study_to_data(study):
    components = []
    for component in study["components"]:
        data = {key: copy.deepcopy(component[key]) for key in _COMPONENT_KEYS if key in component}
        data["bodies"] = [{
            "name": body["name"], "color": list(body["color"]), "opacity": body["opacity"],
            "fused": bool(body.get("fused", False)), "material": body.get("material"),
            "material_auto": bool(body.get("material_auto", False)),
            "power_w": float(body.get("power_w", 0.0)), "emissivity": body.get("emissivity"),
            "profile": copy.deepcopy(body.get("profile")),
            "thermostat": copy.deepcopy(body.get("thermostat")),
            "vertices": np.asarray(body["vertices"], float).tolist(),
            "triangles": [list(t) for t in body["triangles"]],
            "faces": [{"planar": bool(face.get("planar", False)),
                       "tris": [int(i) for i in face.get("tris", ())]}
                      for face in body.get("faces", [])]}
            for body in component["bodies"]]
        components.append(data)
    return {"name": study["name"], "settings": copy.deepcopy(study["settings"]),
            "components": components,
            "conditions": copy.deepcopy(study.get("conditions", [])),
            "probes": copy.deepcopy(study.get("probes", [])),
            "contacts": copy.deepcopy(study.get("contacts", [])),
            "section": copy.deepcopy(study.get("section")),
            "materials": study_used_materials(study),
            "sweep": copy.deepcopy(study.get("sweep"))}


def study_from_data(data):
    study = study_new()
    if not data:
        return study
    study["name"] = str(data.get("name", study["name"]))
    study["settings"] = study_settings(data.get("settings"))
    for name, material in (data.get("materials") or {}).items():
        if str(name).strip() and name not in MATERIAL_THERMAL:
            study["materials"][str(name)] = material_settings(material)
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
                                    body.get("fused", False), body.get("material_auto", False),
                                    body.get("profile"), body.get("thermostat"),
                                    study["materials"])
                       for body in item.get("bodies", [])]}
        for key in ("source", "source_name", "signature", "path", "file_stamp", "file_root",
                    "shape", "zone", "flow"):
            if key in item:
                component[key] = copy.deepcopy(item[key])
        study["components"].append(component)
    study["conditions"] = [condition_settings(item) for item in data.get("conditions", [])]
    study["probes"] = [probe_settings(item) for item in data.get("probes", [])]
    study["contacts"] = [contact_settings(item) for item in data.get("contacts", [])]
    section = data.get("section")
    if isinstance(section, dict):
        study["section"] = {"position": [float(x) for x in section.get("position", (0, 0, 0))],
                            "rotation": [float(x) for x in section.get("rotation", (0, 0, 0))],
                            "flip": bool(section.get("flip", False)),
                            "size": float(section.get("size", 100.0)),
                            "active": bool(section.get("active", False))}
    study["sweep"] = sweep_settings(data.get("sweep"))
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
    return {"name": study["name"], "settings": copy.deepcopy(study["settings"]),
            "components": components,
            "conditions": copy.deepcopy(study.get("conditions", [])),
            "probes": copy.deepcopy(study.get("probes", [])),
            "contacts": copy.deepcopy(study.get("contacts", [])),
            "materials": copy.deepcopy(study.get("materials", {})),
            "sweep": copy.deepcopy(study.get("sweep"))}


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
            _freeze(state.get("conditions", [])), _freeze(state.get("probes", [])),
            _freeze(state.get("contacts", [])), _freeze(state.get("materials", {})),
            _freeze(state.get("sweep")))


# ── simulation input ──────────────────────────────────────────────────────────
def study_jobs(study):
    """The bodies the solver sees: every visible body (Bridge sources are inside their Bridge
    and left out) in world coordinates. Placed heat sources come last so they win overlaps."""
    jobs, primitives = [], []
    for ci, component in enumerate(study["components"]):
        if not component.get("visible", True) or component.get("kind") in FLOW_KINDS:
            continue
        for bi, body in enumerate(component["bodies"]):
            if body.get("fused") or not len(body["triangles"]):
                continue
            label = component["name"] if len(component["bodies"]) == 1 else \
                f"{component['name']} / {body['name']}"
            job = {"key": (ci, bi), "name": label, "material": body.get("material"),
                   "props": material_properties(body.get("material"), study.get("materials", {})),
                   "power": float(body.get("power_w", 0.0)), "emissivity": body.get("emissivity"),
                   "profile": copy.deepcopy(body.get("profile")),
                   "thermostat": resolve_thermostat(study, body.get("thermostat")),
                   "V": cad.assembly_world_vertices(component, body["vertices"]),
                   "T": np.asarray(body["triangles"], np.int64).reshape(-1, 3),
                   "faces": [np.asarray(face.get("tris", ()), np.int64)
                             for face in body.get("faces", [])],
                   "primitive": component["kind"] in PRIMITIVE_KINDS,
                   "zone": bool(component.get("zone"))}
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


def _forced_h(speed, length):
    """Forced-convection coefficient W/(m²·K) of air flowing along a surface of `length` m at
    `speed` m/s (flat plate, laminar / turbulent, average over the length)."""
    length = np.maximum(length, 1e-3)
    re = np.maximum(speed, 0.0) * length / AIR_KINEMATIC_VISCOSITY
    pr3 = AIR_PRANDTL ** (1.0 / 3.0)
    nu = np.where(re < 5e5, 0.664 * np.sqrt(re) * pr3,
                  np.maximum(0.037 * re ** 0.8 - 871.0, 0.664 * np.sqrt(5e5)) * pr3)
    return nu * AIR_CONDUCTIVITY / length


def _scipy():
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        import scipy.ndimage as ndi
        from scipy.spatial import cKDTree
    except ImportError as ex:
        raise ThermalError("The thermal solver needs scipy. Install it with:  pip install scipy") from ex
    return sp, spla, ndi, cKDTree


def _prepare_bodies(jobs, messages):
    """Solver records of the jobs: material data, extent, volume, conductivity per axis."""
    bodies = []
    for job in jobs:
        props = job.get("props") or material_properties(job["material"])
        if props is None:
            raise ThermalError(f"'{job['name']}' has no material. Choose one in the properties."
                               if not job["material"] else
                               f"'{job['name']}': the material '{job['material']}' is unknown "
                               "(a user material that is not in this study or the library).")
        if not props.get("density"):
            props["density"] = 1000.0
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
        # temperature-dependent data: k as a factor on the k at 20 °C (also for the in-plane k),
        # cp as values
        k_curve = cp_curve = None
        if props.get("k_table"):
            table = np.asarray(props["k_table"], float)
            k_curve = (table[:, 0], table[:, 1] / max(float(props["k"]), 1e-12))
        if props.get("cp_table"):
            table = np.asarray(props["cp_table"], float)
            cp_curve = (table[:, 0], table[:, 1])
        bodies.append({"job": job, "name": job["name"], "material": job["material"],
                       "props": props, "V": V, "T": T, "low": low, "high": high,
                       "extent": extent, "volume": volume, "bbox_volume": bbox_volume,
                       "area": _mesh_area(V, T), "k": k, "emissivity": emissivity,
                       "power": max(0.0, float(job.get("power", 0.0))),
                       "profile": profile_settings(job.get("profile")),
                       "thermostat": job.get("thermostat"),
                       "k_curve": k_curve, "cp_curve": cp_curve})
    return bodies


def _kept_frames(count, per_frame, budget):
    """Indices of the frames whose whole field is kept within a memory budget (values): evenly
    spread, always with the first and the last."""
    stride = max(1, int(math.ceil(count * per_frame / max(budget, 1))))
    return np.unique(np.r_[np.arange(0, count, stride), count - 1])


def result_velocity(result, frame=None):
    """(velocity (n0, n1, n2, 3), speed) of an airflow result at a stored frame (the nearest
    kept velocity field of a transient run)."""
    frames = result.get("velocity_frames")
    if frames is None or frame is None or len(frames) <= 1:
        return result["velocity"], result["speed"]
    kept = np.asarray(result["velocity_index"])
    slot = int(np.argmin(np.abs(kept - frame)))
    velocity = frames[slot]
    speed = np.sqrt((velocity ** 2).sum(-1))
    speed[result["grid"]["solid"].reshape(speed.shape)] = 0.0
    return velocity, speed


def _curve_factor(curve, values):
    """Interpolated values of a (temperatures, values) curve."""
    return np.interp(values, curve[0], curve[1])


def _check_thermostats(bodies, conditions):
    for owner, thermostat in [(f"'{b['name']}'", b.get("thermostat")) for b in bodies] + \
            [(f"the face condition '{c['name']}'", c.get("thermostat")) for c in conditions
             if c["kind"] == "flux"]:
        if thermostat is not None and thermostat["sensor"] == "probe" and \
                thermostat.get("point") is None:
            raise ThermalError(f"The thermostat of {owner} reads the probe "
                               f"'{thermostat['probe']}', which does not exist (any more). Place "
                               "the probe or choose another sensor.")


class _Thermostats:
    """On / off state and sensor of every thermostat-controlled source of a run.

    sources: [(name, thermostat, sensor, linearise)]: sensor(T) → °C of a temperature field;
    linearise(T) → a linear sensor function at the field T (the hottest point of a body is a
    fixed point near T)."""

    def __init__(self, sources):
        self.sources = sources
        self.on = np.ones(len(sources), bool)
        self.switches = np.zeros(len(sources), int)
        self.duty = np.ones(len(sources))           # steady state: share of the power
        self.sensor_values = np.full(len(sources), np.nan)
        self.on_steps = np.zeros(len(sources))
        self.steps = 0

    def __len__(self):
        return len(self.sources)

    def start(self, T):
        """Initial state: on unless the sensor is already above the upper switching point."""
        for i, (_name, thermostat, sensor, _linear) in enumerate(self.sources):
            value = sensor(T)
            self.sensor_values[i] = value
            self.on[i] = value < thermostat["setpoint_c"] + thermostat["hysteresis_k"] / 2.0

    def update(self, T):
        """Switch by the sensor values at the start of a time step (two-point control)."""
        for i, (_name, thermostat, sensor, _linear) in enumerate(self.sources):
            value = sensor(T)
            self.sensor_values[i] = value
            upper = thermostat["setpoint_c"] + thermostat["hysteresis_k"] / 2.0
            lower = thermostat["setpoint_c"] - thermostat["hysteresis_k"] / 2.0
            if self.on[i] and value >= upper:
                self.on[i] = False
                self.switches[i] += 1
            elif not self.on[i] and value <= lower:
                self.on[i] = True
                self.switches[i] += 1
        self.steps += 1
        self.on_steps += self.on

    def factors(self, transient):
        return self.on.astype(float) if transient else self.duty

    def solve_duty(self, base, responses):
        """Steady state: shares p (0 … 1) of the full power so that every sensor sits at its
        setpoint, by superposition T = base + Σ p_j · responses[j] (bounded solve on the few
        thermostats). Returns the field T."""
        n = len(self.sources)
        p = np.clip(self.duty, 0.0, 1.0)
        for _outer in range(4):
            guess = base + sum(p[j] * responses[j] for j in range(n))
            linear = [linearise(guess) for _name, _t, _sensor, linearise in self.sources]
            S = np.array([[f(response) for response in responses] for f in linear])
            r = np.array([thermostat["setpoint_c"] - f(base)
                          for (_name, thermostat, _s, _l), f in zip(self.sources, linear)])
            new = np.ones(n)
            free = np.ones(n, bool)
            for _ in range(2 * n + 2):
                trial = new.copy()
                if free.any():
                    fixed_part = S[:, ~free] @ new[~free] if (~free).any() else np.zeros(n)
                    trial[free] = np.linalg.lstsq(S[np.ix_(free, free)], (r - fixed_part)[free],
                                                  rcond=None)[0]
                clipped = np.clip(trial, 0.0, 1.0)
                newly = free & (np.abs(clipped - trial) > 1e-12)
                new = clipped
                if not newly.any():
                    break
                free &= ~newly
            done = float(np.abs(new - p).max()) < 1e-7
            p = new
            if done:
                break
        self.duty = p
        T = base + sum(p[j] * responses[j] for j in range(n))
        self.sensor_values = np.array([sensor(T) for _name, _t, sensor, _l in self.sources])
        return T

    def report(self, transient, nominal, duration=None):
        """Summary lines."""
        lines, notes = [], []
        for i, (name, thermostat, _sensor, _linear) in enumerate(self.sources):
            text = f"Thermostat of '{name}' ({thermostat_text(thermostat)}): "
            if transient:
                share = self.on_steps[i] / max(self.steps, 1)
                text += (f"on {share:.0%} of the time steps, {int(self.switches[i])} switchings, "
                         f"sensor {self.sensor_values[i]:.1f} °C at the end")
                if self.switches[i] == 0 and self.on[i]:
                    notes.append(f"Thermostat of '{name}': the sensor never reached "
                                 f"{thermostat['setpoint_c'] + thermostat['hysteresis_k'] / 2.0:g} "
                                 "°C, so the power stayed on.")
                elif self.steps and self.switches[i] > 0.6 * self.steps:
                    notes.append(f"⚠ Thermostat of '{name}' switches at almost every time step: "
                                 "the steps are too long for its cycle (more time steps, or a "
                                 "larger hysteresis).")
            else:
                p = float(self.duty[i])
                text += (f"{p:.1%} of the power ({p * nominal[i]:.4g} W of {nominal[i]:.4g} W), "
                         f"sensor {self.sensor_values[i]:.2f} °C")
                if p >= 1.0 - 1e-9 and self.sensor_values[i] < thermostat["setpoint_c"] - 0.01:
                    notes.append(f"Thermostat of '{name}': even the full power does not reach "
                                 f"{thermostat['setpoint_c']:g} °C at its sensor.")
                elif p <= 1e-9 and self.sensor_values[i] > thermostat["setpoint_c"] + 0.01:
                    notes.append(f"Thermostat of '{name}': the sensor stays above "
                                 f"{thermostat['setpoint_c']:g} °C even without this power.")
            lines.append(text)
        return lines, notes

    def results(self, transient, nominal):
        return [{"name": name, "text": thermostat_text(thermostat),
                 "setpoint": thermostat["setpoint_c"],
                 "share": float(self.on_steps[i] / max(self.steps, 1)) if transient
                 else float(self.duty[i]),
                 "power": None if transient else float(self.duty[i] * nominal[i]),
                 "nominal": float(nominal[i]), "switches": int(self.switches[i]),
                 "sensor": float(self.sensor_values[i])}
                for i, (name, thermostat, _s, _l) in enumerate(self.sources)]


def _voxel_owner(bodies, settings, step, messages):
    """Voxelise the bodies on one grid: {"owner" (int16 grid: body index, -1 = outside air,
    len(bodies) = enclosed air), "h_mm", "origin", "shape", "air_index", "cavity_cells"}."""
    _sp, _spla, ndi, _tree = _scipy()
    low = np.min([body["low"] for body in bodies], axis=0)
    high = np.max([body["high"] for body in bodies], axis=0)
    h_mm, notes = choose_cell_size([body["volume"] or body["bbox_volume"] for body in bodies],
                                   [body["extent"] for body in bodies], high - low, settings)
    messages += notes
    origin = low - h_mm                        # one cell of air around everything
    shape = tuple(int(n) for n in np.ceil((high - origin) / h_mm).astype(int) + 1)
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
        if len(replaced) and not body["job"].get("zone"):
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
    return {"owner": owner, "h_mm": h_mm, "origin": origin, "shape": shape,
            "air_index": air_index, "cavity_cells": cavity_cells}


def detect_contacts(jobs, settings, progress=None):
    """Touching bodies as the solver sees them (cells of different bodies side by side):
    [(job a, job b, contact area mm²)], largest first."""
    messages = []
    bodies = _prepare_bodies(jobs, messages)

    def step(percent, text):
        if progress is not None:
            progress(int(percent), text)

    grid = _voxel_owner(bodies, study_settings(settings), step, messages)
    owner, air = grid["owner"], grid["air_index"]
    count = collections.Counter()
    for axis in range(3):
        a = np.moveaxis(owner, axis, 0)[:-1].ravel()
        b = np.moveaxis(owner, axis, 0)[1:].ravel()
        touching = (a >= 0) & (b >= 0) & (a != b) & (a != air) & (b != air)
        if not touching.any():
            continue
        low = np.minimum(a[touching], b[touching]).astype(np.int64)
        high = np.maximum(a[touching], b[touching]).astype(np.int64)
        keys, cells = np.unique(low * (air + 1) + high, return_counts=True)
        for key, n in zip(keys.tolist(), cells.tolist()):
            count[divmod(key, air + 1)] += n
    area = grid["h_mm"] ** 2
    return sorted(((a, b, n * area) for (a, b), n in count.items()), key=lambda item: -item[2])


def solve_thermal(jobs, settings, progress=None, cancelled=None, conditions=(), contacts=()):
    """Temperatures of the study bodies: steady state, or over time (settings "analysis").
    `conditions` are the face conditions of study_conditions(), `contacts` the contacts of
    study_contacts(). Returns a result dict (see the end); temperature fields are stored as
    frames of the result-surface nodes (one frame for steady state) and, for section views and
    probes, of all grid nodes."""
    sp, spla, ndi, cKDTree = _scipy()
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
    transient = settings["analysis"] == "transient"
    step(2, "Preparing bodies...")

    # 1. bodies, materials ----------------------------------------------------------------
    bodies = _prepare_bodies(jobs, messages)
    conditions = [condition for condition in (conditions or ())]
    for condition in conditions:
        condition["profile"] = profile_settings(condition.get("profile"))
    has_fixed = any(c["kind"] == "fixed" and c["targets"] for c in conditions)
    has_flux = any(c["kind"] == "flux" and c["targets"] and
                   profile_peak_power(c.get("profile"), 1.0) > 0 for c in conditions)
    if not any(profile_peak_power(body["profile"], body["power"]) > 0 for body in bodies) and \
            not has_flux and not has_fixed:
        raise ThermalError("No heat source: give a body a power (W), add a heat source, or put a "
                           "heat flux or fixed temperature on faces.")
    if settings["convection"] == "fixed" and settings["h"] <= 0 and not settings["radiation"] \
            and not has_fixed:
        raise ThermalError("Without convection (h = 0) and radiation the heat cannot leave.")
    _check_thermostats(bodies, conditions)

    # 2. voxels ---------------------------------------------------------------------------
    grid = _voxel_owner(bodies, settings, step, messages)
    owner, h_mm, origin, shape = grid["owner"], grid["h_mm"], grid["origin"], grid["shape"]
    air_index, cavity_cells = grid["air_index"], grid["cavity_cells"]
    h = h_mm / 1000.0
    if cavity_cells:
        messages.append(f"Air enclosed in a housing ({cavity_cells * h_mm ** 3 / 1000.0:.4g} cm³) "
                        "conducts heat as still air (no airflow in Conduction mode).")
    I, J, K = np.nonzero(owner >= 0)
    element_body = owner[I, J, K].astype(np.int64)
    n_elements = len(I)
    cells = np.bincount(element_body, minlength=air_index + 1)
    for index, body in enumerate(bodies):
        body["cells"] = int(cells[index])
        if not body["cells"]:
            if profile_peak_power(body["profile"], body["power"]) > 0:
                raise ThermalError(f"The heat source '{body['name']}' is thinner than one cell "
                                   f"({h_mm:g} mm). Choose a smaller cell size.")
            messages.append(f"⚠ '{body['name']}' is thinner than one cell ({h_mm:g} mm) and is "
                            "left out. Choose a smaller cell size.")
            continue
        if body["volume"] > 0 and not body["job"].get("zone"):
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
    k_base = np.array([body["k"] for body in bodies] + [[AIR_CONDUCTIVITY] * 3])[element_body]
    k_element = k_base.copy()
    contact_extra = None
    # contacts: half of a contact's resistance goes into each of the two cell layers that touch,
    # as a lower conductivity across the contact, h / (h / k + R / 2) (exact for the heat flow
    # through a plane joint); the touching faces are kept for the heat through each contact
    contacts = list(contacts or ())
    contact_faces = [[] for _contact in contacts]
    if contacts:
        n_owner = air_index + 1
        resistance = np.zeros((n_owner, n_owner))
        number = np.full((n_owner, n_owner), -1)
        for index, contact in enumerate(contacts):
            a, b = contact["pair"]
            resistance[a, b] = resistance[b, a] = max(0.0, float(contact["R"]))
            number[a, b] = number[b, a] = index
        cell_id = (I.astype(np.int64) * shape[1] + J) * shape[2] + K      # ascending (C order)
        extra = np.zeros((n_elements, 3))
        padded_owner = np.pad(owner, 1, constant_values=-1)
        for direction in range(6):
            axis, side = divmod(direction, 2)
            offset = np.zeros(3, int); offset[axis] = 1 if side else -1
            neighbour = padded_owner[I + 1 + offset[0], J + 1 + offset[1],
                                     K + 1 + offset[2]].astype(np.int64)
            which = np.full(n_elements, -1)
            valid = neighbour >= 0
            which[valid] = number[element_body[valid], neighbour[valid]]
            touching = np.nonzero(which >= 0)[0]
            extra[touching, axis] += resistance[element_body[touching], neighbour[touching]] / 2.0
            if side == 1 and len(touching):              # each touching face once
                other = np.searchsorted(cell_id, cell_id[touching] + (offset[0] * shape[1] + offset[1])
                                        * shape[2] + offset[2])
                for index in np.unique(which[touching]):
                    chosen = which[touching] == index
                    contact_faces[int(index)].append((touching[chosen], other[chosen], axis))
        del padded_owner
        contact_extra = extra

    def effective_k(k, elements=slice(None)):
        """Conductivities with the contact layers of the elements."""
        if contact_extra is None:
            return k
        extra = contact_extra[elements]
        return np.where(extra > 0, h / (h / k + extra), k)

    k_element = np.array(effective_k(k_base))        # a copy: k_base stays the 20 °C value
    pairs = collections.defaultdict(list)
    for a in range(8):
        for b in range(8):
            pairs[_offset_index(_CORNERS[b] - _CORNERS[a])].append((a, b))

    def nodal_values(elements, k, pattern=None, corners=None):
        """(27, nodes) conduction-matrix entries of the elements (all: slice(None)) with the
        conductivities k (n, 3); `pattern` receives the non-zero structure. `corners` = the
        elements' corner nodes as (8, n) (contiguous rows; faster when assembled often)."""
        values = np.zeros((27, n_nodes))
        kx, ky, kz = (k[:, axis] * h for axis in range(3))
        nodes = np.ascontiguousarray(conn[elements].T) if corners is None else corners
        for offset, offset_pairs in pairs.items():
            for a, b in offset_pairs:
                values[offset] += np.bincount(nodes[a], kx * _KX[a, b] + ky * _KY[a, b] +
                                              kz * _KZ[a, b], n_nodes)
                if pattern is not None:
                    pattern[offset] |= np.bincount(nodes[a], minlength=n_nodes) > 0
        return values

    present = np.zeros((27, n_nodes), bool)
    values = nodal_values(slice(None), k_element, present)
    rank = (np.cumsum(present, axis=0) - present).astype(np.int32)    # entries before an offset
    indptr = np.zeros(n_nodes + 1, np.int64)
    np.cumsum(present.sum(0), out=indptr[1:])
    nnz = int(indptr[-1])
    indices = np.empty(nnz, np.int32)
    global_offset = (_OFFSETS[:, 0] * ny1 + _OFFSETS[:, 1]) * nz1 + _OFFSETS[:, 2]
    for offset in range(27):
        rows = np.nonzero(present[offset])[0]
        slots = indptr[rows] + rank[offset, rows]
        indices[slots] = np.searchsorted(nodes_global, nodes_global[rows] + global_offset[offset])

    def matrix_data(values):
        """CSR data of (27, nodes) matrix entries."""
        data = np.zeros(nnz)
        for offset in range(27):
            rows = np.nonzero(present[offset])[0]
            data[indptr[rows] + rank[offset, rows]] = values[offset, rows]
        return data

    data_k = matrix_data(values)
    del values
    # temperature-dependent conductivity: the elements of bodies with a k(T) table are
    # re-assembled with the conductivity at their mean temperature (the rest stays as it is)
    k_tabled = [index for index, body in enumerate(bodies) if body["k_curve"] is not None and
                body["cells"]]
    if k_tabled:
        tab_elements = np.nonzero(np.isin(element_body, k_tabled))[0]
        tab_body = element_body[tab_elements]
        tab_corners = np.ascontiguousarray(conn[tab_elements].T)
        data_static = data_k - matrix_data(nodal_values(tab_elements, k_element[tab_elements],
                                                        corners=tab_corners))

    def tabled_ratio(T):
        """k / k(20 °C) of the elements with a k(T) table at the nodal temperatures T."""
        t_element = T[tab_corners].mean(0)
        ratio = np.ones(len(tab_elements))
        for index in k_tabled:
            mask = tab_body == index
            ratio[mask] = _curve_factor(bodies[index]["k_curve"], t_element[mask])
        return ratio

    def assemble_k(ratio):
        nonlocal data_k
        k_tab = effective_k(k_base[tab_elements] * ratio[:, None], tab_elements)
        k_element[tab_elements] = k_tab
        data_k = data_static + matrix_data(nodal_values(tab_elements, k_tab, corners=tab_corners))

    # heat sources: the power of a body is spread evenly over its cells; a source with a time
    # profile or a thermostat gets its own load vector of 1 W, scaled by its power in every
    # time step (and by the thermostat's on / off state or share of the power)
    element_power = np.zeros(n_elements)
    profiled = []    # [(nodal load of 1 W, profile, nominal W, "source" | "flux", index, slot)]
    controlled = []  # thermostat slots: (name, thermostat, "source" | "flux", index)
    constant_power = 0.0
    for index, body in enumerate(bodies):
        if not body["cells"] or profile_peak_power(body["profile"], body["power"]) <= 0:
            continue
        if body["profile"] is None and body["thermostat"] is None:
            element_power[element_body == index] = body["power"] / body["cells"]
            constant_power += body["power"]
        else:
            unit = np.bincount(conn[element_body == index].ravel(), minlength=n_nodes) / \
                (8.0 * body["cells"])
            slot = None
            if body["thermostat"] is not None:
                slot = len(controlled)
                controlled.append((body["name"], body["thermostat"], "source", index))
            profiled.append((unit, body["profile"], body["power"], "source", index, slot))
    load_source = np.bincount(conn.ravel(), np.repeat(element_power / 8.0, 8), n_nodes)
    del element_power

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
    if not k_tabled:
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
    forced_h = None
    if settings["convection"] == "forced":
        # flow along the surface: the length of a face is the body's mean extent in its plane
        along = np.array([[max((body["extent"].sum() - body["extent"][axis]) / 2.0, 2.0) / 1000.0
                           for axis in range(3)] for body in bodies] + [[1.0] * 3])
        forced_h = _forced_h(settings["air_speed"], along[face_body, face_dir // 2])
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
            if condition.get("profile") is None and condition.get("thermostat") is None:
                flux_power[index] = density * area
                load_flux += np.bincount(face_nodes[mask].ravel(),
                                         np.repeat(density * face_area[mask] / 4.0, 4), n_nodes)
            else:
                unit = np.bincount(face_nodes[mask].ravel(),
                                   np.repeat(face_area[mask] / (4.0 * area), 4), n_nodes)
                slot = None
                if condition.get("thermostat") is not None:
                    slot = len(controlled)
                    controlled.append((condition["name"], condition["thermostat"], "flux", index))
                profiled.append((unit, condition["profile"], density * area, "flux", index, slot))
        elif condition["kind"] == "fixed":
            nodes = face_nodes[mask].ravel()
            fixed_value[nodes] = condition["temperature_c"]
            fixed_owner[nodes] = index
    fixed_nodes = np.nonzero(np.isfinite(fixed_value))[0]

    def heat_input(t0=None, t1=None, factors=None):
        """(nodal heat load, source power, flux power per condition) — steady state (no times)
        or the mean over the time step t0 … t1; `factors` scale the thermostat-controlled
        sources (on / off or share of the power)."""
        load = load_source + load_flux
        source, flux = constant_power, flux_power.copy()
        for unit, profile, nominal, kind, index, slot in profiled:
            power = profile_steady_power(profile, nominal) if t0 is None else \
                profile_mean_power(profile, nominal, t0, t1)
            if slot is not None and factors is not None:
                power *= float(factors[slot])
            if power:
                load = load + unit * power
            if kind == "source":
                source += power
            else:
                flux[index] = power
        return load, source, flux

    _load, total_power, flux_steady = heat_input()
    total_in = total_power + float(flux_steady.sum())
    if not transient and total_in <= 0 and not has_fixed:
        raise ThermalError("The heat sources have no power in steady state (an on / off cycle "
                           "with 0 % on, or a table ending at 0 W). Use a transient run.")

    def face_coefficients(T):
        """(convection h, radiation h) per exposed face for the nodal temperatures T."""
        face_t = T[face_nodes].mean(1)
        if settings["convection"] == "fixed":
            convection = np.full(len(face_t), max(0.0, settings["h"]))
        else:
            mean = np.bincount(face_body, face_t * face_area, air_index + 1) / \
                np.maximum(exposed_area, 1e-30)
            convection = _natural_h(mean[face_body] - ambient, face_length, orientation)
            if forced_h is not None:              # mixed convection: natural and forced combined
                convection = np.cbrt(convection ** 3 + forced_h ** 3)
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

    # thermostat sensors: a probe (trilinear between nodes, else the nearest node), or the
    # hottest point / the mean of the heated body (of the bodies under a flux condition)
    def make_sensor(thermostat, owners):
        if thermostat["sensor"] == "probe":
            point = np.asarray(thermostat["point"], float)[None, :]
            idx, w = _surface_weights(point, origin, h_mm, shape, nodes_global)
            if w.sum() <= 0:
                idx[:] = int(cKDTree(node_points).query(point[0])[1])
                w = np.array([[1.0] + [0.0] * 7])
            idx, w = idx[0], np.asarray(w[0], float)

            def probe_sensor(T):
                return float(T[idx] @ w)
            return probe_sensor, lambda _T: probe_sensor
        owners = [index for index in owners if body_nodes[index] is not None]
        if not owners:
            raise ThermalError(f"A thermostat has no body to measure ({thermostat_text(thermostat)}).")
        if thermostat["sensor"] == "mean":
            elements = np.concatenate([body_elements[index] for index in owners])

            def mean_sensor(T):
                return float(T[elements].mean())
            return mean_sensor, lambda _T: mean_sensor
        nodes = np.unique(np.concatenate([body_nodes[index] for index in owners]))

        def max_sensor(T):
            return float(T[nodes].max())

        def linearise(T):
            node = int(nodes[int(np.argmax(T[nodes]))])
            return lambda field: float(field[node])
        return max_sensor, linearise

    thermostats = None
    if controlled:
        sources = []
        for name, thermostat, kind, index in controlled:
            owners = [index] if kind == "source" else list(conditions[index]["targets"])
            sensor, linearise = make_sensor(thermostat, owners)
            sources.append((name, thermostat, sensor, linearise))
        thermostats = _Thermostats(sources)
    controlled_nominal = np.zeros(len(controlled))
    for unit, profile, nominal, kind, index, slot in profiled:
        if slot is not None:
            controlled_nominal[slot] = profile_steady_power(profile, nominal)

    # 5. solve ----------------------------------------------------------------------------
    cg_iterations = [0]

    def cg_callback(_x):
        cg_iterations[0] += 1
        if cg_iterations[0] % 25 == 0 and cancelled is not None and cancelled():
            raise ThermalCancelled()

    face_mass = _FACE_MASS.ravel()

    def system(convection_weight, radiation_weight, load):
        """Matrix data and right-hand side for the surface coefficients · areas and the nodal
        heat load."""
        weight = convection_weight + radiation_weight
        data = data_k + np.bincount(face_slots.ravel(), (weight[:, None] * face_mass[None, :]).ravel(), nnz)
        outside = (convection_weight * face_ambient + radiation_weight * ambient) / 4.0
        rhs = load + np.bincount(face_nodes.ravel(), np.repeat(outside, 4), n_nodes)
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

    nonlinear = settings["convection"] != "fixed" or settings["radiation"] or bool(k_tabled)
    cg_total, solver_short, q_stored = 0, False, 0.0
    flux_now = flux_steady
    factors = None
    if not transient:
        # Picard iteration over the temperature-dependent surface coefficients (and k(T)); a
        # thermostat finds its share of the power by superposition: the field without the
        # controlled sources plus each controlled source's own field at full power
        if thermostats is not None:
            factors = np.zeros(len(controlled))
            base_load = heat_input(factors=factors)[0]
            unit_loads = [None] * len(controlled)
            for unit, profile, nominal, kind, index, slot in profiled:
                if slot is not None:
                    unit_loads[slot] = unit * profile_steady_power(profile, nominal)
            responses = [np.zeros(n_nodes) for _load in unit_loads]
        else:
            base_load = heat_input()[0]
        T = np.full(n_nodes, ambient + total_in / (10.0 * exposed_area.sum()))
        T[fixed_nodes] = fixed_value[fixed_nodes]
        T_base = T.copy()
        convection_c = radiation_c = ratio_c = None
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
            if k_tabled:
                ratio = tabled_ratio(T)
                ratio_c = ratio if ratio_c is None else (1.0 - relaxation) * ratio_c + relaxation * ratio
                assemble_k(ratio_c)
            data, rhs = constrain(*system(convection_c * face_area, radiation_c * face_area,
                                          base_load))
            # loose while the surface coefficients still change, tight near the end
            tolerance = 1e-9 if not nonlinear or change < 0.05 else 1e-6
            T_base, info, steps_used = solve(data, rhs, T_base, tolerance)
            cg_total += steps_used
            solver_short |= info > 0
            if thermostats is not None:
                for j, unit_load in enumerate(unit_loads):
                    rhs_j = unit_load.copy()
                    rhs_j[fixed_nodes] = 0.0              # held temperatures: no response
                    responses[j], info, steps_used = solve(data, rhs_j, responses[j], tolerance,
                                                           refresh=False)
                    cg_total += steps_used
                    solver_short |= info > 0
                solution = thermostats.solve_duty(T_base, responses)
            else:
                solution = T_base
            change = float(np.abs(solution - T).max())
            T = solution
            iterations = iteration + 1
            if not nonlinear or (iteration >= 1 and change < 0.005 and tolerance < 1e-8):
                break
        if nonlinear and change >= 0.005:
            messages.append(f"⚠ The surface coefficients did not fully settle (last change "
                            f"{change:.3g} K).")
        if thermostats is not None:
            factors = thermostats.duty
        load, total_power, flux_now = heat_input(factors=factors)
        total_in = total_power + float(flux_now.sum())
        last_system = system(convection_c * face_area, radiation_c * face_area, load)
        frame_steps = np.array([0])
        frame_times = np.array([0.0])
        frames = T[used][None, :].astype(np.float32)
        volume_frames = T[None, :].astype(np.float32)
        volume_index = np.array([0])
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
        element_capacity = capacity[element_body]
        cp_tabled = [index for index, body in enumerate(bodies) if body["cp_curve"] is not None
                     and body["cells"]]
        cp_elements = np.nonzero(np.isin(element_body, cp_tabled))[0] if cp_tabled else None

        def capacity_at(T):
            """Heat capacity per node / Δt in W/K, with cp(T) at the element temperatures."""
            if cp_tabled:
                t_element = T[conn[cp_elements]].mean(1)
                owner_of = element_body[cp_elements]
                for index in cp_tabled:
                    mask = owner_of == index
                    element_capacity[cp_elements[mask]] = bodies[index]["props"]["density"] * \
                        _curve_factor(bodies[index]["cp_curve"], t_element[mask])
            return np.bincount(conn.ravel(), np.repeat(element_capacity * h ** 3 / 8.0, 8),
                               n_nodes) / dt

        start = ambient if settings["initial_ambient"] else settings["initial_c"]
        T = np.full(n_nodes, float(start))
        T[fixed_nodes] = fixed_value[fixed_nodes]          # held from the start
        node_capacity = capacity_at(T)
        ratio_assembled = None
        frame_steps = np.unique(np.round(np.linspace(0, n_steps, min(n_steps, MAX_FRAMES) + 1))
                                .astype(int))
        frame_of_step = {int(s): f for f, s in enumerate(frame_steps)}
        frame_times = frame_steps * dt
        frames = np.empty((len(frame_steps), len(used)), np.float32)
        # whole fields (section views, probes) for as many frames as the memory budget allows
        stride = max(1, int(math.ceil(len(frame_steps) * n_nodes / VOLUME_FRAME_BUDGET)))
        volume_index = np.unique(np.r_[np.arange(0, len(frame_steps), stride), len(frame_steps) - 1])
        volume_slot = {int(f): s for s, f in enumerate(volume_index)}
        volume_frames = np.empty((len(volume_index), n_nodes), np.float32)
        hot_points = np.empty((len(frame_steps), 3))
        hot_values = np.empty(len(frame_steps))
        stats = np.empty((n_steps + 1, 3, len(bodies)))
        power_series = np.zeros(n_steps + 1)
        t_min, t_max = float(T.min()), float(T.max())

        def record(step_index, T):
            stats[step_index] = body_stats(T)
            frame = frame_of_step.get(step_index)
            if frame is not None:
                frames[frame] = T[used]
                if frame in volume_slot:
                    volume_frames[volume_slot[frame]] = T
                hottest = int(np.argmax(T))
                hot_points[frame] = node_points[hottest]
                hot_values[frame] = T[hottest]

        record(0, T)
        if thermostats is not None:
            thermostats.start(T)
            factors = thermostats.factors(True)
            switch_series = np.zeros((len(controlled), n_steps + 1))
            switch_series[:, 0] = factors
        start_input = heat_input(0.0, 0.0, factors)
        power_series[0] = start_input[1] + float(start_input[2].sum())
        shortest = min([p["period_s"] * min(p["duty_pct"], 100.0 - p["duty_pct"]) / 100.0
                        for _u, p, _n, _k, _i, _s in profiled if p is not None and
                        p["kind"] == "duty" and 0.0 < p["duty_pct"] < 100.0] or [float("inf")])
        if shortest < 2.0 * dt:
            messages.append(f"⚠ An on / off cycle switches faster ({format_time(shortest)}) than "
                            f"the time step ({format_time(dt)}); its power is averaged over each "
                            "step. More time steps show the ripple.")
        previous = T
        for step_index in range(1, n_steps + 1):
            step(45 + 50 * (step_index - 1) / n_steps,
                 f"Time step {step_index} of {n_steps} — t = {format_time(step_index * dt)}")
            if thermostats is not None:          # switch by the sensors at the step start
                thermostats.update(T)
                factors = thermostats.factors(True)
                switch_series[:, step_index] = factors
            refresh = step_index % 10 == 1
            if k_tabled:                         # re-assemble when k changed by over 1 %; the
                ratio = tabled_ratio(T)          # coarse preconditioner may lag behind
                if ratio_assembled is None or float(np.abs(ratio - ratio_assembled).max()) > 1e-2:
                    assemble_k(ratio)
                    ratio_assembled = ratio
            if cp_tabled and step_index > 1:
                node_capacity = capacity_at(T)
            convection, radiation = face_coefficients(T)
            load, total_power, flux_now = heat_input((step_index - 1) * dt, step_index * dt,
                                                     factors)
            power_series[step_index] = total_power + float(flux_now.sum())
            data, rhs = system(convection * face_area, radiation * face_area, load)
            data[diagonal_slots] += node_capacity
            rhs += node_capacity * T
            last_system = (data, rhs)
            data, rhs = constrain(data, rhs)
            # start from the linear extrapolation of the last two steps (far fewer iterations)
            guess = T if step_index == 1 else T + (T - previous)
            previous = T
            # the surface coefficients change slowly: the coarse factorisation is renewed now
            # and then only
            T, info, steps_used = solve(data, rhs, guess, TRANSIENT_TOLERANCE, refresh=refresh)
            cg_total += steps_used
            solver_short |= info > 0
            t_min, t_max = min(t_min, float(T.min())), max(t_max, float(T.max()))
            record(step_index, T)
        q_stored = float((node_capacity * (T - previous)).sum())       # W, heat still stored
        iterations = n_steps
        total_in = total_power + float(flux_now.sum())        # during the last step
        series = {"times": np.arange(n_steps + 1) * dt, "min": stats[:, 0, :].T,
                  "mean": stats[:, 1, :].T, "max": stats[:, 2, :].T, "power": power_series}
        if thermostats is not None:
            series["thermostats"] = switch_series
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
                "flux": float(flux_now[index]),
                "convection": float(convection_flow[mask].sum()),
                "insulated": 0.0}[condition["kind"]]
        condition_results.append({"name": condition["name"], "kind": condition["kind"],
                                  "cell_faces": int(mask.sum()), "heat": heat,
                                  "area_mm2": float(face_area[mask].sum() * 1e6)})
    # heat through each contact (from body a to body b) from the mean temperatures of the two
    # touching cells and the conductance between their centres
    contact_results = []
    for index, contact in enumerate(contacts):
        a, _b = contact["pair"]
        heat, faces = 0.0, 0
        for rows, other, axis in contact_faces[index]:
            t_row, t_other = T[conn[rows]].mean(1), T[conn[other]].mean(1)
            conductance = h * h / (h / (2.0 * k_element[rows, axis]) +
                                   h / (2.0 * k_element[other, axis]))
            sign = np.where(element_body[rows] == a, 1.0, -1.0)
            heat += float((sign * conductance * (t_row - t_other)).sum())
            faces += len(rows)
        area = faces * h * h
        contact_results.append({"name": contact["name"], "kind": contact["kind"],
                                "text": contact.get("text", ""), "heat": heat,
                                "area_mm2": area * 1e6, "faces": faces,
                                "jump": heat * contact["R"] / area if area > 0 else 0.0,
                                "a": bodies[a]["name"], "b": bodies[contact["pair"][1]]["name"]})
        if not faces:
            messages.append(f"⚠ Contact '{contact['name']}': the two bodies do not touch (no "
                            "common cell faces), so it has no effect.")
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
                         ([f"flux {flux_now.sum():.4g} W"] if flux_now.any() else []) +
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
    for entry in contact_results:
        if entry["faces"]:
            summary.append(f"Contact '{entry['name']}' ({entry['text']}, {entry['area_mm2']:.4g} mm²): "
                           f"{abs(entry['heat']):.4g} W from "
                           f"'{entry['a'] if entry['heat'] >= 0 else entry['b']}' to "
                           f"'{entry['b'] if entry['heat'] >= 0 else entry['a']}'" +
                           (f", temperature step ≈ {abs(entry['jump']):.3g} K"
                            if entry["kind"] != "bonded" else ""))
    timed = [entry for entry in profiled if entry[1] is not None]
    if timed and transient:
        summary.append("Time profiles: " + "; ".join(
            f"{bodies[i]['name'] if kind == 'source' else conditions[i]['name']}: "
            f"{profile_text(profile, nominal)}" for _u, profile, nominal, kind, i, _s in timed))
    elif timed:
        summary.append("Steady state with time profiles: on / off cycles count with their mean "
                       "power, ramps with the full power, tables with the last value.")
    thermostat_results = []
    if thermostats is not None:
        lines, notes = thermostats.report(transient, controlled_nominal)
        summary += lines
        messages += notes
        thermostat_results = thermostats.results(transient, controlled_nominal)
    if k_tabled or (transient and cp_tabled):
        summary.append("Temperature-dependent material data: " + ", ".join(
            f"{bodies[i]['name']} ({bodies[i]['material']})"
            for i in sorted(set(k_tabled) | set(cp_tabled if transient else []))))
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
            "power": total_power, "q_flux": float(flux_now.sum()), "q_fixed": q_fixed,
            "q_convection": q_convection, "q_radiation": q_radiation,
            "q_stored": q_stored, "balance_error": balance_error, "iterations": iterations,
            "conditions": condition_results, "contacts": contact_results,
            "thermostats": thermostat_results,
            "volume_frames": volume_frames, "volume_index": volume_index,
            "grid": {"kind": "nodes", "origin": origin, "h_mm": h_mm, "shape": shape,
                     "nodes_global": nodes_global,
                     "cells": (I.astype(np.int64) * shape[1] + J) * shape[2] + K,   # elements
                     "cell_body": element_body.astype(np.int16)},
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


def result_point_weights(result, points, air=False):
    """(index (n, 8), weights (n, 8)) to interpolate the stored whole fields of a result at
    world points (mm): trilinear between grid nodes (Conduction) or cell centres (Airflow; only
    solid cells unless `air`)."""
    grid = result["grid"]
    points = np.asarray(points, float).reshape(-1, 3)
    if grid["kind"] == "cells":
        return _cell_weights(points, grid, None if air else grid["solid"])
    index, weights = _surface_weights(points, grid["origin"], grid["h_mm"], grid["shape"],
                                      grid["nodes_global"])
    lost = weights.sum(1) <= 0
    if lost.any():                                  # outside the bodies: the nearest node
        tree = result.get("_node_tree")
        if tree is None:
            from scipy.spatial import cKDTree
            ny1, nz1 = grid["shape"][1] + 1, grid["shape"][2] + 1
            nodes = grid["nodes_global"]
            node_index = np.stack([nodes // (ny1 * nz1), (nodes // nz1) % ny1, nodes % nz1], axis=1)
            tree = result["_node_tree"] = cKDTree(grid["origin"] + node_index * grid["h_mm"])
        _distance, nearest = tree.query(points[lost])
        index[lost] = np.asarray(nearest)[:, None]
        weights[lost] = 0.0
        weights[lost, 0] = 1.0
    return index, weights


def result_volume_values(result, frame, index, weights):
    """Interpolated temperatures for a stored frame; between two kept whole fields (long
    transient runs keep fewer of them) linearly in time."""
    kept, fields = result["volume_index"], result["volume_frames"]
    position = int(np.searchsorted(kept, frame))
    if position < len(kept) and kept[position] == frame:
        return (fields[position][index] * weights).sum(1)
    position = min(max(position, 1), len(kept) - 1)
    a, b = int(kept[position - 1]), int(kept[position])
    times = result["frame_times"]
    w = 0.0 if times[b] == times[a] else (times[frame] - times[a]) / (times[b] - times[a])
    return (1.0 - w) * (fields[position - 1][index] * weights).sum(1) + \
        w * (fields[position][index] * weights).sum(1)


def result_point_series(result, point, air=False):
    """Temperature at a world point for every stored frame of the result."""
    index, weights = result_point_weights(result, [point], air)
    kept = result["volume_index"]
    values = (result["volume_frames"][:, index[0]] * weights[0]).sum(1)
    if len(kept) == len(result["frame_times"]):
        return values
    return np.interp(result["frame_times"], result["frame_times"][kept], values)


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


# ── airflow + heat (CHT) ──────────────────────────────────────────────────────
# Air and solids on one Cartesian grid: fine cells at the parts that grow towards the domain
# sides. The air follows the steady incompressible Navier–Stokes equations with Boussinesq
# buoyancy (gravity along -Z), discretised by finite volumes on a staggered grid (velocities on
# the cell faces, pressure and temperature in the cells) and solved with SIMPLEC; the energy
# equation covers air and solids together, so no surface coefficient is needed. Laminar by
# default, optionally a mixing-length eddy viscosity. Fans push the air (fixed volume flow or a
# fan curve), vents add a pressure loss; the domain sides are open, walls, inlets or symmetry
# planes.
FLOW_STRETCH = 1.18           # growth of the cells from the parts towards the domain sides
FLOW_MAX_RATIO = 10.0         # largest cell = this × the cell size at the parts
FLOW_RELAX_U = 0.7            # momentum under-relaxation (SIMPLEC)
FLOW_RELAX_P = 1.0            # pressure under-relaxation (SIMPLEC)
FLOW_RELAX_T = 1.0            # temperature
FLOW_MASS_TOLERANCE = 1e-4    # converged: mass residual, temperature change and heat balance
FLOW_DT_TOLERANCE = 2e-5      # temperature change per iteration, share of the temperature span
FLOW_VELOCITY_TOLERANCE = 2e-4    # velocity change per iteration, share of the top speed
FLOW_BALANCE_TOLERANCE = 0.005
MIXING_KAPPA = 0.41


def _stretched_axis(lo_dom, lo_fine, hi_fine, hi_dom, h):
    """Cell faces along one axis: cells of about h between lo_fine and hi_fine, growing
    geometrically (FLOW_STRETCH, at most FLOW_MAX_RATIO · h) towards the domain ends."""
    n = max(1, int(math.ceil((hi_fine - lo_fine) / h - 1e-6)))
    fine = np.linspace(lo_fine, hi_fine, n + 1)
    step = (hi_fine - lo_fine) / n

    def grow(length):
        if length <= 1e-9:
            return np.zeros(0)
        sizes, total, size = [], 0.0, step
        while total < length - 1e-9:
            size = min(size * FLOW_STRETCH, step * FLOW_MAX_RATIO)
            sizes.append(size)
            total += size
        if len(sizes) > 1 and total - length > 0.5 * sizes[-1]:
            sizes.pop()
        sizes = np.asarray(sizes) * (length / sum(sizes))
        return np.cumsum(sizes)

    return np.concatenate([(lo_fine - grow(lo_fine - lo_dom))[::-1], fine,
                           hi_fine + grow(hi_dom - hi_fine)])


def plan_flow_grid(jobs, settings, flow_items=()):
    """Grid of an airflow run: {"faces": (x, y, z) cell faces in mm, "h_mm", "shape", "cells",
    "low", "high", "size", "notes"}, or None without bodies."""
    settings = study_settings(settings)
    boxes = [(np.asarray(job["V"], float).min(0), np.asarray(job["V"], float).max(0))
             for job in jobs if len(job["V"])]
    boxes += [(np.asarray(item["low"], float), np.asarray(item["high"], float)) for item in flow_items]
    if not boxes:
        return None
    low = np.min([box[0] for box in boxes], axis=0)
    high = np.max([box[1] for box in boxes], axis=0)
    size = high - low
    largest = max(float(size.max()), 1.0)
    sides = settings["flow_sides"]
    closed = ("wall", "wall_fixed", "symmetry")

    def space(value, automatic, side):
        if value > 0:
            return float(value)
        return 0.0 if sides[side] in closed else automatic

    beside, below, above = 0.6 * largest, 0.4 * largest, 1.2 * largest
    pad_low = np.array([space(settings["flow_space"], beside, "-X"),
                        space(settings["flow_space"], beside, "-Y"),
                        space(settings["flow_below"], below, "-Z")])
    pad_high = np.array([space(settings["flow_space"], beside, "+X"),
                         space(settings["flow_space"], beside, "+Y"),
                         space(settings["flow_above"], above, "+Z")])
    max_cells = int(settings["flow_max_cells"])
    requested = float(settings["flow_cell_mm"])
    notes = []
    if requested > 0:
        h = requested
    else:
        fine_volume = float(np.prod(np.maximum(size, largest * 0.05)))
        h = _nice((fine_volume / (0.35 * max_cells)) ** (1.0 / 3.0))
        extents = [float(np.ptp(np.asarray(job["V"], float), axis=0).min()) for job in jobs
                   if len(job["V"])]
        thin = [e for e in extents if e > 1e-3]
        if thin:
            h = min(h, _nice(max(min(thin) / 2.0, h / 4.0)))
    for _ in range(60):
        lo_fine = np.where(size < h, (low + high) / 2.0 - h / 2.0, low)
        hi_fine = np.where(size < h, (low + high) / 2.0 + h / 2.0, high)
        dom_low = np.minimum(lo_fine - pad_low, lo_fine)
        dom_high = np.maximum(hi_fine + pad_high, hi_fine)
        faces = [_stretched_axis(dom_low[a], lo_fine[a], hi_fine[a], dom_high[a], h) for a in range(3)]
        shape = tuple(len(f) - 1 for f in faces)
        cells = int(np.prod(shape))
        if cells <= max_cells:
            break
        h = _nice_up(h * 1.08)
    if requested > 0 and h > requested * 1.0001:
        notes.append(f"Cell size raised to {h:g} mm to stay within {max_cells:,} cells "
                     "(Max cells of the air domain).".replace(",", " "))
    return {"faces": faces, "h_mm": h, "shape": shape, "cells": cells, "low": dom_low,
            "high": dom_high, "size": dom_high - dom_low, "notes": notes}


def _voxel_parity_rect(P, centres, axis, pair_limit=4_000_000):
    """Cells of a rectilinear grid (cell centres per axis, mm) whose centres lie inside the
    closed triangle soup P (m, 3, 3) in mm: crossings of rays parallel to `axis`."""
    a, b = [ax for ax in range(3) if ax != axis]
    ca, cb, cc = (np.asarray(centres[i], float) for i in (a, b, axis))
    shape = tuple(len(c) for c in centres)
    toggles = np.zeros(shape, np.uint8)
    spacing = [float(np.min(np.diff(c))) if len(c) > 1 else 1.0 for c in centres]
    A = P[:, :, a] - _RAY_JITTER[a] * spacing[a]
    B = P[:, :, b] - _RAY_JITTER[b] * spacing[b]
    C = P[:, :, axis]
    area = (A[:, 1] - A[:, 0]) * (B[:, 2] - B[:, 0]) - (A[:, 2] - A[:, 0]) * (B[:, 1] - B[:, 0])
    keep = np.abs(area) > 1e-12 * spacing[a] * spacing[b]
    A, B, C, area = A[keep], B[keep], C[keep], area[keep]
    if not len(area):
        return toggles.astype(bool)
    ia0 = np.searchsorted(ca, A.min(1), "left")
    ia1 = np.searchsorted(ca, A.max(1), "right") - 1
    ib0 = np.searchsorted(cb, B.min(1), "left")
    ib1 = np.searchsorted(cb, B.max(1), "right") - 1
    na, nb = ia1 - ia0 + 1, ib1 - ib0 + 1
    count = np.where((na > 0) & (nb > 0), na * nb, 0)
    cum = np.cumsum(count)
    start = 0
    while start < len(count):
        base = int(cum[start - 1]) if start else 0
        end = max(int(np.searchsorted(cum, base + pair_limit, side="right")), start + 1)
        tri = np.repeat(np.arange(start, end), count[start:end])
        start = end
        if not len(tri):
            continue
        offsets = base + np.arange(len(tri)) - (cum[tri] - count[tri])
        ia = ia0[tri] + offsets // nb[tri]
        ib = ib0[tri] + offsets % nb[tri]
        ra, rb = ca[ia], cb[ib]
        a0, a1, a2 = A[tri, 0], A[tri, 1], A[tri, 2]
        b0, b1, b2 = B[tri, 0], B[tri, 1], B[tri, 2]
        w0 = (a1 - a0) * (rb - b0) - (b1 - b0) * (ra - a0)
        w1 = (a2 - a1) * (rb - b1) - (b2 - b1) * (ra - a1)
        w2 = (a0 - a2) * (rb - b2) - (b0 - b2) * (ra - a2)
        inside = ((w0 > 0) & (w1 > 0) & (w2 > 0)) | ((w0 < 0) & (w1 < 0) & (w2 < 0))
        if not inside.any():
            continue
        tri, ia, ib = tri[inside], ia[inside], ib[inside]
        w0, w1, w2 = w0[inside], w1[inside], w2[inside]
        t = (w1 * C[tri, 0] + w2 * C[tri, 1] + w0 * C[tri, 2]) / area[tri]
        k = np.searchsorted(cc, t, "right")
        valid = k < shape[axis]
        index = [None, None, None]
        index[a], index[b], index[axis] = ia[valid], ib[valid], k[valid]
        np.add.at(toggles, tuple(index), 1)
    return (np.cumsum(toggles, axis=axis, dtype=np.uint8) & 1).astype(bool)


def voxelize_rect(P, centres):
    votes = np.zeros(tuple(len(c) for c in centres), np.uint8)
    for axis in range(3):
        votes += _voxel_parity_rect(P, centres, axis)
    return votes >= 2


def _cell_weights(points, grid, mask=None):
    """(index (n, 8), weights (n, 8)) of the trilinear interpolation between cell centres of a
    rectilinear grid; with `mask` (flat bool per cell) only those cells count (the nearest one
    where none is near)."""
    centres = (grid["xc"], grid["yc"], grid["zc"])
    shape = grid["shape"]
    points = np.asarray(points, float).reshape(-1, 3)
    low_index, fraction = [], []
    for axis in range(3):
        c = centres[axis]
        if len(c) == 1:
            low_index.append(np.zeros(len(points), np.int64))
            fraction.append(np.zeros(len(points)))
            continue
        i0 = np.clip(np.searchsorted(c, points[:, axis], "right") - 1, 0, len(c) - 2)
        f = np.clip((points[:, axis] - c[i0]) / (c[i0 + 1] - c[i0]), 0.0, 1.0)
        low_index.append(i0)
        fraction.append(f)
    index = np.empty((len(points), 8), np.int64)
    weights = np.empty((len(points), 8))
    for corner, (ox, oy, oz) in enumerate(_CORNERS):
        ijk = []
        w = np.ones(len(points))
        for axis, offset in enumerate((ox, oy, oz)):
            i = np.minimum(low_index[axis] + offset, shape[axis] - 1)
            ijk.append(i)
            w = w * (fraction[axis] if offset else 1.0 - fraction[axis])
        index[:, corner] = (ijk[0] * shape[1] + ijk[1]) * shape[2] + ijk[2]
        weights[:, corner] = w
    if mask is not None:
        weights = weights * mask[index]
        total = weights.sum(1, keepdims=True)
        lost = total[:, 0] <= 1e-9
        weights = np.where(total > 1e-9, weights / np.maximum(total, 1e-12), 0.0)
        if lost.any():
            tree = grid.get("_mask_tree")
            if tree is None:
                from scipy.spatial import cKDTree
                cells = np.flatnonzero(mask)
                ii, jj, kk = np.unravel_index(cells, shape)
                tree = grid["_mask_tree"] = (cKDTree(np.stack([centres[0][ii], centres[1][jj],
                                                               centres[2][kk]], axis=1)), cells)
            _distance, nearest = tree[0].query(points[lost])
            index[lost] = tree[1][np.asarray(nearest)][:, None]
            weights[lost] = 0.0
            weights[lost, 0] = 1.0
    return index, weights.astype(np.float32)


def airflow_streamlines(result, lattice=(6, 6, 5), frame=None):
    """Streamlines of an airflow result (at a stored frame of a transient run): [(points (n, 3)
    mm, air temperatures (n,))], traced forwards and backwards from seeds at fans, vents, inlet
    sides and around the parts."""
    grid = result["grid"]
    shape = grid["shape"]
    faces = grid["faces"]
    centres = (grid["xc"], grid["yc"], grid["zc"])
    low = np.array([f[0] for f in faces])
    high = np.array([f[-1] for f in faces])
    fluid = result["fluid"].ravel()
    velocity = result_velocity(result, frame)[0].reshape(-1, 3)
    kept = np.asarray(result["volume_index"])
    temperature = result["volume_frames"][int(np.argmin(np.abs(kept - (frame if frame is not None
                                                                         else kept[-1]))))]
    h = grid["h_mm"]
    seeds = []
    for device in result.get("devices", []):
        lo, hi = np.asarray(device["low"], float), np.asarray(device["high"], float)
        axis = device["axis"]
        a, b = [x for x in range(3) if x != axis]
        for s in np.linspace(0.15, 0.85, 4):
            for t in np.linspace(0.15, 0.85, 4):
                point = (lo + hi) / 2.0
                point[a] = lo[a] + s * (hi[a] - lo[a])
                point[b] = lo[b] + t * (hi[b] - lo[b])
                seeds.append(point)
    for axis, pair in enumerate(result.get("sides", [])):
        for side, kind in enumerate(pair):
            if kind != "inlet":
                continue
            a, b = [x for x in range(3) if x != axis]
            for s in np.linspace(0.1, 0.9, 6):
                for t in np.linspace(0.1, 0.9, 6):
                    point = np.empty(3)
                    point[axis] = low[axis] + 1.5 * h if side == 0 else high[axis] - 1.5 * h
                    point[a] = low[a] + s * (high[a] - low[a])
                    point[b] = low[b] + t * (high[b] - low[b])
                    seeds.append(point)
    solid = grid["solid"].reshape(shape)
    if solid.any():
        ii, jj, kk = np.nonzero(solid)
        lo = np.array([centres[0][ii].min(), centres[1][jj].min(), centres[2][kk].min()])
        hi = np.array([centres[0][ii].max(), centres[1][jj].max(), centres[2][kk].max()])
        pad = 0.25 * (hi - lo) + 2.0 * h
        lo, hi = np.maximum(lo - pad, low), np.minimum(hi + pad, high)
        axes = [np.linspace(lo[a], hi[a], n) for a, n in enumerate(lattice)]
        seeds += [np.array([x, y, z]) for x in axes[0] for y in axes[1] for z in axes[2]]
    if not seeds:
        return []
    seeds = np.asarray(seeds, float)

    def sample(points):
        index, weights = _cell_weights(points, grid, None)
        return (velocity[index] * weights[..., None]).sum(1), \
            index[np.arange(len(points)), np.argmax(weights, 1)]

    _vel, cell = sample(seeds)
    seeds = seeds[fluid[cell]]
    if not len(seeds):
        return []
    step = 0.6 * h
    max_steps = int(min(1500, 2.5 * float((high - low).max()) / step))

    def trace(sign):
        position = seeds.copy()
        alive = np.ones(len(position), bool)
        out = [position.copy()]
        for _ in range(max_steps):
            if not alive.any():
                break
            vel, cell = sample(position)
            speed = np.linalg.norm(vel, axis=1)
            alive &= (speed > 1e-4) & fluid[cell] & np.all((position >= low) & (position <= high), 1)
            direction = sign * vel / np.maximum(speed, 1e-12)[:, None]
            middle = position + 0.5 * step * direction
            vel2, _cell = sample(middle)
            direction2 = sign * vel2 / np.maximum(np.linalg.norm(vel2, axis=1), 1e-12)[:, None]
            position = np.where(alive[:, None], position + step * direction2, position)
            out.append(np.where(alive[:, None], position, np.nan))
        return np.stack(out, 1)

    forward, backward = trace(1.0), trace(-1.0)
    lines = []
    for f, b in zip(forward, backward):
        path = np.concatenate([b[::-1][:-1], f])
        path = path[np.all(np.isfinite(path), axis=1)]
        if len(path) < 4 or float(np.linalg.norm(np.ptp(path, axis=0))) < 3.0 * step:
            continue
        index, weights = _cell_weights(path, grid, None)
        lines.append((path.astype(np.float32), (temperature[index] * weights).sum(1)))
    return lines


def _power_law(diffusion, flow):
    """Patankar's power-law weight A(|Pe|) · D for a face with conductance `diffusion` and
    convective flow `flow` (same units)."""
    peclet = np.abs(flow) / np.maximum(diffusion, 1e-300)
    return diffusion * np.maximum(0.0, (1.0 - 0.1 * peclet)) ** 5


def _seven_point(sp, shape, diagonal, neighbours):
    """CSR matrix of a 7-point stencil on a C-ordered grid: `neighbours[(axis, ±1)]` holds the
    (positive) coupling coefficients, entered as -a."""
    n = int(np.prod(shape))
    strides = (shape[1] * shape[2], shape[2], 1)
    diagonals, offsets = [diagonal.ravel()], [0]
    for (axis, sign), coefficient in neighbours.items():
        if shape[axis] == 1 or not coefficient.any():
            continue                       # no neighbours along a flat axis
        offset = sign * strides[axis]
        values = -coefficient.ravel()
        diagonals.append(values[:n - offset] if offset > 0 else values[-offset:])
        offsets.append(offset)
    return sp.diags(diagonals, offsets, shape=(n, n), format="csr")


def _box_cells(centres, low, high):
    """Bool grid: cells whose centres lie in the box [low, high] (mm); at least the cell
    nearest to the box centre along each axis."""
    masks = []
    for axis in range(3):
        c = centres[axis]
        inside = (c >= low[axis]) & (c <= high[axis])
        if not inside.any():
            inside = np.zeros(len(c), bool)
            inside[int(np.argmin(np.abs(c - (low[axis] + high[axis]) / 2.0)))] = True
        masks.append(inside)
    return masks[0][:, None, None] & masks[1][None, :, None] & masks[2][None, None, :]


def solve_airflow(jobs, settings, progress=None, cancelled=None, conditions=(), contacts=(),
                  flow_items=()):
    """Air flow and temperatures of air and solids together (conjugate heat transfer), steady
    or over time (settings "analysis"; implicit time steps, each converged by inner SIMPLEC
    iterations). Returns a result dict like solve_thermal plus the air velocity field (and its
    frames over time)."""
    sp, spla, ndi, cKDTree = _scipy()
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
    ambient_k = ambient + 273.15
    sides = settings["flow_sides"]
    side_kind = [[sides[f"-{axis}"], sides[f"+{axis}"]] for axis in "XYZ"]
    step(2, "Preparing bodies...")
    bodies = _prepare_bodies(jobs, messages)
    conditions = [dict(condition) for condition in (conditions or ())]
    for condition in conditions:
        condition["profile"] = profile_settings(condition.get("profile"))
    if any(c["kind"] == "convection" and c["targets"] for c in conditions):
        messages.append("Convection face conditions are left out: in Airflow mode the air flow "
                        "itself cools the surfaces.")
    conditions = [c for c in conditions if c["kind"] != "convection"]
    _check_thermostats(bodies, conditions)
    transient = settings["analysis"] == "transient"
    has_fixed = any(c["kind"] == "fixed" and c["targets"] for c in conditions) or \
        any("wall_fixed" in pair for pair in side_kind)
    power = [profile_steady_power(body["profile"], body["power"]) for body in bodies]
    if not any(p > 0 for p in power) and not has_fixed and \
            not any(c["kind"] == "flux" and c["targets"] for c in conditions):
        raise ThermalError("No heat source: give a body a power (W), add a heat source, or put a "
                           "heat flux or fixed temperature on faces.")
    closed_box = not any(kind in ("open", "inlet", "wall_fixed") for pair in side_kind for kind in pair)
    if closed_box and not settings["radiation"] and \
            not any(c["kind"] == "fixed" and c["targets"] for c in conditions):
        raise ThermalError("The heat cannot leave: all domain sides are closed walls, radiation is "
                           "off and no face has a fixed temperature (Simulation panel → Air domain).")
    if any(pair.count("inlet") for pair in side_kind) and \
            not any(kind == "open" for pair in side_kind for kind in pair):
        raise ThermalError("Air blown in through an Inlet side needs an Open side to leave the "
                           "domain (Simulation panel → Air domain).")

    # 1. grid and cells -------------------------------------------------------------------
    step(4, "Building the grid...")
    plan = plan_flow_grid(jobs, settings, flow_items)
    messages += plan["notes"]
    faces_mm = plan["faces"]
    shape = plan["shape"]
    n0, n1, n2 = shape
    N = n0 * n1 * n2
    centres_mm = [(f[1:] + f[:-1]) / 2.0 for f in faces_mm]
    h_mm = plan["h_mm"]
    owner = np.full(shape, -1, np.int16)
    for index, body in enumerate(bodies):
        step(5 + 15 * index / len(bodies), f"Meshing {body['name']}...")
        ranges = []
        for axis in range(3):
            c = centres_mm[axis]
            ranges.append((int(np.searchsorted(c, body["low"][axis], "left")),
                           int(np.searchsorted(c, body["high"][axis], "right"))))
        body["cells"] = 0
        if any(hi <= lo for lo, hi in ranges):
            continue
        inside = voxelize_rect(body["V"][body["T"]], [centres_mm[a][lo:hi]
                                                      for a, (lo, hi) in enumerate(ranges)])
        region = tuple(slice(lo, hi) for lo, hi in ranges)
        sub = owner[region]
        replaced = sub[inside]
        replaced = replaced[replaced >= 0]
        if len(replaced) and not body["job"].get("zone"):
            for other, cells in enumerate(np.bincount(replaced, minlength=len(bodies))):
                if cells:
                    messages.append(f"'{bodies[other]['name']}' and '{body['name']}' overlap; the "
                                    f"overlap belongs to '{body['name']}'.")
        sub[inside] = index
    devices = []
    for item in flow_items:
        region = _box_cells(centres_mm, item["low"], item["high"])
        opened = int((owner[region] >= 0).sum())
        owner[region] = -1                       # fans and vents open walls they overlap
        if not item["aligned"]:
            messages.append(f"⚠ '{item['name']}' is not aligned with a grid axis; it acts along "
                            f"{'+' if item['sign'] > 0 else '−'}{'XYZ'[item['axis']]}.")
        devices.append(dict(item, region=region, opened=opened))
    fluid = owner < 0
    solid = ~fluid
    if not fluid.any():
        raise ThermalError("There is no air in the domain.")
    cells_of = np.bincount(owner[solid].astype(np.int64), minlength=len(bodies))
    for index, body in enumerate(bodies):
        body["cells"] = int(cells_of[index])
        if not body["cells"]:
            if power[index] > 0:
                raise ThermalError(f"The heat source '{body['name']}' is thinner than one cell "
                                   f"({h_mm:g} mm). Choose a smaller cell size.")
            messages.append(f"⚠ '{body['name']}' is thinner than one cell ({h_mm:g} mm) and is "
                            "left out.")

    # geometry in SI units
    F = [f / 1000.0 for f in faces_mm]                 # cell faces, m
    C = [c / 1000.0 for c in centres_mm]               # cell centres, m
    D = [np.diff(f) for f in F]                        # cell widths, m

    def along(values, axis, ndim=3):
        shape_ = [1] * ndim
        shape_[axis] = len(values)
        return np.asarray(values, float).reshape(shape_)

    def sl(axis, start, stop, ndim=3):
        index = [slice(None)] * ndim
        index[axis] = slice(start, stop)
        return tuple(index)

    others = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    volume = along(D[0], 0) * along(D[1], 1) * along(D[2], 2)
    face_area = [along(D[others[c][0]], others[c][0]) * along(D[others[c][1]], others[c][1])
                 for c in range(3)]                    # area of the c-faces (broadcast)
    # control-volume length of the staggered c-velocities along c
    cv_length = []
    for c in range(3):
        length = np.empty(shape[c] + 1)
        length[0], length[-1] = D[c][0] / 2.0, D[c][-1] / 2.0
        length[1:-1] = np.diff(C[c])
        cv_length.append(length)

    # 2. material data per cell ---------------------------------------------------------------
    rho, cp, mu_air, k_air = AIR_DENSITY, AIR_HEAT_CAPACITY, AIR_VISCOSITY, AIR_CONDUCTIVITY
    beta = 1.0 / ambient_k
    k_table = np.array([body["k"] for body in bodies] + [[k_air] * 3])
    owner_index = np.where(solid, owner, len(bodies)).astype(np.int64)
    k_cell = [k_table[owner_index, axis] for axis in range(3)]          # W/(m·K) per axis
    # heat sources in W per cell; a source with a time profile or a thermostat is kept as its
    # distribution of 1 W and scaled in every time step (or by its thermostat)
    volume_flat = np.broadcast_to(volume, shape).ravel()
    owner_flat = owner.ravel()
    constant_source = np.zeros(N)
    constant_power = 0.0
    parts = []          # (cells, weights of 1 W, profile, nominal W, "source" | "flux", index, slot)
    controlled = []     # thermostat slots: (name, thermostat, "source" | "flux", index)
    for index, body in enumerate(bodies):
        if not body["cells"] or profile_peak_power(body["profile"], body["power"]) <= 0:
            continue
        cells = np.flatnonzero(owner_flat == index)
        weights = volume_flat[cells] / volume_flat[cells].sum()
        if body["profile"] is None and body["thermostat"] is None:
            constant_source[cells] += body["power"] * weights
            constant_power += body["power"]
        else:
            slot = None
            if body["thermostat"] is not None:
                slot = len(controlled)
                controlled.append((body["name"], body["thermostat"], "source", index))
            parts.append((cells, weights, body["profile"], body["power"], "source", index, slot))
    source = np.zeros(shape)
    power_now = [0.0]                    # W put in at the moment (sources and fluxes)
    # temperature-dependent material data (k(T) per cell, cp(T) for transient runs)
    k_tabled = [index for index, body in enumerate(bodies) if body["k_curve"] is not None and
                body["cells"]]
    cp_tabled = [index for index, body in enumerate(bodies) if body["cp_curve"] is not None and
                 body["cells"]] if transient else []
    cells_of_body = {index: np.flatnonzero(owner_flat == index)
                     for index in set(k_tabled) | set(cp_tabled)}

    # fluid regions connected to the open air (radiation goes from their surfaces)
    labels, _count = ndi.label(fluid)
    border = set()
    for axis in range(3):
        for side in (0, 1):
            if side_kind[axis][side] in ("open", "inlet"):
                border |= set(np.unique(np.take(labels, 0 if side == 0 else shape[axis] - 1,
                                                axis=axis)).tolist())
    border.discard(0)
    outer_air = np.isin(labels, list(border)) if border else np.zeros(shape, bool)
    region_labels = labels
    cavities = int(((labels > 0) & ~outer_air).sum())
    if cavities:
        messages.append(f"Closed air spaces ({cavities:,} cells) have their own internal air "
                        "flow; no air is exchanged with the outside.".replace(",", " "))

    # 3. velocity unknowns: active faces, fixed values --------------------------------------
    active, fixed = [], []
    for c in range(3):
        Sc = list(shape); Sc[c] += 1
        act = np.zeros(Sc, bool)
        val = np.zeros(Sc)
        act[sl(c, 1, -1)] = fluid[sl(c, 0, -1)] & fluid[sl(c, 1, None)]
        for side, index in ((0, 0), (1, shape[c])):
            kind = side_kind[c][side]
            cell = fluid[sl(c, 0, 1)] if side == 0 else fluid[sl(c, -1, None)]
            if kind == "open":
                act[sl(c, index, index + 1)] = cell
            elif kind == "inlet":
                val[sl(c, index, index + 1)] = np.where(cell, settings["inlet_speed"] *
                                                         (1.0 if side == 0 else -1.0), 0.0)
        active.append(act)
        fixed.append(val)
    # fans: fixed flow → prescribed velocity on the middle face layer; curve → pressure jump
    for device in devices:
        c = device["axis"]
        region = device["region"]
        layer_cells = np.nonzero(region.any(axis=tuple(a for a in range(3) if a != c)))[0]
        middle = int(layer_cells[len(layer_cells) // 2])           # face below the middle cell
        mask = np.zeros(active[c].shape, bool)
        cross = region.take(middle, axis=c)
        index = [slice(None)] * 3
        index[c] = middle
        mask[tuple(index)] = cross & (fluid.take(max(middle - 1, 0), axis=c) | (middle == 0)) & \
            fluid.take(middle, axis=c)
        device["faces"] = mask
        device["area"] = float((np.broadcast_to(face_area[c], mask.shape)[mask]).sum()) \
            if mask.any() else 0.0
        if device["area"] <= 0:
            messages.append(f"⚠ '{device['name']}' has no free air face; it has no effect.")
            continue
        if device["kind"] == "fan" and device["flow"]["flow_mode"] == "fixed":
            speed = device["flow"]["flow_m3h"] / 3600.0 / device["area"]
            active[c][mask] = False
            fixed[c][mask] = device["sign"] * speed
            device["speed"] = speed
        device["dp"] = 0.0
    velocity = [fixed[c].copy() for c in range(3)]
    p = np.zeros(shape)
    T = np.full(shape, float(ambient))

    # 4. face conditions on solid faces next to air -------------------------------------------
    face_condition = [np.full(active[c].shape, -1, np.int64) for c in range(3)]
    flux_faces = {}
    if conditions:
        step(21, "Applying face conditions...")
        records = []          # (axis c, face index tuple arrays, solid cell, outward normal sign)
        for c in range(3):
            lo_solid = solid[sl(c, 0, -1)] & fluid[sl(c, 1, None)]     # solid below, air above
            hi_solid = fluid[sl(c, 0, -1)] & solid[sl(c, 1, None)]
            for mask, sign, cell_offset in ((lo_solid, 1.0, 0), (hi_solid, -1.0, 1)):
                idx = np.nonzero(mask)
                if not len(idx[0]):
                    continue
                face_idx = list(idx)
                face_idx[c] = face_idx[c] + 1
                cell_idx = list(idx)
                cell_idx[c] = cell_idx[c] + cell_offset
                records.append((c, tuple(face_idx), tuple(cell_idx), sign))
        targets = sorted({b for condition in conditions for b in condition["targets"]})
        for b in targets:
            body = bodies[b]
            tri_face = np.full(len(body["T"]), -1)
            for face_index, tris in enumerate(body["job"].get("faces", [])):
                tri_face[np.asarray(tris, np.int64)] = face_index
            P = body["V"][body["T"]]
            normal = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
            samples, _triangles, src = refine_triangles(body["V"], body["T"], h_mm, with_source=True)
            tree = cKDTree(samples)
            for c, face_idx, cell_idx, sign in records:
                mine = owner[cell_idx] == b
                if not mine.any():
                    continue
                fi = tuple(arr[mine] for arr in face_idx)
                ci = tuple(arr[mine] for arr in cell_idx)
                centre = np.stack([centres_mm[a][ci[a]] for a in range(3)], axis=1)
                centre[:, c] = faces_mm[c][fi[c]]
                n_vec = np.zeros(3); n_vec[c] = sign
                count = min(8, len(samples))
                distance, nearest = tree.query(centre, k=count)
                distance = distance.reshape(len(centre), -1)
                nearest = nearest.reshape(len(centre), -1)
                fits = ((normal[src[nearest]] @ n_vec) > 0.3) & (distance < 2.0 * h_mm)
                first = np.argmax(fits, axis=1)
                rows = np.arange(len(centre))
                found = fits[rows, first]
                mesh_face = np.where(found, tri_face[src[nearest[rows, first]]], -1)
                for index, condition in enumerate(conditions):
                    faces = condition["targets"].get(b)
                    if not faces:
                        continue
                    hit = np.isin(mesh_face, faces)
                    if hit.any():
                        sel = tuple(arr[hit] for arr in fi)
                        face_condition[c][sel] = index
        for index, condition in enumerate(conditions):
            if condition["targets"] and not any((fc == index).any() for fc in face_condition):
                messages.append(f"⚠ Face condition '{condition['name']}' reaches no surface next "
                                "to air.")
    # heat fluxes go into the solid cell next to each flux face
    flux_power = np.zeros(len(conditions))          # W per condition (steady: long-term value)
    flux_constant = np.zeros(len(conditions))
    for index, condition in enumerate(conditions):
        if condition["kind"] != "flux":
            continue
        area = sum(float(np.broadcast_to(face_area[c], fc.shape)[fc == index].sum())
                   for c, fc in enumerate(face_condition))
        if area <= 0:
            continue
        density = condition["power_w"] / area if condition["flux_mode"] == "total" else condition["flux_w_m2"]
        nominal = density * area
        unit = np.zeros(shape)
        for c in range(3):
            mask = face_condition[c] == index
            if not mask.any():
                continue
            idx = np.nonzero(mask)
            face_a = np.broadcast_to(face_area[c], mask.shape)[mask]
            below = list(idx); below[c] = idx[c] - 1
            for offset_idx, ok in ((tuple(below), below[c] >= 0), (idx, idx[c] < shape[c])):
                cell = tuple(arr[ok] for arr in offset_idx)
                if not len(cell[0]):
                    continue
                is_solid = solid[cell]
                np.add.at(unit, tuple(arr[is_solid] for arr in cell), face_a[ok][is_solid] / area)
        unit = unit.ravel()
        cells = np.flatnonzero(unit)
        if condition.get("profile") is None and condition.get("thermostat") is None:
            constant_source[cells] += nominal * unit[cells]
            flux_constant[index] = flux_power[index] = nominal
        else:
            slot = None
            if condition.get("thermostat") is not None:
                slot = len(controlled)
                controlled.append((condition["name"], condition["thermostat"], "flux", index))
            parts.append((cells, unit[cells], condition["profile"], nominal, "flux", index, slot))
            flux_power[index] = profile_steady_power(condition["profile"], nominal)
    flux_faces.clear()

    def heat_sources(t0=None, t1=None, factors=None):
        """Fill `source` for steady state (no times) or the time step t0 … t1; `factors` scale
        the thermostat-controlled sources. Returns (source W, flux W per condition)."""
        values = constant_source.copy()
        source_w, flux = constant_power, flux_constant.copy()
        for cells, weights, profile, nominal, kind, index, slot in parts:
            value = profile_steady_power(profile, nominal) if t0 is None else \
                profile_mean_power(profile, nominal, t0, t1)
            if slot is not None and factors is not None:
                value *= float(factors[slot])
            if value:
                values[cells] += value * weights
            if kind == "source":
                source_w += value
            else:
                flux[index] = value
        source[...] = values.reshape(shape)
        power_now[0] = source_w + float(flux.sum())
        return source_w, flux

    total_power, flux_power = heat_sources()

    # contacts: extra resistance between the touching cells of two bodies
    contacts = list(contacts or ())
    contact_resistance_grid = [np.zeros(active[c].shape) for c in range(3)]
    contact_id = [np.full(active[c].shape, -1, np.int64) for c in range(3)]
    if contacts:
        n_owner = len(bodies) + 1
        table = np.zeros((n_owner, n_owner))
        number = np.full((n_owner, n_owner), -1)
        for index, contact in enumerate(contacts):
            a, b = contact["pair"]
            table[a, b] = table[b, a] = max(0.0, float(contact["R"]))
            number[a, b] = number[b, a] = index
        for c in range(3):
            lo = owner_index[sl(c, 0, -1)]
            hi = owner_index[sl(c, 1, None)]
            contact_resistance_grid[c][sl(c, 1, -1)] = table[lo, hi]
            contact_id[c][sl(c, 1, -1)] = number[lo, hi]

    # 5. linear-system helpers -------------------------------------------------------------
    cell_index = np.stack(np.unravel_index(np.arange(N), shape), axis=1)
    prolongation = _coarse_space(cell_index, COARSE_BLOCK)
    coarse = {"pressure": None, "energy": None}

    def two_level(matrix, key, refresh):
        inverse = 1.0 / matrix.diagonal()
        if refresh or coarse[key] is None:
            coarse[key] = spla.splu((prolongation.T @ matrix @ prolongation).tocsc())
        lu = coarse[key]
        return spla.LinearOperator(matrix.shape, dtype=float,
                                   matvec=lambda v: inverse * v + prolongation @ lu.solve(prolongation.T @ v))

    # wall distance for the mixing-length model ("auto": turbulent as soon as a fan or an inlet
    # blows; buoyant flows of small parts are laminar)
    forced = any(device["kind"] == "fan" for device in devices) or \
        any(kind == "inlet" for pair in side_kind for kind in pair)
    turbulent = settings["turbulence"] == "mixing" or (settings["turbulence"] == "auto" and forced)
    flow_model = "turbulent (mixing length)" if turbulent else "laminar"
    mu_t = np.zeros(shape)
    if turbulent:
        spacing = [float(np.median(d)) for d in D]
        wall_distance = ndi.distance_transform_edt(fluid, sampling=spacing)
        reference = max(float(max(np.ptp(body["V"], axis=0).max() for body in bodies)) / 1000.0, 0.01)
        mixing = np.minimum(MIXING_KAPPA * wall_distance, 0.09 * reference)

    def cell_velocity():
        return [0.5 * (velocity[c][sl(c, 0, -1)] + velocity[c][sl(c, 1, None)]) for c in range(3)]

    def update_turbulence():
        centre = cell_velocity()
        grads = [[np.gradient(centre[i], C[j], axis=j) if shape[j] > 1 else np.zeros(shape)
                  for j in range(3)] for i in range(3)]
        strain = 0.0
        for i in range(3):
            for j in range(3):
                s = 0.5 * (grads[i][j] + grads[j][i])
                strain = strain + 2.0 * s * s
        new = rho * mixing ** 2 * np.sqrt(strain)
        new[solid] = 0.0
        mu_t[:] = 0.5 * mu_t + 0.5 * np.minimum(new, 1000.0 * mu_air)

    # momentum ---------------------------------------------------------------------------------
    def momentum(c):
        """Assemble and solve the c-momentum equation; returns d (pressure-correction factors)."""
        Sc = active[c].shape
        U = velocity[c]
        mu = mu_air + mu_t
        aP = np.zeros(Sc)
        b = np.zeros(Sc)
        nb = {}
        # neighbours along c (through the cell centres)
        area_c = face_area[c]
        Fc = rho * 0.5 * (U[sl(c, 0, -1)] + U[sl(c, 1, None)]) * area_c
        Dc = mu * area_c / along(D[c], c)
        weight = _power_law(Dc, Fc)
        east = np.zeros(Sc); west = np.zeros(Sc)
        east[sl(c, 0, -1)] = weight + np.maximum(-Fc, 0.0)
        west[sl(c, 1, None)] = weight + np.maximum(Fc, 0.0)
        nb[(c, 1)], nb[(c, -1)] = east, west
        # neighbours along the other axes
        for d in others[c]:
            e = 3 - c - d
            Vd = velocity[d]
            inner = Vd[sl(d, 1, -1)]                              # interior d-faces
            pad = [(0, 0)] * 3
            pad[c] = (1, 1)
            padded = np.pad(inner, pad, mode="edge")
            vd = 0.5 * (padded[sl(c, 0, -1)] + padded[sl(c, 1, None)])   # at the c-faces
            area = along(cv_length[c], c) * along(D[e], e)
            Fd = rho * vd * area
            mu_pad = np.pad(mu, pad, mode="edge")
            mu_c = 0.5 * (mu_pad[sl(c, 0, -1)] + mu_pad[sl(c, 1, None)])
            mu_d = 0.5 * (mu_c[sl(d, 0, -1)] + mu_c[sl(d, 1, None)])
            Dd = mu_d * area / along(np.diff(C[d]), d)
            weight = _power_law(Dd, Fd)
            north = np.zeros(Sc); south = np.zeros(Sc)
            north[sl(d, 0, -1)] = weight + np.maximum(-Fd, 0.0)
            south[sl(d, 1, None)] = weight + np.maximum(Fd, 0.0)
            # a neighbour face that is blocked by a solid is a no-slip wall half a cell away
            area_full = np.broadcast_to(along(cv_length[c], c) * along(D[e], e), Sc)
            wall_plus = np.zeros(Sc, bool); wall_minus = np.zeros(Sc, bool)
            wall_plus[sl(d, 0, -1)] = ~active[c][sl(d, 1, None)] & (fixed[c][sl(d, 1, None)] == 0)
            wall_minus[sl(d, 1, None)] = ~active[c][sl(d, 0, -1)] & (fixed[c][sl(d, 0, -1)] == 0)
            half = np.broadcast_to(along(D[d], d) / 2.0, Sc)
            wall_coefficient = mu_air * area_full / half
            aP += np.where(wall_plus, wall_coefficient, 0.0) + np.where(wall_minus, wall_coefficient, 0.0)
            north[wall_plus] = 0.0
            south[wall_minus] = 0.0
            # domain sides along d: no-slip at walls and inlets, free slip at symmetry / open sides
            for side, index in ((0, 0), (1, shape[d] - 1)):
                if side_kind[d][side] in ("wall", "wall_fixed", "inlet"):
                    edge = sl(d, index, index + 1)
                    aP[edge] += (mu_air * area_full / half)[edge]
            nb[(d, 1)], nb[(d, -1)] = north, south
        total = sum(nb.values())
        aP += total
        # couplings to fixed faces move to the right-hand side
        for (axis, sign), coefficient in nb.items():
            neighbour_fixed = np.zeros(Sc, bool)
            neighbour_value = np.zeros(Sc)
            if sign > 0:
                neighbour_fixed[sl(axis, 0, -1)] = ~active[c][sl(axis, 1, None)]
                neighbour_value[sl(axis, 0, -1)] = velocity[c][sl(axis, 1, None)]
            else:
                neighbour_fixed[sl(axis, 1, None)] = ~active[c][sl(axis, 0, -1)]
                neighbour_value[sl(axis, 1, None)] = velocity[c][sl(axis, 0, -1)]
            b += np.where(neighbour_fixed, coefficient * neighbour_value, 0.0)
            coefficient[neighbour_fixed] = 0.0
        # pressure gradient (open sides: ambient pressure outside)
        pad = [(0, 0)] * 3
        pad[c] = (1, 1)
        p_pad = np.pad(p, pad)                                    # zero = ambient outside
        area_face = np.broadcast_to(face_area[c], Sc)
        b += (p_pad[sl(c, 0, -1)] - p_pad[sl(c, 1, None)]) * area_face
        # buoyancy (Boussinesq), gravity along -Z
        if c == 2:
            T_pad = np.pad(T, pad, mode="edge")
            T_face = 0.5 * (T_pad[sl(2, 0, -1)] + T_pad[sl(2, 1, None)])
            cv = np.broadcast_to(along(cv_length[2], 2) * face_area[2], Sc)
            b += rho * GRAVITY * beta * (T_face - ambient) * cv
        # fans with a curve: pressure jump; vents: pressure loss
        for device in devices:
            if device["axis"] != c or not device.get("area"):
                continue
            mask = device["faces"]
            if device["kind"] == "fan" and device["flow"]["flow_mode"] == "curve":
                b[mask] += device["sign"] * device["dp"] * area_face[mask]
            elif device["kind"] == "vent":
                zeta = vent_loss_coefficient(device["flow"]["free_area"])
                aP[mask] += zeta * 0.5 * rho * np.abs(U[mask]) * area_face[mask]
        # pseudo time step (inertia ρ·V/Δt with Δt = cell / velocity scale) plus under-relaxation:
        # keeps the first iterations, with large buoyancy and no flow yet, from overshooting
        inertia = rho * area_face * velocity_scale[0]
        aP = aP + inertia
        b += inertia * U
        if unsteady["dt"] is not None:              # time step: ρ·V/Δt · (U − U_old)
            mass = rho * np.broadcast_to(along(cv_length[c], c) * face_area[c], Sc) / unsteady["dt"]
            aP = aP + mass
            b += mass * unsteady["velocity"][c]
        aP_relaxed = aP / FLOW_RELAX_U
        b += (aP_relaxed - aP) * U
        act = active[c]
        aP_relaxed = np.where(act, np.maximum(aP_relaxed, 1e-30), 1.0)
        b = np.where(act, b, velocity[c])
        for key in nb:
            nb[key] = np.where(act, nb[key], 0.0)
        matrix = _seven_point(sp, Sc, aP_relaxed, nb)
        inverse = 1.0 / aP_relaxed.ravel()
        preconditioner = spla.LinearOperator(matrix.shape, dtype=float, matvec=lambda v: inverse * v)
        solution, _info = spla.bicgstab(matrix, b.ravel(), x0=U.ravel(), rtol=1e-3, atol=0.0,
                                        maxiter=60, M=preconditioner)
        if not np.all(np.isfinite(solution)):
            raise ThermalError("The air flow did not converge (the velocities grew without "
                               "bound). Try a smaller cell size or the turbulent flow model.")
        velocity[c][...] = solution.reshape(Sc)
        # SIMPLEC: d = A / (a_P - Σ a_nb)
        denominator = aP_relaxed - total
        return np.where(act, area_face / np.maximum(denominator, 1e-30 * aP_relaxed + 1e-300), 0.0)

    # pressure correction ---------------------------------------------------------------------
    pin = []                       # one cell of every air region without an open side
    open_region = set()
    for c in range(3):
        for side, index in ((0, 0), (1, shape[c])):
            if side_kind[c][side] == "open":
                faces = active[c].take(index, axis=c)
                cells = region_labels.take(0 if side == 0 else shape[c] - 1, axis=c)
                open_region |= set(np.unique(cells[faces]).tolist())
    for label in range(1, int(region_labels.max()) + 1):
        if label not in open_region:
            cells = np.flatnonzero(region_labels.ravel() == label)
            if len(cells):
                pin.append(int(cells[0]))

    def pressure_correction(d, refresh):
        aP = np.zeros(shape)
        nb = {}
        imbalance = np.zeros(shape)
        for c in range(3):
            area_face = np.broadcast_to(face_area[c], active[c].shape)
            flow = rho * velocity[c] * area_face                   # kg/s through every c-face
            imbalance += flow[sl(c, 1, None)] - flow[sl(c, 0, -1)]
            coefficient = rho * d[c] * area_face
            aP += coefficient[sl(c, 1, None)] + coefficient[sl(c, 0, -1)]
            east = coefficient[sl(c, 1, None)].copy()
            west = coefficient[sl(c, 0, -1)].copy()
            east[sl(c, -1, None)] = 0.0                       # the open side: ghost p' = 0
            west[sl(c, 0, 1)] = 0.0
            nb[(c, 1)], nb[(c, -1)] = east, west
        rhs = -imbalance.ravel()
        diagonal = aP.ravel().copy()
        empty = diagonal <= 0
        diagonal[empty] = 1.0
        rhs[empty] = 0.0
        for cell in pin:
            diagonal[cell] += 1e6 * max(float(diagonal.max()), 1e-30)
        matrix = _seven_point(sp, shape, diagonal, nb)
        mass_scale = max(float(np.abs(imbalance).sum()), 1e-300)
        correction, _info = spla.cg(matrix, rhs, rtol=0.05, atol=0.0, maxiter=200,
                                    M=two_level(matrix, "pressure", refresh))
        correction = correction.reshape(shape)
        for c in range(3):
            pad = [(0, 0)] * 3
            pad[c] = (1, 1)
            pc = np.pad(correction, pad)
            velocity[c] += d[c] * (pc[sl(c, 0, -1)] - pc[sl(c, 1, None)])
        p[...] += FLOW_RELAX_P * correction
        return mass_scale

    # energy -----------------------------------------------------------------------------------
    k_eff = None
    run_length = np.array([[max((body["extent"].sum() - body["extent"][axis]) / 2.0, 2.0) / 1000.0
                            for axis in range(3)] for body in bodies] + [[1.0] * 3])
    solid_face = []                     # per interior c-face: 1 solid → air, 2 air → solid
    for c in range(3):
        lo_solid, hi_solid = solid[sl(c, 0, -1)], solid[sl(c, 1, None)]
        kind = np.zeros(lo_solid.shape, np.int8)
        kind[lo_solid & ~hi_solid] = 1
        kind[~lo_solid & hi_solid] = 2
        solid_face.append(kind)

    def conductances():
        """(G per c-face array, G_half per cell for each side) with turbulent conductivity."""
        nonlocal k_eff
        k_turbulent = cp * mu_t / 0.9
        k_eff = [k_cell[c] + np.where(fluid, k_turbulent, 0.0) for c in range(3)]
        centre = cell_velocity()
        G = []
        for c in range(3):
            Sc = active[c].shape
            g = np.zeros(Sc)
            area = np.broadcast_to(face_area[c], Sc)
            half = along(D[c], c) / (2.0 * k_eff[c])           # m²K/W of a half cell
            lo_half, hi_half = half[sl(c, 0, -1)], half[sl(c, 1, None)]
            # wall function: the air side of a solid face conducts at least as well as a flat
            # plate in the local air speed (coarse cells cannot resolve thin boundary layers)
            kind = solid_face[c]                     # 1: solid below / air above, 2: the reverse
            if kind.any():
                parallel = np.sqrt(sum(centre[a] ** 2 for a in range(3) if a != c))
                speed_air = np.where(kind == 1, parallel[sl(c, 1, None)], parallel[sl(c, 0, -1)])
                body_side = np.where(kind == 1, owner_index[sl(c, 0, -1)], owner_index[sl(c, 1, None)])
                film = 1.0 / np.maximum(_forced_h(speed_air, run_length[body_side, c]), 1e-12)
                lo_half = np.where(kind == 2, np.minimum(lo_half, film), lo_half)
                hi_half = np.where(kind == 1, np.minimum(hi_half, film), hi_half)
            g[sl(c, 1, -1)] = area[sl(c, 1, -1)] / (lo_half + hi_half +
                                                    contact_resistance_grid[c][sl(c, 1, -1)])
            G.append(g)
        return G

    face_fixed_value = []
    for c in range(3):
        values = np.full(active[c].shape, np.nan)
        for index, condition in enumerate(conditions):
            if condition["kind"] == "fixed":
                values[face_condition[c] == index] = condition["temperature_c"]
        face_fixed_value.append(values)
    emissivity = np.array([body["emissivity"] for body in bodies] + [0.0])[owner_index]
    radiating = np.zeros(shape)
    if settings["radiation"]:
        for c in range(3):
            area = np.broadcast_to(face_area[c], active[c].shape)
            insulated = np.zeros(active[c].shape, bool)
            for index, condition in enumerate(conditions):
                if condition["kind"] in ("insulated", "fixed"):
                    insulated |= face_condition[c] == index
            lo = solid[sl(c, 0, -1)] & outer_air[sl(c, 1, None)] & ~insulated[sl(c, 1, -1)]
            hi = outer_air[sl(c, 0, -1)] & solid[sl(c, 1, None)] & ~insulated[sl(c, 1, -1)]
            radiating[sl(c, 0, -1)] += np.where(lo, area[sl(c, 1, -1)], 0.0)
            radiating[sl(c, 1, None)] += np.where(hi, area[sl(c, 1, -1)], 0.0)
        radiating *= emissivity

    energy_parts = {}
    total_in = total_power + float(flux_power.sum())
    heat_scale = [max(total_in, 1e-3)]          # W; updated from the heat balance
    # time stepping (transient): Δt, velocities and temperatures of the last step
    unsteady = {"dt": None, "velocity": None, "T": None}
    capacity = (np.array([body["props"]["density"] * body["props"]["cp"] for body in bodies] +
                         [rho * cp])[owner_index] * np.broadcast_to(volume, shape))   # J/K per cell

    def energy(refresh):
        G = conductances()
        aP = np.zeros(shape)
        b = source.copy()
        nb = {}
        fixed_heat = []                 # (cells, G_half, T_fixed, condition index)
        for c in range(3):
            Sc = active[c].shape
            area = np.broadcast_to(face_area[c], Sc)
            flow = rho * cp * velocity[c] * area                     # W/K, + along +c
            g = G[c]
            insulated = np.zeros(Sc, bool)
            for index, condition in enumerate(conditions):
                if condition["kind"] == "insulated":
                    insulated |= face_condition[c] == index
            fixed_faces = np.isfinite(face_fixed_value[c])
            g = np.where(insulated | fixed_faces, 0.0, g)
            weight = _power_law(g, flow)
            east = np.zeros(shape); west = np.zeros(shape)
            east[...] = (weight + np.maximum(-flow, 0.0))[sl(c, 1, None)]
            west[...] = (weight + np.maximum(flow, 0.0))[sl(c, 0, -1)]
            aP += (weight + np.maximum(flow, 0.0))[sl(c, 1, None)] + \
                (weight + np.maximum(-flow, 0.0))[sl(c, 0, -1)]
            # domain sides: the face weights there are 0 (G = 0); add the side conditions
            east[sl(c, -1, None)] = 0.0
            west[sl(c, 0, 1)] = 0.0
            for side, face in ((0, 0), (1, shape[c])):
                kind = side_kind[c][side]
                cell = sl(c, 0, 1) if side == 0 else sl(c, -1, None)
                area_side = area[sl(c, face, face + 1)]
                f = flow[sl(c, face, face + 1)] * (1.0 if side == 0 else -1.0)   # + = into the domain
                g_side = area_side * k_eff[c][cell] / (D[c][0 if side == 0 else -1] / 2.0)
                if kind in ("open", "inlet"):
                    # outflow already carries the cell temperature out (a_P above); inflow
                    # brings ambient air, which also conducts over the half cell
                    inflow = np.maximum(f, 0.0)
                    conduction = np.where(f >= 0, g_side, 0.0)
                    aP[cell] += conduction
                    b[cell] += (inflow + conduction) * ambient
                elif kind == "wall_fixed":
                    aP[cell] += g_side
                    b[cell] += g_side * settings["wall_c"]
            # fixed-temperature faces: the face is held at its temperature; the solid cell and
            # the air cell on either side conduct to it over half a cell
            for index, condition in enumerate(conditions):
                if condition["kind"] != "fixed":
                    continue
                mask = face_condition[c] == index
                if not mask.any():
                    continue
                idx = np.nonzero(mask)
                for offset in (-1, 0):
                    cell = list(idx)
                    cell[c] = idx[c] + offset
                    ok = (cell[c] >= 0) & (cell[c] < shape[c])
                    cell = tuple(arr[ok] for arr in cell)
                    if not len(cell[0]):
                        continue
                    g_half = area[tuple(arr[ok] for arr in idx)] * k_eff[c][cell] / \
                        (D[c][cell[c]] / 2.0)
                    np.add.at(aP, cell, g_half)
                    np.add.at(b, cell, g_half * condition["temperature_c"])
                    fixed_heat.append((cell, g_half, condition["temperature_c"], index))
            nb[(c, 1)], nb[(c, -1)] = east, west
        # radiation to the ambient (linearised with the last temperatures)
        if settings["radiation"]:
            tk = T + 273.15
            h_rad = STEFAN_BOLTZMANN * (tk ** 2 + ambient_k ** 2) * (tk + ambient_k) * radiating
            aP += h_rad
            b += h_rad * ambient
            energy_parts["radiation"] = h_rad
        if unsteady["dt"] is not None:              # time step: ρc·V/Δt · (T − T_old)
            stored = capacity / unsteady["dt"]
            aP = aP + stored
            b = b + stored * unsteady["T"]
        aP_relaxed = aP / FLOW_RELAX_T
        b = b + (aP_relaxed - aP) * T
        aP_relaxed = np.maximum(aP_relaxed, 1e-30)
        # absolute tolerance in watts, tied to the heat that flows: a relative tolerance would
        # be dominated by the stiff metal cells, and their tiny temperature errors are large
        # heat errors
        matrix = _seven_point(sp, shape, aP_relaxed, nb)
        # inner tolerance: 1 % of the starting residual (the outer iterations change the
        # coefficients anyway), never looser than needed for the final heat balance
        start = T.ravel()
        initial = float(np.linalg.norm(b.ravel() - matrix @ start))
        tolerance = max(1e-6 * max(heat_scale[0], 1e-12) / math.sqrt(N), 0.01 * initial)
        solution, _info = spla.bicgstab(matrix, b.ravel(), x0=start, rtol=0.0,
                                        atol=tolerance, maxiter=1000,
                                        M=two_level(matrix, "energy", refresh))
        if not np.all(np.isfinite(solution)):
            raise ThermalError("The temperatures did not converge.")
        energy_parts["fixed"] = fixed_heat
        energy_parts["G"] = G
        return solution.reshape(shape)

    def heat_balance():
        """(in W, out by air W, radiation W, fixed faces / walls W, per-condition heat)."""
        out_air = 0.0
        walls = 0.0
        for c in range(3):
            area = np.broadcast_to(face_area[c], active[c].shape)
            for side, face in ((0, 0), (1, shape[c])):
                kind = side_kind[c][side]
                cell = sl(c, 0, 1) if side == 0 else sl(c, -1, None)
                f = (rho * cp * velocity[c] * area)[sl(c, face, face + 1)] * (1.0 if side == 0 else -1.0)
                g_side = area[sl(c, face, face + 1)] * k_eff[c][cell] / (D[c][0 if side == 0 else -1] / 2.0)
                t_cell = T[cell]
                if kind in ("open", "inlet"):
                    out_air += float((np.maximum(-f, 0.0) * (t_cell - ambient)).sum())
                    out_air += float((np.where(f >= 0, g_side, 0.0) * (t_cell - ambient)).sum())
                elif kind == "wall_fixed":
                    walls += float((g_side * (t_cell - settings["wall_c"])).sum())
        radiation = float((energy_parts.get("radiation", 0.0) * (T - ambient)).sum()) \
            if settings["radiation"] else 0.0
        per_condition = np.zeros(len(conditions))
        fixed_out = 0.0
        gross = abs(out_air) + abs(radiation) + abs(walls)
        for cell, g_half, value, index in energy_parts.get("fixed", []):
            heat = g_half * (T[cell] - value)            # + = leaves the part
            per_condition[index] += float(heat.sum())
            fixed_out += float(heat.sum())
            gross += float(np.abs(heat).sum())
        heat_scale[0] = max(power_now[0], gross, 1e-12)
        return out_air, radiation, walls, fixed_out, per_condition

    def stored_heat():
        """W going into heat storage during the time step (0 in steady state)."""
        if unsteady["dt"] is None:
            return 0.0
        return float((capacity * (T - unsteady["T"])).sum()) / unsteady["dt"]

    # result surfaces (and the grid of the result)
    step(21, "Preparing the result surfaces...")
    grid = {"kind": "cells", "xc": centres_mm[0], "yc": centres_mm[1], "zc": centres_mm[2],
            "faces": faces_mm, "shape": shape, "solid": solid.ravel(), "h_mm": h_mm,
            "owner": owner.ravel().copy()}             # body per cell, -1 = air
    # result surfaces of the bodies: trilinear between solid cells
    total_area = sum(body["area"] for body in bodies)
    edge = max(h_mm, math.sqrt(2.0 * total_area / DISPLAY_TRIANGLE_LIMIT))
    surface = {}
    for body in bodies:
        vertices, triangles = refine_triangles(body["V"], body["T"], edge)
        index, weights = _cell_weights(vertices, grid, grid["solid"])
        surface[body["job"]["key"]] = {"vertices": vertices.astype(np.float32),
                                       "triangles": triangles.astype(np.int32),
                                       "index": index, "weights": weights}
    used = np.unique(np.concatenate([item["index"].ravel() for item in surface.values()])) \
        if surface else np.zeros(0, np.int64)
    for item in surface.values():
        item["index"] = np.searchsorted(used, item["index"]).astype(np.int32)

    # 6. iterate ---------------------------------------------------------------------------
    max_iterations = int(settings["flow_iterations"])
    q_stored = 0.0
    speeds = [settings["inlet_speed"] if any(k == "inlet" for pair in side_kind for k in pair) else 0.0]
    speeds += [device.get("speed", 0.0) for device in devices]
    length = max(float(np.max(plan["size"])) / 1000.0 * 0.3, 0.01)
    speeds.append(math.sqrt(GRAVITY * beta * 10.0 * length))
    mass_reference = rho * max(speeds) * length ** 2
    velocity_scale = [max(speeds)]
    history = []
    converged = False
    balance_error = 1.0
    iteration = 0
    T_change = float("inf")
    mass_residual = float("inf")

    def update_k():
        """k(T) of the solid cells of bodies with a table."""
        flat_T = T.ravel()
        for index in k_tabled:
            cells = cells_of_body[index]
            ratio = _curve_factor(bodies[index]["k_curve"], flat_T[cells])
            for axis in range(3):
                k_cell[axis].reshape(-1)[cells] = bodies[index]["k"][axis] * ratio

    def update_capacity():
        flat_T = T.ravel()
        for index in cp_tabled:
            cells = cells_of_body[index]
            capacity.reshape(-1)[cells] = bodies[index]["props"]["density"] * volume_flat[cells] * \
                _curve_factor(bodies[index]["cp_curve"], flat_T[cells])

    def iterate(number, refresh):
        """One SIMPLEC iteration (momentum, pressure correction, energy); returns (mass
        residual, velocity change, temperature change)."""
        nonlocal T
        hottest_now = float(T[solid].max()) if solid.any() else float(T.max())
        velocity_scale[0] = max(math.sqrt(GRAVITY * beta * max(hottest_now - ambient, 0.5) * length),
                                max(speeds[:-1] + [0.0]), 0.01)
        if turbulent and number % 3 == 1:
            update_turbulence()
        previous_velocity = [v.copy() for v in velocity]
        d = [momentum(c) for c in range(3)]
        # fan curves: the pressure jump follows the flow through the fan (under-relaxed)
        for device in devices:
            if device["kind"] != "fan" or device["flow"]["flow_mode"] != "curve" or not device.get("area"):
                continue
            c = device["axis"]
            area_face = np.broadcast_to(face_area[c], active[c].shape)
            flow = float((velocity[c][device["faces"]] * area_face[device["faces"]]).sum()) * \
                device["sign"] * 3600.0
            flows, pressures = flow_curve(device["flow"])
            target = float(np.interp(flow, flows, pressures, left=pressures[0], right=0.0))
            device["dp"] = 0.8 * device["dp"] + 0.2 * target
            device["q"] = flow
        imbalance = pressure_correction(d, refresh)
        top_speed = max(float(np.abs(v).max()) for v in velocity)
        # residuals relative to the flow that is there (a weak flow converges to its own scale)
        reference_flow = rho * max(top_speed, 1e-3 * max(speeds)) * length ** 2
        mass = imbalance / max(min(mass_reference, reference_flow), 1e-300)
        change = max(float(np.abs(v - w).max()) for v, w in zip(velocity, previous_velocity)) / \
            max(top_speed, 1e-9)
        T_new = energy(refresh)
        dT = float(np.abs(T_new - T).max())
        T = T_new
        return mass, change, dT

    def balance():
        out_air, radiation, walls, fixed_out, _per = heat_balance()
        return (out_air + radiation + walls + fixed_out + stored_heat() - power_now[0]) / heat_scale[0]

    # thermostat sensors on the cell field: a probe (trilinear between solid cells when it sits
    # in a part, else between air cells), or the hottest / mean cell of the heated bodies
    def make_sensor(thermostat, owners):
        if thermostat["sensor"] == "probe":
            point = np.asarray(thermostat["point"], float)[None, :]
            ijk = [int(np.clip(np.searchsorted(centres_mm[a], point[0, a]), 0, shape[a] - 1))
                   for a in range(3)]
            in_solid = bool(solid[tuple(ijk)])
            small = {"xc": centres_mm[0], "yc": centres_mm[1], "zc": centres_mm[2], "shape": shape}
            idx, w = _cell_weights(point, small, solid.ravel() if in_solid else None)
            idx, w = idx[0], np.asarray(w[0], float)

            def probe_sensor(field):
                return float(field.ravel()[idx] @ w)
            return probe_sensor
        cells = np.concatenate([np.flatnonzero(owner_flat == index) for index in owners
                                if bodies[index]["cells"]] or [np.zeros(0, np.int64)])
        if not len(cells):
            raise ThermalError(f"A thermostat has no body to measure ({thermostat_text(thermostat)}).")
        if thermostat["sensor"] == "mean":
            weights = volume_flat[cells] / volume_flat[cells].sum()
            return lambda field: float(field.ravel()[cells] @ weights)
        return lambda field: float(field.ravel()[cells].max())

    thermostats = None
    controlled_nominal = np.zeros(len(controlled))
    if controlled:
        sources = []
        for name, thermostat, kind, index in controlled:
            owners = [index] if kind == "source" else list(conditions[index]["targets"])
            sensor = make_sensor(thermostat, owners)
            sources.append((name, thermostat, sensor, lambda _T, f=sensor: f))
        thermostats = _Thermostats(sources)
        for cells, weights, profile, nominal, kind, index, slot in parts:
            if slot is not None:
                controlled_nominal[slot] = profile_steady_power(profile, nominal)

    def control_thermostats():
        """Steady state: move every thermostat's share of the power towards the share that
        holds its sensor at the setpoint (sensitivity estimated from the heating so far).
        Returns True when all sensors sit at their setpoints (or their share is at a limit)."""
        settled = True
        for j, (_name, thermostat, sensor, _l) in enumerate(thermostats.sources):
            value = sensor(T)
            share = float(thermostats.duty[j])
            error = thermostat["setpoint_c"] - value
            slope = max((value - ambient) / max(share, 0.02), 0.5)
            new = min(1.0, max(0.0, share + 0.7 * error / slope))
            thermostats.duty[j] = new
            thermostats.sensor_values[j] = value
            at_limit = (new >= 1.0 and error > 0) or (new <= 0.0 and error < 0)
            settled &= at_limit or abs(error) < max(0.05, 2e-3 * abs(value - ambient))
        heat_sources(factors=thermostats.duty)
        return settled

    series = None
    if not transient:
        if thermostats is not None:
            thermostats.duty[:] = 1.0
            heat_sources(factors=thermostats.duty)
        settled = thermostats is None
        for iteration in range(1, max_iterations + 1):
            refresh = iteration % 10 == 1
            if k_tabled and iteration % 5 == 1:
                update_k()
            if thermostats is not None and iteration >= 10 and iteration % 10 == 0:
                settled = control_thermostats()
            mass_residual, velocity_change, T_change = iterate(iteration, refresh)
            span = max(float(T.max() - T.min()), 0.01)
            if iteration % 5 == 0 or iteration == 1:
                balance_error = balance()
            history.append((mass_residual, T_change, balance_error))
            step(22 + 74 * iteration / max_iterations,
                 f"Iteration {iteration}: mass {mass_residual:.2e}, ΔT {T_change:.2e} K, "
                 f"T max {float(T[solid].max()) if solid.any() else float(T.max()):.1f} °C, "
                 f"balance {balance_error:+.1%}")
            if iteration >= 40 and mass_residual < FLOW_MASS_TOLERANCE and \
                    velocity_change < FLOW_VELOCITY_TOLERANCE and settled and \
                    T_change < FLOW_DT_TOLERANCE * span and abs(balance_error) < FLOW_BALANCE_TOLERANCE:
                converged = True
                break
        if thermostats is not None:
            for j, (_name, _t, sensor, _l) in enumerate(thermostats.sources):
                thermostats.sensor_values[j] = sensor(T)
            total_power, flux_power = heat_sources(factors=thermostats.duty)
    else:
        # implicit time steps; every step is converged by up to `flow_step_iterations` inner
        # iterations (air and solids together, the air flow follows the temperatures)
        duration, n_steps = settings["duration_s"], settings["time_steps"]
        dt = duration / n_steps
        inner_limit = int(settings["flow_step_iterations"])
        start = ambient if settings["initial_ambient"] else settings["initial_c"]
        T[...] = float(start)
        if cp_tabled:
            update_capacity()
        frame_steps = np.unique(np.round(np.linspace(0, n_steps, min(n_steps, MAX_FRAMES) + 1))
                                .astype(int))
        frame_of_step = {int(s): f for f, s in enumerate(frame_steps)}
        frames = np.empty((len(frame_steps), len(used)), np.float32)
        # whole fields (sections, probes, streamlines) for as many frames as the budgets allow
        volume_index = _kept_frames(len(frame_steps), N, VOLUME_FRAME_BUDGET)
        velocity_index = _kept_frames(len(frame_steps), 3 * N, VELOCITY_FRAME_BUDGET)
        volume_slot = {int(f): k for k, f in enumerate(volume_index)}
        velocity_slot = {int(f): k for k, f in enumerate(velocity_index)}
        volume_frames = np.empty((len(volume_index), N), np.float32)
        velocity_frames = np.empty((len(velocity_index),) + tuple(shape) + (3,), np.float32)
        hot_points = np.empty((len(frame_steps), 3))
        hot_values = np.empty(len(frame_steps))
        stats = np.full((n_steps + 1, 3, len(bodies)), np.nan)
        power_series = np.zeros(n_steps + 1)
        body_cells = [np.flatnonzero(owner_flat == index) for index in range(len(bodies))]
        solid_cells = np.flatnonzero(solid.ravel())
        extremes = [np.inf, -np.inf]

        def record(step_index):
            flat = T.ravel()
            for index, cells in enumerate(body_cells):
                if len(cells):
                    values = flat[cells]
                    stats[step_index, :, index] = (
                        values.min(), float(values @ volume_flat[cells]) / float(volume_flat[cells].sum()),
                        values.max())
            pool = flat[solid_cells] if len(solid_cells) else flat
            extremes[0] = min(extremes[0], float(pool.min()))
            extremes[1] = max(extremes[1], float(pool.max()))
            frame = frame_of_step.get(step_index)
            if frame is None:
                return
            frames[frame] = flat[used]
            if frame in volume_slot:
                volume_frames[volume_slot[frame]] = flat
            if frame in velocity_slot:
                velocity_frames[velocity_slot[frame]] = np.stack(cell_velocity(), axis=-1)
            hottest = int(solid_cells[np.argmax(flat[solid_cells])]) if len(solid_cells) else \
                int(np.argmax(flat))
            i, j, k = np.unravel_index(hottest, shape)
            hot_points[frame] = (centres_mm[0][i], centres_mm[1][j], centres_mm[2][k])
            hot_values[frame] = flat[hottest]

        factors = None
        if thermostats is not None:
            thermostats.start(T)
            factors = thermostats.factors(True)
            switch_series = np.zeros((len(controlled), n_steps + 1))
            switch_series[:, 0] = factors
        total_power, flux_power = heat_sources(0.0, 0.0, factors)
        power_series[0] = power_now[0]
        record(0)
        unsettled = []
        for step_index in range(1, n_steps + 1):
            t0, t1 = (step_index - 1) * dt, step_index * dt
            if thermostats is not None:          # switch by the sensors at the step start
                thermostats.update(T)
                factors = thermostats.factors(True)
                switch_series[:, step_index] = factors
            total_power, flux_power = heat_sources(t0, t1, factors)
            power_series[step_index] = power_now[0]
            if k_tabled:
                update_k()
            if cp_tabled:
                update_capacity()
            unsteady["dt"] = dt
            unsteady["velocity"] = [v.copy() for v in velocity]
            unsteady["T"] = T.copy()
            done = False
            for inner in range(1, inner_limit + 1):
                iteration += 1
                mass_residual, velocity_change, T_change = iterate(iteration, inner == 1)
                span = max(float(T.max() - T.min()), 0.01)
                step(22 + 74 * (step_index - 1 + inner / inner_limit) / n_steps,
                     f"Time step {step_index} of {n_steps} (t = {format_time(t1)}), "
                     f"iteration {inner}: mass {mass_residual:.1e}, ΔT {T_change:.1e} K, "
                     f"T max {float(T[solid].max()) if solid.any() else float(T.max()):.1f} °C")
                if inner >= 2 and mass_residual < 10.0 * FLOW_MASS_TOLERANCE and \
                        velocity_change < 10.0 * FLOW_VELOCITY_TOLERANCE and \
                        T_change < 10.0 * FLOW_DT_TOLERANCE * span:
                    done = True
                    break
            if not done:
                unsettled.append(t1)
            balance_error = balance()
            history.append((mass_residual, T_change, balance_error))
            record(step_index)
        q_stored = stored_heat()
        converged = len(unsettled) <= 0.2 * n_steps       # a few warm-up steps are fine
        series = {"times": np.arange(n_steps + 1) * dt, "min": stats[:, 0, :].T,
                  "mean": stats[:, 1, :].T, "max": stats[:, 2, :].T, "power": power_series}
        if thermostats is not None:
            series["thermostats"] = switch_series
        if unsettled:
            when = format_time(unsettled[0]) if len(unsettled) == 1 else \
                f"{format_time(unsettled[0])} … {format_time(unsettled[-1])}"
            messages.append(f"{'⚠ ' if len(unsettled) > 0.2 * n_steps else ''}{len(unsettled)} of "
                            f"{n_steps} time steps (t = {when}) did not fully settle within "
                            f"{inner_limit} iterations — usually the fast warm-up at the start. "
                            "More iterations per step or shorter time steps make them exact.")
    if not converged and not turbulent and forced:
        messages.append("A laminar jet from a fan or an inlet is often unsteady: Flow → Turbulent "
                        "(or Automatic) usually settles.")
    if not converged and not transient:
        messages.append(f"⚠ The air flow did not fully settle in {iteration} iterations (mass "
                        f"residual {mass_residual:.2e}, last temperature change {T_change:.2e} K, "
                        f"heat balance {balance_error:+.1%}). More iterations or a coarser grid "
                        "help; natural convection at large temperature differences may also be "
                        "unsteady.")

    # 7. results -----------------------------------------------------------------------------
    step(97, "Evaluating results...")
    out_air, q_radiation, q_walls, q_fixed, per_condition = heat_balance()
    out = out_air + q_radiation + q_walls + q_fixed
    total_in = power_now[0]
    balance_error = (out + q_stored - total_in) / heat_scale[0]
    centre_velocity = cell_velocity()
    speed = np.sqrt(sum(v ** 2 for v in centre_velocity))
    speed[solid] = 0.0
    flat_T = T.ravel()
    body_results = []
    for index, body in enumerate(bodies):
        mask = owner == index
        entry = {"key": body["job"]["key"], "name": body["name"], "material": body["material"],
                 "power": power[index], "cells": body["cells"],
                 "max_c": body["props"].get("max_c"), "volume_error": None,
                 "min": None, "mean": None, "max": None, "convection": 0.0, "radiation": 0.0}
        if mask.any():
            values = T[mask]
            entry.update(min=float(values.min()),
                         mean=float((values * np.broadcast_to(volume, shape)[mask]).sum() /
                                    np.broadcast_to(volume, shape)[mask].sum()),
                         max=float(values.max()))
        body_results.append(entry)
    if solid.any():
        hottest = int(np.argmax(np.where(solid, T, -np.inf).ravel()))
    else:
        hottest = int(np.argmax(flat_T))
    hi, hj, hk = np.unravel_index(hottest, shape)
    hot_point = np.array([centres_mm[0][hi], centres_mm[1][hj], centres_mm[2][hk]])
    if not transient:
        frames = flat_T[used][None, :].astype(np.float32)
        frame_times, frame_steps = np.array([0.0]), np.array([0])
        hot_points, hot_values = hot_point[None, :], np.array([float(flat_T[hottest])])
        volume_frames, volume_index = flat_T[None, :].astype(np.float32), np.array([0])
        velocity_frames = np.stack(centre_velocity, axis=-1).astype(np.float32)[None]
        velocity_index = np.array([0])
        t_values = T[solid] if solid.any() else flat_T
        t_min, t_max = min(float(t_values.min()), ambient), float(t_values.max())
    else:
        frame_times = frame_steps * dt
        t_min, t_max = min(extremes[0], ambient), extremes[1]
    speed_frames = np.sqrt((velocity_frames.astype(np.float32) ** 2).sum(-1))
    condition_results = []
    for index, condition in enumerate(conditions):
        cell_faces = int(sum(int((fc == index).sum()) for fc in face_condition))
        area = sum(float(np.broadcast_to(face_area[c], fc.shape)[fc == index].sum())
                   for c, fc in enumerate(face_condition))
        heat = {"fixed": float(per_condition[index]), "flux": float(flux_power[index]),
                "insulated": 0.0}.get(condition["kind"], 0.0)
        condition_results.append({"name": condition["name"], "kind": condition["kind"],
                                  "cell_faces": cell_faces, "heat": heat, "area_mm2": area * 1e6})
    contact_results = []
    G = energy_parts.get("G")
    for index, contact in enumerate(contacts):
        a, b = contact["pair"]
        heat, area_total, faces = 0.0, 0.0, 0
        for c in range(3):
            mask = contact_id[c] == index
            if not mask.any():
                continue
            idx = np.nonzero(mask)
            low_cell = list(idx); low_cell[c] = idx[c] - 1
            low_cell, high_cell = tuple(low_cell), idx
            sign = np.where(owner[low_cell] == a, 1.0, -1.0)
            heat += float((sign * G[c][mask] * (T[low_cell] - T[high_cell])).sum())
            area_total += float(np.broadcast_to(face_area[c], mask.shape)[mask].sum())
            faces += int(mask.sum())
        contact_results.append({"name": contact["name"], "kind": contact["kind"],
                                "text": contact.get("text", ""), "heat": heat,
                                "area_mm2": area_total * 1e6, "faces": faces,
                                "jump": heat * contact["R"] / area_total if area_total else 0.0,
                                "a": bodies[a]["name"], "b": bodies[b]["name"]})
        if not faces:
            messages.append(f"⚠ Contact '{contact['name']}': the two bodies do not touch.")
    # air summary: flow through the open sides, fans, air temperatures
    open_flow = 0.0
    for c in range(3):
        area = np.broadcast_to(face_area[c], active[c].shape)
        for side, face in ((0, 0), (1, shape[c])):
            if side_kind[c][side] in ("open", "inlet"):
                f = (velocity[c] * area)[sl(c, face, face + 1)] * (1.0 if side == 0 else -1.0)
                open_flow += float(np.maximum(f, 0.0).sum())
    air_T = T[fluid]
    device_text = []
    for device in devices:
        if not device.get("area"):
            continue
        c = device["axis"]
        area_face = np.broadcast_to(face_area[c], active[c].shape)
        q = float((velocity[c][device["faces"]] * area_face[device["faces"]]).sum()) * device["sign"] * 3600.0
        if device["kind"] == "fan":
            device_text.append(f"{device['name']}: {q:.3g} m³/h" +
                               (f" at {device['dp']:.3g} Pa" if device["flow"]["flow_mode"] == "curve" else ""))
        else:
            device_text.append(f"{device['name']}: {abs(q):.3g} m³/h")
    air_text = (f"Air: up to {float(air_T.max()):.1f} °C, speeds up to {float(speed.max()):.2g} m/s, "
                f"{open_flow * 3600.0:.3g} m³/h through the open sides" +
                (" · " + " · ".join(device_text) if device_text else ""))
    elapsed = time.perf_counter() - started
    summary = [f"Grid: {shape[0]} × {shape[1]} × {shape[2]} = {N:,} cells "
               f"({int(fluid.sum()):,} air, {int(solid.sum()):,} solid), {h_mm:g} mm at the parts; "
               f"domain {plan['size'][0]:.0f} × {plan['size'][1]:.0f} × {plan['size'][2]:.0f} mm"
               .replace(",", " "),
               (f"Airflow + heat, transient: {format_time(settings['duration_s'])} in "
                f"{settings['time_steps']} steps of {format_time(dt)}, {flow_model} flow, "
                f"{iteration} iterations in {elapsed:.1f} s" if transient else
                f"Airflow + heat: {flow_model} flow, {iteration} iterations in "
                f"{elapsed:.1f} s") + ("" if converged else " (not fully converged)"),
               f"Heat balance: in sources {total_power:.4g} W" +
               (f" + flux {flux_power.sum():.4g} W" if flux_power.any() else "") +
               f"; out with the air {out_air:.4g} W, radiation {q_radiation:.4g} W" +
               (f", fixed-temperature walls {q_walls:.4g} W" if abs(q_walls) > 1e-9 else "") +
               (f", fixed-temperature faces {q_fixed:.4g} W" if abs(q_fixed) > 1e-9 else "") +
               (f", still stored {q_stored:.4g} W" if transient else "") +
               f" (difference {balance_error:+.2%})" + (" at the end" if transient else ""),
               air_text]
    for entry in condition_results:
        title = CONDITION_KINDS[entry["kind"]][0]
        effect = {"fixed": f"{entry['heat']:.4g} W leave" if entry["heat"] >= 0 else
                  f"{-entry['heat']:.4g} W enter", "flux": f"{entry['heat']:.4g} W enter",
                  "insulated": "no heat crosses"}[entry["kind"]]
        summary.append(f"Face condition '{entry['name']}' ({title}, {entry['area_mm2']:.4g} mm²): "
                       f"{effect}")
    for entry in contact_results:
        if entry["faces"]:
            summary.append(f"Contact '{entry['name']}' ({entry['text']}, {entry['area_mm2']:.4g} mm²): "
                           f"{abs(entry['heat']):.4g} W" +
                           (f", temperature step ≈ {abs(entry['jump']):.3g} K"
                            if entry["kind"] != "bonded" else ""))
    # Rayleigh / Reynolds estimate and model hint
    solid_max = float(T[solid].max()) if solid.any() else ambient
    rayleigh = GRAVITY * beta * max(solid_max - ambient, 0.1) * length ** 3 / \
        (AIR_KINEMATIC_VISCOSITY * k_air / (rho * cp))
    reynolds = max(speeds[:-1] + [0.0]) * length / AIR_KINEMATIC_VISCOSITY
    summary.append(f"Rayleigh number ≈ {rayleigh:.2g}" +
                   (f", Reynolds number ≈ {reynolds:.2g}" if reynolds > 0 else ""))
    if not turbulent and (rayleigh > 1e9 or reynolds > 2e4):
        messages.append("The flow is probably turbulent (Rayleigh > 10⁹ or Reynolds > 20 000): "
                        "choose Flow → Turbulent.")
    thermostat_results = []
    if thermostats is not None:
        lines, notes = thermostats.report(transient, controlled_nominal)
        summary += lines
        messages += notes
        thermostat_results = thermostats.results(transient, controlled_nominal)
    if k_tabled or cp_tabled:
        summary.append("Temperature-dependent material data: " + ", ".join(
            f"{bodies[i]['name']} ({bodies[i]['material']})"
            for i in sorted(set(k_tabled) | set(cp_tabled))))
    peaks = [float(np.nanmax(series["max"][i])) if series is not None and
             np.isfinite(series["max"][i]).any() else entry["max"]
             for i, entry in enumerate(body_results)]
    for entry, peak in zip(body_results, peaks):
        if peak is not None and entry["max_c"] is not None and peak > entry["max_c"]:
            messages.append(f"⚠ '{entry['name']}' reaches {peak:.1f} °C, above the "
                            f"{entry['max_c']:g} °C limit of {entry['material']}.")
    step(100, "Done")
    return {"mode": "airflow", "analysis": "transient" if transient else "steady",
            "cell_mm": h_mm, "elements": N, "nodes": N,
            "ambient": ambient, "t_min": t_min, "t_max": t_max,
            "hot_point": hot_points[-1], "hot_value": float(hot_values[-1]),
            "bodies": body_results, "surface": surface, "frames": frames,
            "frame_times": frame_times, "frame_steps": frame_steps,
            "hot_points": hot_points, "hot_values": hot_values,
            "series": series, "power": total_power, "q_flux": float(flux_power.sum()),
            "q_fixed": q_fixed, "q_convection": out_air, "q_radiation": q_radiation,
            "q_stored": q_stored, "balance_error": balance_error, "iterations": iteration,
            "conditions": condition_results, "contacts": contact_results,
            "thermostats": thermostat_results,
            "volume_frames": volume_frames, "volume_index": volume_index,
            "velocity_frames": velocity_frames, "velocity_index": velocity_index,
            "grid": grid, "velocity": np.stack(centre_velocity, axis=-1).astype(np.float32),
            "speed": speed.astype(np.float32), "fluid": fluid,
            "speed_max": float(max(speed.max(), speed_frames.max())),
            "air_text": air_text, "history": history, "converged": converged,
            "devices": [{"name": dv["name"], "kind": dv["kind"], "low": dv["low"], "high": dv["high"],
                         "axis": dv["axis"], "sign": dv["sign"]} for dv in devices],
            "sides": side_kind,
            "seconds": elapsed, "summary": summary, "messages": messages}


# ── parameter sweeps ──────────────────────────────────────────────────────────
# A sweep runs the study once per value of one parameter (the study itself stays unchanged)
# and collects the temperatures of every body and probe; each run's result can be shown.
SWEEP_PARAMETERS = {   # kind: (title, unit, target: "body" | "part" | "flux" | "contact" | "fan" | None)
    "power": ("Power of a body", "W", "body"),
    "material": ("Material of a body", "", "body"),
    "emissivity": ("Emissivity of a body", "", "body"),
    "setpoint": ("Thermostat setpoint of a body", "°C", "body"),
    "position": ("Position of a part", "mm", "part"),
    "flux": ("Heat flux of a face condition", "", "flux"),
    "contact": ("Contact resistance / layer thickness", "", "contact"),
    "ambient": ("Ambient temperature", "°C", None),
    "h": ("Convection coefficient h (fixed)", "W/m²K", None),
    "air_speed": ("Air speed (forced air)", "m/s", None),
    "fan_flow": ("Volume flow of a fan", "m³/h", "fan"),
    "inlet_speed": ("Inlet speed (airflow)", "m/s", None),
    "cell": ("Cell size (mesh study)", "mm", None),
}
SWEEP_MAX_RUNS = 50


def sweep_new(kind="power"):
    return {"parameter": kind, "target": None, "axis": 2, "values": [1.0, 2.0, 3.0],
            "start": 1.0, "stop": 3.0, "count": 3, "spacing": "range"}


def sweep_settings(data):
    """A validated sweep, or None."""
    if not isinstance(data, dict) or data.get("parameter") not in SWEEP_PARAMETERS:
        return None
    sweep = sweep_new(data["parameter"])
    target = data.get("target")
    if isinstance(target, (list, tuple)) and len(target) == 2:
        sweep["target"] = [str(target[0]), int(target[1])]
    elif isinstance(target, str):
        sweep["target"] = target
    try:
        sweep["axis"] = min(2, max(0, int(data.get("axis", 2))))
        sweep["start"] = float(data.get("start", 1.0))
        sweep["stop"] = float(data.get("stop", 3.0))
        sweep["count"] = min(SWEEP_MAX_RUNS, max(1, int(data.get("count", 3))))
    except (TypeError, ValueError):
        pass
    sweep["spacing"] = "list" if data.get("spacing") == "list" else "range"
    values = data.get("values") or []
    if sweep["parameter"] == "material":
        sweep["values"] = [str(v) for v in values if str(v).strip()][:SWEEP_MAX_RUNS]
    else:
        out = []
        for value in values:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                out.append(value)
        sweep["values"] = out[:SWEEP_MAX_RUNS]
    return sweep


def sweep_range(start, stop, count):
    if count <= 1:
        return [float(start)]
    return [float(v) for v in np.linspace(start, stop, int(count))]


def sweep_unit(study, sweep):
    kind = sweep["parameter"]
    if kind == "flux":
        condition = next((c for c in study.get("conditions", []) if c["name"] == sweep["target"]), None)
        return "W/m²" if condition is not None and condition["flux_mode"] == "density" else "W"
    if kind == "contact":
        contact = next((c for c in study.get("contacts", []) if c["name"] == sweep["target"]), None)
        return "mm" if contact is not None and contact["kind"] == "layer" else "K·cm²/W"
    return SWEEP_PARAMETERS[kind][1]


def sweep_title(study, sweep):
    """'Power of Chip (W)', 'Position Z of Heater (mm)'."""
    kind = sweep["parameter"]
    target = sweep.get("target")
    name = f"{target[0]}" + (f" / body {target[1] + 1}" if isinstance(target, list) and
                             len(next((c["bodies"] for c in study["components"]
                                       if c["name"] == target[0]), [])) > 1 else "") \
        if isinstance(target, list) else (target or "")
    titles = {"power": f"Power of {name}", "material": f"Material of {name}",
              "emissivity": f"Emissivity of {name}", "setpoint": f"Thermostat setpoint of {name}",
              "position": f"Position {'XYZ'[sweep['axis']]} of {name}",
              "flux": f"Heat flux '{name}'", "fan_flow": f"Volume flow of {name}",
              "contact": f"Contact '{name}' " + ("thickness" if sweep_unit(study, sweep) == "mm"
                                                  else "resistance")}
    title = titles.get(kind, SWEEP_PARAMETERS[kind][0])
    unit = sweep_unit(study, sweep)
    return title + (f" ({unit})" if unit else "")


def sweep_value_text(sweep, value):
    return str(value) if sweep["parameter"] == "material" else f"{value:g}"


def sweep_apply(study, sweep, value):
    """A copy of the study with the swept parameter set to `value`."""
    copy_ = study_from_state(study_state(study))
    copy_["section"] = None
    kind, target = sweep["parameter"], sweep.get("target")
    settings = copy_["settings"]

    def component(name, kinds=None):
        found = next((c for c in copy_["components"] if c["name"] == name), None)
        if found is None or (kinds and found.get("kind") not in kinds):
            raise ThermalError(f"The swept part '{name}' is not in the study (any more).")
        return found

    if kind in ("ambient", "h", "air_speed", "inlet_speed"):
        settings[{"ambient": "ambient_c", "h": "h", "air_speed": "air_speed",
                  "inlet_speed": "inlet_speed"}[kind]] = float(value)
        if kind == "h":
            settings["convection"] = "fixed"
        elif kind == "air_speed":
            settings["convection"] = "forced"
    elif kind == "cell":
        settings["flow_cell_mm" if settings["mode"] == "airflow" else "cell_mm"] = max(1e-3, float(value))
    elif kind in ("power", "material", "emissivity", "setpoint"):
        if not isinstance(target, list):
            raise ThermalError("Choose the body whose value the sweep changes.")
        part = component(target[0])
        if not 0 <= target[1] < len(part["bodies"]):
            raise ThermalError(f"The swept body of '{target[0]}' is not there (any more).")
        body = part["bodies"][target[1]]
        if kind == "power":
            body["power_w"] = max(0.0, float(value))
        elif kind == "material":
            if not material_known(value, copy_.get("materials", {})):
                raise ThermalError(f"Unknown material '{value}'.")
            body["material"] = str(value)
        elif kind == "emissivity":
            body["emissivity"] = min(1.0, max(0.0, float(value)))
        else:
            if body.get("thermostat") is None:
                raise ThermalError(f"'{target[0]}' has no thermostat whose setpoint could change.")
            body["thermostat"]["setpoint_c"] = float(value)
    elif kind == "position":
        part = component(target)
        part["position"][int(sweep["axis"])] = float(value)
    elif kind == "fan_flow":
        part = component(target, ("fan",))
        part["flow"] = flow_settings(part.get("flow"), "fan")
        part["flow"]["flow_mode"] = "fixed"
        part["flow"]["flow_m3h"] = max(0.0, float(value))
    elif kind == "flux":
        condition = next((c for c in copy_["conditions"] if c["name"] == target and
                          c["kind"] == "flux"), None)
        if condition is None:
            raise ThermalError(f"The heat-flux condition '{target}' is not in the study (any more).")
        condition["power_w" if condition["flux_mode"] == "total" else "flux_w_m2"] = max(0.0, float(value))
    elif kind == "contact":
        contact = next((c for c in copy_["contacts"] if c["name"] == target), None)
        if contact is None:
            raise ThermalError(f"The contact '{target}' is not in the study (any more).")
        contact["enabled"] = True
        if contact["kind"] == "layer":
            contact["thickness_mm"] = max(0.0, float(value))
        else:
            contact["kind"] = "resistance"
            contact["resistance"] = max(0.0, float(value))
    return copy_


def study_inputs(study):
    """(jobs, settings, conditions, contacts, flow items, notes) for a run of the study."""
    jobs = study_jobs(study)
    settings = copy.deepcopy(study["settings"])
    conditions = study_conditions(study, jobs)
    contacts = study_contacts(study, jobs)
    flow_items = study_flow_items(study)
    notes = []
    if flow_items and settings["mode"] != "airflow":
        notes.append("Fans and vents act in the Airflow + heat solver only; Conduction leaves "
                     "them out.")
        flow_items = []
    return jobs, settings, conditions, contacts, flow_items, notes


def solve_study(study, progress=None, cancelled=None):
    """Solve a study with the solver chosen in its settings."""
    jobs, settings, conditions, contacts, flow_items, notes = study_inputs(study)
    if settings["mode"] == "airflow":
        result = solve_airflow(jobs, settings, progress, cancelled, conditions, contacts, flow_items)
    else:
        result = solve_thermal(jobs, settings, progress, cancelled, conditions, contacts)
    result["messages"] = notes + result["messages"]
    return result


def _trim_frames(result, budget):
    """Keep fewer whole fields of a transient result (first, last and evenly between) so a
    sweep holds all its runs in memory."""
    for key, index_key in (("volume_frames", "volume_index"), ("velocity_frames", "velocity_index")):
        fields = result.get(key)
        if fields is None or len(fields) <= 2:
            continue
        per_frame = int(np.prod(fields.shape[1:]))
        keep = max(2, int(budget // max(per_frame, 1)))
        if keep >= len(fields):
            continue
        chosen = np.unique(np.round(np.linspace(0, len(fields) - 1, keep)).astype(int))
        result[key] = fields[chosen].copy()
        result[index_key] = np.asarray(result[index_key])[chosen]


def run_sweep(study, sweep, progress=None, cancelled=None):
    """Run the study for every value of the sweep: {"sweep", "title", "unit", "runs": [{"value",
    "text", "error", "result", "bodies": {name: {...}}, "probes": {name: {...}}}]}."""
    values = list(sweep["values"])
    if not values:
        raise ThermalError("The sweep has no values.")
    runs = []
    budget = VOLUME_FRAME_BUDGET / max(len(values), 1)
    for number, value in enumerate(values):
        if cancelled is not None and cancelled():
            raise ThermalCancelled()
        text = sweep_value_text(sweep, value)

        def run_progress(percent, message, number=number, text=text):
            if progress is not None:
                progress(int((number + percent / 100.0) * 100.0 / len(values)),
                         f"Run {number + 1} of {len(values)} ({text}): {message}")

        run = {"value": value, "text": text, "error": None, "result": None,
               "bodies": {}, "probes": {}}
        try:
            variant = sweep_apply(study, sweep, value)
            result = solve_study(variant, run_progress, cancelled)
        except ThermalError as ex:
            run["error"] = str(ex)
            runs.append(run)
            continue
        transient = result["analysis"] == "transient"
        for index, body in enumerate(result["bodies"]):
            peak = body["max"]
            if transient and result["series"] is not None:
                values_over_time = result["series"]["max"][index]
                if np.isfinite(values_over_time).any():
                    peak = float(np.nanmax(values_over_time))
            run["bodies"][body["name"]] = {"min": body["min"], "mean": body["mean"],
                                           "max": body["max"], "peak": peak,
                                           "max_c": body["max_c"]}
        for probe in variant.get("probes", []):
            point = probe_world_point(variant, probe)
            if point is None:
                continue
            if probe.get("component") is not None:
                index = next((i for i, c in enumerate(variant["components"])
                              if c["name"] == probe["component"]), None)
                if index is None or (index, probe["body"]) not in result["surface"]:
                    continue
            series = result_point_series(result, point, air=probe.get("component") is None)
            run["probes"][probe["name"]] = {"final": float(series[-1]),
                                            "peak": float(np.max(series))}
        _trim_frames(result, budget)
        run["result"] = result
        runs.append(run)
    return {"sweep": copy.deepcopy(sweep), "title": sweep_title(study, sweep),
            "unit": sweep_unit(study, sweep), "runs": runs}


def sweep_table(sweep_result, transient=None):
    """(column titles, rows of texts) of a sweep result: the value, then the maximum (peak in
    transient runs) of every body and every probe."""
    runs = sweep_result["runs"]
    bodies, probes = [], []
    for run in runs:
        for name in run["bodies"]:
            if name not in bodies:
                bodies.append(name)
        for name in run["probes"]:
            if name not in probes:
                probes.append(name)
    if transient is None:
        transient = any(run["result"] is not None and run["result"]["analysis"] == "transient"
                        for run in runs)
    key = "peak" if transient else "max"
    columns = [sweep_result["title"]] + [f"{name} max °C" for name in bodies] + \
        [f"◉ {name} °C" for name in probes]
    rows = []
    for run in runs:
        if run["error"]:
            rows.append([run["text"]] + ["—"] * (len(bodies) + len(probes)))
            continue
        row = [run["text"]]
        for name in bodies:
            value = run["bodies"].get(name, {}).get(key)
            row.append("—" if value is None else f"{value:.2f}")
        for name in probes:
            value = run["probes"].get(name, {}).get("peak" if transient else "final")
            row.append("—" if value is None else f"{value:.2f}")
        rows.append(row)
    return columns, rows, bodies, probes


def sweep_plot_data(sweep_result, max_curves=10):
    """(x values, curves [(name, values)], x labels or None, run order) for a chart of a sweep:
    the maximum (transient: the peak) of every body and the probes over the parameter."""
    runs = sweep_result["runs"]
    transient = any(r["result"] is not None and r["result"]["analysis"] == "transient" for r in runs)
    _columns, _rows, bodies, probes = sweep_table(sweep_result, transient)
    categorical = sweep_result["sweep"]["parameter"] == "material"
    good = {i for i, run in enumerate(runs) if not run["error"]}
    xs = np.arange(len(runs), dtype=float) if categorical else \
        np.array([float(run["value"]) for run in runs])
    order = sorted(range(len(runs)), key=lambda i: xs[i])
    key = "peak" if transient else "max"
    curves = []
    for name in probes[:max_curves // 2]:
        curves.append((f"◉ {name}", np.array(
            [runs[i]["probes"].get(name, {}).get("peak" if transient else "final", np.nan)
             if i in good else np.nan for i in order], float)))
    for name in bodies[:max_curves - len(curves)]:
        values = [runs[i]["bodies"].get(name, {}).get(key) if i in good else None for i in order]
        curves.append((name, np.array([np.nan if v is None else v for v in values], float)))
    labels = [runs[i]["text"] for i in order] if categorical or len(runs) <= 12 else None
    return xs[order], curves, labels, order


# ── export: CSV tables and the PDF report ─────────────────────────────────────
def write_csv(path, rows):
    """Plain CSV (comma, decimal point, UTF-8 with BOM so spreadsheets read °C correctly)."""
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        csv.writer(stream).writerows(rows)


def write_sweep_csv(sweep_result, path):
    columns, rows, _bodies, _probes = sweep_table(sweep_result)
    write_csv(path, [columns] + [["" if value == "—" else value for value in row] for row in rows])


def write_pdf(path, html, images, title="Thermal study"):
    """Render an HTML report (Qt rich text) with its images {name: QImage} to an A4 PDF."""
    from PyQt5.QtGui import QTextDocument, QPdfWriter, QPageSize, QPageLayout
    from PyQt5.QtCore import QUrl, QMarginsF
    document = QTextDocument()
    document.setDefaultFont(QFont("Arial", 9))
    for name, image in images.items():
        document.addResource(QTextDocument.ImageResource, QUrl(name), image)
    document.setHtml(html)
    writer = QPdfWriter(path)
    writer.setTitle(title)
    writer.setCreator(f"SimpleCAD {ADDON_TITLE} {ADDON_VERSION}")
    writer.setPageLayout(QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Portrait,
                                     QMarginsF(15, 15, 15, 15), QPageLayout.Millimeter))
    document.print_(writer)


# VTU (VTK XML unstructured grid) for ParaView: the cells of a result as hexahedra, binary
# ("appended raw") data. Conduction: temperature at the nodes; airflow: temperature, velocity
# and speed per cell. Coordinates in mm; "Body" = index into the body list (-1 = air).
_VTK_HEX_CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))     # VTK_HEXAHEDRON node order
_VTK_TYPES = {np.dtype(np.float32): "Float32", np.dtype(np.float64): "Float64",
              np.dtype(np.int32): "Int32", np.dtype(np.int64): "Int64", np.dtype(np.uint8): "UInt8"}


def result_volume_field(result, frame):
    """The whole stored field (nodes or cells) at a frame; between two kept fields of a long
    transient run linearly in time."""
    kept, fields = np.asarray(result["volume_index"]), result["volume_frames"]
    position = int(np.searchsorted(kept, frame))
    if position < len(kept) and kept[position] == frame:
        return np.asarray(fields[position], np.float32)
    position = min(max(position, 1), len(kept) - 1)
    a, b = int(kept[position - 1]), int(kept[position])
    times = result["frame_times"]
    w = 0.0 if times[b] == times[a] else (times[frame] - times[a]) / (times[b] - times[a])
    return ((1.0 - w) * fields[position - 1] + w * fields[position]).astype(np.float32)


def vtu_geometry(result):
    """(points (n, 3) mm, connectivity (m, 8), body per cell (m,), "point" | "cell": where the
    temperatures sit) of a result's cells."""
    grid = result["grid"]
    shape = grid["shape"]
    n_bodies = len(result["bodies"])
    if grid["kind"] == "nodes":
        if "cells" not in grid:
            raise ValueError("this result has no element list (run the simulation again)")
        ny1, nz1 = shape[1] + 1, shape[2] + 1
        nodes = grid["nodes_global"]
        node_index = np.stack([nodes // (ny1 * nz1), (nodes // nz1) % ny1, nodes % nz1], axis=1)
        points = np.asarray(grid["origin"], float) + node_index * grid["h_mm"]
        cells = grid["cells"]
        I, J, K = cells // (shape[1] * shape[2]), (cells // shape[2]) % shape[1], cells % shape[2]
        base = (I * ny1 + J) * nz1 + K
        connectivity = np.empty((len(cells), 8), np.int64)
        for corner, (dx, dy, dz) in enumerate(_VTK_HEX_CORNERS):
            connectivity[:, corner] = np.searchsorted(nodes, base + (dx * ny1 + dy) * nz1 + dz)
        body = grid["cell_body"].astype(np.int32)
        body[body >= n_bodies] = -1                  # air enclosed in a housing
        return points, connectivity, body, "point"
    faces = grid["faces"]
    n0, n1, n2 = shape
    X, Y, Z = np.meshgrid(faces[0], faces[1], faces[2], indexing="ij")
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    I, J, K = (a.ravel() for a in np.meshgrid(np.arange(n0), np.arange(n1), np.arange(n2),
                                              indexing="ij"))
    connectivity = np.empty((n0 * n1 * n2, 8), np.int64)
    for corner, (dx, dy, dz) in enumerate(_VTK_HEX_CORNERS):
        connectivity[:, corner] = ((I + dx) * (n1 + 1) + (J + dy)) * (n2 + 1) + (K + dz)
    owner = grid.get("owner")
    if owner is None:
        owner = np.where(grid["solid"], 0, -1)
    body = np.asarray(owner, np.int32).copy()
    body[body >= n_bodies] = -1
    return points, connectivity, body, "cell"


def write_vtu(path, points, connectivity, point_data=(), cell_data=(), comment=""):
    """One VTK XML unstructured grid of hexahedra. point_data / cell_data: [(name, array (n,)
    or (n, 3))]. Binary appended data, little endian, 64-bit block headers."""
    n_points, n_cells = len(points), len(connectivity)
    index_type = np.int32 if n_points < 2 ** 31 - 1 and n_cells * 8 < 2 ** 31 - 1 else np.int64
    blocks, offset = [], 0

    def array_xml(name, values, components=1):
        nonlocal offset
        values = np.ascontiguousarray(values)
        if values.dtype not in _VTK_TYPES:
            values = values.astype(np.float32 if values.dtype.kind == "f" else np.int32)
        data = values.astype(values.dtype.newbyteorder("<"), copy=False).tobytes()
        blocks.append(np.uint64(len(data)).tobytes() + data)
        text = (f'<DataArray type="{_VTK_TYPES[values.dtype]}" Name="{name}" '
                + (f'NumberOfComponents="{components}" ' if components > 1 else "")
                + f'format="appended" offset="{offset}"/>')
        offset += 8 + len(data)
        return text

    def section(tag, arrays):
        if not arrays:
            return f"      <{tag}/>\n"
        lines = [f"      <{tag}>"]
        for name, values in arrays:
            values = np.asarray(values)
            components = values.shape[1] if values.ndim == 2 else 1
            lines.append("        " + array_xml(name, values.reshape(len(values), -1).ravel()
                                                if components > 1 else values, components))
        lines.append(f"      </{tag}>")
        return "\n".join(lines) + "\n"

    point_section = section("PointData", list(point_data))
    cell_section = section("CellData", list(cell_data))
    points_xml = array_xml("Points", np.asarray(points, np.float64).ravel(), 3)
    conn_xml = array_xml("connectivity", np.asarray(connectivity, index_type).ravel())
    offsets_xml = array_xml("offsets", np.arange(8, 8 * n_cells + 1, 8, dtype=index_type))
    types_xml = array_xml("types", np.full(n_cells, 12, np.uint8))       # VTK_HEXAHEDRON
    comment = comment.replace("--", "- -")
    header = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt64">\n'
        + (f"  <!--\n{comment}\n  -->\n" if comment else "") +
        "  <UnstructuredGrid>\n"
        f'    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">\n'
        + point_section + cell_section +
        f"      <Points>\n        {points_xml}\n      </Points>\n"
        f"      <Cells>\n        {conn_xml}\n        {offsets_xml}\n        {types_xml}\n"
        "      </Cells>\n"
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        '  <AppendedData encoding="raw">\n   _')
    with open(path, "wb") as stream:
        stream.write(header.encode("utf-8"))
        for block in blocks:
            stream.write(block)
        stream.write(b"\n  </AppendedData>\n</VTKFile>\n")


def export_vtu(result, path, frames=None, frame=None, bodies_note=True, progress=None):
    """Write a result for ParaView. frames=None: one frame (`frame`, default the last) into
    `path` (.vtu); a list of frames: one .vtu per frame (path stem + _0000.vtu …) and a .pvd
    time series at `path` (.pvd). Returns the list of written files."""
    points, connectivity, body, where = vtu_geometry(result)
    names ="\n".join(f"  Body {i}: {b['name']} ({b['material'] or 'no material'})"
                      for i, b in enumerate(result["bodies"]))
    comment = ("  SimpleCAD Thermal " + ADDON_VERSION + " result, coordinates in mm.\n"
               "  Body = index below, -1 = air.\n" + (names if bodies_note else ""))

    def write_frame(file_path, frame):
        temperature = result_volume_field(result, frame)
        cell_data = [("Body", body)]
        point_data = []
        if where == "point":
            point_data.append(("Temperature [degC]", temperature))
        else:
            velocity, speed = result_velocity(result, frame)
            cell_data += [("Temperature [degC]", temperature),
                          ("Velocity [m/s]", np.asarray(velocity, np.float32).reshape(-1, 3)),
                          ("Speed [m/s]", np.asarray(speed, np.float32).ravel())]
        time_text = f"  Time {float(result['frame_times'][frame]):g} s\n" \
            if result["analysis"] == "transient" else ""
        write_vtu(file_path, points, connectivity, point_data, cell_data, time_text + comment)

    last = len(result["frame_times"]) - 1
    if frames is None:
        write_frame(path, last if frame is None else int(frame))
        return [path]
    stem = os.path.splitext(path)[0]
    folder, base = os.path.dirname(stem), os.path.basename(stem)
    written, entries = [], []
    for number, frame in enumerate(frames):
        if progress is not None:
            progress(number, len(frames))
        name = f"{base}_{number:04d}.vtu"
        write_frame(os.path.join(folder, name), int(frame))
        written.append(os.path.join(folder, name))
        entries.append(f'    <DataSet timestep="{float(result["frame_times"][int(frame)]):.9g}" '
                       f'group="" part="0" file="{name}"/>')
    pvd = stem + ".pvd"
    with open(pvd, "w", encoding="utf-8") as stream:
        stream.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" '
                     'byte_order="LittleEndian">\n  <Collection>\n' + "\n".join(entries) +
                     "\n  </Collection>\n</VTKFile>\n")
    return [pvd] + written


def vtu_size_estimate(result):
    """Approximate bytes of one exported frame."""
    grid = result["grid"]
    if grid["kind"] == "nodes":
        points = len(grid["nodes_global"])
        cells = len(grid.get("cells", ()))
        return points * (24 + 4) + cells * (32 + 4 + 1 + 4)
    n = int(np.prod(grid["shape"]))
    return n * 24 + n * (32 + 4 + 1 + 4 + 4 + 12 + 4)


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

    def set_range(self, low, high, note="", title="T  [°C]"):
        self._range = (float(low), float(high))
        self._note = note
        self._title = title
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
                         getattr(self, "_title", "T  [°C]"))
        font.setBold(False)
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        steps = 6
        for i in range(steps + 1):
            value = low + (high - low) * i / steps
            y = bottom - (bottom - top) * i / steps
            painter.drawLine(QPoint(28, int(y)), QPoint(33, int(y)))
            text = f"{value:.1f}" if high - low >= 1.0 else f"{value:.3g}"
            painter.drawText(QRectF(36.0, y - metrics.height() / 2, self.width() - 38.0,
                                    metrics.height()), Qt.AlignLeft | Qt.AlignVCenter, text)
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
        self._x_format = None             # text of an x value (default: a time)
        self._x_labels = None             # text at every x value (categories)
        self._markers = False

    def set_data(self, times, curves, limits=(), x_format=None, x_labels=None, markers=False):
        """curves: [(name, values)] over `times`, or [(name, values, own times)]. A plot over
        something else than time gives `x_format` (value → text) or `x_labels` (one per x)."""
        self._x_format, self._x_labels, self._markers = x_format, x_labels, markers
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
        if self._x_labels is not None:                       # categories / sweep values
            ticks = list(zip(self._times.tolist(), self._x_labels))
            room = max(1, int(len(ticks) * 60 / max(frame.width(), 1.0)))
            ticks = ticks[::room]
        else:
            ticks = [(t0 + (t1 - t0) * fraction, None) for fraction in (0.0, 0.5, 1.0)]
        for t, label in ticks:                               # time (x) axis
            if label is not None:
                text = str(label)
            elif self._x_format is not None:
                text = self._x_format(t)
            else:
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
            if self._markers:
                painter.setBrush(QColor(colour))
                for point in points:
                    painter.drawEllipse(point, 3.0, 3.0)
                painter.setBrush(Qt.NoBrush)
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
        self.pick_mode = None           # None, "faces", "extrude" (a face for an extrusion) or "probe"
        self.face_overlays = []         # [(entry, face index, rgb, strong)] of face conditions
        self.probe_markers = []         # [(world point, label, selected)]
        self._face_hover = None         # (entry, face index) under the cursor while picking
        self.section = None             # {"position", "rotation", "flip", ...} while a section is on
        self.section_caps = []          # [(entry or None, vertices, triangles, rgb or per-vertex rgb)]
        self.section_arrows = None      # (segments (n, 2, 3), colours (n, 3)) of air velocities
        self.streamlines = []           # [(points (n, 3), colours (n, 3))]
        self.flow_markers = []          # [(entry, axis vector, label)] of fans / vents

    def set_pick_mode(self, mode):
        self.pick_mode = mode
        self._face_hover = None
        self.setCursor(Qt.PointingHandCursor if mode else self._default_cursor())
        self.update()

    # ── section view: the part on the hidden side of the plane is clipped away ──
    def _section_plane(self):
        """(origin, normal towards the hidden side) of the active section, or None."""
        section = self.section
        if not section or not section.get("active"):
            return None
        origin, _u, _v, normal = cad.cut_surface_frame(section["position"], section["rotation"])
        return origin, (-normal if section.get("flip") else normal)

    def _clip_on(self):
        plane = self._section_plane()
        if plane is None:
            return False
        origin, hidden = plane
        gl.glClipPlane(gl.GL_CLIP_PLANE0, [float(-hidden[0]), float(-hidden[1]), float(-hidden[2]),
                                           float(hidden @ origin)])
        gl.glEnable(gl.GL_CLIP_PLANE0)
        return True

    def _pick_face(self, origin, direction):
        plane = self._section_plane()
        if plane is None:
            return super()._pick_face(origin, direction)
        point, hidden = plane
        side = float((np.asarray(origin, float) - point) @ hidden)     # > 0: eye on the hidden side
        along = float(np.asarray(direction, float) @ hidden)
        if side > 0:
            if along >= 0:
                return None
            start = -side / along
            hit = super()._pick_face(origin + start * direction, direction)
            return None if hit is None else (hit[0] + start,) + tuple(hit[1:])
        hit = super()._pick_face(origin, direction)
        if hit is not None and along > 0 and hit[0] > -side / along:
            return None                               # behind the plane: hidden
        return hit

    def _cap_hit(self, origin, direction):
        """(t, entry, point) where the ray meets a section cap of a body, or None."""
        plane = self._section_plane()
        if plane is None or not self.section_caps:
            return None
        point, hidden = plane
        along = float(direction @ hidden)
        if abs(along) < 1e-12:
            return None
        t = float((point - origin) @ hidden) / along
        if t <= 0:
            return None
        hit = origin + t * direction
        for entry, vertices, triangles, _colors in self.section_caps:
            if entry is None:
                continue
            P = np.asarray(vertices, float)[np.asarray(triangles, np.int64)]
            a, b, c = P[:, 0], P[:, 1], P[:, 2]
            n = np.cross(b - a, c - a)
            inside = ((np.cross(b - a, hit - a) * n).sum(1) >= 0) & \
                ((np.cross(c - b, hit - b) * n).sum(1) >= 0) & ((np.cross(a - c, hit - c) * n).sum(1) >= 0)
            if inside.any():
                return t, entry, hit
        return None

    def _surface_hit(self, pos, caps=False):
        """(entry, face index (-1 on a section cap), world point) under the cursor, or None."""
        if self._mvp is None:
            return None
        origin, direction = self._ray(pos)
        hit = self._pick_face(origin, direction)
        cap = self._cap_hit(origin, direction) if caps else None
        if cap is not None and (hit is None or cap[0] < hit[0]):
            return cap[1], -1, cap[2]
        if hit is None:
            return None
        t, entry, face = hit
        return entry, face, origin + t * direction

    def _click_select3d(self, pos, ctrl=False):
        if self.pick_mode in ("faces", "extrude"):
            hit = self._surface_hit(pos)
            if hit is not None and hit[1] >= 0:
                self.facePicked.emit(hit[0], int(hit[1]))
            return
        if self.pick_mode == "probe":
            hit = self._surface_hit(pos, caps=True)
            if hit is not None:
                self.pointPicked.emit(hit[0], np.asarray(hit[2], float))
            return
        super()._click_select3d(pos, ctrl)

    def mouseMoveEvent(self, e):
        if self.pick_mode in ("faces", "extrude") and not e.buttons():
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
            self._clip_on()
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
        clipped = self._clip_on()
        try:
            if self.result_mode:
                self._draw_result_solids()
            else:
                super()._draw_solids_inner()
        finally:
            if clipped:
                gl.glDisable(gl.GL_CLIP_PLANE0)
        if clipped:
            self._draw_section_caps()
        self._draw_flow_extras()

    def _draw_section_caps(self):
        """The cut faces of the bodies (temperature colours, or the body colour), plus the air
        velocity arrows of an airflow result."""
        if not self.section_caps and self.section_arrows is None:
            return
        gl.glPushAttrib(gl.GL_ENABLE_BIT | gl.GL_CURRENT_BIT | gl.GL_LINE_BIT |
                        gl.GL_COLOR_BUFFER_BIT | gl.GL_POLYGON_BIT | gl.GL_DEPTH_BUFFER_BIT)
        try:
            gl.glDisable(gl.GL_LIGHTING)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
            gl.glPolygonOffset(1.0, 1.0)          # the arrows stay in front of the cut face
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            for _entry, vertices, triangles, colors in self.section_caps:
                vertices = np.ascontiguousarray(vertices, np.float32)
                triangles = np.ascontiguousarray(triangles, np.uint32)
                if not len(triangles):
                    continue
                gl.glVertexPointer(3, gl.GL_FLOAT, 0, vertices)
                colors = np.asarray(colors, np.float32)
                if colors.ndim == 2:
                    gl.glEnableClientState(gl.GL_COLOR_ARRAY)
                    gl.glColorPointer(3, gl.GL_FLOAT, 0, np.ascontiguousarray(colors))
                else:
                    gl.glColor3f(*[float(c) for c in colors[:3]])
                gl.glDrawElements(gl.GL_TRIANGLES, triangles.size, gl.GL_UNSIGNED_INT, triangles)
                gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
            if self.section_arrows is not None and len(self.section_arrows[0]):
                points, colors = self.section_arrows          # line pairs (2n, 3), per point rgb
                gl.glLineWidth(1.4)
                gl.glEnableClientState(gl.GL_COLOR_ARRAY)
                gl.glVertexPointer(3, gl.GL_FLOAT, 0, np.ascontiguousarray(points, np.float32))
                gl.glColorPointer(3, gl.GL_FLOAT, 0, np.ascontiguousarray(colors, np.float32))
                gl.glDrawArrays(gl.GL_LINES, 0, len(points))
                gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        finally:
            gl.glPopAttrib()

    def _draw_flow_extras(self):
        """Streamlines of an airflow result and the flow arrows of fans / vents."""
        if not self.streamlines and not self.flow_markers:
            return
        gl.glPushAttrib(gl.GL_ENABLE_BIT | gl.GL_CURRENT_BIT | gl.GL_LINE_BIT)
        try:
            gl.glDisable(gl.GL_LIGHTING)
            clipped = self._clip_on()
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glLineWidth(1.6)
            for points, colors in self.streamlines:
                if len(points) < 2:
                    continue
                gl.glVertexPointer(3, gl.GL_FLOAT, 0, np.ascontiguousarray(points, np.float32))
                gl.glColorPointer(3, gl.GL_FLOAT, 0, np.ascontiguousarray(colors, np.float32))
                gl.glDrawArrays(gl.GL_LINE_STRIP, 0, len(points))
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            if clipped:
                gl.glDisable(gl.GL_CLIP_PLANE0)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glLineWidth(2.5)
            for entry, axis, _label in self.flow_markers:
                if not cad.entry_visible(entry) or entry.get("_centre") is None:
                    continue
                centre = np.asarray(entry["_centre"], float)
                points = np.asarray(entry["solid"][0], float)
                length = max(float(np.ptp(points, axis=0).max()) * 0.6, 3.0)
                tip = centre + axis * length
                side = np.cross(axis, np.eye(3)[int(np.argmin(np.abs(axis)))])
                side = side / max(float(np.linalg.norm(side)), 1e-12) * length * 0.18
                gl.glColor3f(0.35, 0.9, 1.0)
                gl.glBegin(gl.GL_LINES)
                for a, b in ((centre, tip), (tip, tip - axis * length * 0.3 + side),
                             (tip, tip - axis * length * 0.3 - side)):
                    gl.glVertex3f(*[float(x) for x in a]); gl.glVertex3f(*[float(x) for x in b])
                gl.glEnd()
        finally:
            gl.glPopAttrib()

    def _draw_result_solids(self):
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

    def __init__(self, jobs, settings, conditions=(), contacts=(), flow_items=(), parent=None):
        super().__init__(parent)
        self._jobs, self._settings, self._conditions = jobs, settings, list(conditions)
        self._contacts, self._flow_items = list(contacts), list(flow_items)
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
            if self._settings.get("mode") == "airflow":
                result = solve_airflow(self._jobs, self._settings, self._progress,
                                       lambda: self._cancel, self._conditions, self._contacts,
                                       self._flow_items)
            else:
                result = solve_thermal(self._jobs, self._settings, self._progress,
                                       lambda: self._cancel, self._conditions, self._contacts)
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


def _fill_material_combo(combo, current=None):
    """(Re)fill a material combo: built-in materials, then the user's (study and library)."""
    blocked = combo.blockSignals(True)
    try:
        combo.clear()
        combo.addItem("— select material —", None)
        for name in material_names():
            combo.addItem(name + ("  (user)" if name not in MATERIAL_THERMAL else ""), name)
            combo.setItemData(combo.count() - 1, material_summary(name) + "\n" + material_note(name),
                              Qt.ToolTipRole)
        if current and combo.findData(current) < 0:
            combo.addItem(f"{current}  (unknown)", current)
        combo.setCurrentIndex(max(0, combo.findData(current)) if current else 0)
    finally:
        combo.blockSignals(blocked)


def _material_combo(current=None):
    combo = QComboBox()
    _fill_material_combo(combo, current)
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


class ProfileEditor(QWidget):
    """Time profile of a power (constant, on / off cycle, ramp, CSV table) for transient runs."""
    changed = pyqtSignal()

    def __init__(self, label_width, value_width, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnMinimumWidth(0, label_width)
        grid.setColumnMinimumWidth(1, value_width)
        self._loading = False
        self._points = []
        self._power = 0.0

        def caption(text):
            label = QLabel(text)
            label.setFixedWidth(label_width)
            return label

        self.cb_kind = QComboBox()
        self.cb_kind.setFixedWidth(value_width)
        for kind, title in PROFILE_KINDS.items():
            self.cb_kind.addItem(title, kind)
        self.cb_kind.setToolTip("How the power changes over time in a transient run")
        self.sp_period = _spin(0.001, 1e7, " s", 1.0, decimals=3)
        self.sp_period.setToolTip("Length of one on / off cycle")
        self.sp_duty = _spin(0.0, 100.0, " % on", 5.0, decimals=1)
        self.sp_duty.setToolTip("Share of each cycle with the power on (at the start of the cycle)")
        self.sp_ramp = _spin(0.001, 1e7, " s", 1.0, decimals=3)
        self.sp_ramp.setToolTip("Time to rise linearly from 0 to the full power")
        self.btn_table = QPushButton("Import CSV…")
        self.btn_table.setFixedWidth(value_width)
        self.btn_table.setToolTip("Text / CSV file with two columns: time in s, power in W "
                                  "(separated by comma, semicolon or tab; a header line is fine)")
        self.btn_table.clicked.connect(self._import_table)
        self.lbl_info = _hint("")
        self.lbl_info.setFixedWidth(label_width + grid.horizontalSpacing() + value_width)
        rows = (("Profile:", self.cb_kind), ("Period:", self.sp_period), ("Duty:", self.sp_duty),
                ("Ramp time:", self.sp_ramp), ("Table:", self.btn_table))
        self._captions = {}
        for row, (text, widget) in enumerate(rows):
            label = caption(text)
            self._captions[widget] = label
            grid.addWidget(label, row, 0)
            grid.addWidget(widget, row, 1, Qt.AlignLeft)
        grid.addWidget(self.lbl_info, len(rows), 0, 1, 2, Qt.AlignLeft)
        self.cb_kind.currentIndexChanged.connect(self._edited)
        for spin in (self.sp_period, self.sp_duty, self.sp_ramp):
            spin.valueChanged.connect(self._edited)
        self.set_profile(None, 0.0)

    def set_profile(self, profile, power):
        self._loading = True
        try:
            self._power = float(power)
            data = profile or profile_new("constant")
            self.cb_kind.setCurrentIndex(max(0, self.cb_kind.findData(data["kind"])))
            self.sp_period.setValue(data["period_s"])
            self.sp_duty.setValue(data["duty_pct"])
            self.sp_ramp.setValue(data["ramp_s"])
            self._points = [list(p) for p in data.get("points", [])]
            self._refresh()
        finally:
            self._loading = False

    def profile(self):
        kind = self.cb_kind.currentData()
        if kind == "constant":
            return None
        return profile_settings({"kind": kind, "period_s": self.sp_period.value(),
                                 "duty_pct": self.sp_duty.value(), "ramp_s": self.sp_ramp.value(),
                                 "points": self._points})

    def is_table(self):
        return self.cb_kind.currentData() == "table"

    def _refresh(self):
        kind = self.cb_kind.currentData()
        shown = {"duty": (self.sp_period, self.sp_duty), "ramp": (self.sp_ramp,),
                 "table": (self.btn_table,)}.get(kind, ())
        for widget in (self.sp_period, self.sp_duty, self.sp_ramp, self.btn_table):
            widget.setVisible(widget in shown)
            self._captions[widget].setVisible(widget in shown)
        profile = self.profile()
        text = "Constant power from the start." if profile is None else \
            (lambda t: t[:1].upper() + t[1:])(profile_text(profile, self._power)) + "."
        self.lbl_info.setText(text + " Profiles act in transient runs; steady state uses the mean "
                              "of a cycle, the full power of a ramp, the last table value.")

    def _edited(self, *_args):
        if self._loading:
            return
        self._refresh()
        self.changed.emit()

    def _import_table(self):
        path, _ = QFileDialog.getOpenFileName(self, "Power over time (CSV)", "",
                                              "CSV / text files (*.csv *.txt *.dat);;All files (*)")
        if not path:
            return
        try:
            self._points = read_profile_csv(path)
        except (OSError, ValueError) as ex:
            QMessageBox.warning(self, "Power table", f"Could not read\n{path}\n\n{ex}")
            return
        self._refresh()
        self.changed.emit()


class FlowItemDialog(QDialog):
    """Place a fan or a vent: size, flow direction, position (next to the selected part)."""
    DIRECTIONS = (("+Z (up)", (0.0, 0.0, 1.0)), ("−Z (down)", (0.0, 0.0, -1.0)),
                  ("+X", (1.0, 0.0, 0.0)), ("−X", (-1.0, 0.0, 0.0)),
                  ("+Y", (0.0, 1.0, 0.0)), ("−Y", (0.0, -1.0, 0.0)))

    def __init__(self, parent, kind, anchor, anchor_name):
        super().__init__(parent)
        self._kind = kind
        self._anchor = anchor                       # (low, high) of the selected part or None
        self.setWindowTitle("Fan" if kind == "fan" else "Vent / opening")
        self.setWindowIcon(_icon("air_flow_on.png"))
        form = QFormLayout(self)
        number = 1
        self.ed_name = QLineEdit(f"{'Fan' if kind == 'fan' else 'Vent'} {number}")
        form.addRow("Name:", self.ed_name)
        self.cb_form = QComboBox()
        self.cb_form.addItem("Round", "disc")
        self.cb_form.addItem("Rectangular", "rect")
        self.cb_form.setCurrentIndex(0 if kind == "fan" else 1)
        form.addRow("Shape:", self.cb_form)
        defaults = flow_new(kind)
        self.sp_diameter = _spin(0.1, 10000.0, " mm", 1.0, prefix="Ø ")
        self.sp_diameter.setValue(defaults["diameter"])
        self.sp_width = _spin(0.1, 10000.0, " mm", 1.0, prefix="W ")
        self.sp_width.setValue(defaults["width"])
        self.sp_height = _spin(0.1, 10000.0, " mm", 1.0, prefix="H ")
        self.sp_height.setValue(defaults["height"])
        self.sp_thickness = _spin(0.1, 10000.0, " mm", 0.5, prefix="T ")
        self.sp_thickness.setValue(defaults["thickness"])
        size = QWidget()
        size_layout = QVBoxLayout(size)
        size_layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self.sp_diameter, self.sp_width, self.sp_height, self.sp_thickness):
            size_layout.addWidget(widget)
        form.addRow("Size:", size)
        self.cb_direction = QComboBox()
        for text, vector in self.DIRECTIONS:
            self.cb_direction.addItem(text, vector)
        self.cb_direction.setCurrentIndex(1 if kind == "fan" and anchor is not None else 0)
        form.addRow("Blows along:" if kind == "fan" else "Air passes along:", self.cb_direction)
        if kind == "fan":
            self.sp_flow = _spin(0.0, 100000.0, " m³/h", 1.0, decimals=2)
            self.sp_flow.setValue(defaults["flow_m3h"])
            form.addRow("Volume flow:", self.sp_flow)
            form.addRow(_hint("A fan curve (pressure over flow) can be set in the properties."))
        else:
            self.sp_free = _spin(1.0, 100.0, " % open", 5.0, decimals=1)
            self.sp_free.setValue(defaults["free_area"])
            form.addRow("Free area:", self.sp_free)
        self.chk_anchor = QCheckBox(
            (f"Next to {anchor_name}, blowing onto it" if kind == "fan" else
             f"In the face of {anchor_name}") if anchor is not None else "Next to the part")
        self.chk_anchor.setVisible(anchor is not None)
        self.chk_anchor.setChecked(anchor is not None)
        form.addRow("", self.chk_anchor)
        self.sp_position = [_spin(-100000.0, 100000.0, " mm", 1.0, prefix=f"{axis} ") for axis in "XYZ"]
        position = QWidget()
        position_layout = QVBoxLayout(position)
        position_layout.setContentsMargins(0, 0, 0, 0)
        for spin in self.sp_position:
            position_layout.addWidget(spin)
        form.addRow("Centre:", position)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.cb_form.currentIndexChanged.connect(self._update)
        self.cb_direction.currentIndexChanged.connect(self._update)
        self.chk_anchor.toggled.connect(self._update)
        for spin in (self.sp_diameter, self.sp_width, self.sp_height, self.sp_thickness):
            spin.valueChanged.connect(self._update)
        cad._style_checkbox_like_explorer(self.chk_anchor)
        self._update()

    def _flow(self):
        flow = flow_new(self._kind)
        flow.update(form=self.cb_form.currentData(), diameter=self.sp_diameter.value(),
                    width=self.sp_width.value(), height=self.sp_height.value(),
                    thickness=self.sp_thickness.value())
        if self._kind == "fan":
            flow["flow_m3h"] = self.sp_flow.value()
        else:
            flow["free_area"] = self.sp_free.value()
        return flow

    def _update(self, *_args):
        disc = self.cb_form.currentData() == "disc"
        self.sp_diameter.setVisible(disc)
        self.sp_width.setVisible(not disc)
        self.sp_height.setVisible(not disc)
        anchored = self._anchor is not None and self.chk_anchor.isChecked()
        if anchored:
            low, high = (np.asarray(v, float) for v in self._anchor)
            direction = np.asarray(self.cb_direction.currentData(), float)
            axis = int(np.argmax(np.abs(direction)))
            forward = direction[axis] > 0
            thickness = self.sp_thickness.value()
            centre = (low + high) / 2.0
            if self._kind == "fan":            # upstream of the part, blowing onto it
                centre[axis] = low[axis] - thickness / 2.0 if forward else high[axis] + thickness / 2.0
            else:                               # in the wall the air leaves through
                centre[axis] = high[axis] - thickness / 2.0 if forward else low[axis] + thickness / 2.0
            for spin, value in zip(self.sp_position, centre):
                spin.blockSignals(True); spin.setValue(float(value)); spin.blockSignals(False)
        for spin in self.sp_position:
            spin.setEnabled(not anchored)

    def values(self):
        """(name, flow settings, centre, rotation)."""
        direction = np.asarray(self.cb_direction.currentData(), float)
        matrix = cad._assembly_rotation_between(np.array([0.0, 0.0, 1.0]), direction)
        rotation = [float(x) for x in cad._import_euler_angles(matrix)]
        name = self.ed_name.text().strip() or ("Fan" if self._kind == "fan" else "Vent")
        return name, self._flow(), [spin.value() for spin in self.sp_position], rotation


class ContactDialog(QDialog):
    """Choose the two bodies of a contact."""

    def __init__(self, parent, bodies):
        super().__init__(parent)
        self.setWindowTitle("Contact between two bodies")
        self.setWindowIcon(_icon("fit_together.png"))
        form = QFormLayout(self)
        self._bodies = bodies                         # [(label, (part name, body index))]
        self.cb_a = QComboBox()
        self.cb_b = QComboBox()
        for label, ref in bodies:
            self.cb_a.addItem(label, ref)
            self.cb_b.addItem(label, ref)
        if len(bodies) > 1:
            self.cb_b.setCurrentIndex(1)
        form.addRow("Body A:", self.cb_a)
        form.addRow("Body B:", self.cb_b)
        form.addRow(_hint("The contact acts where the two bodies touch. Use Contacts → Detect "
                          "touching parts to find all pairs."))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self):
        return self.cb_a.currentData(), self.cb_b.currentData()


class ExtrusionDialog(QDialog):
    """Heat Source Extrusion: a rectangular or round footprint on a picked face, extruded out of
    the face (a component on it) or into the body (a heat zone in the body's material)."""

    def __init__(self, parent, name, face_text, host_material):
        super().__init__(parent)
        self.setWindowTitle("Heat Source Extrusion")
        self.setWindowIcon(_icon("heat_source_extrusion.png"))
        self._host_material = host_material
        form = QFormLayout(self)
        form.addRow(_hint(face_text))
        self.ed_name = QLineEdit(name)
        form.addRow("Name:", self.ed_name)
        self.cb_shape = QComboBox()
        self.cb_shape.addItem("Rectangle", "block")
        self.cb_shape.addItem("Circle", "cylinder")
        form.addRow("Footprint:", self.cb_shape)
        self.sp_u = _spin(0.01, 10000.0, " mm", 1.0, prefix="U ")
        self.sp_u.setValue(10.0)
        self.sp_v = _spin(0.01, 10000.0, " mm", 1.0, prefix="V ")
        self.sp_v.setValue(10.0)
        self.sp_diameter = _spin(0.01, 10000.0, " mm", 1.0, prefix="Ø ")
        self.sp_diameter.setValue(8.0)
        size = QWidget()
        size_layout = QVBoxLayout(size)
        size_layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self.sp_u, self.sp_v, self.sp_diameter):
            size_layout.addWidget(widget)
        form.addRow("Size:", size)
        self.sp_offset_u = _spin(-100000.0, 100000.0, " mm", 1.0, prefix="U ")
        self.sp_offset_v = _spin(-100000.0, 100000.0, " mm", 1.0, prefix="V ")
        offset = QWidget()
        offset_layout = QVBoxLayout(offset)
        offset_layout.setContentsMargins(0, 0, 0, 0)
        offset_layout.addWidget(self.sp_offset_u)
        offset_layout.addWidget(self.sp_offset_v)
        form.addRow("Offset:", offset)
        form.addRow(_hint("Offset of the footprint centre from the face centre along the face "
                          "(U follows X where possible, else Y; V is at right angles)."))
        self.sp_height = _spin(0.01, 10000.0, " mm", 0.5)
        self.sp_height.setValue(2.0)
        form.addRow("Height:", self.sp_height)
        self.cb_direction = QComboBox()
        self.cb_direction.addItem("Out of the face — a component on it", "out")
        self.cb_direction.addItem("Into the body — heat zone in its material", "in")
        form.addRow("Extrude:", self.cb_direction)
        self.sp_power = _spin(0.0, 100000.0, " W", 0.1)
        self.sp_power.setValue(1.0)
        form.addRow("Power:", self.sp_power)
        self.cb_material = _material_combo()
        form.addRow("Material:", self.cb_material)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)
        self.cb_shape.currentIndexChanged.connect(self._update)
        self.cb_direction.currentIndexChanged.connect(self._on_direction)
        self.cb_material.currentIndexChanged.connect(self._update)
        self._update()

    def _on_direction(self, *_args):
        if self.cb_direction.currentData() == "in" and self._host_material:
            self.cb_material.setCurrentIndex(max(0, self.cb_material.findData(self._host_material)))
        self._update()

    def _update(self, *_args):
        block = self.cb_shape.currentData() == "block"
        self.sp_u.setVisible(block)
        self.sp_v.setVisible(block)
        self.sp_diameter.setVisible(not block)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(self.cb_material.currentData() is not None)

    def values(self):
        """(kind, shape, name, offset u, offset v, height, inward, power, material)."""
        kind = self.cb_shape.currentData()
        height = self.sp_height.value()
        shape = {"size": [self.sp_u.value(), self.sp_v.value(), height]} if kind == "block" else \
            {"diameter": self.sp_diameter.value(), "height": height, "axis": "Z"}
        return (kind, shape, self.ed_name.text().strip() or "Heat source",
                self.sp_offset_u.value(), self.sp_offset_v.value(), height,
                self.cb_direction.currentData() == "in", self.sp_power.value(),
                self.cb_material.currentData())


class ThermostatEditor(QWidget):
    """Thermostat of a heat source or heat flux: switches the power by a sensor temperature."""
    changed = pyqtSignal()

    def __init__(self, label_width, value_width, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnMinimumWidth(0, label_width)
        grid.setColumnMinimumWidth(1, value_width)
        self._loading = False
        self.chk_on = QCheckBox("Thermostat")
        self.chk_on.setToolTip("Switches the power off above the setpoint and on again below it, "
                               "e.g. a heated bed with its thermistor")
        self.cb_sensor = QComboBox()
        self.cb_sensor.setFixedWidth(value_width)
        self.cb_sensor.setToolTip("Where the thermostat measures: a probe (place one with the "
                                  "Probe tool), or the hottest point / the mean of the heated body")
        self.sp_setpoint = _spin(-273.0, 5000.0, " °C", 1.0, decimals=1)
        self.sp_hysteresis = _spin(0.0, 1000.0, " K", 0.5, decimals=2)
        self.sp_hysteresis.setToolTip("Switching band: off at setpoint + half of it, on again at "
                                      "setpoint − half of it")
        self.lbl_info = _hint("")
        self.lbl_info.setFixedWidth(label_width + grid.horizontalSpacing() + value_width)
        grid.addWidget(self.chk_on, 0, 0, 1, 2, Qt.AlignLeft)
        self._captions = []
        for row, (text, widget) in enumerate((("Sensor:", self.cb_sensor),
                                              ("Setpoint:", self.sp_setpoint),
                                              ("Hysteresis:", self.sp_hysteresis)), start=1):
            label = QLabel(text)
            label.setFixedWidth(label_width)
            self._captions.append(label)
            grid.addWidget(label, row, 0)
            grid.addWidget(widget, row, 1, Qt.AlignLeft)
        grid.addWidget(self.lbl_info, 4, 0, 1, 2, Qt.AlignLeft)
        self.chk_on.toggled.connect(self._edited)
        self.cb_sensor.currentIndexChanged.connect(self._edited)
        self.sp_setpoint.valueChanged.connect(self._edited)
        self.sp_hysteresis.valueChanged.connect(self._edited)
        self.set_thermostat(None, [])

    def set_thermostat(self, thermostat, probes, owner="body"):
        self._loading = True
        try:
            data = thermostat or thermostat_new()
            self.chk_on.setChecked(thermostat is not None)
            self.cb_sensor.clear()
            self.cb_sensor.addItem(f"Hottest point of the {owner}", ("max", ""))
            self.cb_sensor.addItem(f"Mean of the {owner}", ("mean", ""))
            for name in probes:
                self.cb_sensor.addItem(f"Probe {name}", ("probe", name))
            if data["sensor"] == "probe" and data["probe"] not in probes:
                self.cb_sensor.addItem(f"Probe {data['probe'] or '?'} (missing)", ("probe", data["probe"]))
            for index in range(self.cb_sensor.count()):
                if tuple(self.cb_sensor.itemData(index)) == \
                        (data["sensor"], data["probe"] if data["sensor"] == "probe" else ""):
                    self.cb_sensor.setCurrentIndex(index)
                    break
            self.sp_setpoint.setValue(data["setpoint_c"])
            self.sp_hysteresis.setValue(data["hysteresis_k"])
            self._refresh()
        finally:
            self._loading = False

    def thermostat(self):
        if not self.chk_on.isChecked():
            return None
        sensor, probe = self.cb_sensor.currentData() or ("max", "")
        return thermostat_settings({"sensor": sensor, "probe": probe,
                                    "setpoint_c": self.sp_setpoint.value(),
                                    "hysteresis_k": self.sp_hysteresis.value()})

    def _refresh(self):
        on = self.chk_on.isChecked()
        for widget in [self.cb_sensor, self.sp_setpoint, self.sp_hysteresis] + self._captions:
            widget.setVisible(on)
        thermostat = self.thermostat()
        if thermostat is None:
            self.lbl_info.setText("")
            self.lbl_info.setVisible(False)
            return
        self.lbl_info.setVisible(True)
        band = thermostat["hysteresis_k"] / 2.0
        self.lbl_info.setText(
            f"Off above {thermostat['setpoint_c'] + band:g} °C, on below "
            f"{thermostat['setpoint_c'] - band:g} °C (checked at every time step). Steady state: "
            "the share of the power that holds the sensor at the setpoint.")

    def _edited(self, *_args):
        if self._loading:
            return
        self._refresh()
        self.changed.emit()


class PropertyTable(QWidget):
    """Editable table of a temperature-dependent property ([[°C, value], …])."""
    changed = pyqtSignal()

    def __init__(self, title, unit, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(title))
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["T  [°C]", unit])
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setFixedHeight(118)
        self.table.itemChanged.connect(lambda _item: None if self._loading else self.changed.emit())
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        self.btn_add = QPushButton("Add row")
        self.btn_remove = QPushButton("Remove row")
        self.btn_import = QPushButton("Import CSV…")
        self.btn_import.setToolTip("Two columns: temperature in °C, value (comma, semicolon or "
                                   "tab separated; a header line is fine)")
        for button in (self.btn_add, self.btn_remove, self.btn_import):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.btn_add.clicked.connect(self._add_row)
        self.btn_remove.clicked.connect(self._remove_row)
        self.btn_import.clicked.connect(self._import)
        self._loading = False

    def set_points(self, points):
        self._loading = True
        try:
            self.table.setRowCount(0)
            for t, value in points or []:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(f"{t:g}"))
                self.table.setItem(row, 1, QTableWidgetItem(f"{value:g}"))
        finally:
            self._loading = False

    def points(self):
        rows = []
        for row in range(self.table.rowCount()):
            items = [self.table.item(row, column) for column in range(2)]
            try:
                rows.append([float(items[0].text().replace(",", ".")),
                             float(items[1].text().replace(",", "."))])
            except (AttributeError, ValueError):
                continue
        return _property_table(rows)

    def set_read_only(self, read_only):
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers if read_only else
                                   QAbstractItemView.AllEditTriggers)
        for button in (self.btn_add, self.btn_remove, self.btn_import):
            button.setEnabled(not read_only)

    def _add_row(self):
        points = self.points()
        t = points[-1][0] + 20.0 if points else 20.0
        value = points[-1][1] if points else 1.0
        self._loading = True
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(f"{t:g}"))
        self.table.setItem(row, 1, QTableWidgetItem(f"{value:g}"))
        self._loading = False
        self.changed.emit()

    def _remove_row(self):
        row = self.table.currentRow()
        if row < 0:
            row = self.table.rowCount() - 1
        if row >= 0:
            self.table.removeRow(row)
            self.changed.emit()

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Property table (CSV)", "",
                                              "CSV / text files (*.csv *.txt *.dat);;All files (*)")
        if not path:
            return
        try:
            points = read_table_csv(path)
        except (OSError, ValueError) as ex:
            QMessageBox.warning(self, "Property table", f"Could not read\n{path}\n\n{ex}")
            return
        self.set_points(points)
        self.changed.emit()


class MaterialsDialog(QDialog):
    """The materials: built-in ones (read only) and the user's own, kept in the user's library
    (thermal_materials.json next to SimpleCAD) and copied into every study that uses them. User
    materials can have k(T) and cp(T) tables."""

    def __init__(self, parent, study, select=None):
        super().__init__(parent)
        self.setWindowTitle("Materials")
        self.setWindowIcon(_icon("objects.png"))
        self.resize(860, 640)
        self._study = study
        self._library_names = set(USER_MATERIALS)
        self._user = copy.deepcopy(USER_MATERIALS)
        self._user.update(copy.deepcopy(study.get("materials", {})))     # the study's copy wins
        self._origin = {name: name for name in self._user}               # current → original
        self._touched = set()
        self._deleted = set()
        self._current = None
        self._loading = False
        used = collections.Counter(body.get("material") for component in study["components"]
                                   for body in component["bodies"])
        self._used = used

        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.list = QListWidget()
        self.list.setMinimumWidth(240)
        self.list.currentItemChanged.connect(self._on_selected)
        left.addWidget(self.list, 1)
        row = QHBoxLayout()
        self.btn_new = QPushButton("New")
        self.btn_duplicate = QPushButton("Duplicate")
        self.btn_delete = QPushButton("Delete")
        for button in (self.btn_new, self.btn_duplicate, self.btn_delete):
            row.addWidget(button)
        left.addLayout(row)
        left.addWidget(_hint(f"Your materials are saved in {material_library_path()} and in every "
                             "study that uses them."))
        root.addLayout(left)

        right = QVBoxLayout()
        form = QFormLayout()
        self.ed_name = QLineEdit()
        form.addRow("Name:", self.ed_name)
        self.sp_k = _spin(1e-6, 1e5, " W/m·K", 1.0, decimals=4, width=170)
        form.addRow("k:", self.sp_k)
        self.chk_plane = QCheckBox("Laminate: in-plane k")
        self.sp_k_plane = _spin(1e-6, 1e5, " W/m·K", 1.0, decimals=4, width=170)
        plane = QWidget()
        plane_layout = QHBoxLayout(plane)
        plane_layout.setContentsMargins(0, 0, 0, 0)
        plane_layout.addWidget(self.chk_plane)
        plane_layout.addWidget(self.sp_k_plane)
        plane_layout.addStretch(1)
        form.addRow("", plane)
        self.sp_cp = _spin(1e-3, 1e6, " J/kg·K", 10.0, decimals=2, width=170)
        form.addRow("cp:", self.sp_cp)
        self.sp_density = _spin(1e-3, 1e5, " kg/m³", 10.0, decimals=2, width=170)
        form.addRow("Density:", self.sp_density)
        self.sp_emissivity = _spin(0.0, 1.0, "", 0.05, decimals=3, width=170)
        form.addRow("Emissivity:", self.sp_emissivity)
        self.chk_max = QCheckBox("Limit")
        self.sp_max = _spin(-273.0, 5000.0, " °C", 5.0, decimals=1, width=130)
        limit = QWidget()
        limit_layout = QHBoxLayout(limit)
        limit_layout.setContentsMargins(0, 0, 0, 0)
        limit_layout.addWidget(self.chk_max)
        limit_layout.addWidget(self.sp_max)
        limit_layout.addStretch(1)
        form.addRow("Max. service:", limit)
        self.ed_note = QLineEdit()
        form.addRow("Note:", self.ed_note)
        right.addLayout(form)
        tables = QHBoxLayout()
        self.tbl_k = PropertyTable("k(T), optional", "k  [W/m·K]")
        self.tbl_cp = PropertyTable("cp(T), optional (transient runs)", "cp  [J/kg·K]")
        tables.addWidget(self.tbl_k)
        tables.addWidget(self.tbl_cp)
        right.addLayout(tables)
        self.lbl_info = _hint("")
        right.addWidget(self.lbl_info)
        right.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        right.addWidget(buttons)
        root.addLayout(right, 1)

        self.btn_new.clicked.connect(self._new)
        self.btn_duplicate.clicked.connect(self._duplicate)
        self.btn_delete.clicked.connect(self._delete)
        for widget in (self.sp_k, self.sp_k_plane, self.sp_cp, self.sp_density, self.sp_emissivity,
                       self.sp_max):
            widget.valueChanged.connect(self._edited)
        for widget in (self.chk_plane, self.chk_max):
            widget.toggled.connect(self._edited)
        self.ed_name.editingFinished.connect(self._rename)
        self.ed_note.textEdited.connect(self._edited)
        self.tbl_k.changed.connect(self._edited)
        self.tbl_cp.changed.connect(self._edited)
        for checkbox in self.findChildren(QCheckBox):
            cad._style_checkbox_like_explorer(checkbox)
        self._fill_list(select)

    # list ------------------------------------------------------------------------------------
    def _fill_list(self, select=None):
        self._loading = True
        try:
            self.list.clear()
            italic = QFont(self.list.font())
            italic.setItalic(True)
            chosen = None
            for name in [n for n in material_names({}) if n in MATERIAL_THERMAL]:
                item = QListWidgetItem(name)
                item.setData(Qt.UserRole, ("builtin", name))
                item.setForeground(QColor("#9aa3ad"))
                item.setToolTip("Built-in (read only)")
                self.list.addItem(item)
                if name == select:
                    chosen = item
            for name in sorted(self._user, key=str.lower):
                origin = self._origin.get(name, name)
                where = "library" if origin in self._library_names or name in self._touched else \
                    "this study"
                count = self._used.get(origin, 0)
                item = QListWidgetItem(f"{name}   ({where}" + (f", {count} bod{'y' if count == 1 else 'ies'}"
                                                              if count else "") + ")")
                item.setData(Qt.UserRole, ("user", name))
                item.setFont(italic if where == "this study" else self.list.font())
                self.list.addItem(item)
                if name == select:
                    chosen = item
            if chosen is None and self.list.count():
                chosen = self.list.item(0)
        finally:
            self._loading = False
        if chosen is not None:
            self.list.setCurrentItem(chosen)

    def _kind_name(self, item=None):
        item = item or self.list.currentItem()
        return tuple(item.data(Qt.UserRole)) if item is not None else (None, None)

    def _on_selected(self, current, _previous):
        if self._loading or current is None:
            return
        kind, name = self._kind_name(current)
        self._current = (kind, name)
        data = material_properties(name, {}) if kind == "builtin" else self._user[name]
        self._loading = True
        try:
            self.ed_name.setText(name)
            self.sp_k.setValue(data["k"])
            self.chk_plane.setChecked(bool(data.get("k_plane")))
            self.sp_k_plane.setValue(data.get("k_plane") or data["k"])
            self.sp_cp.setValue(data["cp"])
            self.sp_density.setValue(data.get("density") or 1000.0)
            self.sp_emissivity.setValue(data["emissivity"])
            self.chk_max.setChecked(data.get("max_c") is not None)
            self.sp_max.setValue(data["max_c"] if data.get("max_c") is not None else 100.0)
            self.ed_note.setText(data.get("note", "") or "")
            self.tbl_k.set_points(data.get("k_table", []))
            self.tbl_cp.set_points(data.get("cp_table", []))
            read_only = kind == "builtin"
            for widget in (self.ed_name, self.sp_k, self.chk_plane, self.sp_k_plane, self.sp_cp,
                           self.sp_density, self.sp_emissivity, self.chk_max, self.sp_max,
                           self.ed_note):
                widget.setEnabled(not read_only)
            self.tbl_k.set_read_only(read_only)
            self.tbl_cp.set_read_only(read_only)
            self.btn_delete.setEnabled(not read_only)
            self._refresh()
        finally:
            self._loading = False

    def _refresh(self):
        kind, name = self._current or (None, None)
        if kind is None:
            return
        self.sp_k_plane.setEnabled(kind == "user" and self.chk_plane.isChecked())
        self.sp_max.setEnabled(kind == "user" and self.chk_max.isChecked())
        has_k_table = bool(self.tbl_k.points())
        has_cp_table = bool(self.tbl_cp.points())
        self.sp_k.setEnabled(kind == "user" and not has_k_table)
        self.sp_cp.setEnabled(kind == "user" and not has_cp_table)
        lines = []
        if kind == "builtin":
            lines.append("Built-in material: read only. Duplicate makes an editable copy (e.g. "
                         "to add k(T) or cp(T) tables).")
        if has_k_table:
            lines.append(f"k follows the table; k at {REFERENCE_C:g} °C = "
                         f"{float(table_value(self.tbl_k.points(), REFERENCE_C)):.4g} W/m·K"
                         + (" (the in-plane k scales with it)" if self.chk_plane.isChecked() else ""))
        if has_cp_table:
            lines.append("cp follows the table in transient runs (a peak models melting, e.g. of a "
                         "phase-change material).")
        if kind == "user":
            count = self._used.get(self._origin.get(name, name), 0)
            lines.append(f"Used by {count} bod{'y' if count == 1 else 'ies'} of this study." if count
                         else "Not used in this study.")
        lines.append("Between the table temperatures the values are interpolated linearly; "
                     "beyond them the end values hold.")
        self.lbl_info.setText("\n".join(lines))

    # editing ---------------------------------------------------------------------------------
    def _form_material(self):
        return material_settings({
            "k": self.sp_k.value(), "k_plane": self.sp_k_plane.value() if self.chk_plane.isChecked()
            else None, "cp": self.sp_cp.value(), "density": self.sp_density.value(),
            "emissivity": self.sp_emissivity.value(),
            "max_c": self.sp_max.value() if self.chk_max.isChecked() else None,
            "note": self.ed_note.text(), "k_table": self.tbl_k.points(),
            "cp_table": self.tbl_cp.points()})

    def _edited(self, *_args):
        if self._loading or self._current is None or self._current[0] != "user":
            return
        name = self._current[1]
        self._user[name] = self._form_material()
        self._touched.add(name)
        self._loading = True
        try:
            if self.tbl_k.points():
                self.sp_k.setValue(self._user[name]["k"])
            if self.tbl_cp.points():
                self.sp_cp.setValue(self._user[name]["cp"])
        finally:
            self._loading = False
        self._refresh()

    def _valid_name(self, name, own=None):
        name = name.strip()
        if not name:
            return "The name must not be empty."
        if name in MATERIAL_THERMAL:
            return f"'{name}' is a built-in material."
        if name != own and name in self._user:
            return f"There is already a material '{name}'."
        return None

    def _rename(self):
        if self._loading or self._current is None or self._current[0] != "user":
            return
        old = self._current[1]
        new = self.ed_name.text().strip()
        if new == old:
            return
        problem = self._valid_name(new, old)
        if problem:
            QMessageBox.warning(self, "Materials", problem)
            self.ed_name.setText(old)
            return
        self._user[new] = self._user.pop(old)
        self._origin[new] = self._origin.pop(old, old)
        self._touched.discard(old)
        self._touched.add(new)
        self._current = ("user", new)
        self._fill_list(new)

    def _unique(self, base):
        name, number = base, 2
        while name in self._user or name in MATERIAL_THERMAL:
            name = f"{base} {number}"
            number += 1
        return name

    def _new(self):
        name = self._unique("My material")
        self._user[name] = material_settings({"k": 1.0, "cp": 1000.0, "density": 1000.0,
                                              "emissivity": 0.9})
        self._touched.add(name)
        self._fill_list(name)
        self.ed_name.setFocus()
        self.ed_name.selectAll()

    def _duplicate(self):
        kind, name = self._kind_name()
        if kind is None:
            return
        source = material_properties(name, {}) if kind == "builtin" else self._user[name]
        new = self._unique(f"{name} (copy)")
        data = {key: source.get(key) for key in ("k", "k_plane", "cp", "density", "emissivity",
                                                 "max_c", "note", "k_table", "cp_table")}
        self._user[new] = material_settings(data)
        self._touched.add(new)
        self._fill_list(new)

    def _delete(self):
        kind, name = self._kind_name()
        if kind != "user":
            return
        count = self._used.get(self._origin.get(name, name), 0)
        if count:
            QMessageBox.information(self, "Materials", f"'{name}' is used by {count} "
                                    f"bod{'y' if count == 1 else 'ies'} of this study. Choose "
                                    "another material for them first.")
            return
        del self._user[name]
        self._deleted.add(self._origin.pop(name, name))
        self._touched.discard(name)
        self._fill_list()

    def _accept(self):
        self._rename()
        self.accept()

    def changes(self):
        """(user materials, renames {old: new}, deleted names, names for the library)."""
        renames = {origin: name for name, origin in self._origin.items() if origin != name}
        library = {name for name in self._user
                   if self._origin.get(name, name) in self._library_names or name in self._touched}
        return self._user, renames, self._deleted, library


class SweepDialog(QDialog):
    """Set up a parameter sweep: the parameter, its target and the values."""

    def __init__(self, parent, study, sweep=None):
        super().__init__(parent)
        self.setWindowTitle("Parameter sweep")
        self.setWindowIcon(_icon("settings2.png"))
        self._study = study
        self._loading = False
        form = QFormLayout(self)
        form.addRow(_hint("Runs the study once for every value of one parameter (the study itself "
                          "stays unchanged) and compares the temperatures of all bodies and "
                          "probes. Each run can be shown afterwards."))
        self.cb_parameter = QComboBox()
        for kind, (title, _unit, _target) in SWEEP_PARAMETERS.items():
            self.cb_parameter.addItem(title, kind)
        form.addRow("Parameter:", self.cb_parameter)
        self.cb_target = QComboBox()
        self.lbl_target = QLabel("Of:")
        form.addRow(self.lbl_target, self.cb_target)
        self.cb_axis = QComboBox()
        for index, axis in enumerate("XYZ"):
            self.cb_axis.addItem(axis, index)
        self.lbl_axis = QLabel("Axis:")
        form.addRow(self.lbl_axis, self.cb_axis)
        self.cb_spacing = QComboBox()
        self.cb_spacing.addItem("Evenly from … to …", "range")
        self.cb_spacing.addItem("List of values", "list")
        self.lbl_spacing = QLabel("Values:")
        form.addRow(self.lbl_spacing, self.cb_spacing)
        self.sp_start = _spin(-1e7, 1e7, "", 1.0, decimals=4)
        self.sp_stop = _spin(-1e7, 1e7, "", 1.0, decimals=4)
        self.sp_count = QSpinBox()
        self.sp_count.setRange(1, SWEEP_MAX_RUNS)
        self.sp_count.setKeyboardTracking(False)
        self.lbl_start, self.lbl_stop, self.lbl_count = QLabel("From:"), QLabel("To:"), QLabel("Runs:")
        form.addRow(self.lbl_start, self.sp_start)
        form.addRow(self.lbl_stop, self.sp_stop)
        form.addRow(self.lbl_count, self.sp_count)
        self.ed_list = QLineEdit()
        self.ed_list.setPlaceholderText("e.g. 0.5, 1, 2, 5")
        self.lbl_list = QLabel("List:")
        form.addRow(self.lbl_list, self.ed_list)
        self.lst_materials = QListWidget()
        self.lst_materials.setStyleSheet(_LIST_INDICATOR_STYLE)
        self.lst_materials.setFixedHeight(170)
        for name in material_names(study.get("materials", {})):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, name)
            self.lst_materials.addItem(item)
        self.lbl_materials = QLabel("Materials:")
        form.addRow(self.lbl_materials, self.lst_materials)
        self.lbl_info = _hint("")
        self.lbl_info.setMinimumWidth(380)
        form.addRow(self.lbl_info)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Run sweep")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)
        self.cb_parameter.currentIndexChanged.connect(self._on_parameter)
        self.cb_target.currentIndexChanged.connect(self._on_target)
        self.cb_axis.currentIndexChanged.connect(self._on_target)
        self.cb_spacing.currentIndexChanged.connect(self._update)
        for widget in (self.sp_start, self.sp_stop):
            widget.valueChanged.connect(self._update)
        self.sp_count.valueChanged.connect(self._update)
        self.ed_list.textChanged.connect(self._update)
        self.lst_materials.itemChanged.connect(self._update)
        self._apply(sweep_settings(sweep) or sweep_new())

    # targets of the parameter ----------------------------------------------------------------
    def _targets(self, kind):
        study = self._study
        target = SWEEP_PARAMETERS[kind][2]
        out = []
        if target == "body":
            for component in study["components"]:
                if component.get("kind") in FLOW_KINDS:
                    continue
                for index, body in enumerate(component["bodies"]):
                    if body.get("fused"):
                        continue
                    if kind == "setpoint" and body.get("thermostat") is None:
                        continue
                    label = component["name"] if len(component["bodies"]) == 1 else \
                        f"{component['name']} / {body['name']}"
                    out.append((label, [component["name"], index]))
        elif target == "part":
            out = [(c["name"], c["name"]) for c in study["components"]]
        elif target == "fan":
            out = [(c["name"], c["name"]) for c in study["components"] if c.get("kind") == "fan"]
        elif target == "flux":
            out = [(c["name"], c["name"]) for c in study.get("conditions", []) if c["kind"] == "flux"]
        elif target == "contact":
            out = [(c["name"], c["name"]) for c in study.get("contacts", [])]
        return out

    def _current_value(self):
        """The study's current value of the chosen parameter, or None."""
        study = self._study
        kind = self.cb_parameter.currentData()
        target = self.cb_target.currentData()
        settings = study["settings"]
        if kind in ("ambient", "h", "air_speed", "inlet_speed"):
            return settings[{"ambient": "ambient_c", "h": "h", "air_speed": "air_speed",
                             "inlet_speed": "inlet_speed"}[kind]]
        if kind == "cell":
            value = settings["flow_cell_mm" if settings["mode"] == "airflow" else "cell_mm"]
            if value <= 0:
                estimate = estimate_cells(study) if settings["mode"] != "airflow" else None
                value = estimate[0] if estimate else 1.0
            return value
        if target is None:
            return None
        if isinstance(target, list):
            component = next((c for c in study["components"] if c["name"] == target[0]), None)
            if component is None or target[1] >= len(component["bodies"]):
                return None
            body = component["bodies"][target[1]]
            return {"power": body.get("power_w", 0.0), "emissivity": body.get("emissivity") or
                    (material_properties(body.get("material")) or {}).get("emissivity", 0.9),
                    "setpoint": (body.get("thermostat") or {}).get("setpoint_c")}.get(kind)
        if kind == "position":
            component = next((c for c in study["components"] if c["name"] == target), None)
            return None if component is None else component["position"][self.cb_axis.currentData()]
        if kind == "fan_flow":
            component = next((c for c in study["components"] if c["name"] == target), None)
            return None if component is None else flow_settings(component.get("flow"), "fan")["flow_m3h"]
        if kind == "flux":
            condition = next((c for c in study["conditions"] if c["name"] == target), None)
            return None if condition is None else (condition["power_w"] if condition["flux_mode"] == "total"
                                                   else condition["flux_w_m2"])
        if kind == "contact":
            contact = next((c for c in study["contacts"] if c["name"] == target), None)
            return None if contact is None else (contact["thickness_mm"] if contact["kind"] == "layer"
                                                 else contact["resistance"])
        return None

    def _apply(self, sweep):
        self._loading = True
        try:
            self.cb_parameter.setCurrentIndex(max(0, self.cb_parameter.findData(sweep["parameter"])))
            self._fill_targets(sweep.get("target"))
            self.cb_axis.setCurrentIndex(int(sweep.get("axis", 2)))
            self.cb_spacing.setCurrentIndex(0 if sweep["spacing"] == "range" else 1)
            self.sp_start.setValue(sweep["start"])
            self.sp_stop.setValue(sweep["stop"])
            self.sp_count.setValue(sweep["count"])
            if sweep["parameter"] == "material":
                for row in range(self.lst_materials.count()):
                    item = self.lst_materials.item(row)
                    item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in sweep["values"]
                                       else Qt.Unchecked)
            else:
                self.ed_list.setText(", ".join(f"{v:g}" for v in sweep["values"]))
        finally:
            self._loading = False
        self._update()

    def _fill_targets(self, current=None):
        kind = self.cb_parameter.currentData()
        blocked = self.cb_target.blockSignals(True)
        self.cb_target.clear()
        for label, ref in self._targets(kind):
            self.cb_target.addItem(label, ref)
        for index in range(self.cb_target.count()):
            if self.cb_target.itemData(index) == current:
                self.cb_target.setCurrentIndex(index)
        self.cb_target.blockSignals(blocked)

    def _on_parameter(self, *_args):
        if self._loading:
            return
        self._fill_targets()
        self._on_target()

    def _on_target(self, *_args):
        """New parameter / target: values around the current one."""
        if self._loading:
            return
        value = self._current_value()
        kind = self.cb_parameter.currentData()
        self._loading = True
        try:
            if kind == "position":
                start, stop = (value or 0.0) - 10.0, (value or 0.0) + 10.0
            elif kind == "cell":
                start, stop = (value or 1.0) * 2.0, (value or 1.0) * 0.5
            elif kind == "ambient":
                start, stop = 20.0, 40.0
            elif kind == "emissivity":
                start, stop = 0.1, 0.9
            elif kind == "setpoint":
                start, stop = (value or 60.0) - 10.0, (value or 60.0) + 10.0
            elif value:
                start, stop = value * 0.5, value * 1.5
            else:
                start, stop = 0.0, 1.0
            self.sp_start.setValue(round(start, 4))
            self.sp_stop.setValue(round(stop, 4))
            if self.sp_count.value() < 2:
                self.sp_count.setValue(3)
        finally:
            self._loading = False
        self._update()

    def sweep(self):
        kind = self.cb_parameter.currentData()
        data = {"parameter": kind, "target": self.cb_target.currentData(),
                "axis": self.cb_axis.currentData(), "spacing": self.cb_spacing.currentData(),
                "start": self.sp_start.value(), "stop": self.sp_stop.value(),
                "count": self.sp_count.value()}
        if kind == "material":
            data["values"] = [self.lst_materials.item(row).data(Qt.UserRole)
                              for row in range(self.lst_materials.count())
                              if self.lst_materials.item(row).checkState() == Qt.Checked]
        elif data["spacing"] == "range":
            data["values"] = sweep_range(data["start"], data["stop"], data["count"])
        else:
            values = []
            for text in self.ed_list.text().replace(";", ",").split(","):
                try:
                    values.append(float(text.strip()))
                except ValueError:
                    continue
            data["values"] = values
        return sweep_settings(data)

    def _update(self, *_args):
        if self._loading:
            return
        kind = self.cb_parameter.currentData()
        material = kind == "material"
        needs_target = SWEEP_PARAMETERS[kind][2] is not None
        listed = self.cb_spacing.currentData() == "list"
        for widget in (self.lbl_target, self.cb_target):
            widget.setVisible(needs_target)
        for widget in (self.lbl_axis, self.cb_axis):
            widget.setVisible(kind == "position")
        for widget in (self.lbl_spacing, self.cb_spacing):
            widget.setVisible(not material)
        for widget in (self.lbl_start, self.sp_start, self.lbl_stop, self.sp_stop, self.lbl_count,
                       self.sp_count):
            widget.setVisible(not material and not listed)
        for widget in (self.lbl_list, self.ed_list):
            widget.setVisible(not material and listed)
        for widget in (self.lbl_materials, self.lst_materials):
            widget.setVisible(material)
        unit = sweep_unit(self._study, {"parameter": kind, "target": self.cb_target.currentData()})
        for spin in (self.sp_start, self.sp_stop):
            spin.setSuffix(f" {unit}" if unit else "")
        sweep = self.sweep()
        problems = []
        if needs_target and self.cb_target.count() == 0:
            problems.append({"setpoint": "No body has a thermostat yet.",
                             "fan_flow": "There is no fan in the study.",
                             "flux": "There is no heat-flux face condition.",
                             "contact": "There is no contact."}.get(kind, "Nothing to choose."))
        values = sweep["values"] if sweep else []
        if not values:
            problems.append("No values yet.")
        mode = self._study["settings"]["mode"]
        if kind in ("fan_flow", "inlet_speed") and mode != "airflow":
            problems.append("This parameter acts in the Airflow + heat solver only (Simulation "
                            "panel → Solver).")
        if kind in ("h", "air_speed") and mode == "airflow":
            problems.append("This parameter acts in the Conduction solver only.")
        if problems:
            text = "⚠ " + " ".join(problems)
        else:
            shown = ", ".join(sweep_value_text(sweep, v) for v in values[:12]) + \
                (" …" if len(values) > 12 else "")
            text = (f"{len(values)} run{'s' if len(values) != 1 else ''}: {shown}"
                    f"{(' ' + unit) if unit and not material else ''}. Each run takes about as long "
                    "as one simulation.")
            target = self.cb_target.currentData()
            if kind == "power" and isinstance(target, list):
                component = next((c for c in self._study["components"] if c["name"] == target[0]), None)
                if component is not None and target[1] < len(component["bodies"]) and \
                        component["bodies"][target[1]].get("thermostat") is not None:
                    text += ("\nThis body has a thermostat: while it controls, more power only "
                             "changes how often it switches, not the temperatures.")
        self.lbl_info.setText(text)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not problems)


class SweepResultsDialog(QDialog):
    """Table and chart of a parameter sweep; a run can be shown in the view."""
    showRun = pyqtSignal(int)

    def __init__(self, parent, sweep_result, outdated=False):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        self.setWindowIcon(_icon("settings2.png"))
        self.resize(820, 620)
        self._result = sweep_result
        layout = QVBoxLayout(self)
        self.lbl_title = QLabel()
        self.lbl_title.setWordWrap(True)
        layout.addWidget(self.lbl_title)
        self.table = QTableWidget(0, 0)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemDoubleClicked.connect(lambda _item: self._show())
        layout.addWidget(self.table, 1)
        self.plot = TimePlot()
        self.plot.setMinimumHeight(240)
        self.plot.timeClicked.connect(self._on_plot_clicked)
        layout.addWidget(self.plot, 1)
        buttons = QHBoxLayout()
        self.btn_show = QPushButton("Show run in the view")
        self.btn_show.clicked.connect(self._show)
        self.btn_csv = QPushButton("Export CSV…")
        self.btn_csv.clicked.connect(self._export_csv)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(self.btn_show)
        buttons.addWidget(self.btn_csv)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.set_result(sweep_result, outdated)

    def set_result(self, sweep_result, outdated=False):
        self._result = sweep_result
        self.setWindowTitle(f"Parameter sweep — {sweep_result['title']}")
        runs = sweep_result["runs"]
        transient = any(r["result"] is not None and r["result"]["analysis"] == "transient"
                        for r in runs)
        failed = [r for r in runs if r["error"]]
        self.lbl_title.setText(
            f"<b>{sweep_result['title']}</b>: {len(runs)} runs" +
            (" · transient: the highest temperature during each run" if transient else
             " · steady state") +
            (f" · <span style='color:#ff6b6b'>{len(failed)} failed</span>" if failed else "") +
            ("<br><span style='color:#ffb454'>The study changed after this sweep.</span>"
             if outdated else "") +
            "<br>Double-click a row (or click the chart) to show that run in the 3-D view.")
        columns, rows, bodies, probes = sweep_table(sweep_result, transient)
        self.table.clear()
        self.table.setColumnCount(len(columns))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(columns)
        for row, (values, run) in enumerate(zip(rows, runs)):
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if run["error"]:
                    item.setToolTip(run["error"])
                    item.setForeground(QColor("#ff6b6b"))
                elif column and column <= len(bodies):
                    entry = run["bodies"].get(bodies[column - 1], {})
                    value = entry.get("peak" if transient else "max")
                    if value is not None and entry.get("max_c") is not None and value > entry["max_c"]:
                        item.setForeground(QColor("#ff6b6b"))
                        item.setToolTip(f"Above the {entry['max_c']:g} °C limit")
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        # chart: every body's maximum and the probes over the parameter
        self._xs, curves, labels, self._order = sweep_plot_data(sweep_result)
        self.plot.set_data(self._xs, curves, x_labels=labels,
                           x_format=lambda v: f"{v:g}", markers=True)

    def _on_plot_clicked(self, value):
        if not len(self._xs):
            return
        index = self._order[int(np.argmin(np.abs(self._xs - value)))]
        self.table.selectRow(index)
        self._show()

    def _show(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.showRun.emit(rows[0].row())

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export sweep results", "sweep.csv",
                                              "CSV files (*.csv)")
        if not path:
            return
        try:
            write_sweep_csv(self._result, path)
        except OSError as ex:
            QMessageBox.critical(self, "Export failed", f"Could not write\n{path}\n\n{ex}")


class SweepWorker(QThread):
    """Runs a parameter sweep in the background."""
    progressed = pyqtSignal(int, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, study, sweep, parent=None):
        super().__init__(parent)
        self._study, self._sweep = study, sweep
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
            result = run_sweep(self._study, self._sweep, self._progress, lambda: self._cancel)
        except ThermalCancelled:
            self.failed.emit("")
        except ThermalError as ex:
            self.failed.emit(str(ex))
        except MemoryError:
            self.failed.emit("Not enough memory for the sweep. Fewer runs or a coarser mesh help.")
        except Exception as ex:          # report, never crash the application
            cad._trace("Thermal sweep error:\n" + traceback.format_exc())
            self.failed.emit(f"{type(ex).__name__}: {ex}")
        else:
            self.succeeded.emit(result)


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
        self.study.setdefault("contacts", [])
        self.study.setdefault("section", None)
        self.study.setdefault("materials", {})
        self.study.setdefault("sweep", None)
        set_active_study(self.study)
        self._sweep_result = None            # the last parameter sweep
        self._sweep_key = None               # study state it was run for
        self._sweep_dialog = None
        self._result_note = ""               # e.g. which sweep run is shown
        self._selected = None
        self._selected_body = None
        self._selected_condition = None      # index into study["conditions"]
        self._selected_probe = None          # index into study["probes"]
        self._selected_contact = None        # index into study["contacts"]
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
        self._sync_section_ui()
        self._update_section()
        self._set_status("Import parts or place a heat source, set the materials and powers, "
                         "then Start Simulation.")
        if _LIBRARY_ERROR:
            self.txt_messages.setPlainText("⚠ " + _LIBRARY_ERROR)

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
        self.export_menu = QMenu("&Export", self)
        self.export_menu.setIcon(_icon("export.png"))
        self.act_report = self.export_menu.addAction("PDF report…", self._export_pdf)
        self.export_menu.addSeparator()
        self.act_export_csv = self.export_menu.addAction("Results table (CSV)…", self._export_csv)
        self.act_export_series = self.export_menu.addAction("Values over time (CSV)…",
                                                            self._export_series_csv)
        self.act_export_vtu = self.export_menu.addAction("Result for ParaView (VTU)…",
                                                         self._export_vtu)
        self.act_export_vtu.setToolTip("Temperatures (and air velocities) of the whole mesh as a "
                                       "VTK file for ParaView; a transient run also as a series")
        self.act_export_png = self.export_menu.addAction("Image of the view (PNG)…", self._export_png)
        self.export_menu.aboutToShow.connect(self._update_export_menu)
        self.act_export = action(
            "export.png", "Export", "Export — a PDF report (setup, pictures, temperatures, sweep), "
            "the results as CSV, the result for ParaView (VTU), or the view as an image",
            lambda: None)
        self.act_export.setMenu(self.export_menu)
        self.act_materials = action(
            "objects.png", "Materials", "Materials — the material library: your own materials, "
            "also with temperature-dependent k(T) and cp(T)", self._open_materials)
        self.act_sweep = action(
            "settings2.png", "Parameter sweep",
            "Parameter sweep — run the study for several values of one parameter (power, material, "
            "position, ambient, h, fan flow, cell size …) and compare the temperatures",
            self._open_sweep)
        self.act_sweep_results = QAction("Sweep results", self,
                                         toolTip="Show the table and chart of the last sweep")
        self.act_sweep_results.triggered.connect(lambda _checked=False: self._show_sweep_results())
        self.act_sweep_results.setEnabled(False)
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
        self.heat_menu.addAction("Extrusion on a face…", self._start_extrusion)
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
        self.contact_menu = QMenu("&Contacts", self)
        self.contact_menu.setIcon(_icon("fit_together.png"))
        self.contact_menu.addAction("Detect touching parts", self._detect_contacts)
        self.contact_menu.addAction("Contact between two bodies…", self._add_contact)
        self.act_contact = action(
            "fit_together.png", "Contacts",
            "Contacts — touching bodies are bonded; a contact adds a contact resistance or an "
            "interface layer (thermal paste, pad, insulating film) between two bodies", lambda: None)
        self.act_contact.setMenu(self.contact_menu)
        self.act_section = action(
            "view_section.png", "Section view",
            "Section view — cut the view at a movable plane to see the temperatures inside the "
            "parts (and in the air of an airflow result)", self._on_section_toggled, checkable=True)
        self.flow_menu = QMenu("&Airflow", self)
        self.flow_menu.setIcon(_icon("air_flow_on.png"))
        self.flow_menu.addAction("Fan…", lambda: self._add_flow_item("fan"))
        self.flow_menu.addAction("Vent / opening…", lambda: self._add_flow_item("vent"))
        self.act_flow = action(
            "air_flow_on.png", "Airflow",
            "Airflow — fans and vents for the Airflow + heat (CHT) solver", lambda: None)
        self.act_flow.setMenu(self.flow_menu)
        self.act_streamlines = action(
            "air_flow_off.png", "Streamlines",
            "Streamlines — show the paths of the air in an airflow result",
            self._on_streamlines_toggled, checkable=True)
        self.act_streamlines.setEnabled(False)
        self.act_probe = action(
            "measurement.png", "Probe",
            "Probe — click on a part to place a temperature probe (Esc ends)", self._on_probe_toggled,
            checkable=True)
        self.act_start = action(
            "start_thermal_simulation.png", "Start Simulation",
            "Start Simulation (F5) — mesh the visible bodies and solve with the solver chosen in "
            "the Simulation panel (Conduction or Airflow + heat)", self._start_simulation,
            shortcut="F5")
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
                    self.act_probe, self.act_remove, None, self.act_heat, self.act_condition,
                    self.act_contact, self.act_flow, self.act_materials, None,
                    self.act_start, self.act_sweep, self.act_results, self.act_section, None,
                    *self._view_actions, self.act_origin, self.act_show_all, None,
                    self.act_update_selected, self.act_update_all):
            toolbar.addSeparator() if act is None else toolbar.addAction(act)

        menu_bar = QMenuBar(self)
        file_menu = menu_bar.addMenu("&File")
        for item in (self.act_new, self.act_open, self.act_save, self.act_save_as, None,
                     self.import_menu, self.export_menu, None, self.act_close):
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
        for item in self._view_actions + [None, self.act_origin, self.act_show_all, self.act_results,
                                          self.act_section, self.act_streamlines]:
            view_menu.addSeparator() if item is None else view_menu.addAction(item)
        simulation_menu = menu_bar.addMenu("&Simulation")
        simulation_menu.addMenu(self.heat_menu)
        simulation_menu.addMenu(self.condition_menu)
        simulation_menu.addMenu(self.contact_menu)
        simulation_menu.addMenu(self.flow_menu)
        simulation_menu.addAction(self.act_probe)
        simulation_menu.addAction(self.act_materials)
        simulation_menu.addSeparator()
        for item in (self.act_start, self.act_sweep, self.act_sweep_results, self.act_cancel,
                     self.act_results):
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

        # body of the part (material, power, profile, emissivity)
        self.w_body = QWidget()
        body_grid = QGridLayout(self.w_body)
        body_grid.setContentsMargins(0, 0, 0, 0)
        body_grid.setColumnMinimumWidth(0, label_width)
        body_grid.setColumnMinimumWidth(1, value_width)
        body_row = [0]

        def body_add(widget, label=None, span=False):
            if span:
                body_grid.addWidget(widget, body_row[0], 0, 1, 2, Qt.AlignLeft)
            else:
                if label is not None:
                    caption = QLabel(label)
                    caption.setFixedWidth(label_width)
                    body_grid.addWidget(caption, body_row[0], 0)
                body_grid.addWidget(widget, body_row[0], 1, Qt.AlignLeft)
            body_row[0] += 1

        body_add(_header("Body"), span=True)
        self.lbl_body = value_label()
        body_add(self.lbl_body, "Name:")
        self.cb_material = _material_combo()
        self.cb_material.setFixedWidth(value_width)
        self.cb_material.currentIndexChanged.connect(self._on_material_changed)
        body_add(self.cb_material, "Material:")
        self.lbl_material = _hint("")
        self.lbl_material.setFixedWidth(span_width)
        body_add(self.lbl_material, span=True)
        self.sp_power = _spin(0.0, 100000.0, " W", 0.1)
        self.sp_power.setToolTip("Heat produced inside this body (spread evenly over its volume)")
        self.sp_power.valueChanged.connect(self._on_power_changed)
        body_add(self.sp_power, "Power:")
        self.lbl_power = _hint("")
        self.lbl_power.setFixedWidth(span_width)
        body_add(self.lbl_power, span=True)
        self.profile_body = ProfileEditor(label_width, value_width)
        self.profile_body.changed.connect(self._on_body_profile_changed)
        body_add(self.profile_body, span=True)
        self.thermostat_body = ThermostatEditor(label_width, value_width)
        self.thermostat_body.changed.connect(self._on_body_thermostat_changed)
        body_add(self.thermostat_body, span=True)
        self.chk_emissivity = QCheckBox("Own surface emissivity")
        self.chk_emissivity.setToolTip("e.g. 0.8 for anodised or painted metal; unticked = the "
                                       "material's value")
        self.chk_emissivity.toggled.connect(self._on_emissivity_changed)
        body_add(self.chk_emissivity, span=True)
        self.sp_emissivity = _spin(0.0, 1.0, "", 0.05, decimals=2, prefix="ε ")
        self.sp_emissivity.valueChanged.connect(self._on_emissivity_changed)
        body_add(self.sp_emissivity, span=True)
        add(self.w_body, span=True)
        self.w_flow = self._build_flow_widget(label_width, value_width, span_width)
        add(self.w_flow, span=True)
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
        self.profile_condition = ProfileEditor(label_width, value_width)
        self.profile_condition.changed.connect(self._on_condition_edited)
        self.thermostat_condition = ThermostatEditor(label_width, value_width)
        self.thermostat_condition.changed.connect(self._on_condition_edited)
        self._condition_rows = {
            "fixed": [self.sp_condition_temperature,
                      put_row(self.sp_condition_temperature, "Temperature:")],
            "flux": [self.cb_flux_mode, put_row(self.cb_flux_mode, "Given as:"),
                     self.sp_condition_power, put_row(self.sp_condition_power, "Power:"),
                     self.sp_condition_flux, put_row(self.sp_condition_flux, "Flux:"),
                     self.profile_condition, put_row(self.profile_condition, span=True),
                     self.thermostat_condition, put_row(self.thermostat_condition, span=True)],
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

        # contact page
        contact_page, put_row, finish = page_form()
        put_row(_header("Contact"), span=True)
        self.lbl_contact_name = value_label()
        put_row(self.lbl_contact_name, "Name:")
        self.lbl_contact_bodies = value_label()
        put_row(self.lbl_contact_bodies, "Bodies:")
        self.chk_contact_enabled = QCheckBox("Active (untick: the bodies are bonded)")
        put_row(self.chk_contact_enabled, span=True)
        self.cb_contact_kind = QComboBox()
        self.cb_contact_kind.setFixedWidth(value_width)
        for kind, title in CONTACT_KINDS.items():
            self.cb_contact_kind.addItem(title, kind)
        put_row(self.cb_contact_kind, "Type:")
        self.sp_contact_resistance = _spin(0.0, 100000.0, " K·cm²/W", 0.1, decimals=3)
        self.sp_contact_resistance.setToolTip("Thermal contact resistance per area "
                                              "(1 K·cm²/W = 1·10⁻⁴ m²K/W)")
        self.cb_tim_preset = QComboBox()
        self.cb_tim_preset.setFixedWidth(value_width)
        self.cb_tim_preset.addItem("Custom", None)
        for name, thickness, k in TIM_PRESETS:
            self.cb_tim_preset.addItem(name, (thickness, k))
            self.cb_tim_preset.setItemData(self.cb_tim_preset.count() - 1,
                                           f"{thickness:g} mm, k {k:g} W/m·K", Qt.ToolTipRole)
        self.sp_contact_thickness = _spin(0.001, 1000.0, " mm", 0.05, decimals=3)
        self.sp_contact_k = _spin(0.001, 1000.0, " W/m·K", 0.1, decimals=3)
        self._contact_rows = {
            "resistance": [self.sp_contact_resistance,
                           put_row(self.sp_contact_resistance, "R:")],
            "layer": [self.cb_tim_preset, put_row(self.cb_tim_preset, "Preset:"),
                      self.sp_contact_thickness, put_row(self.sp_contact_thickness, "Thickness:"),
                      self.sp_contact_k, put_row(self.sp_contact_k, "k:")]}
        self.lbl_contact_info = _hint("")
        self.lbl_contact_info.setFixedWidth(span_width)
        put_row(self.lbl_contact_info, span=True)
        contact_hint = _hint("Bonded: the bodies share their temperature where they touch. "
                             "Contact resistance: dry joints, typically 1–10 K·cm²/W for metal on "
                             "metal, 0.1–0.5 with thermal paste. Interface layer: a thin layer "
                             "between the bodies that is not modelled as a part.")
        contact_hint.setFixedWidth(span_width)
        put_row(contact_hint, span=True)
        finish()
        self.chk_contact_enabled.toggled.connect(self._on_contact_edited)
        self.cb_contact_kind.currentIndexChanged.connect(self._on_contact_edited)
        self.cb_tim_preset.currentIndexChanged.connect(self._on_tim_preset)
        for spin in (self.sp_contact_resistance, self.sp_contact_thickness, self.sp_contact_k):
            spin.valueChanged.connect(self._on_contact_edited)

        self._props_stack = QStackedWidget()
        for page in (props, condition_page, probe_page, contact_page):
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
        self.view.section_editor = self           # draws the section plane and its handles
        self._section_timer = QTimer(self)
        self._section_timer.setSingleShot(True)
        self._section_timer.setInterval(60)
        self._section_timer.timeout.connect(self._update_section)
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

        # section view (shown while the Section view button is on)
        self.w_section = QWidget()
        section_grid = QGridLayout(self.w_section)
        section_grid.setContentsMargins(0, 0, 0, 0)
        section_grid.addWidget(_header("Section view"), 0, 0, 1, 2)
        self.chk_cut_show_move = QCheckBox("Move arrows")
        self.chk_cut_show_move.setChecked(True)
        self.chk_cut_show_rotate = QCheckBox("Rotation rings")
        self.chk_cut_show_rotate.setChecked(True)
        self.chk_cut_show_plane = QCheckBox("Show the plane and its handles")
        self.chk_cut_show_plane.setChecked(True)
        self.chk_cut_flip = QCheckBox("Show the other side")
        self.sp_cut_position = [_spin(-1000000.0, 1000000.0, " mm", 1.0, prefix=f"{axis} ")
                                for axis in "XYZ"]
        self.sp_cut_rotation = [_spin(-360000.0, 360000.0, "°", 5.0, prefix=f"{axis} ")
                                for axis in "XYZ"]
        section_row = 1
        for text, widgets in (("Position:", self.sp_cut_position),
                              ("Rotation:", self.sp_cut_rotation)):
            for index, spin in enumerate(widgets):
                if index == 0:
                    section_grid.addWidget(QLabel(text), section_row, 0)
                section_grid.addWidget(spin, section_row, 1)
                section_row += 1
        checks = QWidget()
        checks_layout = QGridLayout(checks)
        checks_layout.setContentsMargins(0, 0, 0, 0)
        checks_layout.addWidget(self.chk_cut_flip, 0, 0)
        checks_layout.addWidget(self.chk_cut_show_plane, 1, 0, 1, 2)
        checks_layout.addWidget(self.chk_cut_show_move, 2, 0)
        checks_layout.addWidget(self.chk_cut_show_rotate, 2, 1)
        section_grid.addWidget(checks, section_row, 0, 1, 2)
        section_row += 1
        self.cb_section_field = QComboBox()
        self.cb_section_field.addItem("Temperature", "temperature")
        self.cb_section_field.addItem("Air speed", "speed")
        self.cb_section_field.setToolTip("What the section through an airflow result shows")
        self.lbl_section_field = QLabel("Shows:")
        section_grid.addWidget(self.lbl_section_field, section_row, 0)
        section_grid.addWidget(self.cb_section_field, section_row, 1)
        section_row += 1
        self.chk_section_arrows = QCheckBox("Air velocity arrows")
        self.chk_section_arrows.setChecked(True)
        section_grid.addWidget(self.chk_section_arrows, section_row, 0, 1, 2)
        section_row += 1
        section_grid.addWidget(_hint("The part on the orange arrow's side of the grey plane is "
                                     "hidden. Drag the arrows / rings in the view or type values. "
                                     "Probes can be placed on the cut faces."),
                               section_row, 0, 1, 2)
        for spin in self.sp_cut_position + self.sp_cut_rotation:
            spin.valueChanged.connect(self._on_section_field_changed)
        self.chk_cut_flip.toggled.connect(self._on_section_field_changed)
        for box in (self.chk_cut_show_plane, self.chk_cut_show_move, self.chk_cut_show_rotate):
            box.toggled.connect(lambda _on: (self._update_gizmo(), self.view.update()))
        self.cb_section_field.currentIndexChanged.connect(lambda _i: self._schedule_section())
        self.chk_section_arrows.toggled.connect(lambda _on: self._schedule_section())
        put(self.w_section, span=True)
        self.w_section.hide()

        put(_header("Solver"), span=True)
        self.cb_mode = QComboBox()
        self.cb_mode.addItem("Conduction (FEM)", "conduction")
        self.cb_mode.addItem("Airflow + heat (CHT)", "airflow")
        self.cb_mode.setToolTip(
            "Conduction (FEM): heat conduction in the bodies; air cooling and radiation act on "
            "their surfaces (fast).\nAirflow + heat (CHT): the air around and inside the parts is "
            "simulated as well, with buoyancy, fans and vents (minutes).")
        put(self.cb_mode, "Solver:")
        self.lbl_mode_hint = _hint("")
        put(self.lbl_mode_hint, span=True)
        self.cb_analysis = QComboBox()
        self.cb_analysis.addItem("Steady state", "steady")
        self.cb_analysis.addItem("Transient (over time)", "transient")
        self.cb_analysis.setToolTip(
            "Steady state: the final temperatures after a long time.\n"
            "Transient: how the temperatures develop after the heat sources are switched on.")
        put(self.cb_analysis, "Analysis:")
        self._conduction_rows = []
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
        self.cb_convection.addItem("Forced air (air speed)", "forced")
        self.cb_convection.addItem("Fixed coefficient h", "fixed")
        self.cb_convection.setToolTip(
            "Natural convection: h from correlations for vertical, upward and downward facing "
            "surfaces (gravity along −Z).\nForced air: air flows along the surfaces at the given "
            "speed (flat-plate correlation, combined with natural convection).\nFixed: one heat "
            "transfer coefficient for all surfaces (≈ 5–10 still air, 20–100 with a fan).")
        self._conduction_rows += [self.cb_convection, put(self.cb_convection, "Convection:")]
        self.sp_air_speed = _spin(0.0, 100.0, " m/s", 0.1, decimals=2)
        self.sp_air_speed.setToolTip("Speed of the air along the surfaces (a small fan gives "
                                     "≈ 1–3 m/s)")
        self._speed_rows = [self.sp_air_speed, put(self.sp_air_speed, "Air speed:")]
        self.sp_h = _spin(0.0, 100000.0, " W/m²K", 1.0, decimals=2)
        self._h_rows = [self.sp_h, put(self.sp_h, "h:")]
        self.chk_radiation = QCheckBox("Radiation to the surroundings")
        put(self.chk_radiation, span=True)
        self.lbl_environment_hint = _hint("")
        put(self.lbl_environment_hint, span=True)

        # airflow domain (Airflow + heat only)
        self._airflow_rows = [_header("Air domain")]
        put(self._airflow_rows[0], span=True)
        self.sp_flow_space = _spin(0.0, 100000.0, " mm", 5.0, decimals=1)
        self.sp_flow_space.setToolTip("Air beside the parts in X and Y (0 = automatic)")
        self.sp_flow_below = _spin(0.0, 100000.0, " mm", 5.0, decimals=1)
        self.sp_flow_below.setToolTip("Air below the parts (0 = automatic)")
        self.sp_flow_above = _spin(0.0, 100000.0, " mm", 5.0, decimals=1)
        self.sp_flow_above.setToolTip("Air above the parts, where the warm air rises "
                                      "(0 = automatic)")
        for spin, text in ((self.sp_flow_space, "Beside:"), (self.sp_flow_below, "Below:"),
                           (self.sp_flow_above, "Above:")):
            spin.setSpecialValueText("automatic")
            self._airflow_rows += [spin, put(spin, text)]
        sides = QWidget()
        sides_layout = QGridLayout(sides)
        sides_layout.setContentsMargins(0, 0, 0, 0)
        sides_layout.setHorizontalSpacing(4)
        self.cb_sides = {}
        for index, side in enumerate(FLOW_SIDES):
            combo = QComboBox()
            for kind, title in SIDE_KINDS.items():
                combo.addItem(title, kind)
            combo.setToolTip(SIDE_HINTS)
            label = QLabel(FLOW_SIDE_TITLES[side])
            sides_layout.addWidget(label, index, 0)
            sides_layout.addWidget(combo, index, 1)
            combo.currentIndexChanged.connect(self._on_settings_edited)
            self.cb_sides[side] = combo
        self._airflow_rows += [sides, put(sides, "Domain sides:")]
        self.sp_inlet_speed = _spin(0.0, 100.0, " m/s", 0.1, decimals=2)
        self.sp_inlet_speed.setToolTip("Air speed through Inlet sides (into the domain)")
        self._inlet_rows = [self.sp_inlet_speed, put(self.sp_inlet_speed, "Inlet speed:")]
        self.sp_wall_temperature = _spin(-100.0, 1000.0, " °C", 1.0, decimals=1)
        self.sp_wall_temperature.setToolTip("Temperature of sides set to 'Wall, fixed temperature'")
        self._wall_rows = [self.sp_wall_temperature, put(self.sp_wall_temperature, "Wall at:")]
        self.cb_turbulence = QComboBox()
        self.cb_turbulence.addItem("Automatic", "auto")
        self.cb_turbulence.addItem("Laminar", "laminar")
        self.cb_turbulence.addItem("Turbulent (mixing length)", "mixing")
        self.cb_turbulence.setToolTip("Automatic: laminar for natural convection, turbulent as soon "
                                      "as a fan or an inlet blows.\nLaminar fits natural "
                                      "convection in small housings; with fans or fast inlets the "
                                      "flow is usually turbulent.")
        self._airflow_rows += [self.cb_turbulence, put(self.cb_turbulence, "Flow:")]
        self.chk_auto_flow_cell = QCheckBox("Automatic grid")
        put(self.chk_auto_flow_cell, span=True)
        self._airflow_rows.append(self.chk_auto_flow_cell)
        self.sp_flow_cell = _spin(0.05, 1000.0, " mm", 0.5, decimals=2)
        self.sp_flow_cell.setToolTip("Cell size at the parts; the cells grow towards the domain "
                                     "sides")
        self._airflow_rows += [self.sp_flow_cell, put(self.sp_flow_cell, "Cell size:")]
        self.sp_flow_max_cells = QSpinBox()
        self.sp_flow_max_cells.setRange(5000, 3000000)
        self.sp_flow_max_cells.setSingleStep(25000)
        self.sp_flow_max_cells.setKeyboardTracking(False)
        self.sp_flow_max_cells.setMinimumWidth(130)
        self.sp_flow_max_cells.setToolTip("Upper limit of grid cells (air and solid)")
        self._airflow_rows += [self.sp_flow_max_cells, put(self.sp_flow_max_cells, "Max cells:")]
        self.sp_flow_iterations = QSpinBox()
        self.sp_flow_iterations.setRange(20, 20000)
        self.sp_flow_iterations.setSingleStep(100)
        self.sp_flow_iterations.setKeyboardTracking(False)
        self.sp_flow_iterations.setMinimumWidth(130)
        self.sp_flow_iterations.setToolTip("Upper limit of solver iterations; the run stops "
                                           "earlier once the flow and the heat balance settle")
        self._flow_steady_rows = [self.sp_flow_iterations, put(self.sp_flow_iterations, "Iterations:")]
        self._airflow_rows += self._flow_steady_rows
        self.sp_flow_step_iterations = QSpinBox()
        self.sp_flow_step_iterations.setRange(2, 500)
        self.sp_flow_step_iterations.setSingleStep(5)
        self.sp_flow_step_iterations.setKeyboardTracking(False)
        self.sp_flow_step_iterations.setMinimumWidth(130)
        self.sp_flow_step_iterations.setToolTip("Upper limit of iterations within each time step; "
                                                "a step ends earlier once air and heat settle")
        self._flow_transient_rows = [self.sp_flow_step_iterations,
                                     put(self.sp_flow_step_iterations, "Per time step:")]
        self._airflow_rows += self._flow_transient_rows
        self.lbl_flow_estimate = _hint("")
        put(self.lbl_flow_estimate, span=True)
        self._airflow_rows.append(self.lbl_flow_estimate)

        self._mesh_rows = [_header("Mesh")]
        put(self._mesh_rows[0], span=True)
        self.chk_auto_cell = QCheckBox("Automatic cell size")
        put(self.chk_auto_cell, span=True)
        self.sp_cell = _spin(0.01, 1000.0, " mm", 0.1, decimals=3)
        self.sp_cell.setToolTip("Edge length of the cubic cells. Thin walls need at least two "
                                "cells across.")
        self.sp_max_cells = QSpinBox()
        self.sp_max_cells.setRange(10000, 5000000)
        self.sp_max_cells.setSingleStep(50000)
        self.sp_max_cells.setKeyboardTracking(False)
        self.sp_max_cells.setMinimumWidth(130)
        self.sp_max_cells.setToolTip("Upper limit of solid cells; more cells = finer result, "
                                     "more time and memory")
        self.lbl_estimate = _hint("")
        self._mesh_rows += [self.chk_auto_cell, self.sp_cell, put(self.sp_cell, "Cell size:"),
                            self.sp_max_cells, put(self.sp_max_cells, "Max cells:"),
                            self.lbl_estimate]
        put(self.lbl_estimate, span=True)
        for widget in (self.sp_ambient, self.sp_h, self.sp_cell, self.sp_air_speed,
                       self.sp_flow_space, self.sp_flow_below, self.sp_flow_above,
                       self.sp_inlet_speed, self.sp_wall_temperature, self.sp_flow_cell):
            widget.valueChanged.connect(self._on_settings_edited)
        for widget in (self.sp_max_cells, self.sp_flow_max_cells, self.sp_flow_iterations,
                       self.sp_flow_step_iterations):
            widget.valueChanged.connect(self._on_settings_edited)
        for widget in (self.cb_convection, self.cb_mode, self.cb_turbulence):
            widget.currentIndexChanged.connect(self._on_settings_edited)
        for widget in (self.chk_radiation, self.chk_auto_cell, self.chk_auto_flow_cell):
            widget.toggled.connect(self._on_settings_edited)
        for widget in (self.cb_analysis, self.cb_duration_unit):
            widget.currentIndexChanged.connect(self._on_settings_edited)
        for widget in (self.sp_duration, self.sp_time_steps, self.sp_initial):
            widget.valueChanged.connect(self._on_settings_edited)
        self.chk_initial_ambient.toggled.connect(self._on_settings_edited)
        self._setting_inputs = (self.sp_ambient, self.cb_convection, self.sp_h, self.chk_radiation,
                                self.chk_auto_cell, self.sp_cell, self.sp_max_cells,
                                self.cb_analysis, self.sp_duration, self.cb_duration_unit,
                                self.sp_time_steps, self.chk_initial_ambient, self.sp_initial,
                                self.sp_air_speed, self.cb_mode, self.sp_flow_space,
                                self.sp_flow_below, self.sp_flow_above, self.sp_inlet_speed,
                                self.sp_wall_temperature, self.cb_turbulence,
                                self.chk_auto_flow_cell, self.sp_flow_cell, self.sp_flow_max_cells,
                                self.sp_flow_iterations, self.sp_flow_step_iterations,
                                *self.cb_sides.values())

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
            if button.defaultAction() in (self.act_import, self.act_heat, self.act_condition,
                                          self.act_contact, self.act_flow, self.act_export):
                button.setPopupMode(QToolButton.InstantPopup)
        root.addWidget(toolbar)
        root.addWidget(split, 1)
        root.addLayout(bottom)
        for checkbox in self.findChildren(QCheckBox):       # as the object explorer
            cad._style_checkbox_like_explorer(checkbox)

    def _build_flow_widget(self, label_width, value_width, span_width):
        """Properties of a fan or vent (shown instead of the body group)."""
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnMinimumWidth(0, label_width)
        grid.setColumnMinimumWidth(1, value_width)
        row = [0]

        def put_row(item, label=None, span=False):
            caption = None
            if span:
                grid.addWidget(item, row[0], 0, 1, 2, Qt.AlignLeft)
            else:
                if label is not None:
                    caption = QLabel(label)
                    caption.setFixedWidth(label_width)
                    grid.addWidget(caption, row[0], 0)
                grid.addWidget(item, row[0], 1, Qt.AlignLeft)
            row[0] += 1
            return caption

        self.lbl_flow_header = _header("Fan")
        put_row(self.lbl_flow_header, span=True)
        self.cb_flow_form = QComboBox()
        self.cb_flow_form.setFixedWidth(value_width)
        self.cb_flow_form.addItem("Round", "disc")
        self.cb_flow_form.addItem("Rectangular", "rect")
        put_row(self.cb_flow_form, "Shape:")
        self.sp_flow_diameter = _spin(0.1, 10000.0, " mm", 1.0, prefix="Ø ")
        self.sp_flow_width = _spin(0.1, 10000.0, " mm", 1.0, prefix="W ")
        self.sp_flow_height = _spin(0.1, 10000.0, " mm", 1.0, prefix="H ")
        self.sp_flow_thickness = _spin(0.1, 10000.0, " mm", 0.5, prefix="T ")
        self.sp_flow_thickness.setToolTip("Thickness along the flow direction (at least one cell)")
        self._flow_size_rows = {
            "disc": [self.sp_flow_diameter, put_row(self.sp_flow_diameter, "Size:")],
            "rect": [self.sp_flow_width, put_row(self.sp_flow_width, "Size:"), self.sp_flow_height,
                     put_row(self.sp_flow_height, "")]}
        put_row(self.sp_flow_thickness, "")
        self.cb_flow_mode = QComboBox()
        self.cb_flow_mode.setFixedWidth(value_width)
        self.cb_flow_mode.addItem("Fixed volume flow", "fixed")
        self.cb_flow_mode.addItem("Fan curve", "curve")
        self.cb_flow_mode.setToolTip("Fixed: the fan always moves this flow.\nFan curve: the flow "
                                     "follows from the fan's pressure curve and the resistance of "
                                     "the housing, vents and fins.")
        self.sp_flow_m3h = _spin(0.0, 100000.0, " m³/h", 1.0, decimals=2)
        self.sp_flow_max_m3h = _spin(0.001, 100000.0, " m³/h", 1.0, decimals=2)
        self.sp_flow_max_m3h.setToolTip("Free-blowing flow from the datasheet (at 0 Pa)")
        self.sp_flow_max_pa = _spin(0.0, 100000.0, " Pa", 1.0, decimals=1)
        self.sp_flow_max_pa.setToolTip("Maximum static pressure from the datasheet (at 0 flow)")
        self.btn_flow_curve = QPushButton("Import curve CSV…")
        self.btn_flow_curve.setFixedWidth(value_width)
        self.btn_flow_curve.setToolTip("Fan curve as two columns: flow in m³/h, pressure in Pa "
                                       "(replaces the two datasheet values)")
        self.btn_flow_curve.clicked.connect(self._import_fan_curve)
        self._fan_rows = {
            "mode": [self.cb_flow_mode, put_row(self.cb_flow_mode, "Fan:")],
            "fixed": [self.sp_flow_m3h, put_row(self.sp_flow_m3h, "Flow:")],
            "curve": [self.sp_flow_max_m3h, put_row(self.sp_flow_max_m3h, "Max flow:"),
                      self.sp_flow_max_pa, put_row(self.sp_flow_max_pa, "Max pressure:"),
                      self.btn_flow_curve, put_row(self.btn_flow_curve, "")]}
        self.sp_free_area = _spin(1.0, 100.0, " % open", 5.0, decimals=1)
        self.sp_free_area.setToolTip("Open share of the vent area (grille, perforated plate); "
                                     "100 % = a plain opening")
        self._vent_rows = [self.sp_free_area, put_row(self.sp_free_area, "Free area:")]
        self.lbl_flow_info = _hint("")
        self.lbl_flow_info.setFixedWidth(span_width)
        put_row(self.lbl_flow_info, span=True)
        flow_hint = _hint("The arrow shows the flow direction (the part's local +Z axis); turn the "
                          "part to change it. A fan or vent overlapping a housing wall opens the "
                          "wall there. Fans and vents act in the Airflow + heat solver only.")
        flow_hint.setFixedWidth(span_width)
        put_row(flow_hint, span=True)
        self.cb_flow_form.currentIndexChanged.connect(self._on_flow_edited)
        self.cb_flow_mode.currentIndexChanged.connect(self._on_flow_edited)
        for spin in (self.sp_flow_diameter, self.sp_flow_width, self.sp_flow_height,
                     self.sp_flow_thickness, self.sp_flow_m3h, self.sp_flow_max_m3h,
                     self.sp_flow_max_pa, self.sp_free_area):
            spin.valueChanged.connect(self._on_flow_edited)
        widget.hide()
        return widget

    # ── Cut Surface interface of AssemblyView3D (plane and handles of the section view) ──
    @property
    def assembly(self):
        return self.study

    def section_editing(self):
        section = self.study.get("section")
        return bool(section and section.get("active")) and self.chk_cut_show_plane.isChecked()

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
            section = self.study.get("section")
            self.study.clear()
            self.study.update(restored)
            self.study["section"] = section
            set_active_study(self.study)
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
        for component in self.study["components"]:      # the study keeps copies of user materials
            for body in component["bodies"]:
                name = body.get("material")
                if name and name not in MATERIAL_THERMAL and name not in self.study["materials"] \
                        and user_material(name) is not None:
                    self.study["materials"][name] = copy.deepcopy(user_material(name))
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
        section = self.study.get("section")          # a view setting: not part of the history
        self._restoring = True
        try:
            self._history_index = index
            self.study.clear()
            self.study.update(restored)
            self.study["section"] = section
            set_active_study(self.study)
            if self._selected_contact is not None and \
                    self._selected_contact >= len(self.study["contacts"]):
                self._selected_contact = None
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
        self._refresh_flow_markers()
        self._schedule_section()
        self.view.update()

    def _refresh_flow_markers(self):
        """Flow-direction arrows of the fans and vents."""
        markers = []
        for component in self.study["components"]:
            if component.get("kind") not in FLOW_KINDS or not component["bodies"]:
                continue
            entry = component["bodies"][0].get("_render")
            if entry is None:
                continue
            axis = cad.assembly_world_direction(component, (0.0, 0.0, 1.0))
            markers.append((entry, axis / max(float(np.linalg.norm(axis)), 1e-12), component["name"]))
        self.view.flow_markers = markers

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
            entry.pop("_cap", None)
        if component is self._selected:
            self._update_gizmo()
        if component.get("kind") in FLOW_KINDS:
            self._refresh_flow_markers()
        if getattr(self, "_ready", False):
            self._schedule_section()

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
                self._worker is None and self.view.pick_mode is None and \
                not (hasattr(self, "chk_cut_show_plane") and self.section_editing()):
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
        profile = body.get("profile")
        if profile is not None and profile["kind"] == "table":
            parts.append("power table")
        elif body.get("power_w", 0.0) > 0:
            parts.append(f"{body['power_w']:g} W" +
                         ("" if profile is None else
                          {"duty": ", on / off", "ramp": ", ramp"}.get(profile["kind"], "")))
        if body.get("thermostat") is not None and body_has_power(body):
            parts[-1] += ", " + thermostat_text(body["thermostat"], short=True)
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
            flow_icon = _icon("air_flow_on.png")
            for ci, component in enumerate(self.study["components"]):
                bodies = component["bodies"]
                single = len(bodies) == 1
                if component.get("kind") in FLOW_KINDS:
                    flow = flow_settings(component.get("flow"), component["kind"])
                    item = QTreeWidgetItem([f"{component['name']}   [{component['kind'].capitalize()}: "
                                            f"{flow_text(component['kind'], flow)}]"])
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(0, Qt.Checked if component["visible"] else Qt.Unchecked)
                    item.setData(0, Qt.UserRole, ci)
                    item.setData(0, Qt.UserRole + 1, -1)
                    item.setIcon(0, flow_icon)
                    item.setForeground(0, QColor.fromRgbF(*FLOW_COLOR))
                    root_item.addChild(item)
                    if component is self._selected and self._selected_condition is None and \
                            self._selected_probe is None and self._selected_contact is None:
                        selected_item = item
                    continue
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
                if component["kind"] in PRIMITIVE_KINDS or any(body_has_power(b) for b in bodies):
                    item.setIcon(0, heat_icon)
                if component.get("zone"):
                    item.setToolTip(0, "Heat zone inside another body (in its material)")
                if not single:
                    for bi, body in enumerate(bodies):
                        child = QTreeWidgetItem([self._body_text(body)])
                        child.setData(0, Qt.UserRole, ci)
                        child.setData(0, Qt.UserRole + 1, bi)
                        if not body.get("material") and not body.get("fused"):
                            child.setForeground(0, QColor("#ffb454"))
                        if body_has_power(body):
                            child.setIcon(0, heat_icon)
                        item.addChild(child)
                        if component is self._selected and self._selected_body == bi:
                            selected_item = child
                root_item.addChild(item)
                if component is self._selected and (single or self._selected_body is None) and \
                        self._selected_condition is None and self._selected_probe is None and \
                        self._selected_contact is None:
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
            if self.study["contacts"]:
                group = QTreeWidgetItem(["Contacts"])
                group.setFont(0, group_font)
                group.setIcon(0, _icon("fit_together.png"))
                root_item.addChild(group)
                for index, contact in enumerate(self.study["contacts"]):
                    text = self._contact_item_text(contact)
                    item = QTreeWidgetItem([text])
                    item.setToolTip(0, text)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(0, Qt.Checked if contact.get("enabled", True) else Qt.Unchecked)
                    item.setData(0, Qt.UserRole + 2, f"contact:{index}")
                    if not self._contact_bodies_exist(contact):
                        item.setForeground(0, QColor("#ff6b6b"))
                    group.addChild(item)
                    if index == self._selected_contact:
                        selected_item = item
            if self.study["probes"]:
                group = QTreeWidgetItem(["Probes"])
                group.setFont(0, group_font)
                group.setIcon(0, _icon("measurement.png"))
                root_item.addChild(group)
                for index, probe in enumerate(self.study["probes"]):
                    where = f"on {probe['component']}" if probe.get("component") else "in the air"
                    item = QTreeWidgetItem([f"{probe['name']}   {where}"])
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
                 "flux": lambda c: (f"{c['power_w']:g} W" if c["flux_mode"] == "total"
                                    else f"{c['flux_w_m2']:g} W/m²") +
                 ("" if not c.get("profile") else
                  {"duty": ", on / off", "ramp": ", ramp", "table": ", table"}.get(c["profile"]["kind"], "")) +
                 ("" if not c.get("thermostat") else ", " + thermostat_text(c["thermostat"], short=True)),
                 "convection": lambda c: f"h {c['h']:g}" + ("" if c["fluid_ambient"]
                                                           else f", air {c['fluid_c']:g} °C"),
                 "insulated": lambda c: ""}[kind](condition)
        count = len(condition["faces"])
        return (f"{condition['name']}   [{CONDITION_KINDS[kind][0]}" + (f", {value}" if value else "")
                + f"]   {count} face{'s' if count != 1 else ''}" + ("" if count else "  ⚠"))

    def _body_label(self, ref):
        """'Part' or 'Part / Body' of a (part name, body index) reference."""
        component = next((c for c in self.study["components"] if c["name"] == ref[0]), None)
        if component is None or not 0 <= ref[1] < len(component["bodies"]):
            return f"{ref[0]} (missing)"
        if len(component["bodies"]) == 1:
            return component["name"]
        return f"{component['name']} / {component['bodies'][ref[1]]['name']}"

    def _contact_bodies_exist(self, contact):
        return all(not self._body_label(contact[side]).endswith("(missing)") for side in ("a", "b"))

    def _contact_item_text(self, contact):
        return (f"{contact['name']}   [{contact_text(contact)}]   {self._body_label(contact['a'])} ↔ "
                f"{self._body_label(contact['b'])}")

    @staticmethod
    def _item_extra(item):
        """("condition" | "probe" | "contact", index) of such an item, else (None, None)."""
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
        if kind == "contact":
            self._select_contact(index, update_tree=False)
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
        if kind == "contact" and 0 <= index < len(self.study["contacts"]):
            self.study["contacts"][index]["enabled"] = item.checkState(0) == Qt.Checked
            self._changed()
            if index == self._selected_contact:
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
            elif kind == "contact":
                self._select_contact(index)
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
        power = menu.addAction("Set power…") if self._body() is not None and \
            component.get("kind") not in FLOW_KINDS else None
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
        self._selected_contact = None
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
        self._selected_contact = index if kind == "contact" else None
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

    def _select_contact(self, index, update_tree=True):
        if 0 <= index < len(self.study["contacts"]):
            self._select_extra("contact", index, update_tree)

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
        if self._selected_contact is not None:
            self._props_stack.setCurrentIndex(3)
            self._show_contact()
            return
        self._props_stack.setCurrentIndex(0)
        component, body = self._selected, self._body()
        flow_item = component is not None and component.get("kind") in FLOW_KINDS
        self.w_body.setVisible(not flow_item)
        self.w_flow.setVisible(flow_item)
        if flow_item:
            self._show_flow_item(component)
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
            for widget in (self.cb_material, self.sp_power, self.chk_emissivity, self.profile_body,
                           self.thermostat_body):
                widget.setEnabled(body is not None)
            if body is None or flow_item:
                self.lbl_body.setText("Select a body of the part" if component is not None else "—")
                self.cb_material.setCurrentIndex(0)
                self.lbl_material.setText("")
                self.sp_power.setValue(0.0)
                self.lbl_power.setText("")
                self.profile_body.set_profile(None, 0.0)
                self.thermostat_body.set_thermostat(None, [])
                self.chk_emissivity.setChecked(False)
                self.sp_emissivity.setEnabled(False)
                return
            self.lbl_body.setText(body["name"] + ("  (inside a Bridge, not simulated)"
                                                  if body.get("fused") else ""))
            material = body.get("material")
            _fill_material_combo(self.cb_material, material)
            note = material_note(material)
            self.lbl_material.setText(material_summary(material) + ("\n" + note if note else ""))
            self.lbl_material.setStyleSheet("color:#aab2bd; font-size:9px;" if material else
                                            "color:#ffb454; font-size:9px;")
            power = float(body.get("power_w", 0.0))
            self.sp_power.setValue(power)
            profile = body.get("profile")
            self.profile_body.set_profile(profile, power)
            self.thermostat_body.set_thermostat(body.get("thermostat"),
                                                [p["name"] for p in self.study["probes"]])
            table = profile is not None and profile["kind"] == "table"
            self.sp_power.setEnabled(not table)
            volume = body_volume_mm3(body)
            if table:
                self.lbl_power.setText(f"Power from the table: up to "
                                       f"{profile_peak_power(profile, power):.4g} W")
            elif power > 0 and volume > 0:
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
        name = body["material"]
        if name and name not in MATERIAL_THERMAL and name not in self.study["materials"] and \
                user_material(name) is not None:   # the study keeps its own copy
            self.study["materials"][name] = copy.deepcopy(user_material(name))
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

    def _on_body_thermostat_changed(self):
        body = self._body()
        if self._loading or body is None:
            return
        body["thermostat"] = self.thermostat_body.thermostat()
        self._rebuild_tree()
        self._changed()

    def _on_body_profile_changed(self):
        body = self._body()
        if self._loading or body is None:
            return
        body["profile"] = self.profile_body.profile()
        self._show_properties()
        self._rebuild_tree()
        self._changed()

    # ── fans and vents ────────────────────────────────────────────────────────
    def _show_flow_item(self, component):
        kind = component["kind"]
        flow = flow_settings(component.get("flow"), kind)
        self._loading = True
        try:
            self.lbl_flow_header.setText(f"<b>{'Fan' if kind == 'fan' else 'Vent / opening'}</b>")
            self.cb_flow_form.setCurrentIndex(0 if flow["form"] == "disc" else 1)
            self.sp_flow_diameter.setValue(flow["diameter"])
            self.sp_flow_width.setValue(flow["width"])
            self.sp_flow_height.setValue(flow["height"])
            self.sp_flow_thickness.setValue(flow["thickness"])
            self.cb_flow_mode.setCurrentIndex(0 if flow["flow_mode"] == "fixed" else 1)
            self.sp_flow_m3h.setValue(flow["flow_m3h"])
            self.sp_flow_max_m3h.setValue(flow["max_m3h"])
            self.sp_flow_max_pa.setValue(flow["max_pa"])
            self.sp_free_area.setValue(flow["free_area"])
            for form, widgets in self._flow_size_rows.items():
                for widget in widgets:
                    if widget is not None:
                        widget.setVisible(form == flow["form"])
            fan = kind == "fan"
            for key, widgets in self._fan_rows.items():
                for widget in widgets:
                    if widget is not None:
                        widget.setVisible(fan and (key == "mode" or key == flow["flow_mode"]))
            for widget in self._vent_rows:
                if widget is not None:
                    widget.setVisible(not fan)
            area = (flow["width"] * flow["height"] if flow["form"] == "rect"
                    else math.pi * flow["diameter"] ** 2 / 4.0) * 1e-6
            if fan and flow["flow_mode"] == "fixed":
                info = f"Mean air speed through the fan ≈ {flow['flow_m3h'] / 3600.0 / area:.2f} m/s."
            elif fan:
                flows, pressures = flow_curve(flow)
                info = (f"Curve from {pressures[0]:g} Pa (no flow) to {flows[-1]:g} m³/h (free "
                        "blowing)" + (f", {len(flow['curve'])} points from CSV" if flow["curve"]
                                      else ", straight line") + ".")
            else:
                info = (f"Pressure-loss coefficient ζ ≈ {vent_loss_coefficient(flow['free_area']):.3g} "
                        "(Δp = ζ·ρv²/2).")
            self.lbl_flow_info.setText(info)
        finally:
            self._loading = False

    def _on_flow_edited(self, *_args):
        component = self._selected
        if self._loading or component is None or component.get("kind") not in FLOW_KINDS:
            return
        flow = flow_settings(component.get("flow"), component["kind"])
        flow.update(form=self.cb_flow_form.currentData(), diameter=self.sp_flow_diameter.value(),
                    width=self.sp_flow_width.value(), height=self.sp_flow_height.value(),
                    thickness=self.sp_flow_thickness.value(),
                    flow_mode=self.cb_flow_mode.currentData(), flow_m3h=self.sp_flow_m3h.value(),
                    max_m3h=self.sp_flow_max_m3h.value(), max_pa=self.sp_flow_max_pa.value(),
                    free_area=self.sp_free_area.value())
        if (self.sp_flow_max_m3h.hasFocus() or self.sp_flow_max_pa.hasFocus()) and flow["curve"]:
            flow["curve"] = []                     # typed datasheet values replace a CSV curve
        flow_set_shape(component, flow)
        self._selected_body = 0
        self._rebuild_scene()
        self._rebuild_tree()
        self._show_properties()
        self._changed()

    def _import_fan_curve(self):
        component = self._selected
        if component is None or component.get("kind") != "fan":
            return
        path, _ = QFileDialog.getOpenFileName(self, "Fan curve (CSV)", self._start_directory(),
                                              "CSV / text files (*.csv *.txt *.dat);;All files (*)")
        if not path:
            return
        try:
            points = read_profile_csv(path)              # same two-column format
        except (OSError, ValueError) as ex:
            QMessageBox.warning(self, "Fan curve", f"Could not read\n{path}\n\n{ex}")
            return
        if len(points) < 2:
            QMessageBox.warning(self, "Fan curve", "The curve needs at least two points.")
            return
        flow = flow_settings(component.get("flow"), "fan")
        flow["curve"] = points
        flow["flow_mode"] = "curve"
        component["flow"] = flow
        self._show_properties()
        self._rebuild_tree()
        self._changed()

    def _add_flow_item(self, kind):
        if self._worker is not None:
            return
        anchor = cad.assembly_bounds(self._selected) if self._selected is not None and \
            self._selected.get("kind") not in FLOW_KINDS else None
        dialog = FlowItemDialog(self, kind, anchor, self._selected["name"] if anchor else None)
        if dialog.exec_() != QDialog.Accepted:
            return
        name, flow, position, rotation = dialog.values()
        component = study_add_flow_item(self.study, kind, flow, name, position, rotation)
        self._after_components_changed(select=component)
        if self.study["settings"].get("mode") != "airflow":
            self._set_status(f"Placed {component['name']} — fans and vents act in the Airflow + "
                             "heat solver (Simulation panel → Solver)")
        else:
            self._set_status(f"Placed {component['name']} — move or turn it with the arrows / rings")

    def _on_streamlines_toggled(self, on):
        self._update_streamlines()
        self.view.update()

    # ── contacts ──────────────────────────────────────────────────────────────
    def _contact(self):
        index = self._selected_contact
        if index is None or not 0 <= index < len(self.study["contacts"]):
            return None
        return self.study["contacts"][index]

    def _unique_contact_name(self, base):
        names = {c["name"] for c in self.study["contacts"]}
        name, number = base, 2
        while name in names:
            name = f"{base} ({number})"; number += 1
        return name

    def _detect_contacts(self):
        if self._worker is not None:
            return
        jobs = study_jobs(self.study)
        missing = [job["name"] for job in jobs if job["props"] is None]
        if len(jobs) < 2:
            self._set_status("Contacts need at least two visible bodies")
            return
        if missing:
            QMessageBox.information(self, "Detect touching parts", "Choose the materials first "
                                    "(" + ", ".join(missing[:5]) + ").")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            found = detect_contacts(jobs, self.study["settings"])
        except ThermalError as ex:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Detect touching parts", str(ex))
            return
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
        components = self.study["components"]
        existing = {frozenset((tuple(c["a"]), tuple(c["b"]))) for c in self.study["contacts"]}
        added = 0
        for a, b, area in found:
            refs = []
            for index in (a, b):
                ci, bi = jobs[index]["key"]
                refs.append((components[ci]["name"], bi))
            if frozenset(refs) in existing:
                continue
            contact = contact_new(self._unique_contact_name(
                f"{self._body_label(refs[0]).split(' / ')[-1]} – {self._body_label(refs[1]).split(' / ')[-1]}"),
                refs[0], refs[1])
            contact["area_mm2"] = float(area)
            self.study["contacts"].append(contact)
            existing.add(frozenset(refs))
            added += 1
        if not found:
            self._set_status("No touching bodies found (at the current cell size)")
            return
        self._rebuild_tree()
        self._changed()
        self._set_status(f"{len(found)} touching pair{'s' if len(found) != 1 else ''} found, "
                         f"{added} new contact{'s' if added != 1 else ''} added (bonded — choose a "
                         "type in the properties)")
        if added:
            self._select_contact(len(self.study["contacts"]) - 1)

    def _add_contact(self):
        if self._worker is not None:
            return
        bodies = []
        for component in self.study["components"]:
            if component.get("kind") in FLOW_KINDS:
                continue
            for bi, body in enumerate(component["bodies"]):
                if not body.get("fused"):
                    bodies.append((self._body_label((component["name"], bi)), (component["name"], bi)))
        if len(bodies) < 2:
            self._set_status("Contacts need at least two bodies")
            return
        dialog = ContactDialog(self, bodies)
        if dialog.exec_() != QDialog.Accepted:
            return
        a, b = dialog.values()
        if a == b:
            self._set_status("Choose two different bodies")
            return
        contact = contact_new(self._unique_contact_name(
            f"{self._body_label(a).split(' / ')[-1]} – {self._body_label(b).split(' / ')[-1]}"), a, b)
        contact["kind"] = "layer"
        contact["thickness_mm"], contact["k"] = TIM_PRESETS[1][1], TIM_PRESETS[1][2]
        self.study["contacts"].append(contact)
        self._select_contact(len(self.study["contacts"]) - 1)
        self._changed()

    def _show_contact(self):
        contact = self._contact()
        if contact is None:
            return
        self._loading = True
        try:
            self.lbl_contact_name.setText(contact["name"])
            self.lbl_contact_bodies.setText(f"{self._body_label(contact['a'])}\n↔ "
                                            f"{self._body_label(contact['b'])}")
            self.chk_contact_enabled.setChecked(contact.get("enabled", True))
            self.cb_contact_kind.setCurrentIndex(max(0, self.cb_contact_kind.findData(contact["kind"])))
            self.sp_contact_resistance.setValue(contact["resistance"])
            self.sp_contact_thickness.setValue(contact["thickness_mm"])
            self.sp_contact_k.setValue(contact["k"])
            preset = next((i + 1 for i, (_n, t, k) in enumerate(TIM_PRESETS)
                           if abs(t - contact["thickness_mm"]) < 1e-9 and abs(k - contact["k"]) < 1e-9), 0)
            self.cb_tim_preset.setCurrentIndex(preset)
            for kind, widgets in self._contact_rows.items():
                for widget in widgets:
                    if widget is not None:
                        widget.setVisible(kind == contact["kind"])
            resistance = contact_resistance(contact)
            area = contact.get("area_mm2")
            result = next((entry for entry in (self._result or {}).get("contacts", [])
                           if entry["name"] == contact["name"]), None)
            if result is not None and result["faces"]:
                area = result["area_mm2"]
            if contact["kind"] == "bonded":
                info = "No resistance: the bodies share their temperature where they touch."
            else:
                info = f"R = {resistance:.3g} m²K/W = {resistance * 1e4:.3g} K·cm²/W"
                if area:
                    info += f"; over {area:.4g} mm² that is {resistance / (area * 1e-6):.3g} K/W"
            if result is not None and result["faces"]:
                info += (f"\nLast result: {abs(result['heat']):.3g} W cross the contact" +
                         (f", step ≈ {abs(result['jump']):.3g} K" if contact["kind"] != "bonded"
                          else ""))
            self.lbl_contact_info.setText(info)
        finally:
            self._loading = False

    def _on_contact_edited(self, *_args):
        contact = self._contact()
        if self._loading or contact is None:
            return
        contact["enabled"] = self.chk_contact_enabled.isChecked()
        contact["kind"] = self.cb_contact_kind.currentData()
        contact["resistance"] = float(self.sp_contact_resistance.value())
        contact["thickness_mm"] = float(self.sp_contact_thickness.value())
        contact["k"] = max(1e-4, float(self.sp_contact_k.value()))
        self._show_contact()
        self._rebuild_tree()
        self._changed()

    def _on_tim_preset(self, _index):
        values = self.cb_tim_preset.currentData()
        if self._loading or values is None:
            return
        self._loading = True
        self.sp_contact_thickness.setValue(values[0])
        self.sp_contact_k.setValue(values[1])
        self._loading = False
        self._on_contact_edited()

    # ── Heat Source Extrusion ─────────────────────────────────────────────────
    def _start_extrusion(self):
        if self._worker is not None:
            return
        if not any(c.get("kind") not in FLOW_KINDS for c in self.study["components"]):
            self._set_status("Import a part or place a block first, then extrude a heat source "
                             "from one of its faces")
            return
        self._selected_condition = None
        self._set_pick_mode("extrude")

    def _extrude_on_face(self, component, body_index, face):
        body = component["bodies"][body_index]
        if not body["faces"][face].get("planar"):
            self._set_status("Choose a flat face for the extrusion")
            return
        self._set_pick_mode(None)
        frame = face_frame(component, body, face)
        area = face_signature(body, face)[6]
        normal = frame[3]
        number = 1 + sum(1 for c in self.study["components"] if c["kind"] in PRIMITIVE_KINDS)
        text = (f"Face of {component['name']}: {area:.4g} mm², normal "
                f"({normal[0]:+.2f}, {normal[1]:+.2f}, {normal[2]:+.2f})")
        dialog = ExtrusionDialog(self, f"Heat source {number}", text, body.get("material"))
        if dialog.exec_() != QDialog.Accepted:
            return
        kind, shape, name, offset_u, offset_v, height, inward, power, material = dialog.values()
        position, rotation = extrusion_placement(frame, offset_u, offset_v, height, inward)
        new = study_add_primitive(self.study, kind, shape, name, position, power, material,
                                  rotation=rotation, zone=inward,
                                  source=f"Extrusion on {component['name']}" +
                                         (" (heat zone)" if inward else ""))
        self._after_components_changed(select=new)
        self._set_status(f"Placed {new['name']} ({power:g} W) " +
                         (f"as a heat zone inside {component['name']}" if inward else
                          f"on {component['name']}"))

    # ── section view ──────────────────────────────────────────────────────────
    def _section_size(self):
        boxes = [cad.assembly_bounds(c) for c in self.study["components"] if c["visible"]]
        boxes = [box for box in boxes if box is not None]
        if not boxes:
            return 100.0
        low = np.min([box[0] for box in boxes], axis=0)
        high = np.max([box[1] for box in boxes], axis=0)
        return max(float(np.linalg.norm(high - low)) * 1.3, 10.0)

    def _on_section_toggled(self, on):
        if self._loading:
            return
        section = self.study.get("section")
        if on and section is None:
            boxes = [cad.assembly_bounds(c) for c in self.study["components"] if c["visible"]]
            boxes = [box for box in boxes if box is not None]
            centre = ((np.min([box[0] for box in boxes], axis=0) +
                       np.max([box[1] for box in boxes], axis=0)) / 2.0) if boxes else np.zeros(3)
            # a vertical plane across the view; the half towards the viewer is hidden
            rx, rz = math.radians(self.view.rot_x), math.radians(self.view.rot_z)
            Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
            Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
            viewer = (Rx @ Rz).T @ np.array([0.0, 0.0, 1.0])
            rotation = [90.0, 0.0, 0.0] if abs(viewer[1]) >= abs(viewer[0]) else [0.0, 90.0, 0.0]
            normal = cad.cut_surface_frame(centre, rotation)[3]
            section = self.study["section"] = {
                "position": [round(float(x), 3) for x in centre], "rotation": rotation,
                "flip": bool(normal @ viewer < 0), "size": 100.0, "active": False}
        if section is not None:
            section["active"] = bool(on)
            if on:
                section["size"] = self._section_size()
        self._sync_section_ui()
        self._update_section()
        self._update_gizmo()
        self._update_title()
        if on:
            self._set_status("Section view: move / turn the plane with the arrows and rings or the "
                             "fields in the Simulation panel")

    def _sync_section_ui(self):
        section = self.study.get("section")
        active = bool(section and section.get("active"))
        self.act_section.blockSignals(True)
        self.act_section.setChecked(active)
        self.act_section.blockSignals(False)
        self.w_section.setVisible(active)
        airflow = self._result is not None and self._result.get("mode") == "airflow"
        for widget in (self.lbl_section_field, self.cb_section_field, self.chk_section_arrows):
            widget.setVisible(airflow)
        if section:
            self._loading = True
            try:
                for spins, key in ((self.sp_cut_position, "position"),
                                   (self.sp_cut_rotation, "rotation")):
                    for spin, value in zip(spins, section[key]):
                        spin.setValue(float(value))
                self.chk_cut_flip.setChecked(bool(section.get("flip")))
            finally:
                self._loading = False

    def _on_section_field_changed(self, *_args):
        section = self.study.get("section")
        if self._loading or section is None or not section.get("active"):
            return
        section["position"] = [spin.value() for spin in self.sp_cut_position]
        section["rotation"] = [spin.value() for spin in self.sp_cut_rotation]
        section["flip"] = self.chk_cut_flip.isChecked()
        self.view.section = section          # the plane follows at once, the caps shortly after
        self.view.update()
        self._schedule_section()

    def _schedule_section(self):
        if getattr(self, "_section_timer", None) is not None:
            self._section_timer.start()

    def _update_section(self):
        """Cut faces of the bodies at the section plane: temperature colours while results are
        shown (and the air of an airflow result), else the body colour."""
        section = self.study.get("section")
        active = bool(section and section.get("active"))
        self.view.section = section if active else None
        self.view.section_caps = []
        self.view.section_arrows = None
        if not active:
            self.view.update()
            return
        origin, u, v, normal = cad.cut_surface_frame(section["position"], section["rotation"])
        hidden = -normal if section.get("flip") else normal
        key = (tuple(np.round(origin, 6)), tuple(np.round(hidden, 9)))
        result = self._result if self.view.result_mode else None
        if result is not None and result.get("mode") == "airflow":
            self._update_airflow_section(result, origin, u, v, hidden)
            return
        low_high = self._color_range() if result is not None else None
        caps = []
        for ci, component in enumerate(self.study["components"]):
            if not component["visible"] or component.get("kind") in FLOW_KINDS:
                continue
            for bi, body in enumerate(component["bodies"]):
                entry = body.get("_render")
                if entry is None or "solid" not in entry or body.get("fused"):
                    continue
                cached = entry.get("_cap")
                if cached is None or cached[0] != key:
                    try:
                        mesh = cad.cut_mesh_by_plane((entry["solid"][0], entry["solid"][1],
                                                      body["faces"]), origin, hidden, u, v)
                    except Exception:
                        cad._trace("Thermal section cut failed:\n" + traceback.format_exc())
                        mesh = ([], [], [])
                    cap_faces = [face for face in mesh[2] if face.get("section") is not None]
                    vertices = np.asarray(mesh[0], float).reshape(-1, 3)
                    triangles = np.asarray(mesh[1], np.int64).reshape(-1, 3)
                    tris = np.concatenate([np.asarray(f["tris"], np.int64) for f in cap_faces]) \
                        if cap_faces else np.zeros(0, np.int64)
                    cached = (key, vertices, triangles[tris] if len(tris) else np.zeros((0, 3), np.int64),
                              None)
                    entry["_cap"] = cached
                _key, vertices, triangles, _extra = cached
                if not len(triangles):
                    continue
                surface = None if result is None else result["surface"].get((ci, bi))
                if surface is None:
                    color = np.asarray(body["color"][:3], float) * 0.62 + 0.08
                    caps.append((entry, vertices, triangles, color))
                    continue
                weights = entry.get("_cap_weights")
                if weights is None or weights[0] != (key, id(result)):
                    fine, fine_tris = refine_triangles(vertices, triangles, result["cell_mm"])
                    index, w = result_point_weights(result, fine)
                    weights = ((key, id(result)), fine, fine_tris, index, w)
                    entry["_cap_weights"] = weights
                _k, fine, fine_tris, index, w = weights
                values = result_volume_values(result, self._frame, index, w)
                caps.append((entry, fine, fine_tris, temperature_colors(values, *low_high)))
        self.view.section_caps = caps
        self.view.update()

    def _update_airflow_section(self, result, origin, u, v, hidden):
        """Section through an airflow result: the plane sampled across the whole domain (air
        and solids), coloured by temperature or air speed, with velocity arrows in the air."""
        grid = result["grid"]
        faces = grid["faces"]
        low = np.array([f[0] for f in faces])
        high = np.array([f[-1] for f in faces])
        corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1])
                            for z in (low[2], high[2])])
        cu, cv = (corners - origin) @ u, (corners - origin) @ v
        span = max(float(np.ptp(cu)), float(np.ptp(cv)), 1e-6)
        step = max(grid["h_mm"] * 0.5, span / 240.0)
        au = np.arange(cu.min(), cu.max() + step, step)
        av = np.arange(cv.min(), cv.max() + step, step)
        A, B = np.meshgrid(au, av, indexing="ij")
        points = origin + A[..., None] * u + B[..., None] * v
        tolerance = 1e-6
        inside = np.all((points >= low - tolerance) & (points <= high + tolerance), axis=-1)
        speed_mode = self.cb_section_field.currentData() == "speed"
        flat = points.reshape(-1, 3)
        valid = inside.ravel()
        values = np.zeros(len(flat))
        velocity_field, speed_field = result_velocity(result, self._frame)
        if valid.any():
            index, weights = _cell_weights(flat[valid], grid, None)
            if speed_mode:
                values[valid] = (speed_field.ravel()[index] * weights).sum(1)
            else:
                values[valid] = result_volume_values(result, self._frame, index, weights)
        nu_, nv_ = A.shape
        ids = np.arange(nu_ * nv_).reshape(nu_, nv_)
        quad = valid.reshape(nu_, nv_)
        ok = quad[:-1, :-1] & quad[1:, :-1] & quad[:-1, 1:] & quad[1:, 1:]
        a, b, c, d = ids[:-1, :-1][ok], ids[1:, :-1][ok], ids[1:, 1:][ok], ids[:-1, 1:][ok]
        triangles = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
        if speed_mode:
            top = max(float(result["speed_max"]), 1e-6)
            colors = temperature_colors(values, 0.0, top)
            self.legend.set_range(0.0, top, "Section: air speed; parts: temperature",
                                  title="v  [m/s]")
        else:
            low_t, high_t = self._color_range()
            colors = temperature_colors(values, low_t, high_t)
            self.legend.set_range(low_t, high_t)
        caps = [(None, flat.astype(np.float32), triangles, colors)] if len(triangles) else []
        arrows = None
        if self.chk_section_arrows.isChecked() and valid.any():
            stride = max(1, int(round(max(nu_, nv_) / 30)))
            sample = np.zeros((nu_, nv_), bool)
            sample[::stride, ::stride] = True
            chosen = np.flatnonzero(sample.ravel() & valid)
            if len(chosen):
                index, weights = _cell_weights(flat[chosen], grid, None)
                fluid = result["fluid"].ravel()
                in_air = fluid[index[np.arange(len(chosen)), np.argmax(weights, 1)]]
                velocity = velocity_field.reshape(-1, 3)
                vel = (velocity[index] * weights[..., None]).sum(1)
                planar = vel - np.outer(vel @ hidden, hidden)
                magnitude = np.linalg.norm(planar, axis=1)
                top = max(float(magnitude[in_air].max()) if in_air.any() else 0.0, 1e-9)
                keep = in_air & (magnitude > 0.03 * top)
                start = flat[chosen][keep] - hidden * step * 0.2
                direction = planar[keep] / magnitude[keep, None]
                length = step * stride * 0.9 * np.sqrt(magnitude[keep] / top)[:, None]
                tip = start + direction * length
                side = np.cross(hidden, direction)
                head = 0.3 * length
                lines = np.concatenate([np.stack([start, tip], 1),
                                        np.stack([tip, tip - direction * head + side * head * 0.5], 1),
                                        np.stack([tip, tip - direction * head - side * head * 0.5], 1)])
                arrows = (lines.reshape(-1, 3).astype(np.float32),
                          np.full((lines.shape[0] * 2, 3), 0.95, np.float32))
        self.view.section_caps = caps
        self.view.section_arrows = arrows
        self.view.update()

    def _update_streamlines(self):
        """Paths of the air of an airflow result, coloured by the air temperature."""
        result = self._result
        show = result is not None and result.get("mode") == "airflow" and \
            self.act_streamlines.isChecked() and self.view.result_mode
        if not show:
            self.view.streamlines = []
            return
        cache = result.setdefault("_streamlines", {})
        kept = np.asarray(result.get("velocity_index", [0]))
        slot = int(np.argmin(np.abs(kept - self._frame))) if len(kept) > 1 else 0
        lines = cache.get(slot)
        if lines is None:
            if self._play_timer.isActive() and self.view.streamlines:
                return                         # playing: keep the lines until it stops
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                lines = cache[slot] = airflow_streamlines(result, frame=self._frame)
            finally:
                QApplication.restoreOverrideCursor()
        low, high = self._color_range()
        # lifted towards white, so lines in ambient-coloured air stay visible on the background
        self.view.streamlines = [(points, 0.55 * temperature_colors(values, low, high) + 0.45)
                                 for points, values in lines]

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
                self.profile_condition.set_profile(condition.get("profile"), condition["power_w"]
                                                   if total else condition["flux_w_m2"])
                self.thermostat_condition.set_thermostat(
                    condition.get("thermostat"), [p["name"] for p in self.study["probes"]],
                    owner="heated bodies")
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
        if condition["kind"] == "flux":
            condition["profile"] = self.profile_condition.profile()
            condition["thermostat"] = self.thermostat_condition.thermostat()
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
            self._set_status("Click on a part (or on a cut face of the section view) to place a "
                             "probe (Esc ends)")
        elif mode == "extrude":
            self._set_status("Heat Source Extrusion: click the flat face the heat source stands on "
                             "(Esc cancels)")
        self.view.update()

    def _on_face_picked(self, entry, face):
        component = entry.get("_thermal_component")
        body_index = entry.get("_body_index")
        if self.view.pick_mode == "extrude":
            if component is not None and body_index is not None and \
                    component.get("kind") not in FLOW_KINDS:
                self._extrude_on_face(component, body_index, face)
            return
        condition = self._condition()
        if condition is None or component is None or body_index is None or \
                component.get("kind") in FLOW_KINDS:
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
        component = entry.get("_thermal_component") if entry is not None else None
        body_index = entry.get("_body_index") if entry is not None else None
        if entry is not None and (component is None or body_index is None or
                                  component.get("kind") in FLOW_KINDS):
            return
        names = {p["name"] for p in self.study["probes"]}
        number = 1
        while f"P{number}" in names:
            number += 1
        if component is None:                 # on the section through the air
            probe = {"name": f"P{number}", "component": None, "body": 0,
                     "point": [float(x) for x in point]}
        else:
            local = cad.assembly_local_point(component, point)
            probe = {"name": f"P{number}", "component": component["name"],
                     "body": int(body_index), "point": [float(x) for x in local]}
        self.study["probes"].append(probe)
        self._changed(results=False)          # probes read the existing results
        self._rebuild_tree()
        self._refresh_overlays()
        self._fill_probe_table()
        self._update_plot()
        self._set_status(f"Probe P{number} placed " +
                         (f"on {component['name']}" if component is not None else "in the air") +
                         " — click for more, Esc ends")

    def _probe_series(self, probe):
        """Temperatures of a probe for every stored frame of the result, or None
        (interpolated in the stored whole fields at the probe point)."""
        result = self._result
        if result is None:
            return None
        if probe.get("component") is not None:
            index = next((i for i, c in enumerate(self.study["components"])
                          if c["name"] == probe["component"]), None)
            if index is None or (index, probe["body"]) not in result["surface"]:
                return None                   # the part was not simulated
        point = probe_world_point(self.study, probe)
        if point is None:
            return None
        key = (id(result), probe.get("component"), probe["body"], tuple(probe["point"]),
               tuple(np.round(point, 6)))
        if key not in self._probe_cache:
            self._probe_cache[key] = result_point_series(result, point,
                                                         air=probe.get("component") is None)
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
            for column, text in enumerate((probe["name"], probe["component"] or "air",
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
        self.lbl_probe_part.setText(probe["component"] or "— (air, fixed point)")
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
        contact = self._contact()
        if condition is not None or self._selected_probe is not None or contact is not None:
            if contact is not None:
                item, items = contact, self.study["contacts"]
            elif condition is not None:
                item, items = condition, self.study["conditions"]
            else:
                item, items = self.study["probes"][self._selected_probe], self.study["probes"]
            name, ok = QInputDialog.getText(self, "Rename", "New name:", text=item["name"])
            name = name.strip()
            if not ok or not name or name == item["name"]:
                return
            if any(other["name"] == name for other in items):
                self._set_status(f"'{name}' is already used")
                return
            if items is self.study["probes"]:           # thermostats reading this probe follow
                owners = [b for c in self.study["components"] for b in c["bodies"]] + \
                    self.study["conditions"]
                for owner in owners:
                    thermostat = owner.get("thermostat")
                    if thermostat and thermostat["sensor"] == "probe" and \
                            thermostat["probe"] == item["name"]:
                        thermostat["probe"] = name
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
            airflow = settings["mode"] == "airflow"
            self.cb_mode.setCurrentIndex(1 if airflow else 0)
            self.lbl_mode_hint.setText(
                "Airflow + heat (CHT): air and solids are solved together on one grid — buoyancy, "
                "fans and vents decide where the heat goes. Takes minutes (transient runs longer)."
                if airflow
                else "Conduction (FEM): heat conduction in the bodies; air cooling and radiation act "
                     "on their surfaces. Fast.")
            self.sp_ambient.setValue(settings["ambient_c"])
            self.cb_convection.setCurrentIndex(max(0, self.cb_convection.findData(
                settings["convection"])))
            self.sp_h.setValue(settings["h"])
            self.sp_air_speed.setValue(settings["air_speed"])
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
            self.sp_flow_space.setValue(settings["flow_space"])
            self.sp_flow_below.setValue(settings["flow_below"])
            self.sp_flow_above.setValue(settings["flow_above"])
            for side, combo in self.cb_sides.items():
                combo.setCurrentIndex(max(0, combo.findData(settings["flow_sides"].get(side, "open"))))
            self.sp_inlet_speed.setValue(settings["inlet_speed"])
            self.sp_wall_temperature.setValue(settings["wall_c"])
            self.cb_turbulence.setCurrentIndex(max(0, self.cb_turbulence.findData(settings["turbulence"])))
            flow_auto = settings["flow_cell_mm"] <= 0
            self.chk_auto_flow_cell.setChecked(flow_auto)
            self.sp_flow_cell.setEnabled(not flow_auto)
            if not flow_auto:
                self.sp_flow_cell.setValue(settings["flow_cell_mm"])
            self.sp_flow_max_cells.setValue(int(settings["flow_max_cells"]))
            self.sp_flow_iterations.setValue(int(settings["flow_iterations"]))
            self.sp_flow_step_iterations.setValue(int(settings["flow_step_iterations"]))

            def show(widgets, on):
                for widget in widgets:
                    if widget is not None:
                        widget.setVisible(on)

            show(self._conduction_rows, not airflow)
            show(self._transient_rows, transient)
            show(self._speed_rows, not airflow and settings["convection"] == "forced")
            show(self._h_rows, not airflow and settings["convection"] == "fixed")
            show(self._mesh_rows, not airflow)
            show(self._airflow_rows, airflow)
            show(self._flow_steady_rows, airflow and not transient)
            show(self._flow_transient_rows, airflow and transient)
            sides = set(settings["flow_sides"].values())
            show(self._inlet_rows, airflow and "inlet" in sides)
            show(self._wall_rows, airflow and "wall_fixed" in sides)
            self.lbl_environment_hint.setText(
                "Gravity points along −Z. The air is simulated in a box around the parts (Air "
                "domain); radiation goes from the outer surfaces to the ambient." if airflow else
                "Gravity points along −Z. Surfaces that do not touch another body exchange heat "
                "with the ambient air; air enclosed in a housing conducts as still air.")
        finally:
            self._loading = False

    def _on_settings_edited(self, *_args):
        if self._loading:
            return
        settings = self.study["settings"]
        settings["mode"] = self.cb_mode.currentData()
        settings["ambient_c"] = float(self.sp_ambient.value())
        settings["convection"] = self.cb_convection.currentData()
        settings["h"] = float(self.sp_h.value())
        settings["air_speed"] = float(self.sp_air_speed.value())
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
        settings["flow_space"] = float(self.sp_flow_space.value())
        settings["flow_below"] = float(self.sp_flow_below.value())
        settings["flow_above"] = float(self.sp_flow_above.value())
        settings["flow_sides"] = {side: combo.currentData() for side, combo in self.cb_sides.items()}
        settings["inlet_speed"] = float(self.sp_inlet_speed.value())
        settings["wall_c"] = float(self.sp_wall_temperature.value())
        settings["turbulence"] = self.cb_turbulence.currentData()
        settings["flow_cell_mm"] = 0.0 if self.chk_auto_flow_cell.isChecked() else \
            float(self.sp_flow_cell.value())
        settings["flow_max_cells"] = int(self.sp_flow_max_cells.value())
        settings["flow_iterations"] = int(self.sp_flow_iterations.value())
        settings["flow_step_iterations"] = int(self.sp_flow_step_iterations.value())
        self._show_settings()
        self._changed()

    def _update_estimate(self):
        settings = self.study["settings"]
        if settings["mode"] == "airflow":
            try:
                plan = plan_flow_grid(study_jobs(self.study), settings)
            except Exception:
                plan = None
            if plan is None:
                self.lbl_flow_estimate.setText("No visible bodies yet.")
            else:
                self.lbl_flow_estimate.setText(
                    f"≈ {plan['cells']:,} cells ({plan['shape'][0]} × {plan['shape'][1]} × "
                    f"{plan['shape'][2]}), {plan['h_mm']:g} mm at the parts; domain "
                    f"{plan['size'][0]:.0f} × {plan['size'][1]:.0f} × {plan['size'][2]:.0f} mm"
                    .replace(",", " ") + (" (automatic)" if settings["flow_cell_mm"] <= 0 else "") +
                    (f". Transient: {settings['time_steps']} steps × up to "
                     f"{settings['flow_step_iterations']} iterations"
                     if settings["analysis"] == "transient" else ""))
                if settings["flow_cell_mm"] <= 0:
                    self.sp_flow_cell.blockSignals(True)
                    self.sp_flow_cell.setValue(plan["h_mm"])
                    self.sp_flow_cell.blockSignals(False)
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
        contact = self._contact()
        if condition is not None or self._selected_probe is not None or contact is not None:
            if contact is not None:
                items, index = self.study["contacts"], self._selected_contact
            elif condition is not None:
                items, index = self.study["conditions"], self._selected_condition
            else:
                items, index = self.study["probes"], self._selected_probe
            name = items[index]["name"]
            if self.view.pick_mode is not None:
                self._set_pick_mode(None)
            del items[index]
            self._selected_condition = self._selected_probe = self._selected_contact = None
            self._rebuild_tree()
            self._show_properties()
            self._refresh_overlays()
            self._fill_probe_table()
            self._changed(results=condition is not None or contact is not None)
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
        set_active_study(self.study)
        self._forget_sweep()
        self._window._thermal_file = None
        self._selected, self._selected_body = None, None
        self._selected_condition = self._selected_probe = self._selected_contact = None
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
        self._sync_section_ui()
        self._update_section()
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
        set_active_study(self.study)
        self._forget_sweep()
        self._window._thermal_file = path
        self._selected, self._selected_body = None, None
        self._selected_condition = self._selected_probe = self._selected_contact = None
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
        self._sync_section_ui()
        self._update_section()
        self.view.fit_all()
        self._set_status(f"Opened {path}")

    # ── simulation ────────────────────────────────────────────────────────────
    def _start_simulation(self):
        if self._worker is not None:
            return
        if not self._check_runnable():
            return
        jobs, settings, conditions, contacts, flow_items, notes = study_inputs(self.study)
        self.txt_messages.setPlainText("\n".join(notes))
        if self.view.pick_mode is not None:
            self._set_pick_mode(None)
        self._discard_results()
        worker = ThermalWorker(jobs, settings, conditions, contacts, flow_items)
        worker.progressed.connect(self._on_progress)
        worker.succeeded.connect(self._on_simulation_done)
        worker.failed.connect(self._on_simulation_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        self._set_running(True)
        self._set_status("Simulation started...")
        worker.start()

    def _check_runnable(self):
        """Ask for what is missing before a run (parts, materials, a heat source). True = ok."""
        jobs = study_jobs(self.study)
        if not jobs:
            QMessageBox.information(self, ADDON_TITLE, "There is nothing to simulate. Import parts "
                                    "or place a heat source first (hidden parts are left out).")
            return False
        missing = [job for job in jobs if job["props"] is None]
        if missing:
            ci, bi = missing[0]["key"]
            self._select(self.study["components"][ci], bi)
            names = "\n".join("•  " + job["name"] for job in missing[:10]) + \
                ("\n…" if len(missing) > 10 else "")
            QMessageBox.warning(self, "Material missing",
                                f"These bodies have no material yet:\n\n{names}\n\n"
                                "Choose a material in the properties (Body → Material).")
            return False
        settings = self.study["settings"]
        airflow = settings["mode"] == "airflow"
        conditions = study_conditions(self.study, jobs)
        drives = any(c["kind"] in ("flux", "fixed") and c["targets"] for c in conditions) or \
            (airflow and "wall_fixed" in settings["flow_sides"].values())
        if not any(profile_peak_power(job.get("profile"), job["power"]) > 0 for job in jobs) and \
                not drives:
            QMessageBox.warning(self, "No heat source",
                                "No body produces heat. Place a heat source (Heat Source → Block / "
                                "Cylinder / Extrusion), give a body a power in the properties, or "
                                "put a heat flux or fixed temperature on faces (Face Condition).")
            return False
        return True

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
                    self.act_update_all, self.act_condition, self.act_probe, self.act_contact,
                    self.act_flow, self.act_materials, self.act_sweep, self.act_export,
                    self.act_sweep_results):
            act.setEnabled(not running)
        self.act_cancel.setEnabled(running)
        self.act_results.setEnabled(not running and self._result is not None)
        self.act_sweep_results.setEnabled(not running and self._sweep_result is not None)
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
        self._result_note = ""
        self._show_result()
        hottest = max((b for b in result["bodies"] if b["max"] is not None),
                      key=lambda b: b["max"], default=None)
        self._set_status(f"Simulation finished in {result['seconds']:.1f} s — maximum "
                         f"{result['hot_value']:.1f} °C" +
                         (f" in {hottest['name']}" if hottest else ""))

    def _on_simulation_failed(self, text):
        if text:
            notes = self.txt_messages.toPlainText().strip()
            self.txt_messages.setPlainText(text + ("\n\n" + notes if notes else ""))
            QMessageBox.warning(self, "Simulation not possible", text)
            self._set_status("Simulation stopped: " + text)
        else:
            self._set_status("Simulation cancelled")

    def _on_worker_finished(self):
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()
        self._set_running(False)

    # ── materials ─────────────────────────────────────────────────────────────
    def _open_materials(self):
        if self._worker is not None:
            return
        body = self._body()
        dialog = MaterialsDialog(self, self.study, body.get("material") if body else None)
        if dialog.exec_() != QDialog.Accepted:
            return
        user, renames, deleted, library = dialog.changes()
        before = copy.deepcopy(self.study.get("materials", {}))
        for name in list(USER_MATERIALS):              # the library
            if name in deleted or name in renames:
                del USER_MATERIALS[name]
        for name in library:
            USER_MATERIALS[name] = copy.deepcopy(user[name])
        try:
            save_material_library()
        except OSError as ex:
            QMessageBox.warning(self, "Materials", f"Could not save the material library\n"
                                f"{material_library_path()}\n\n{ex}")
        renamed_bodies = 0                             # the study follows renames
        for component in self.study["components"]:
            for body in component["bodies"]:
                if body.get("material") in renames:
                    body["material"] = renames[body["material"]]
                    renamed_bodies += 1
        sweep = self.study.get("sweep")
        if sweep and sweep["parameter"] == "material":
            sweep["values"] = [renames.get(v, v) for v in sweep["values"]]
        materials = {}
        for name in before:
            new = renames.get(name, name)
            if new in user:
                materials[new] = copy.deepcopy(user[new])
        for component in self.study["components"]:
            for body in component["bodies"]:
                name = body.get("material")
                if name in user and name not in materials:
                    materials[name] = copy.deepcopy(user[name])
        self.study["materials"] = materials
        set_active_study(self.study)
        self._rebuild_tree()
        self._show_properties()
        if materials != before or renamed_bodies:
            self._changed()
        self._set_status(f"Materials saved: {len(USER_MATERIALS)} in your library "
                         f"({material_library_path()})")

    # ── parameter sweep ───────────────────────────────────────────────────────
    def _open_sweep(self):
        if self._worker is not None or not self._check_runnable():
            return
        dialog = SweepDialog(self, self.study, self.study.get("sweep"))
        if dialog.exec_() != QDialog.Accepted:
            return
        sweep = dialog.sweep()
        if sweep is None or not sweep["values"]:
            return
        self.study["sweep"] = sweep
        self._changed(results=False)
        if self.view.pick_mode is not None:
            self._set_pick_mode(None)
        snapshot = study_from_state(study_state(self.study))
        snapshot["section"] = None
        self._sweep_key = state_key(study_state(self.study))
        worker = SweepWorker(snapshot, sweep)
        worker.progressed.connect(self._on_progress)
        worker.succeeded.connect(self._on_sweep_done)
        worker.failed.connect(self._on_simulation_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        self._sweep_started = time.perf_counter()
        self._set_running(True)
        self._set_status(f"Parameter sweep started: {sweep_title(self.study, sweep)}, "
                         f"{len(sweep['values'])} runs")
        worker.start()

    def _on_sweep_done(self, result):
        self._sweep_result = result
        runs = result["runs"]
        failed = [run for run in runs if run["error"]]
        seconds = time.perf_counter() - getattr(self, "_sweep_started", time.perf_counter())
        self._set_status(f"Parameter sweep finished: {len(runs) - len(failed)} of {len(runs)} runs "
                         f"in {format_time(seconds)}" + (f", {len(failed)} failed" if failed else ""))
        self._show_sweep_results()

    def _forget_sweep(self):
        self._sweep_result = None
        self._sweep_key = None
        self.act_sweep_results.setEnabled(False)
        if self._sweep_dialog is not None:
            self._sweep_dialog.close()
            self._sweep_dialog = None

    def _sweep_outdated(self):
        return self._sweep_result is not None and \
            state_key(study_state(self.study)) != self._sweep_key

    def _show_sweep_results(self):
        if self._sweep_result is None:
            return
        self.act_sweep_results.setEnabled(self._worker is None)
        if self._sweep_dialog is None:
            self._sweep_dialog = SweepResultsDialog(self, self._sweep_result, self._sweep_outdated())
            self._sweep_dialog.showRun.connect(self._show_sweep_run)
            cad._set_dark_title_bar(self._sweep_dialog)
        else:
            self._sweep_dialog.set_result(self._sweep_result, self._sweep_outdated())
        self._sweep_dialog.show()
        self._sweep_dialog.raise_()
        self._sweep_dialog.activateWindow()

    def _show_sweep_run(self, index):
        if self._worker is not None or self._sweep_result is None:
            return
        runs = self._sweep_result["runs"]
        if not 0 <= index < len(runs):
            return
        run = runs[index]
        if run["error"] or run["result"] is None:
            self._set_status(f"This run failed: {run['error']}")
            return
        self._discard_results()
        self.txt_messages.clear()
        self._result = run["result"]
        self._result_note = f"Sweep run: {self._sweep_result['title']} = {run['text']}"
        self._show_result()
        self._set_status(f"Showing the sweep run {self._sweep_result['title']} = {run['text']} "
                         "(the study itself is unchanged)")

    # ── export ────────────────────────────────────────────────────────────────
    def _update_export_menu(self):
        result = self._result
        self.act_export_csv.setEnabled(result is not None)
        self.act_export_series.setEnabled(result is not None and result["analysis"] == "transient")
        self.act_export_vtu.setEnabled(result is not None)

    def _export_vtu(self):
        result = self._result
        if result is None:
            return
        frames = None
        kept = list(np.asarray(result["volume_index"]).tolist())
        if result["analysis"] == "transient" and len(kept) > 1:
            size = vtu_size_estimate(result)
            box = QMessageBox(self)
            box.setWindowTitle("Export for ParaView")
            box.setIcon(QMessageBox.Question)
            shown = format_time(result["frame_times"][self._frame])
            box.setText(f"Export the time shown (t = {shown}) or all {len(kept)} stored times as a "
                        f"time series (one file each, about {size * len(kept) / 1e6:.0f} MB "
                        "together)?")
            one = box.addButton("Shown time", QMessageBox.AcceptRole)
            series = box.addButton("All times", QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.exec_()
            if box.clickedButton() is series:
                frames = kept
            elif box.clickedButton() is not one:
                return
        if frames is None:
            path = self._export_path("Export for ParaView", ".vtu", "VTK unstructured grid (*.vtu)")
        else:
            path = self._export_path("Export a time series for ParaView", ".pvd",
                                     "ParaView time series (*.pvd)")
        if not path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            def progress(number, total):
                self.status.setText(f"Writing VTU file {number + 1} of {total}...")
                QApplication.processEvents()
            files = export_vtu(result, path, frames=frames, frame=self._frame, progress=progress)
        except (OSError, ValueError, MemoryError) as ex:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Export failed", f"Could not write\n{path}\n\n{ex}")
            return
        QApplication.restoreOverrideCursor()
        size = sum(os.path.getsize(f) for f in files if os.path.exists(f))
        self._set_status(f"Saved for ParaView: {files[0]}" +
                         (f" and {len(files) - 1} frame files" if len(files) > 1 else "") +
                         f" ({size / 1e6:.1f} MB). In ParaView: open it, Apply, colour by "
                         "Temperature" + (" (airflow values are per cell: the Cell Data to Point "
                                          "Data filter makes them smooth)."
                                          if result.get("mode") == "airflow" else "."))

    def _export_path(self, title, suffix, file_filter):
        base = os.path.splitext(self._window._thermal_file or "")[0] or \
            os.path.join(self._start_directory(), self.study["name"] or THERMAL_DEFAULT_NAME)
        path, _ = QFileDialog.getSaveFileName(self, title, base + suffix, file_filter)
        if path and not path.lower().endswith(os.path.splitext(suffix)[1]):
            path += os.path.splitext(suffix)[1]
        return path

    def _view_image(self):
        """The 3-D view as an image, with the colour legend beside it while it is shown."""
        from PyQt5.QtGui import QImage
        image = self.view.grabFrameBuffer()
        if not self.legend.isVisible():
            return image
        legend = self.legend.grab().toImage()
        combined = QImage(image.width() + legend.width(), max(image.height(), legend.height()),
                          QImage.Format_RGB32)
        combined.fill(QColor("#1c1c21"))
        painter = QPainter(combined)
        painter.drawImage(0, 0, image)
        painter.drawImage(image.width(), 0, legend)
        painter.end()
        return combined

    def _export_png(self):
        path = self._export_path("Save the view as an image", ".png", "PNG images (*.png)")
        if not path:
            return
        if self._view_image().save(path):
            self._set_status(f"Image saved to {path}")
        else:
            QMessageBox.critical(self, "Export failed", f"Could not write\n{path}")

    def _probe_rows(self):
        """[(probe, part, value at the end, highest value)] of the shown result."""
        rows = []
        for probe in self.study["probes"]:
            series = self._probe_series(probe)
            rows.append((probe["name"], probe.get("component") or "air",
                         None if series is None else float(series[-1]),
                         None if series is None else float(np.max(series))))
        return rows

    def _export_csv(self):
        result = self._result
        if result is None:
            return
        path = self._export_path("Export results", "_results.csv", "CSV files (*.csv)")
        if not path:
            return
        transient = result["analysis"] == "transient"
        number = (lambda v: "" if v is None or not np.isfinite(v) else f"{v:.4g}")
        rows = [["Thermal study", self.study["name"]],
                ["Solver", "Airflow + heat" if result.get("mode") == "airflow" else "Conduction"],
                ["Analysis", result["analysis"]], ["Ambient °C", number(result["ambient"])]]
        if self._result_note:
            rows.append([self._result_note])
        rows += [[], ["Body", "Material", "Power W", "Min °C", "Mean °C", "Max °C",
                      "Highest in the run °C", "Material limit °C"]]
        for index, body in enumerate(result["bodies"]):
            peak = body["max"]
            if transient and result["series"] is not None and \
                    np.isfinite(result["series"]["max"][index]).any():
                peak = float(np.nanmax(result["series"]["max"][index]))
            rows.append([body["name"], body["material"] or "", number(body["power"]),
                         number(body["min"]), number(body["mean"]), number(body["max"]),
                         number(peak), number(body["max_c"])])
        probes = self._probe_rows()
        if probes:
            rows += [[], ["Probe", "On part", "°C at the end", "Highest °C"]]
            rows += [[name, part, number(end), number(top)] for name, part, end, top in probes]
        if result.get("thermostats"):
            rows += [[], ["Thermostat of", "Setpoint °C", "Share of the power", "Switchings",
                          "Sensor °C at the end"]]
            rows += [[t["name"], number(t["setpoint"]), f"{t['share']:.4f}", t["switches"],
                      number(t["sensor"])] for t in result["thermostats"]]
        rows += [[], ["Summary"]] + [[line] for line in result["summary"]]
        if result["messages"]:
            rows += [[], ["Messages"]] + [[line] for line in result["messages"]]
        try:
            write_csv(path, rows)
        except OSError as ex:
            QMessageBox.critical(self, "Export failed", f"Could not write\n{path}\n\n{ex}")
            return
        self._set_status(f"Results saved to {path}")

    def _export_series_csv(self):
        result = self._result
        if result is None or result["analysis"] != "transient":
            return
        path = self._export_path("Export values over time", "_over_time.csv", "CSV files (*.csv)")
        if not path:
            return
        series = result["series"]
        times = series["times"]
        header = ["Time s", "Power W"] + [f"{b['name']} max °C" for b in result["bodies"]] + \
            [f"{b['name']} mean °C" for b in result["bodies"]]
        probes = []
        for probe in self.study["probes"]:
            values = self._probe_series(probe)
            if values is not None:
                probes.append(np.interp(times, result["frame_times"], values))
                header.append(f"Probe {probe['name']} °C")
        rows = [header]
        for k, t in enumerate(times):
            row = [f"{t:.6g}", f"{series['power'][k]:.6g}"]
            row += [f"{series['max'][i, k]:.5g}" for i in range(len(result["bodies"]))]
            row += [f"{series['mean'][i, k]:.5g}" for i in range(len(result["bodies"]))]
            row += [f"{values[k]:.5g}" for values in probes]
            rows.append(row)
        try:
            write_csv(path, rows)
        except OSError as ex:
            QMessageBox.critical(self, "Export failed", f"Could not write\n{path}\n\n{ex}")
            return
        self._set_status(f"Values over time saved to {path}")

    @staticmethod
    def _plot_image(configure, width=900, height=320):
        """A chart drawn off screen into an image."""
        from PyQt5.QtGui import QImage
        plot = TimePlot()
        configure(plot)
        plot.resize(width, height)
        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(QColor("#171a1f"))
        plot.render(image)
        return image

    def _export_pdf(self):
        path = self._export_path("Save the PDF report", "_report.pdf", "PDF files (*.pdf)")
        if not path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            html, images = self._report()
            write_pdf(path, html, images, f"Thermal study {self.study['name']}")
        except Exception as ex:
            QApplication.restoreOverrideCursor()
            cad._trace("Thermal report error:\n" + traceback.format_exc())
            QMessageBox.critical(self, "Report failed", f"Could not write\n{path}\n\n{ex}")
            return
        QApplication.restoreOverrideCursor()
        self._set_status(f"Report saved to {path}")
        cad.log_action(f"Thermal: report {path}")

    def _report(self):
        """(HTML, images) of the PDF report: setup, picture of the view, temperatures of all
        bodies and probes, heat balance, the course over time, the last sweep, the materials."""
        import html as html_module
        import datetime
        esc = html_module.escape
        result = self._result
        settings = self.study["settings"]
        airflow = settings["mode"] == "airflow"
        images = {}
        out = ["<html><head><style>"
               "body { font-family: Arial; font-size: 9pt; color: #202428; }"
               "h1 { font-size: 17pt; color: #17324d; margin-bottom: 2px; }"
               "h2 { font-size: 12pt; color: #17324d; margin-top: 14px; margin-bottom: 4px; }"
               "th { background-color: #e3e8ee; text-align: left; font-weight: bold; }"
               ".num { text-align: right; } .hot { color: #c62828; font-weight: bold; }"
               ".note { color: #5f6b77; font-size: 8pt; } .warn { color: #b26a00; }"
               "</style></head><body>"]

        def table(header, rows, numeric_from=1, widths=None):
            parts = ["<table border='1' cellspacing='0' cellpadding='3' width='100%' "
                     "style='border-color: #b8c0c8; border-collapse: collapse;'><tr>"]
            for index, title in enumerate(header):
                width = f" width='{widths[index]}%'" if widths else ""
                parts.append(f"<th{width}>{esc(str(title))}</th>")
            parts.append("</tr>")
            for row in rows:
                parts.append("<tr>")
                for index, cell in enumerate(row):
                    text = cell if isinstance(cell, tuple) else (str(cell), "")
                    try:
                        float(text[0])
                        numeric = True
                    except ValueError:
                        numeric = text[0] == "—"
                    css = "num" if index >= numeric_from and numeric else ""
                    parts.append(f"<td class='{css} {text[1]}'>{esc(text[0])}</td>")
                parts.append("</tr>")
            parts.append("</table>")
            return "".join(parts)

        def c(value, digits=1):
            return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"

        path = self._window._thermal_file
        out.append(f"<h1>Thermal study: {esc(self.study['name'])}</h1>")
        out.append(f"<p class='note'>{esc(path) if path else 'not saved yet'} · "
                   f"{datetime.datetime.now():%Y-%m-%d %H:%M} · SimpleCAD {ADDON_TITLE} "
                   f"{ADDON_VERSION}</p>")
        if self._result_note:
            out.append(f"<p><b>{esc(self._result_note)}</b> (the study itself has the values "
                       "listed below)</p>")
        # picture
        image = self._view_image()
        if image.width() > 1600:
            image = image.scaledToWidth(1600, Qt.SmoothTransformation)
        images["view.png"] = image
        caption = ("Temperatures" + (f" at t = {format_time(result['frame_times'][self._frame])}"
                                     if result["analysis"] == "transient" else "")
                   if result is not None and self.view.result_mode else "Model")
        out.append(f"<p align='center'><img src='view.png' width='520'><br>"
                   f"<span class='note'>{esc(caption)} (as shown in the Thermal window)</span></p>")
        # setup
        out.append("<h2>Setup</h2>")
        rows = [("Solver", "Airflow + heat (CHT)" if airflow else "Conduction (FEM)")]
        if settings["analysis"] == "transient":
            start = settings["ambient_c"] if settings["initial_ambient"] else settings["initial_c"]
            rows.append(("Analysis", f"Transient: {format_time(settings['duration_s'])} in "
                                     f"{settings['time_steps']} steps, start at {start:g} °C"))
        else:
            rows.append(("Analysis", "Steady state"))
        rows.append(("Ambient", f"{settings['ambient_c']:g} °C"))
        if airflow:
            sides = settings["flow_sides"]
            rows.append(("Air domain sides", ", ".join(f"{FLOW_SIDE_TITLES[s]} {SIDE_KINDS[sides[s]].lower()}"
                                                       for s in FLOW_SIDES)))
            rows.append(("Flow model", {"auto": "Automatic", "laminar": "Laminar",
                                        "mixing": "Turbulent (mixing length)"}[settings["turbulence"]]))
        else:
            rows.append(("Convection", {"natural": "Natural convection (correlations)",
                                        "forced": f"Forced air, {settings['air_speed']:g} m/s",
                                        "fixed": f"Fixed h = {settings['h']:g} W/m²K"}[settings["convection"]]))
        rows.append(("Radiation", "to the surroundings" if settings["radiation"] else "off"))
        if result is not None:
            rows.append(("Mesh", result["summary"][0].split(": ", 1)[-1]))
        out.append(table(["Item", "Value"], rows, numeric_from=9, widths=[24, 76]))
        sources = []
        for component in self.study["components"]:
            if component.get("kind") in FLOW_KINDS:
                sources.append((component["name"], f"{component['kind']}: "
                                f"{flow_text(component['kind'], flow_settings(component.get('flow'), component['kind']))}"))
                continue
            for body in component["bodies"]:
                if not body_has_power(body):
                    continue
                name = component["name"] if len(component["bodies"]) == 1 else \
                    f"{component['name']} / {body['name']}"
                text = f"{body['power_w']:g} W"
                if body.get("profile"):
                    text += f", {profile_text(body['profile'], body['power_w'])}"
                if body.get("thermostat"):
                    text += f", {thermostat_text(body['thermostat'])}"
                sources.append((name, text))
        for condition in self.study.get("conditions", []):
            if condition.get("enabled", True):
                sources.append((condition["name"], self._condition_text(condition)))
        for contact in self.study.get("contacts", []):
            if contact.get("enabled", True):
                sources.append((contact["name"], self._contact_item_text(contact)))
        if sources:
            out.append("<p><b>Heat sources, face conditions, contacts, fans and vents</b></p>")
            out.append(table(["Name", "Setting"], sources, numeric_from=9, widths=[30, 70]))
        # results
        if result is None:
            out.append("<h2>Results</h2><p>No results yet: start the simulation, then export the "
                       "report again.</p>")
        else:
            transient = result["analysis"] == "transient"
            out.append("<h2>Results</h2>")
            out.append(f"<p>Highest temperature <b>{float(np.max(result['hot_values'])):.1f} °C</b> · "
                       f"ambient {result['ambient']:g} °C · heat balance {result['balance_error']:+.2%}"
                       + (" (at the end)" if transient else "") + "</p>")
            rows = []
            for index, body in enumerate(result["bodies"]):
                peak = body["max"]
                if transient and result["series"] is not None and \
                        np.isfinite(result["series"]["max"][index]).any():
                    peak = float(np.nanmax(result["series"]["max"][index]))
                over = peak is not None and body["max_c"] is not None and peak > body["max_c"]
                row = [body["name"], body["material"] or "—", f"{body['power']:g}", c(body["min"]),
                       c(body["mean"]), c(body["max"])]
                if transient:
                    row.append((c(peak), "hot" if over else ""))
                row.append(("over the limit" if over else "ok" if body["max_c"] is not None else "—",
                            "hot" if over else ""))
                row.append("—" if body["max_c"] is None else f"{body['max_c']:g}")
                rows.append(row)
            header = ["Body", "Material", "W", "Min °C", "Mean °C",
                      "Max °C" + (" (end)" if transient else "")] + \
                (["Highest °C"] if transient else []) + ["Status", "Limit °C"]
            out.append(table(header, rows, numeric_from=2))
            probes = self._probe_rows()
            if probes:
                out.append("<p><b>Probes</b></p>")
                out.append(table(["Probe", "On part", "°C" + (" (end)" if transient else "")] +
                                 (["Highest °C"] if transient else []),
                                 [[name, part, c(end, 2)] + ([c(top, 2)] if transient else [])
                                  for name, part, end, top in probes], numeric_from=2))
            out.append("<p><b>Summary of the run</b></p><p class='note'>" +
                       "<br>".join(esc(line) for line in result["summary"]) + "</p>")
            if result["messages"]:
                out.append("<p><b>Messages</b></p><p class='warn'>" +
                           "<br>".join(esc(line) for line in result["messages"]) + "</p>")
            if transient and self.plot._times is not None:
                def copy_plot(plot):
                    for key in ("_times", "_curves", "_limits", "_x_format", "_x_labels", "_markers"):
                        setattr(plot, key, getattr(self.plot, key))
                images["over_time.png"] = self._plot_image(copy_plot)
                out.append("<p><b>Over time</b> (bodies: maximum, ◉ probes; dashed: material "
                           "limits)</p><p align='center'><img src='over_time.png' width='520'></p>")
        # sweep
        if self._sweep_result is not None:
            sweep = self._sweep_result
            out.append(f"<h2>Parameter sweep: {esc(sweep['title'])}</h2>")
            if self._sweep_outdated():
                out.append("<p class='warn'>The study changed after this sweep was run.</p>")
            columns, rows, _bodies, _probes = sweep_table(sweep)
            out.append(table(columns, rows, numeric_from=1))
            failed = [run for run in sweep["runs"] if run["error"]]
            for run in failed:
                out.append(f"<p class='warn'>{esc(run['text'])}: {esc(run['error'])}</p>")
            xs, curves, labels, _order = sweep_plot_data(sweep)
            if len(xs) > 1:
                images["sweep.png"] = self._plot_image(
                    lambda plot: plot.set_data(xs, curves, x_labels=labels,
                                               x_format=lambda v: f"{v:g}", markers=True))
                out.append("<p align='center'><img src='sweep.png' width='520'></p>")
        # materials
        used = sorted({body.get("material") for component in self.study["components"]
                       if component.get("kind") not in FLOW_KINDS
                       for body in component["bodies"] if body.get("material")})
        if used:
            out.append("<h2>Materials</h2>")
            rows = []
            for name in used:
                props = material_properties(name, self.study.get("materials", {}))
                if props is None:
                    rows.append([name, "unknown", "", "", "", "", ""])
                    continue
                k = f"{props['k']:.4g}" + (f" / {props['k_plane']:.4g} in-plane" if props.get("k_plane") else "")
                extra = ", ".join(x for x in ("k(T) table" if props.get("k_table") else "",
                                              "cp(T) table" if props.get("cp_table") else "",
                                              "user material" if props.get("user") else "") if x)
                rows.append([name, k, f"{props['cp']:.4g}", f"{props['density'] or 0:g}",
                             f"{props['emissivity']:g}",
                             "—" if props.get("max_c") is None else f"{props['max_c']:g}", extra or "—"])
            out.append(table(["Material", "k W/m·K", "cp J/kg·K", "ρ kg/m³", "ε", "Limit °C", "Notes"],
                             rows, numeric_from=1))
        out.append("<p class='note'>Simulation results are estimates. Check the heat balance and "
                   "the messages, compare with a finer mesh (Parameter sweep → Cell size) and keep "
                   "a margin to material limits.</p></body></html>")
        return "".join(out), images

    # ── results ───────────────────────────────────────────────────────────────
    def _discard_results(self, outdated=False):
        self._result = None
        self._result_note = ""
        self._frame = 0
        self._play_timer.stop()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.w_playback.hide()
        self.plot.clear()
        self.plot.hide()
        self.lbl_plot.hide()
        self.view.result_mode = False
        self.view.hot_marker = None
        self.view.streamlines = []
        self.act_streamlines.blockSignals(True)
        self.act_streamlines.setChecked(False)
        self.act_streamlines.blockSignals(False)
        self.act_streamlines.setEnabled(False)
        for entry in self._scene:
            self.view.set_entry_colors(entry, None)
            entry.pop("_cap_weights", None)
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
        if self._selected_contact is not None:
            self._show_contact()
        self._sync_section_ui()
        self._schedule_section()
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
            self._schedule_section()
            self._update_streamlines()
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
        notes = self.txt_messages.toPlainText().strip()
        self.txt_messages.setPlainText("\n".join(result["summary"] + [""] + result["messages"] +
                                                 ([notes] if notes else [])))
        airflow = result.get("mode") == "airflow"
        self.act_streamlines.setEnabled(airflow)
        if airflow and not self.act_streamlines.isChecked():
            self.act_streamlines.blockSignals(True)
            self.act_streamlines.setChecked(True)
            self.act_streamlines.blockSignals(False)
        self._set_frame(last)
        self.act_results.setEnabled(True)
        self.act_results.setChecked(True)          # → _on_results_toggled
        self._on_results_toggled(True)
        self._sync_section_ui()
        if self._selected_contact is not None:
            self._show_contact()

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
                f"{result['power']:.4g} W<br>At the end: convection {result['q_convection']:.3g} W, "
                f"radiation {result['q_radiation']:.3g} W, still stored {result['q_stored']:.3g} W")
        else:
            self.lbl_result.setText(
                f"Maximum <b>{value:.1f} °C</b> · ambient {result['ambient']:g} °C · "
                f"{result['power']:.4g} W<br>Heat balance: convection {result['q_convection']:.3g} W, "
                f"radiation {result['q_radiation']:.3g} W ({result['balance_error']:+.2%})")
        lines = []
        for entry in result.get("conditions", [])[:4]:        # heat through the face conditions
            if entry["kind"] == "insulated":
                continue
            heat = entry["heat"]
            lines.append(f"{entry['name']}: {abs(heat):.3g} W "
                         f"{'in' if (heat < 0) == (entry['kind'] != 'flux') else 'out'}")
        for entry in result.get("contacts", [])[:3]:          # heat through the contacts
            if entry["faces"]:
                lines.append(f"{entry['name']}: {abs(entry['heat']):.3g} W" +
                             (f", ΔT {abs(entry['jump']):.3g} K" if entry["kind"] != "bonded" else ""))
        switching = (result.get("series") or {}).get("thermostats")
        for j, entry in enumerate(result.get("thermostats", [])[:4]):   # thermostats
            if transient and switching is not None:
                state = switching[j, int(result["frame_steps"][frame])] > 0.5
                lines.append(f"{entry['name']}: heater {'on' if state else 'off'}")
            else:
                lines.append(f"{entry['name']}: {entry['share']:.1%} of {entry['nominal']:.3g} W")
        if lines:
            self.lbl_result.setText(self.lbl_result.text() + "<br>" + " · ".join(lines))
        if self._result_note:
            self.lbl_result.setText(f"<span style='color:#80c8ff'>{self._result_note}</span><br>" +
                                    self.lbl_result.text())
        if result.get("mode") == "airflow":
            self.lbl_result.setText(self.lbl_result.text() + "<br>" + result.get("air_text", "") +
                                    (" (at the end)" if transient else ""))
            if transient and self.act_streamlines.isChecked():
                self._update_streamlines()
        self._fill_probe_table()
        self._refresh_overlays()
        if self._selected_probe is not None:
            self._show_probe()
        self._schedule_section()
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
        self._update_streamlines()
        self._schedule_section()
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
