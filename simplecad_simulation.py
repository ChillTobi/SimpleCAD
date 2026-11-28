"""
SimpleCAD Simulation (file simplecad_simulation.py) — optional add-on: heat conduction (FEM),
airflow + heat (CHT) and, with simplecad_structural.py, Structural studies (linear static
stress and natural frequencies of the parts; gmsh tetrahedra).

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

Liquid cooling (water, water–glycol, oils; properties over temperature):
  * Coolant channel: the picked inner faces of a pipe or passage give its length, hydraulic
    diameter and cross-section; the liquid (fixed flow or a pump curve) warms along it, with the
    wall coefficient from pipe-flow correlations (laminar with the entry region, transitional,
    turbulent) and the pressure drop from Darcy–Weisbach. Channels can feed each other.
  * Liquid bath: the liquid fills a container up to a fill level and is well mixed (natural
    convection on the wetted faces, the open surface loses heat by convection, radiation and
    evaporation), or the faces sit in a large volume at a fixed temperature.
  * In Airflow mode the liquid itself can be solved instead: a bath with buoyancy from the
    density table, or a channel fed by its pump at the inlet.

Printed circuit boards: imported from KiCad (.kicad_pcb, KiCad 5 … 10) or Gerber files (RS-274X
with Excellon drill files, optionally a Gerber job). The outline is extruded to the board; every
copper layer is rasterised, and each cell's conductivity follows the layer stack (FR-4 or an
aluminium-core IMS board), the copper share there and the plated holes. Power components are
placed as package blocks with a two-resistor junction model (θ to the package top, θ to the
board); the results give their junction temperatures.

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

ADDON_TITLE = "Simulation"
ADDON_VERSION = "2.4"           # add-on release, shown in SimpleCAD's Help → About
THERMAL_FORMAT = "simplecad-thermal-study"
THERMAL_EXTENSION = ".thml"
THERMAL_DEFAULT_NAME = "Untitled"
HEAT_SOURCE_COLOR = (1.0, 0.45, 0.15)
PRIMITIVE_KINDS = ("block", "cylinder")
COMPONENT_KINDS = ("project", "file", "assembly") + PRIMITIVE_KINDS + ("fan", "vent", "pcb")
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
    "Mould compound": {"k": 0.8, "cp": 880.0, "emissivity": 0.90, "max_c": None, "density": 1900.0,
                       "note": "epoxy mould compound of IC packages (the die is modelled by the "
                               "package's θ values)"},
}
# Mechanical data for Structural studies (typical handbook values at room temperature):
#   E        Young's modulus, GPa          nu      Poisson's ratio
#   yield    yield strength in MPa (brittle materials: tensile strength)
#   alpha    thermal expansion, µm/(m·K)   brittle  True: fails by fracture, no yielding
# Laminates and wood are anisotropic; the values are in-plane / along the grain.
MATERIAL_MECHANICAL = {
    "Aluminium": {"E": 69.0, "nu": 0.33, "yield": 240.0, "alpha": 23.4,
                  "mech_note": "6061-T6 / 6082-T6 (yield 240–275 MPa); soft alloys far lower"},
    "Aluminium Black Anodized": {"E": 69.0, "nu": 0.33, "yield": 160.0, "alpha": 23.4,
                                 "mech_note": "6063-T5 / T6 extrusion alloy"},
    "Copper": {"E": 117.0, "nu": 0.34, "yield": 70.0, "alpha": 16.8,
               "mech_note": "annealed; cold-worked copper 200–300 MPa"},
    "Tin": {"E": 50.0, "nu": 0.36, "yield": 14.0, "alpha": 22.0, "mech_note": "creeps at room temperature"},
    "Brass": {"E": 100.0, "nu": 0.34, "yield": 200.0, "alpha": 20.0, "mech_note": "CuZn37, half hard"},
    "Bronze": {"E": 115.0, "nu": 0.34, "yield": 250.0, "alpha": 18.0, "mech_note": "CuSn8"},
    "Steel": {"E": 210.0, "nu": 0.30, "yield": 235.0, "alpha": 12.0,
              "mech_note": "structural steel S235; stainless 1.4301 ≈ 210 MPa, α 16 µm/m·K"},
    "ABS": {"E": 2.2, "nu": 0.35, "yield": 40.0, "alpha": 90.0,
            "mech_note": "moulded; 3D-printed parts are weaker, especially across the layers"},
    "PLA": {"E": 3.5, "nu": 0.36, "yield": 55.0, "alpha": 68.0,
            "mech_note": "moulded; 3D-printed parts are weaker, especially across the layers; creeps"},
    "PET": {"E": 2.1, "nu": 0.38, "yield": 50.0, "alpha": 60.0,
            "mech_note": "PET / PETG moulded; 3D-printed parts are weaker across the layers"},
    "Micarta": {"E": 8.0, "nu": 0.30, "yield": 90.0, "alpha": 25.0, "brittle": True,
                "mech_note": "phenolic laminate, in-plane"},
    "G10": {"E": 18.0, "nu": 0.15, "yield": 280.0, "alpha": 12.0, "brittle": True,
            "mech_note": "in-plane values; across the layers far weaker"},
    "FR-4": {"E": 22.0, "nu": 0.15, "yield": 300.0, "alpha": 14.0, "brittle": True,
             "mech_note": "in-plane values (α through the board ≈ 60 µm/m·K)"},
    "GFK": {"E": 20.0, "nu": 0.25, "yield": 200.0, "alpha": 15.0, "brittle": True,
            "mech_note": "glass-fibre mat laminate, in-plane"},
    "CFK": {"E": 60.0, "nu": 0.30, "yield": 500.0, "alpha": 2.0, "brittle": True,
            "mech_note": "quasi-isotropic lay-up, in-plane"},
    "Wood (Oak)": {"E": 11.0, "nu": 0.30, "yield": 50.0, "alpha": 5.0, "brittle": True,
                   "mech_note": "along the grain; across it about 1/10 of the stiffness and strength"},
    "Wood (Pine)": {"E": 9.0, "nu": 0.30, "yield": 35.0, "alpha": 5.0, "brittle": True,
                    "mech_note": "along the grain; across it about 1/10 of the stiffness and strength"},
    "Wood (Beech)": {"E": 14.0, "nu": 0.30, "yield": 55.0, "alpha": 5.0, "brittle": True,
                     "mech_note": "along the grain; across it about 1/10 of the stiffness and strength"},
    "Wood (Birch)": {"E": 14.0, "nu": 0.30, "yield": 50.0, "alpha": 5.0, "brittle": True,
                     "mech_note": "along the grain; across it about 1/10 of the stiffness and strength"},
    "Wood (Walnut)": {"E": 11.0, "nu": 0.30, "yield": 45.0, "alpha": 5.0, "brittle": True,
                      "mech_note": "along the grain; across it about 1/10 of the stiffness and strength"},
    "Mould compound": {"E": 25.0, "nu": 0.25, "yield": 120.0, "alpha": 10.0, "brittle": True,
                       "mech_note": "epoxy mould compound (flexural strength)"},
}
for _name, _data in MATERIAL_MECHANICAL.items():
    MATERIAL_THERMAL[_name].update(_data)
MECHANICAL_KEYS = ("E", "nu", "yield", "alpha", "brittle", "mech_note")

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
    # mechanical data (Structural studies), optional: E in GPa, yield in MPa, α in µm/(m·K)
    material["E"] = number("E", 0.0, 0.0) or None
    material["nu"] = number("nu", 0.3, -0.99, 0.49)
    material["yield"] = number("yield", 0.0, 0.0) or None
    material["alpha"] = number("alpha", 0.0, -100.0, 1000.0)
    material["brittle"] = bool(data.get("brittle", False))
    material["mech_note"] = str(data.get("mech_note", "") or "")
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
        props["density"] = data.get("density") if density is None else density * 1000.0
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
    mechanical = mechanical_text(props)
    if mechanical:
        parts.append(mechanical)
    if props.get("user"):
        parts.append("user material")
    return " · ".join(parts)


def mechanical_text(props):
    """'E 69 GPa, ν 0.33, yield 240 MPa, α 23.4 µm/m·K' or '' without elastic data."""
    if not props or not props.get("E"):
        return ""
    text = f"E {props['E']:g} GPa, ν {props.get('nu', 0.3):g}"
    if props.get("yield"):
        text += f", {'strength' if props.get('brittle') else 'yield'} {props['yield']:g} MPa"
    if props.get("alpha"):
        text += f", α {props['alpha']:g} µm/m·K"
    return text


def material_note(name, materials=None):
    props = material_properties(name, materials) or {}
    notes = [props.get("note", "") or "", props.get("mech_note", "") or ""]
    return "; ".join(note for note in notes if note)


_LIBRARY_ERROR = load_material_library()


# ── liquids ───────────────────────────────────────────────────────────────────
# Coolants and bath liquids at about 1 bar, tabled over temperature (°C): density kg/m³, specific
# heat J/(kg·K), conductivity W/(m·K), dynamic viscosity mPa·s. Water follows IAPWS; the glycol
# mixtures (ethylene glycol by volume) and the oils are typical values (actual products vary).
#   limit_c / limit_text   warning when a wetted face gets hotter (boiling, oil ageing)
#   evaporation            share of the evaporation of plain water from an open surface
FLUIDS = {
    "Water": {
        "t": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "rho": [999.8, 999.7, 998.2, 995.7, 992.2, 988.0, 983.2, 977.8, 971.8, 965.3, 958.4],
        "cp": [4217, 4192, 4182, 4178, 4179, 4181, 4185, 4190, 4197, 4205, 4216],
        "k": [0.561, 0.580, 0.598, 0.615, 0.631, 0.644, 0.654, 0.663, 0.670, 0.675, 0.679],
        "mu": [1.792, 1.307, 1.002, 0.797, 0.653, 0.547, 0.467, 0.404, 0.355, 0.315, 0.282],
        "limit_c": 100.0, "limit_text": "boils at 100 °C (1 bar); boiling is not modelled",
        "evaporation": 1.0, "emissivity": 0.95, "note": "freezes at 0 °C"},
    "Water–glycol 30 %": {
        "t": [-10, 0, 20, 40, 60, 80, 100],
        "rho": [1050, 1048, 1041, 1032, 1021, 1009, 995],
        "cp": [3680, 3700, 3750, 3800, 3850, 3900, 3950],
        "k": [0.45, 0.46, 0.48, 0.50, 0.52, 0.53, 0.54],
        "mu": [5.0, 3.4, 2.0, 1.25, 0.85, 0.62, 0.48],
        "limit_c": 104.0, "limit_text": "boils near 104 °C (1 bar); boiling is not modelled",
        "evaporation": 0.85, "emissivity": 0.95,
        "note": "ethylene glycol 30 % by volume, frost protection to about −15 °C"},
    "Water–glycol 50 %": {
        "t": [-20, 0, 20, 40, 60, 80, 100],
        "rho": [1090, 1082, 1072, 1061, 1049, 1036, 1022],
        "cp": [3200, 3280, 3360, 3440, 3520, 3600, 3680],
        "k": [0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.41],
        "mu": [17.0, 7.5, 3.9, 2.3, 1.5, 1.05, 0.78],
        "limit_c": 107.0, "limit_text": "boils near 107 °C (1 bar); boiling is not modelled",
        "evaporation": 0.65, "emissivity": 0.95,
        "note": "ethylene glycol 50 % by volume, frost protection to about −37 °C"},
    "Transformer oil": {
        "t": [-20, 0, 20, 40, 60, 80, 100],
        "rho": [908, 895, 882, 869, 856, 843, 830],
        "cp": [1720, 1800, 1860, 1930, 2000, 2070, 2140],
        "k": [0.136, 0.134, 0.132, 0.130, 0.128, 0.126, 0.124],
        "mu": [420, 70, 22, 9.5, 5.2, 3.3, 2.3],
        "limit_c": 105.0, "limit_text": "ages quickly above about 105 °C",
        "evaporation": 0.0, "emissivity": 0.9,
        "note": "mineral insulating oil (IEC 60296), viscosity rises steeply in the cold"},
    "Hydraulic oil (ISO VG 32)": {
        "t": [0, 20, 40, 60, 80, 100],
        "rho": [878, 865, 852, 839, 826, 813],
        "cp": [1840, 1900, 1970, 2040, 2110, 2180],
        "k": [0.136, 0.134, 0.132, 0.130, 0.128, 0.126],
        "mu": [262, 73, 27.2, 13.0, 7.2, 4.4],
        "limit_c": 90.0, "limit_text": "ages quickly above about 90 °C",
        "evaporation": 0.0, "emissivity": 0.9, "note": "mineral oil, 32 mm²/s at 40 °C"},
}
DEFAULT_FLUID = "Water"


def fluid_props(name, t):
    """Properties of a liquid at the temperature(s) t °C (clamped to its table): {"rho", "cp",
    "k", "mu" (Pa·s), "nu" (m²/s), "alpha" (m²/s), "pr", "beta" (1/K)}; arrays for array t."""
    f = FLUIDS.get(name) or FLUIDS[DEFAULT_FLUID]
    temps = np.asarray(f["t"], float)
    scalar = np.ndim(t) == 0
    tt = np.clip(np.asarray(t, float), temps[0], temps[-1])
    rho = np.interp(tt, temps, f["rho"])
    cp = np.interp(tt, temps, f["cp"])
    k = np.interp(tt, temps, f["k"])
    mu = np.exp(np.interp(tt, temps, np.log(np.asarray(f["mu"], float)))) * 1e-3   # log-linear
    slope = np.gradient(np.asarray(f["rho"], float), temps)
    beta = np.maximum(-np.interp(tt, temps, slope) / rho, 0.0)
    out = {"rho": rho, "cp": cp, "k": k, "mu": mu, "nu": mu / rho, "alpha": k / (rho * cp),
           "pr": mu * cp / k, "beta": beta}
    return {key: float(value) for key, value in out.items()} if scalar else out


def fluid_density(name, t):
    f = FLUIDS.get(name) or FLUIDS[DEFAULT_FLUID]
    temps = np.asarray(f["t"], float)
    return np.interp(np.clip(np.asarray(t, float), temps[0], temps[-1]), temps, f["rho"])


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
    # structural (mode "structural", simplecad_structural.py)
    "struct_analysis": "static",   # "static" (stresses) or "modal" (natural frequencies)
    "struct_mesh_mm": 0.0,         # element size; 0 = automatic
    "struct_max_elements": 30000,
    "struct_order": 2,             # 2: quadratic tetrahedra (default), 1: linear
    "struct_gravity": False,       # the parts' own weight, along -Z
    "struct_thermal": "none",      # thermal expansion: "none", "uniform" or "conduction"
    "struct_reference_c": 20.0,    # stress-free temperature
    "struct_uniform_c": 70.0,      # temperature of all parts for "uniform"
    "struct_modes": 6,             # natural frequencies to find
    "struct_scale": 0.0,           # exaggeration of the deformed shape in the view; 0 = automatic
}
STRUCTURAL_ANALYSES = {"static": "Static (stress, deformation)", "modal": "Natural frequencies"}
STRUCTURAL_THERMAL = {"none": "None", "uniform": "Uniform temperature",
                      "conduction": "Temperatures of a Conduction run"}
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
                   "flow", "pcb", "package")


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
    "coolant": ("Coolant channel", (0.1, 0.65, 1.0)),
    "bath": ("Liquid bath", (0.45, 0.45, 1.0)),
    # Structural studies
    "support_fixed": ("Fixed support", (1.0, 0.35, 0.3)),
    "support_roller": ("Sliding support", (1.0, 0.62, 0.25)),
    "support_pin": ("Pinned support", (0.9, 0.5, 1.0)),
    "force": ("Force", (0.35, 0.95, 0.35)),
    "pressure": ("Pressure", (0.3, 0.8, 1.0)),
    "torque": ("Torque", (1.0, 0.9, 0.3)),
}
STRUCTURAL_KINDS = ("support_fixed", "support_roller", "support_pin", "force", "pressure", "torque")
THERMAL_KINDS = tuple(kind for kind in CONDITION_KINDS if kind not in STRUCTURAL_KINDS)


def condition_kinds(mode):
    """The face condition kinds of a solver mode."""
    return STRUCTURAL_KINDS if mode == "structural" else THERMAL_KINDS


LIQUID_KINDS = ("coolant", "bath")
LIQUID_MODELS = {"simple": "Correlations (fast)", "flow": "Liquid flow solved (Airflow mode)"}
BATH_MODES = {"container": "Fills the container", "fixed": "Large volume, fixed temperature"}
CONDITION_HINTS = {
    "coolant": "Pick the inner faces of a pipe, drilled channel or cold-plate passage. The liquid "
               "enters at the lower end (or reversed), takes up heat from the wall and warms "
               "along the channel; the heat transfer follows pipe-flow correlations (laminar with "
               "the entry region, transitional, turbulent).",
    "bath": "Pick faces that touch the liquid, e.g. the inside of the container and the heater. "
            "'Fills the container': the liquid fills the container up to the fill level and is "
            "well mixed (one temperature), with natural convection on the wetted faces and the "
            "open surface losing heat to the room. 'Large volume': the faces see liquid at a "
            "fixed temperature.",
    "fixed": "The faces are held at this temperature, e.g. a part clamped to a cold plate or a "
             "water-cooled surface.",
    "flux": "This heat enters through the faces (e.g. a heating foil or a known load). The faces "
            "still exchange heat with the air.",
    "convection": "Own heat transfer coefficient and air temperature for these faces, e.g. a fan-"
                  "cooled side (h ≈ 20–100 W/m²K). Radiation stays as set for the environment.",
    "insulated": "No heat crosses these faces (no convection, no radiation), e.g. a face against "
                 "insulation or a mounting surface that hardly conducts.",
    "support_fixed": "The faces cannot move at all: clamped, welded or bolted firmly to something "
                     "rigid.",
    "support_roller": "The faces cannot move along their normal but slide freely along the surface: "
                      "resting on a smooth flat support, or a symmetry plane.",
    "support_pin": "Pick the round face of a hole or a shaft: it may turn about its axis but not move "
                   "sideways, like a part on a pin, bolt or in a bearing.",
    "force": "A total force spread evenly over the faces, e.g. a weight hanging on them or a pull. "
             "X, Y and Z are world directions (gravity points along −Z).",
    "pressure": "Pushes on the faces along their normals (negative values pull). 1 bar = 0.1 MPa; "
                "MPa = N/mm².",
    "torque": "Turns the faces about their axis: a round face about its own axis, a flat face about "
              "its normal through its centre (right-hand rule).",
}


def condition_new(kind, name):
    return {"name": name, "kind": kind, "enabled": True, "faces": [],
            "temperature_c": 25.0,                                  # fixed
            "flux_mode": "total", "power_w": 1.0, "flux_w_m2": 1000.0,  # flux
            "profile": None, "thermostat": None,                    # flux: time profile, control
            "h": 25.0, "fluid_ambient": True, "fluid_c": 25.0,      # convection
            # coolant channel and liquid bath
            "fluid": DEFAULT_FLUID, "liquid_model": "simple",
            "flow_mode": "fixed", "flow_lpm": 1.0,                   # coolant: L/min or a pump
            "pump_max_lpm": 4.0, "pump_max_kpa": 20.0, "pump_curve": [],
            # a pump of the project (its group name): its estimated curve at its motor speed,
            # copied in, for water (pump_density); the solver scales it to the liquid's density
            "pump_source": "", "pump_rpm": 0.0, "pump_density": 0.0,
            "inlet_c": 25.0, "inlet_from": "", "reverse": False,
            "bath_mode": "container", "fill_auto": True, "fill_z": 0.0,   # bath
            "bath_c": 25.0, "bath_start_auto": True, "evaporation": True,
            # structural: force (N, world axes), pressure (MPa), torque (N·m), pinned support
            "fx": 0.0, "fy": 0.0, "fz": -100.0, "pressure_mpa": 0.1, "torque_nm": 1.0,
            "torque_reverse": False, "pin_axial_free": False}


def condition_settings(data):
    """A validated condition from loaded data."""
    kind = data.get("kind")
    if kind not in CONDITION_KINDS:
        raise ValueError(f"invalid face condition kind: {kind!r}")
    condition = condition_new(kind, str(data.get("name", CONDITION_KINDS[kind][0])))
    condition["enabled"] = bool(data.get("enabled", True))
    for key in ("temperature_c", "power_w", "flux_w_m2", "h", "fluid_c", "flow_lpm",
                "pump_max_lpm", "pump_max_kpa", "inlet_c", "fill_z", "bath_c",
                "fx", "fy", "fz", "pressure_mpa", "torque_nm"):
        if key in data:
            condition[key] = float(data[key])
    for key in ("flow_lpm", "pump_max_lpm", "pump_max_kpa"):
        condition[key] = max(0.0, condition[key])
    if data.get("fluid") in FLUIDS:
        condition["fluid"] = data["fluid"]
    condition["liquid_model"] = "flow" if data.get("liquid_model") == "flow" else "simple"
    condition["flow_mode"] = "pump" if data.get("flow_mode") == "pump" else "fixed"
    condition["pump_curve"] = [[float(q), float(p)] for q, p in data.get("pump_curve") or []]
    condition["pump_source"] = str(data.get("pump_source") or "")
    condition["pump_rpm"] = max(0.0, float(data.get("pump_rpm") or 0.0))
    condition["pump_density"] = max(0.0, float(data.get("pump_density") or 0.0))
    condition["inlet_from"] = str(data.get("inlet_from") or "")
    condition["bath_mode"] = "fixed" if data.get("bath_mode") == "fixed" else "container"
    for key, default in (("reverse", False), ("fill_auto", True), ("bath_start_auto", True),
                         ("evaporation", True), ("torque_reverse", False), ("pin_axial_free", False)):
        condition[key] = bool(data.get(key, default))
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


def study_conditions(study, jobs, kinds=None):
    """Face conditions for the solver: [{..condition values.., "targets": {job index: [face
    indices]}}] for the enabled conditions (of `kinds`), in order (later ones win on shared
    faces)."""
    job_of = {job["key"]: index for index, job in enumerate(jobs)}
    component_index = {c["name"]: i for i, c in enumerate(study["components"])}
    out = []
    for condition in study.get("conditions", []):
        if not condition.get("enabled", True) or (kinds is not None and condition["kind"] not in kinds):
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
    if settings["mode"] not in ("conduction", "airflow", "structural"):
        settings["mode"] = "conduction"
    if settings["struct_analysis"] not in STRUCTURAL_ANALYSES:
        settings["struct_analysis"] = "static"
    if settings["struct_thermal"] not in STRUCTURAL_THERMAL:
        settings["struct_thermal"] = "none"
    settings["struct_order"] = 1 if settings["struct_order"] == 1 else 2
    settings["struct_mesh_mm"] = max(0.0, settings["struct_mesh_mm"])
    settings["struct_max_elements"] = min(2_000_000, max(1000, settings["struct_max_elements"]))
    settings["struct_modes"] = min(30, max(1, settings["struct_modes"]))
    settings["struct_scale"] = max(0.0, settings["struct_scale"])
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
        data = {key: pack_arrays(component[key]) if key == "pcb" else copy.deepcopy(component[key])
                for key in _COMPONENT_KEYS if key in component}
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
        if isinstance(item.get("pcb"), dict):
            component["pcb"] = unpack_arrays(item["pcb"])
        if isinstance(item.get("package"), dict):
            component["package"] = package_settings(item["package"])
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
        data = {key: copy_sharing_arrays(value) if key == "pcb" else copy.deepcopy(value)
                for key, value in component.items() if key != "bodies" and not key.startswith("_")}
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
            if component.get("pcb") is not None:
                job["pcb"] = component["pcb"]
                job["pcb_frame"] = pcb_frame_of(component)
            if component.get("package") is not None:
                job["junction"] = dict(component["package"])
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


def _forced_h(speed, length, nu_fluid=AIR_KINEMATIC_VISCOSITY, prandtl=AIR_PRANDTL,
              k_fluid=AIR_CONDUCTIVITY):
    """Forced-convection coefficient W/(m²·K) of air (or another fluid) flowing along a surface
    of `length` m at `speed` m/s (flat plate, laminar / turbulent, average over the length)."""
    length = np.maximum(length, 1e-3)
    re = np.maximum(speed, 0.0) * length / nu_fluid
    pr3 = prandtl ** (1.0 / 3.0)
    nu = np.where(re < 5e5, 0.664 * np.sqrt(re) * pr3,
                  np.maximum(0.037 * re ** 0.8 - 871.0, 0.664 * np.sqrt(5e5)) * pr3)
    return nu * k_fluid / length


# ── liquid cooling: pipe flow, channel shape, natural convection in liquids ───────────
def _duct_constants(geometry):
    """Fully developed laminar Nu (uniform wall temperature, uniform heat flux) and f·Re (Darcy)
    of a channel cross-section: round, or rectangular by its aspect ratio (Shah & London)."""
    if geometry.get("shape") == "rect":
        a = float(np.clip(geometry.get("aspect", 1.0), 0.0, 1.0))
        nu_t = 7.541 * (1 - 2.610 * a + 4.970 * a ** 2 - 5.119 * a ** 3 + 2.702 * a ** 4 - 0.548 * a ** 5)
        nu_h = 8.235 * (1 - 2.0421 * a + 3.0853 * a ** 2 - 2.4765 * a ** 3 + 1.0578 * a ** 4
                        - 0.1861 * a ** 5)
        f_re = 96.0 * (1 - 1.3553 * a + 1.9467 * a ** 2 - 1.7012 * a ** 3 + 0.9564 * a ** 4
                       - 0.2537 * a ** 5)
        return nu_t, nu_h, f_re
    return 3.657, 4.364, 64.0


def pipe_flow(fluid, flow_m3s, geometry, t_bulk, t_wall=None, wall_mode=0.5):
    """Heat transfer and pressure drop of a liquid flowing through a channel (geometry from
    channel_geometry). Laminar: mean Nu over the channel length with the entry region
    (Gnielinski, VDI Heat Atlas), blended between uniform wall temperature (wall_mode 0, Nu → 3.66
    in a long round pipe) and uniform heat flux (wall_mode 1, Nu → 4.36); turbulent: Gnielinski;
    in between interpolated. Wall-temperature property correction when t_wall is given. Pressure
    drop: Darcy–Weisbach, smooth walls, plus 1.5 dynamic heads for entry and exit."""
    p = fluid_props(fluid, t_bulk)
    dh = max(geometry["dh_mm"], 1e-3) / 1000.0
    area = max(geometry["area_mm2"], 1e-6) * 1e-6
    length = max(geometry["length_mm"] / 1000.0, dh)
    velocity = max(flow_m3s, 0.0) / area
    re = p["rho"] * velocity * dh / p["mu"]
    pr = p["pr"]
    nu_t, nu_h, f_re = _duct_constants(geometry)
    w = float(np.clip(wall_mode, 0.0, 1.0))

    def laminar(re_):
        gz = re_ * pr * dh / length
        mean_t = np.cbrt(nu_t ** 3 + 0.7 ** 3 + (1.615 * np.cbrt(gz) - 0.7) ** 3 +
                         ((2.0 / (1.0 + 22.0 * pr)) ** (1.0 / 6.0) * math.sqrt(gz)) ** 3)
        mean_h = np.cbrt(nu_h ** 3 + 0.6 ** 3 + (1.953 * np.cbrt(gz) - 0.6) ** 3 +
                         (0.924 * np.cbrt(pr) * math.sqrt(re_ * dh / length)) ** 3)
        return float(mean_t + w * (mean_h - mean_t))

    def turbulent(re_):
        xi = (1.8 * math.log10(re_) - 1.5) ** -2
        return float((xi / 8.0) * (re_ - 1000.0) * pr /
                     (1.0 + 12.7 * math.sqrt(xi / 8.0) * (pr ** (2.0 / 3.0) - 1.0)) *
                     (1.0 + (dh / length) ** (2.0 / 3.0)))

    if re <= 2300.0:
        nu, regime = laminar(re), "laminar"
    elif re >= 1e4:
        nu, regime = turbulent(re), "turbulent"
    else:
        gamma = (re - 2300.0) / (1e4 - 2300.0)
        nu, regime = (1.0 - gamma) * laminar(2300.0) + gamma * turbulent(1e4), "transitional"
    if t_wall is not None and np.isfinite(t_wall):
        pw = fluid_props(fluid, t_wall)
        nu *= (p["mu"] / pw["mu"]) ** 0.14 if regime == "laminar" else (pr / pw["pr"]) ** 0.11
    h = nu * p["k"] / dh

    def turbulent_friction(re_):
        return (1.8 * math.log10(re_) - 1.5) ** -2

    if re < 2300.0:
        friction = f_re / max(re, 1e-9)
    elif re > 4000.0:
        friction = turbulent_friction(re)
    else:
        gamma = (re - 2300.0) / 1700.0
        friction = (1.0 - gamma) * f_re / 2300.0 + gamma * turbulent_friction(4000.0)
    dynamic = 0.5 * p["rho"] * velocity ** 2
    dp = (friction * length / dh + 1.5) * dynamic if re > 0 else 0.0
    return {"velocity": velocity, "re": re, "pr": pr, "regime": regime, "nu": nu, "h": h,
            "friction": friction, "dp": dp, "rho": p["rho"], "cp": p["cp"], "k": p["k"],
            "mu": p["mu"], "gz": re * pr * dh / length, "wall_mode": w}


def pump_curve(condition):
    """(flows m³/s, pressures Pa) of a coolant pump, falling with the flow: a table
    [[L/min, kPa], …] or the line from the shut-off pressure to the free-flowing flow rate. A
    table made for another liquid (pump_density, kg/m³) is scaled to this liquid's density: a
    centrifugal pump gives the same head, so its pressure goes with the density."""
    table = condition.get("pump_curve") or []
    if len(table) >= 2:
        points = np.array(sorted((float(q), float(p)) for q, p in table))
        scale = 1.0
        if float(condition.get("pump_density") or 0.0) > 0.0:
            scale = float(fluid_density(condition.get("fluid", DEFAULT_FLUID), 20.0)) / float(condition["pump_density"])
        return points[:, 0] / 60000.0, np.maximum(points[:, 1], 0.0) * 1000.0 * scale
    q_max = max(float(condition.get("pump_max_lpm", 4.0)), 1e-6) / 60000.0
    return np.array([0.0, q_max]), np.array([max(float(condition.get("pump_max_kpa", 20.0)), 0.0)
                                             * 1000.0, 0.0])


def project_pumps(objects):
    """Pumps made with the pump tool in the project: {group name: the housing's parameters}."""
    pumps = {}
    for entry in objects:
        parameters = entry.get("pump_parameters") if isinstance(entry, dict) else None
        if not parameters or entry.get("preview") or parameters.get("role") != "housing":
            continue
        pumps[str(parameters.get("group") or entry.get("name"))] = parameters
    return pumps


def project_pump_curve(parameters, samples=24):
    """(table [[L/min, kPa], …] for water, motor speed rpm) of a project pump: the pump tool's
    estimate at the motor speed saved with it."""
    rpm = float(parameters.get("motor_speed") or 1500.0)
    d = cad.pump_dimensions(parameters)
    curve = cad.pump_curve(d, rpm, (1.0, d["d_out"], 0.0), samples=samples)
    return [[round(q, 6), round(p, 6)] for q, p in zip(curve["flow"], curve["pump"])], rpm


def pump_operating_point(curve, system_dp):
    """Flow (m³/s) where the pump curve meets the rising system curve system_dp(Q) (Pa)."""
    flows, pressures = curve
    low, high = 0.0, float(flows[-1])
    if high <= 0.0:
        return 0.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if np.interp(middle, flows, pressures) > system_dp(middle):
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def channel_march(area, wall, h, mdot_cp, t_in):
    """Liquid warming along a channel, segment by segment from the inlet (exact for a constant
    wall temperature within a segment). area (n,) m², wall (n,) °C, h W/(m²·K), mdot_cp W/K.
    Returns (t_face (n,): the liquid temperature the faces of each segment see, so that
    h·A·(T_wall − t_face) is exactly the heat the liquid takes up there; t_after (n,); heat (n,) W)."""
    n = len(area)
    h = np.broadcast_to(np.asarray(h, float), (n,))      # one value, or one per segment
    t_face, t_after, heat = np.empty(n), np.empty(n), np.zeros(n)
    t = float(t_in)
    for i in range(n):
        if area[i] <= 0 or mdot_cp <= 0 or h[i] <= 0:
            t_face[i] = t_after[i] = t
            continue
        ntu = h[i] * area[i] / mdot_cp
        t_new = wall[i] + (t - wall[i]) * math.exp(-ntu)
        heat[i] = mdot_cp * (t_new - t)
        t_face[i] = wall[i] - heat[i] / (h[i] * area[i])
        t_after[i] = t = t_new
    return t_face, t_after, heat


def wall_mode_of(area, wall, liquid):
    """0 … 1: how much the wall behaves like uniform heat flux (wall − liquid difference even
    along the channel) rather than uniform wall temperature (the wall itself even)."""
    weights = np.asarray(area, float)
    if weights.sum() <= 0 or len(weights) < 3:
        return 0.5
    weights = weights / weights.sum()

    def spread(values):
        mean = float(weights @ values)
        return float(weights @ (values - mean) ** 2)
    a, b = spread(np.asarray(wall, float)), spread(np.asarray(wall, float) - liquid)
    return 0.5 if a + b < 1e-12 else a / (a + b)


def _liquid_natural_h(fluid, dT, length, orientation, t_film):
    """Natural-convection coefficient W/(m²·K) of a surface in a liquid at rest. dT = surface −
    liquid (signed); orientation 0 = vertical, 1 = the surface faces up (into the liquid above
    it), 2 = faces down. Churchill & Chu (vertical), Lloyd & Moran (hot face up / cold face down),
    0.27 Ra^¼ for the stable cases."""
    p = fluid_props(fluid, t_film)
    dT = np.asarray(dT, float)
    length = np.maximum(np.asarray(length, float), 1e-3)
    ra = GRAVITY * np.maximum(p["beta"], 1e-5) * np.maximum(np.abs(dT), 0.02) * length ** 3 / \
        (p["nu"] * p["alpha"])
    psi = (1.0 + (0.492 / p["pr"]) ** (9.0 / 16.0)) ** (-16.0 / 9.0)
    vertical = (0.825 + 0.387 * (ra * psi) ** (1.0 / 6.0)) ** 2
    rising = np.where(ra < 1e7, 0.54 * ra ** 0.25, 0.15 * np.cbrt(ra))
    stable = 0.27 * ra ** 0.25
    plume = (orientation == 1) == (dT > 0)          # hot face up, or cold face down
    nu = np.where(orientation == 0, vertical, np.where(plume, rising, stable))
    return np.maximum(nu, 1.0) * p["k"] / length


def _saturation_pressure(t):
    """Saturation pressure of water vapour (Pa) at t °C (Magnus, at most 1 atm)."""
    t = np.minimum(np.asarray(t, float), 100.0)
    return np.minimum(610.94 * np.exp(17.625 * t / (t + 243.04)), 101325.0)


def _evaporation_h(t_surface, t_air, h_air, share, humidity=0.5):
    """Extra heat transfer coefficient W/(m²·K) for evaporation from an open liquid surface into
    room air of 50 % humidity (Lewis analogy from the convective coefficient h_air)."""
    if share <= 0:
        return np.zeros_like(np.asarray(t_surface, float))
    pressure = 101325.0
    ps = _saturation_pressure(np.clip(t_surface, -40.0, 100.0))
    pa = humidity * _saturation_pressure(t_air)
    # vapour mass fractions at the surface and in the room (bounded by 1 at boiling)
    ys = 0.622 * ps / (pressure - 0.378 * ps)
    ya = 0.622 * pa / (pressure - 0.378 * pa)
    q = share * np.asarray(h_air, float) / 1005.0 * np.maximum(ys - ya, 0.0) * 2.45e6
    return q / np.maximum(np.asarray(t_surface, float) - t_air, 0.5)


def _evaporation_q(t_surface, t_air, length, share):
    """Evaporation heat flux W/m² of an open liquid surface at t_surface into room air (natural
    convection above it for the Lewis analogy)."""
    t_surface = np.asarray(t_surface, float)
    h_air = _natural_h(t_surface - t_air, length, np.ones(t_surface.shape, int))
    return _evaporation_h(t_surface, t_air, h_air, share) * np.maximum(t_surface - t_air, 0.5) * \
        (t_surface > t_air)


def channel_geometry(triangles, reverse=False):
    """Shape of a coolant passage from its wall triangles (n, 3, 3) in mm: the position along the
    passage comes from shortest paths over points sampled on the wall (works for bent and
    serpentine channels), the cross-section from slices across it.
    Returns {"length_mm", "wall_area_mm2", "area_mm2", "perimeter_mm", "dh_mm", "shape"
    ("round" | "rect"), "aspect", "size_mm" (diameter, or width and height), "points" (samples),
    "s" (mm from the inlet per sample), "centre" (slice centres from the inlet), "centre_s",
    "tangent", "inlet", "outlet", "inlet_dir", "outlet_dir"}. The inlet is the lower end unless
    `reverse`."""
    from scipy.sparse import coo_matrix, csgraph
    from scipy.spatial import ConvexHull, cKDTree
    P = np.asarray(triangles, float).reshape(-1, 3, 3)
    cross = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    tri_area = 0.5 * np.linalg.norm(cross, axis=1)
    keep = tri_area > 1e-12
    P, tri_area = P[keep], tri_area[keep]
    wall_area = float(tri_area.sum())
    if wall_area <= 0:
        raise ThermalError("A coolant channel has no faces (pick the inner faces of the passage).")
    rng = np.random.default_rng(12345)
    cumulative = np.cumsum(tri_area)
    unit_normal = cross[keep] / (2.0 * tri_area[:, None])

    def sample(n):
        """(points, unit normals of their triangles) spread evenly over the wall."""
        u = (np.arange(n) + rng.random(n)) / n * cumulative[-1]
        tri = np.minimum(np.searchsorted(cumulative, u, side="right"), len(P) - 1)
        r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
        a, b, c = 1.0 - r1, r1 * (1.0 - r2), r1 * r2
        return (a[:, None] * P[tri, 0] + b[:, None] * P[tri, 1] + c[:, None] * P[tri, 2],
                unit_normal[tri])

    # open ends: boundary loops of the welded wall mesh (edges of one triangle only); the two
    # longest are the inlet and outlet rims, and the distance from a whole rim measures the
    # position along the passage without leaning around the circumference
    corners = P.reshape(-1, 3)
    _keys, first, vid = np.unique(np.round(corners / 1e-4).astype(np.int64), axis=0,
                                  return_index=True, return_inverse=True)
    vid = vid.reshape(-1, 3)
    coords = corners[first]
    edges = np.sort(np.concatenate([vid[:, [0, 1]], vid[:, [1, 2]], vid[:, [2, 0]]]), axis=1)
    edges, uses = np.unique(edges, axis=0, return_counts=True)
    boundary = edges[uses == 1]
    rims = []
    if len(boundary):
        n_vertices = len(coords)
        loop_graph = coo_matrix((np.ones(len(boundary)), (boundary[:, 0], boundary[:, 1])),
                                shape=(n_vertices, n_vertices))
        _n, label = csgraph.connected_components(loop_graph, directed=False)
        edge_label = label[boundary[:, 0]]
        edge_length = np.linalg.norm(coords[boundary[:, 0]] - coords[boundary[:, 1]], axis=1)
        totals = np.bincount(edge_label, edge_length)
        order = np.argsort(-totals)
        if len(order) >= 2 and totals[order[1]] > 0.3 * totals[order[0]]:
            for lab in order[:2]:
                spacing = totals[lab] / 64.0
                pieces = []
                for (a, b), size in zip(boundary[edge_label == lab], edge_length[edge_label == lab]):
                    count = max(1, int(math.ceil(size / spacing)))
                    f = np.arange(count)[:, None] / count
                    pieces.append(coords[a] + f * (coords[b] - coords[a]))
                rims.append(np.concatenate(pieces))

    def along(points, use_rims=True):
        """(position along the passage per sample in path units, share of connected samples,
        whether the rims were used)."""
        rim_list = rims if use_rims else []
        nodes = np.concatenate([points] + rim_list)
        n, n_points = len(nodes), len(points)
        k = min(13, n)
        distance, neighbour = cKDTree(nodes).query(nodes, k=k)
        rows = np.repeat(np.arange(n), k - 1)
        graph = coo_matrix((distance[:, 1:].ravel() + 1e-9, (rows, neighbour[:, 1:].ravel())),
                           shape=(n, n)).tocsr()
        graph = graph.maximum(graph.T)
        _count, labels = csgraph.connected_components(graph, directed=False)
        main = labels == np.argmax(np.bincount(labels))
        if rim_list:
            first_rim = np.arange(n_points, n_points + len(rims[0]))
            da = csgraph.dijkstra(graph, indices=first_rim, min_only=True)
            db = csgraph.dijkstra(graph, indices=np.arange(n_points + len(rims[0]), n),
                                  min_only=True)
            reached = main[:n_points] & np.isfinite(da[:n_points]) & np.isfinite(db[:n_points])
            if reached.sum() < 0.5 * main[:n_points].sum():
                # the rims lie on pieces apart from the main one (faces that do not join up):
                # between the farthest samples instead
                return along(points, use_rims=False)
        else:                       # no open ends found: between the two farthest samples
            start = int(np.flatnonzero(main)[0])
            d0 = csgraph.dijkstra(graph, indices=start)
            a = int(np.argmax(np.where(main, d0, -np.inf)))
            da = csgraph.dijkstra(graph, indices=a)
            b = int(np.argmax(np.where(main, da, -np.inf)))
            db = csgraph.dijkstra(graph, indices=b)
        main &= np.isfinite(da) & np.isfinite(db)
        s = np.full(n, np.nan)
        s[main] = (da[main] - db[main]) / 2.0
        s -= s[main].min()
        if not main.all():                    # stray samples: position of the nearest main one
            tree = cKDTree(nodes[main])
            s[~main] = s[main][tree.query(nodes[~main])[1]]
        return s[:n_points], float(main[:n_points].mean()), bool(rim_list)

    points, normals = sample(6000)
    s, share, used_rims = along(points)
    if not used_rims:
        rims.clear()
    if rims:
        # faces of separate parts (a duct built from plates) leave open edges everywhere: the
        # rims are the ends only when they give about the length between the farthest samples
        s_far, share_far, _used = along(points, use_rims=False)
        if float(s.max()) < 0.7 * float(s_far.max()):
            rims.clear()
            s, share = s_far, share_far
    length = float(s.max())
    if length <= 0:
        raise ThermalError("A coolant channel needs the faces along the whole passage.")
    estimate = wall_area / length / math.pi                       # diameter of a round passage
    wanted = int(np.clip(wall_area / (estimate / 8.0) ** 2, 6000, 60000))
    if wanted > 9000:
        points, normals = sample(wanted)
        s, share, used_rims = along(points)
        if not used_rims:
            rims.clear()
        length = float(s.max())
    if share < 0.8:
        raise ThermalError("The faces of a coolant channel must form one connected passage (pick "
                           "the inner faces of one channel; make a condition for each channel).")

    def slices(s, length, count):
        edges = np.linspace(0.0, length, count + 1)
        which = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, count - 1)
        counts = np.bincount(which, minlength=count)
        centre = np.stack([np.bincount(which, points[:, a], count) for a in range(3)],
                          axis=1) / np.maximum(counts, 1)[:, None]
        filled = counts > 0
        middle = 0.5 * (edges[1:] + edges[:-1])
        for a in range(3):
            centre[~filled, a] = np.interp(middle[~filled], middle[filled], centre[filled, a])
        return which, centre, middle

    if not rims:
        # between the farthest samples the position leans around the circumference near the
        # ends: projected onto a centre line through wide slices (which average the lean out)
        n_wide = int(np.clip(round(length / max(2.0 * estimate, 1e-3)), 3, 100))
        _which, wide, _middle = slices(s, length, n_wide)
        segment = np.diff(wide, axis=0)
        seg_length = np.maximum(np.linalg.norm(segment, axis=1), 1e-12)
        arc = np.r_[0.0, np.cumsum(seg_length)]
        best = np.full(len(points), np.inf)
        new_s = np.zeros(len(points))
        for j in range(len(segment)):
            t = ((points - wide[j]) @ segment[j]) / seg_length[j] ** 2
            t = np.clip(t, -np.inf if j == 0 else 0.0, np.inf if j == len(segment) - 1 else 1.0)
            dist = np.linalg.norm(points - (wide[j] + t[:, None] * segment[j]), axis=1)
            better = dist < best
            best[better] = dist[better]
            new_s[better] = (arc[j] + t * seg_length[j])[better]
        s = new_s - new_s.min()
        length = float(s.max())

    # path lengths over the samples run a few per cent long: scaled to the length of the line
    # through the slice centres
    n_coarse = int(np.clip(round(length / max(estimate, 1e-3)), 3, 200))
    _which, centre, middle = slices(s, length, n_coarse)
    polyline = float(np.linalg.norm(np.diff(centre, axis=0), axis=1).sum())
    if middle[-1] > middle[0] and polyline > 0:
        scale = polyline / (middle[-1] - middle[0])
        s, length, estimate = s * scale, length * scale, estimate / scale
    # slices across the passage: centre line, direction and cross-section
    n_slices = int(np.clip(round(length / max(estimate / 2.0, 1e-3)), 4, 400))
    which, centre, middle = slices(s, length, n_slices)
    tangent = np.gradient(centre, axis=0) if n_slices > 1 else np.array([[1.0, 0.0, 0.0]])
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    areas, perimeters = [], []
    lo, hi = int(0.1 * n_slices), max(int(0.9 * n_slices), int(0.1 * n_slices) + 1)
    for i in range(lo, hi):
        mine = points[which == i] - centre[i]
        if len(mine) < 8:
            continue
        t = tangent[i]
        u = np.cross(t, [0.0, 0.0, 1.0] if abs(t[2]) < 0.9 else [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(t, u)
        try:
            hull = ConvexHull(np.stack([mine @ u, mine @ v], axis=1))
        except Exception:
            continue
        areas.append(hull.volume)
        perimeters.append(hull.area)
    if areas:
        area = float(np.median(areas))
        perimeter = float(np.median(perimeters))
    else:                                                 # too few samples: assume round
        perimeter = wall_area / length
        area = perimeter ** 2 / (4.0 * math.pi)
    dh = 4.0 * area / max(perimeter, 1e-9)
    circularity = 4.0 * math.pi * area / max(perimeter, 1e-9) ** 2
    if circularity > 0.9:
        shape, aspect, size = "round", 1.0, [dh]
    else:
        half = perimeter / 2.0
        disc = max(half * half - 4.0 * area, 0.0)
        width, height = (half + math.sqrt(disc)) / 2.0, (half - math.sqrt(disc)) / 2.0
        shape, aspect, size = "rect", height / max(width, 1e-9), [width, height]
    # inlet at the lower end (then smaller X, then Y) unless reversed; the ends are compared by
    # the mean of the samples near them (robust against the exact end slices)
    delta = points[s > 0.85 * length].mean(0) - points[s < 0.15 * length].mean(0)
    lower_first = True
    for axis in (2, 0, 1):
        if abs(delta[axis]) > 0.01 * length:
            lower_first = bool(delta[axis] > 0)
            break
    if lower_first == bool(reverse):
        s = length - s
        centre, tangent, middle = centre[::-1], -tangent[::-1], length - middle[::-1]
    return {"length_mm": length, "wall_area_mm2": wall_area, "area_mm2": area,
            "perimeter_mm": perimeter, "dh_mm": dh, "shape": shape, "aspect": aspect,
            "size_mm": size, "points": points.astype(np.float32), "s": s,
            "normals": normals.astype(np.float32),
            "centre": centre, "centre_s": middle, "tangent": tangent,
            "inlet": centre[0], "outlet": centre[-1], "inlet_dir": tangent[0],
            "outlet_dir": tangent[-1]}


def channel_text(geometry):
    if geometry["shape"] == "round":
        cross = f"round, Ø {geometry['size_mm'][0]:.3g} mm"
    else:
        cross = (f"rectangular {geometry['size_mm'][0]:.3g} × {geometry['size_mm'][1]:.3g} mm "
                 f"(hydraulic Ø {geometry['dh_mm']:.3g} mm)")
    return f"{cross}, {geometry['length_mm']:.4g} mm long"


BATH_CONDUCTIVITY = 3000.0     # W/(m·K) inside a well-mixed bath body (far above its film)
BATH_FILM_GUESS = 300.0        # W/(m²·K) first guess of the film on the wetted faces


def _condition_triangles(condition, bodies):
    """World triangles (n, 3, 3) in mm of the faces a condition picked, and their unit normals."""
    tris, normals = [], []
    for b, faces in condition["targets"].items():
        body = bodies[b]
        face_tris = body["job"].get("faces", [])
        chosen = [np.asarray(face_tris[f], np.int64) for f in faces if 0 <= f < len(face_tris)]
        if not chosen:
            continue
        P = body["V"][body["T"][np.concatenate(chosen)]]
        normal = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
        tris.append(P)
        normals.append(normal)
    if not tris:
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    return np.concatenate(tris), np.concatenate(normals)


def condition_world_triangles(study, condition):
    """World triangles (n, 3, 3) in mm of the faces a study condition picked (for the window)."""
    components = {c["name"]: c for c in study["components"]}
    tris = []
    for ref in condition["faces"]:
        component = components.get(ref["component"])
        if component is None or not 0 <= ref["body"] < len(component["bodies"]):
            continue
        body = component["bodies"][ref["body"]]
        faces = body.get("faces", [])
        if not 0 <= ref["face"] < len(faces):
            continue
        V = np.asarray(cad.assembly_world_vertices(component, body["vertices"]), float)
        T = np.asarray(body["triangles"], np.int64).reshape(-1, 3)
        tris.append(V[T[np.asarray(faces[ref["face"]]["tris"], np.int64)]])
    return np.concatenate(tris) if tris else np.zeros((0, 3, 3))


def _cells_of(points, faces):
    """Grid cells (index tuple) of points inside a grid with cell faces `faces` (per axis, mm)."""
    inside = np.ones(len(points), bool)
    index = []
    for a in range(3):
        inside &= (points[:, a] >= faces[a][0]) & (points[:, a] <= faces[a][-1])
        index.append(np.clip(np.searchsorted(faces[a], points[:, a]) - 1, 0, len(faces[a]) - 2))
    return tuple(i[inside] for i in index)


def _liquid_seed_cells(tris, normals, faces, open_cells, offset):
    """Open grid cells (bool grid open_cells) just in front of the picked triangles (`offset` mm
    along the normal, else behind it for meshes facing inwards), as an index tuple."""
    P = np.asarray(tris, float)
    centre = P.mean(1)
    points = np.concatenate([centre] + [(2.0 * P[:, v] + centre) / 3.0 for v in range(3)])
    direction = np.tile(normals, (4, 1))
    for sign in (1.0, -1.0):
        cells = _cells_of(points + sign * offset * direction, faces)
        keep = open_cells[cells]
        if keep.any():
            return tuple(i[keep] for i in cells)
    return tuple(np.zeros(0, np.int64) for _ in range(3))


def _bath_region(open_cells, faces, seeds, fill_z):
    """Liquid cells of a container: the open cells below the fill level (cell centres) that are
    connected to the seed cells. Returns (mask, leaks: the region reaches the grid border)."""
    _sp, _spla, ndi, _tree = _scipy()
    zc = 0.5 * (faces[2][1:] + faces[2][:-1])
    below = open_cells & (zc[None, None, :] <= fill_z)
    labels, _count = ndi.label(below)
    found = np.unique(labels[seeds])
    found = found[found > 0]
    region = np.isin(labels, found) if len(found) else np.zeros(open_cells.shape, bool)
    leaks = bool(region[0].any() or region[-1].any() or region[:, 0].any() or
                 region[:, -1].any() or region[:, :, 0].any() or region[:, :, -1].any())
    return region, leaks


def _bath_fill(condition, tris):
    """Fill level (world Z, mm) of a bath: the top of its picked faces or the value entered."""
    if condition.get("fill_auto", True) and len(tris):
        return float(np.asarray(tris)[:, :, 2].max())
    return float(condition.get("fill_z", 0.0))


def _bath_body(condition, number, low, high):
    """Solver record of the liquid of a bath that fills its container (a well-mixed body)."""
    fluid = condition["fluid"]
    table = FLUIDS.get(fluid) or FLUIDS[DEFAULT_FLUID]
    p20 = fluid_props(fluid, 20.0)
    name = f"{condition['name']} ({fluid})"
    extent = np.maximum(high - low, 1e-6)
    return {"job": {"key": (-1, number), "name": name, "faces": [], "zone": False,
                    "material": fluid},
            "name": name, "material": fluid,
            "props": {"k": BATH_CONDUCTIVITY, "cp": p20["cp"], "density": p20["rho"],
                      "emissivity": table["emissivity"], "max_c": table["limit_c"], "note": ""},
            "V": np.zeros((0, 3)), "T": np.zeros((0, 3), np.int64), "low": low, "high": high,
            "extent": extent, "volume": 0.0, "bbox_volume": float(np.prod(extent)), "area": 0.0,
            "k": np.full(3, BATH_CONDUCTIVITY), "emissivity": table["emissivity"], "power": 0.0,
            "profile": None, "thermostat": None, "k_curve": None,
            "cp_curve": (np.asarray(table["t"], float), np.asarray(table["cp"], float)),
            "synthetic": True, "bath": condition}


def _channel_cells(geometry, faces, open_cells, cell_size):
    """Open grid cells inside a coolant channel (geometry from channel_geometry): nearer to the
    centre line than the wall in that direction, between the inlet and the outlet; the largest
    connected piece. Returns (mask, inlet mask, outlet mask) — inlet / outlet: the first and
    last cell layer along the channel."""
    from scipy.spatial import cKDTree
    _sp, _spla, ndi, _tree = _scipy()
    points = np.asarray(geometry["points"], float)
    margin = geometry["dh_mm"]
    centres = [0.5 * (f[1:] + f[:-1]) for f in faces]
    box = []
    for a in range(3):
        lo = int(np.searchsorted(centres[a], points[:, a].min() - margin))
        hi = int(np.searchsorted(centres[a], points[:, a].max() + margin, side="right"))
        box.append(slice(lo, max(hi, lo)))
    box = tuple(box)
    sub_open = open_cells[box]
    empty = np.zeros(open_cells.shape, bool)
    if not sub_open.any():
        return empty, empty, empty
    idx = np.nonzero(sub_open)
    X = np.stack([centres[a][box[a]][idx[a]] for a in range(3)], axis=1)
    line, tangent, line_s = geometry["centre"], geometry["tangent"], geometry["centre_s"]
    _d, j = cKDTree(line).query(X)
    relative = X - line[j]
    axial = (relative * tangent[j]).sum(1)
    r_cell = np.linalg.norm(relative - axial[:, None] * tangent[j], axis=1)
    s_cell = line_s[j] + axial
    _d, w = cKDTree(points).query(X)
    along_ok = (s_cell > 0.0) & (s_cell < geometry["length_mm"])
    normals = geometry.get("normals")
    if normals is not None:
        # on the open side of the nearest wall point (the wall normals point into the channel;
        # a mesh facing inwards is noticed by almost nothing passing)
        side = ((X - points[w]) * np.asarray(normals, float)[w]).sum(1)
        inside = (side > 0) & along_ok
        if inside.sum() < 0.05 * along_ok.sum():
            inside = (side < 0) & along_ok
    else:
        relative_wall = points[w] - line[j]
        r_wall = np.linalg.norm(relative_wall - (relative_wall * tangent[j]).sum(1)[:, None] *
                                tangent[j], axis=1)
        inside = (r_cell < r_wall) & along_ok
    mask_sub = np.zeros(sub_open.shape, bool)
    mask_sub[tuple(i[inside] for i in idx)] = True
    s_sub = np.full(sub_open.shape, np.nan)
    s_sub[idx] = s_cell
    # open cells the channel encloses (corners the test missed) belong to the liquid, too
    mask_sub = ndi.binary_fill_holes(mask_sub | ~sub_open) & sub_open & np.isfinite(s_sub)
    labels, count = ndi.label(mask_sub)
    if count > 1:
        mask_sub = labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)
    if not mask_sub.any():
        return empty, empty, empty
    s_in = s_sub[mask_sub]
    layer = 1.2 * cell_size
    inlet_sub = mask_sub & (s_sub <= s_in.min() + layer)
    outlet_sub = mask_sub & (s_sub >= s_in.max() - layer)
    out = []
    for sub in (mask_sub, inlet_sub, outlet_sub):
        full = np.zeros(open_cells.shape, bool)
        full[box] = sub
        out.append(full)
    return tuple(out)


def _ordered_channels(records):
    """Coolant records in the order their liquid flows (an inlet fed by another channel's outlet
    comes after it); raises on a loop."""
    by_name = {record["condition"]["name"]: record for record in records}
    ordered, state = [], {}

    def visit(record):
        name = record["condition"]["name"]
        if state.get(name) == 1:
            raise ThermalError(f"The coolant channels feed each other in a loop ('{name}'); one of "
                               "them needs its own inlet temperature.")
        if state.get(name) == 2:
            return
        state[name] = 1
        upstream = by_name.get(record["condition"].get("inlet_from") or "")
        record["upstream"] = upstream
        if upstream is not None:
            visit(upstream)
        state[name] = 2
        ordered.append(record)

    for record in records:
        visit(record)
    return ordered


def _channel_flow(record, wall_faces, face_areas):
    """Flow of a coolant channel (fixed, or the pump operating point), its heat transfer
    coefficient and the wall temperature per segment, from the wall temperatures of its cell
    faces. Sets "flow", "flow_m3s", "mdot", "mdot_cp", "h", "t_in", "wall" of the record."""
    c = record["condition"]
    geometry = record["geometry"]
    seg, n_seg, area = record["seg"], record["n_seg"], record["area_seg"]
    wall = np.bincount(seg, wall_faces * face_areas, n_seg) / np.maximum(area, 1e-30)
    upstream = record.get("upstream")
    t_in = float(upstream["t_out"]) if upstream is not None and "t_out" in upstream \
        else float(c["inlet_c"])
    t_bulk = 0.5 * (t_in + record.get("t_out", t_in))
    t_wall = float((wall_faces * face_areas).sum() / max(face_areas.sum(), 1e-30))
    mode = record.get("mode", 0.5)
    if c["flow_mode"] == "pump":
        flow_m3s = pump_operating_point(
            pump_curve(c), lambda q: pipe_flow(c["fluid"], q, geometry, t_bulk, t_wall, mode)["dp"])
    else:
        flow_m3s = c["flow_lpm"] / 60000.0
    flow = pipe_flow(c["fluid"], flow_m3s, geometry, t_bulk, t_wall, mode)
    mdot = flow_m3s * fluid_props(c["fluid"], t_in)["rho"]
    record.update(flow=flow, flow_m3s=flow_m3s, mdot=mdot, mdot_cp=mdot * flow["cp"],
                  h=flow["h"], t_in=t_in, wall=wall)
    return flow


def _update_channel(record, wall_faces, face_areas, relax=1.0):
    """One coolant channel: flow (fixed or pump operating point), heat transfer coefficient and
    the liquid temperature along it from the wall temperatures of its cell faces. Updates the
    record ("h", "t_face" per segment, "t_out", "flow", …) and returns the liquid temperature
    every face sees."""
    flow = _channel_flow(record, wall_faces, face_areas)
    seg, n_seg, area = record["seg"], record["n_seg"], record["area_seg"]
    wall, t_in, mdot, mdot_cp = record["wall"], record["t_in"], record["mdot"], record["mdot_cp"]
    flow_m3s = record["flow_m3s"]
    # the wall follows the liquid by a share alpha per segment (≈ 1 where a heat flux drives the
    # wall, ≈ 0 where its temperature is held), estimated from the last two updates; marching
    # with the wall predicted that way converges in a few iterations even when the liquid warms
    # a lot (a plain update from the last wall temperatures would crawl)
    previous = record.get("t_face")
    alpha = record.get("alpha", np.zeros(n_seg))
    if previous is not None and record.get("prev_face") is not None:
        d_face = previous - record["prev_face"]
        d_wall = wall - record["prev_wall"]
        moved = np.abs(d_face) > 1e-6
        estimate = np.where(moved, d_wall / np.where(moved, d_face, 1.0), alpha)
        alpha = np.clip(0.5 * alpha + 0.5 * np.clip(estimate, 0.0, 0.95), 0.0, 0.95)
    base = previous if previous is not None else np.zeros(n_seg)
    t_face, t_after, heat = channel_march(area, (wall - alpha * base) / (1.0 - alpha),
                                          flow["h"] * (1.0 - alpha), mdot_cp, t_in)
    if previous is not None and relax < 1.0:
        t_face = (1.0 - relax) * previous + relax * t_face
    filled = area > 0
    record.update(alpha=alpha, prev_face=previous, prev_wall=wall)
    record.update(t_face=t_face, t_after=t_after, wall=wall, t_in=t_in,
                  t_out=float(t_after[-1]) if n_seg else t_in, flow=flow, flow_m3s=flow_m3s,
                  mdot=mdot, mdot_cp=mdot_cp, h=flow["h"], heat_march=float(heat.sum()),
                  mode=wall_mode_of(area[filled], wall[filled], t_face[filled]))
    return t_face[seg]


def liquid_rows(result, number):
    """(header, rows) of the liquids of a result (coolant channels and baths)."""
    header = ["Liquid item", "Kind", "Liquid", "Flow L/min", "In °C", "Out / liquid °C", "Heat W",
              "h W/m²K", "Pressure drop kPa"]
    rows = []
    for entry in result.get("liquids") or []:
        if entry["kind"] == "coolant":
            rows.append([entry["name"], "Coolant channel" + (" (flow solved)" if entry.get("solved") else ""),
                         entry["fluid"], number(entry["flow_lpm"]), number(entry["t_in"]),
                         number(entry["t_out"]), number(entry["heat"]), number(entry.get("h")),
                         number(entry["dp_pa"] / 1000.0)])
        else:
            kind = "Bath, large volume" if entry.get("mode") == "fixed" else \
                "Bath, flow solved" if entry.get("solved") else "Bath, well mixed"
            rows.append([entry["name"], kind, entry["fluid"], "", "", number(entry["t_liquid"]),
                         number(entry["heat"]), number(entry.get("h")), ""])
    return header, rows


def channel_summary(record, heat):
    """Result entry and summary line of a coolant channel; `heat` = W taken up from the faces."""
    c, flow, geometry = record["condition"], record["flow"], record["geometry"]
    lpm = record["flow_m3s"] * 60000.0
    entry = {"name": c["name"], "kind": "coolant", "fluid": c["fluid"], "flow_lpm": lpm,
             "velocity": flow["velocity"], "re": flow["re"], "pr": flow["pr"],
             "regime": flow["regime"], "nu": flow["nu"], "h": flow["h"], "dp_pa": flow["dp"],
             "wall_mode": record.get("mode", 0.5),
             "t_in": record["t_in"], "t_out": record["t_out"], "heat": heat,
             "check_t_out": record["t_in"] + heat / max(record["mdot_cp"], 1e-12),
             "wall_max": float(np.nanmax(record["wall"][record["area_seg"] > 0]))
             if (record["area_seg"] > 0).any() else None,
             "geometry": channel_text(geometry), "length_mm": geometry["length_mm"],
             "dh_mm": geometry["dh_mm"],
             "profile_s": (np.arange(record["n_seg"]) + 1.0) / record["n_seg"] * geometry["length_mm"],
             "profile_liquid": record["t_after"], "profile_wall": record["wall"]}
    pump = f" (pump operating point at {flow['dp'] / 1000.0:.3g} kPa)" if c["flow_mode"] == "pump" else ""
    text = (f"Coolant '{c['name']}' ({c['fluid']}, {lpm:.3g} L/min{pump}, {channel_text(geometry)}): "
            f"{flow['regime']}, Re {flow['re']:.0f}, Nu {flow['nu']:.3g}, h {flow['h']:.4g} W/m²K; "
            f"in {record['t_in']:.2f} °C → out {record['t_out']:.2f} °C, {heat:.4g} W taken up, "
            f"pressure drop ≈ {flow['dp'] / 1000.0:.3g} kPa")
    return entry, text


# ── printed circuit boards ────────────────────────────────────────────────────
# A board comes from KiCad (.kicad_pcb) or from Gerber files (+ Excellon drill, + Gerber job)
# and becomes one body: the outline extruded to the board thickness. Its conductivity varies
# from cell to cell: every copper layer is rasterised into a coverage map (share of copper per
# area), and plated holes add copper through the board. A cell's in-plane conductivity is the
# thickness-weighted mean of the layers it stands for, its through-plane conductivity the
# series value, plus the copper barrels of the vias in parallel.
COPPER_K = 390.0               # W/(m·K), PCB copper
PCB_RESIN_K = 0.25             # W/(m·K), resin between the copper of a layer
COPPER_WEIGHTS_UM = (18, 35, 70, 105, 140)        # ½, 1, 2, 3, 4 oz
PCB_CORES = {
    "FR-4": {"title": "FR-4", "k_plane": 0.81, "k_through": 0.29, "material": "FR-4"},
    "IMS": {"title": "Aluminium core (IMS)", "k_plane": 2.0, "k_through": 2.0,
            "material": "Aluminium", "base_k": 167.0},
}
PCB_DEFAULT_PLATING_UM = 25.0
PCB_COLOR = (0.12, 0.42, 0.20)
PCB_MAX_PIXELS = 2400          # raster size limit per side


def _sexpr_parse(text):
    """Nested lists of a KiCad S-expression file (strings unquoted)."""
    import re
    stack = [[]]
    for match in re.finditer(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+', text):
        token = match.group(0)
        if token == "(":
            stack.append([])
        elif token == ")":
            if len(stack) < 2:
                raise ValueError("unbalanced parentheses")
            item = stack.pop()
            stack[-1].append(item)
        elif token[0] == '"':
            stack[-1].append(token[1:-1].replace('\\"', '"').replace("\\\\", "\\"))
        else:
            stack[-1].append(token)
    if len(stack) != 1 or not stack[0]:
        raise ValueError("not an S-expression file")
    return stack[0][0]


def _sx_children(node, name):
    return [item for item in node[1:] if isinstance(item, list) and item and item[0] == name]


def _sx_child(node, name):
    for item in node[1:]:
        if isinstance(item, list) and item and item[0] == name:
            return item
    return None


def _sx_value(node, name, default=None, index=1):
    item = _sx_child(node, name)
    if item is None or len(item) <= index:
        return default
    return item[index]


def _sx_xy(node, name):
    item = _sx_child(node, name)
    if item is None or len(item) < 3:
        return None
    return float(item[1]), float(item[2])


def _sx_pts(node):
    pts = _sx_child(node, "pts")
    if pts is None:
        return []
    out = []
    for item in pts[1:]:
        if isinstance(item, list) and item and item[0] == "xy":
            out.append((float(item[1]), float(item[2])))
        elif isinstance(item, list) and item and item[0] == "arc":   # arc inside a polygon
            start, mid, end = _sx_xy(item, "start"), _sx_xy(item, "mid"), _sx_xy(item, "end")
            if start and mid and end:
                out += _arc3(start, mid, end)[:-1] + [end]
    return out


def _sx_layers(node):
    """Layer names of an item: (layer "F.Cu") or (layers "F.Cu" "B.Cu")."""
    single = _sx_child(node, "layer")
    names = [single[1]] if single is not None and len(single) > 1 else []
    many = _sx_child(node, "layers")
    if many is not None:
        names += [str(x) for x in many[1:] if not isinstance(x, list)]
    return names


def _arc3(start, mid, end, step_mm=0.25):
    """Points of the circular arc through start, mid and end."""
    (x1, y1), (x2, y2), (x3, y3) = start, mid, end
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        return [start, end]
    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) +
          (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) +
          (x3 * x3 + y3 * y3) * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    a1, a2, a3 = (math.atan2(y - uy, x - ux) for x, y in (start, mid, end))
    sweep = (a3 - a1) % (2 * math.pi)
    if (a2 - a1) % (2 * math.pi) > sweep:            # the mid point lies the other way round
        sweep -= 2 * math.pi
    return _arc_from(ux, uy, r, a1, sweep, step_mm)


def _arc_from(cx, cy, r, a0, sweep, step_mm=0.25):
    count = max(2, int(math.ceil(abs(sweep) * r / step_mm)) + 1, int(abs(sweep) / 0.2) + 1)
    return [(cx + r * math.cos(a0 + sweep * i / (count - 1)),
             cy + r * math.sin(a0 + sweep * i / (count - 1))) for i in range(count)]


def _circle_pts(cx, cy, r, step_mm=0.25):
    return _arc_from(cx, cy, r, 0.0, 2 * math.pi, step_mm)[:-1]


def _rounded_rect(x0, y0, x1, y1, radius):
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    radius = min(max(radius, 0.0), (x1 - x0) / 2, (y1 - y0) / 2)
    if radius <= 1e-9:
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    pts = []
    for cx, cy, start in ((x1 - radius, y0 + radius, -math.pi / 2), (x1 - radius, y1 - radius, 0.0),
                          (x0 + radius, y1 - radius, math.pi / 2), (x0 + radius, y0 + radius, math.pi)):
        pts += _arc_from(cx, cy, radius, start, math.pi / 2, 0.2)
    return pts


def _poly_area(pts):
    if len(pts) < 3:
        return 0.0
    p = np.asarray(pts, float)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _chain_loops(paths, tolerance=0.05):
    """Closed loops from open polylines whose ends meet (board outlines)."""
    loops, open_paths = [], []
    for path in paths:
        if len(path) < 2:
            continue
        if math.dist(path[0], path[-1]) <= tolerance and len(path) > 2:
            loops.append(list(path[:-1]))
        else:
            open_paths.append(list(path))
    while open_paths:
        current = open_paths.pop(0)
        grown = True
        while grown:
            grown = False
            if math.dist(current[0], current[-1]) <= tolerance:
                break
            for i, path in enumerate(open_paths):
                if math.dist(current[-1], path[0]) <= tolerance:
                    current += path[1:]
                elif math.dist(current[-1], path[-1]) <= tolerance:
                    current += path[::-1][1:]
                elif math.dist(current[0], path[-1]) <= tolerance:
                    current = path[:-1] + current
                elif math.dist(current[0], path[0]) <= tolerance:
                    current = path[::-1][:-1] + current
                else:
                    continue
                open_paths.pop(i)
                grown = True
                break
        if math.dist(current[0], current[-1]) <= tolerance and len(current) > 3:
            loops.append(current[:-1])
    return loops


def _rotate_kicad(x, y, degrees):
    """KiCad rotation (counter-clockwise on screen, Y pointing down)."""
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return x * c + y * s, -x * s + y * c


def _kicad_graphic(item, transform=None, step_mm=0.25):
    """('poly', pts) | ('stroke', pts, width) | None of a gr_* / fp_* item (KiCad coordinates)."""
    head = item[0][3:] if item[0].startswith(("gr_", "fp_")) else item[0]
    tf = transform or (lambda x, y: (x, y))
    stroke = _sx_child(item, "stroke")
    width = float(_sx_value(stroke, "width", 0.0)) if stroke is not None else \
        float(_sx_value(item, "width", 0.0))
    fill = _sx_value(item, "fill", "no")
    filled = fill in ("yes", "solid", "true")
    if head == "line":
        a, b = _sx_xy(item, "start"), _sx_xy(item, "end")
        return ("stroke", [tf(*a), tf(*b)], width) if a and b else None
    if head == "arc":
        start, mid, end = _sx_xy(item, "start"), _sx_xy(item, "mid"), _sx_xy(item, "end")
        if start and mid and end:
            pts = _arc3(start, mid, end, step_mm)
        else:                                  # KiCad 5: (start centre) (end point) (angle)
            centre, point = _sx_xy(item, "start"), _sx_xy(item, "end")
            angle = float(_sx_value(item, "angle", 0.0))
            if not centre or not point:
                return None
            r = math.dist(centre, point)
            a0 = math.atan2(point[1] - centre[1], point[0] - centre[0])
            pts = _arc_from(centre[0], centre[1], r, a0, math.radians(angle), step_mm)
        return ("stroke", [tf(*p) for p in pts], width)
    if head == "circle":
        centre, point = _sx_xy(item, "center"), _sx_xy(item, "end")
        if not centre or not point:
            return None
        pts = [tf(*p) for p in _circle_pts(centre[0], centre[1], math.dist(centre, point), step_mm)]
        return ("poly", pts) if filled else ("stroke", pts + [pts[0]], width)
    if head == "rect":
        a, b = _sx_xy(item, "start"), _sx_xy(item, "end")
        if not a or not b:
            return None
        pts = [tf(*p) for p in _rounded_rect(a[0], a[1], b[0], b[1],
                                              float(_sx_value(item, "radius", 0.0)))]
        return ("poly", pts) if filled else ("stroke", pts + [pts[0]], width)
    if head == "poly":
        pts = [tf(*p) for p in _sx_pts(item)]
        if len(pts) < 3:
            return None
        return ("poly", pts) if filled or width <= 0 else ("stroke", pts + [pts[0]], width)
    if head == "curve":
        pts = [tf(*p) for p in _sx_pts(item)]
        return ("stroke", pts, width) if len(pts) >= 2 else None
    return None


def _pad_shapes(pad, place):
    """Copper shapes of a footprint pad; place(x, y) maps pad-local points to the board."""
    kind = pad[3] if len(pad) > 3 and not isinstance(pad[3], list) else "rect"
    size = _sx_child(pad, "size")
    w, h = (float(size[1]), float(size[2])) if size is not None and len(size) > 2 else (1.0, 1.0)
    out = []
    if kind == "circle":
        out.append(("poly", [place(*p) for p in _circle_pts(0.0, 0.0, w / 2, 0.1)]))
    elif kind == "oval":
        if abs(w - h) < 1e-9:
            out.append(("poly", [place(*p) for p in _circle_pts(0.0, 0.0, w / 2, 0.1)]))
        elif w > h:
            out.append(("stroke", [place(-(w - h) / 2, 0.0), place((w - h) / 2, 0.0)], h))
        else:
            out.append(("stroke", [place(0.0, -(h - w) / 2), place(0.0, (h - w) / 2)], w))
    elif kind == "roundrect":
        ratio = float(_sx_value(pad, "roundrect_rratio", 0.25))
        out.append(("poly", [place(*p) for p in _rounded_rect(-w / 2, -h / 2, w / 2, h / 2,
                                                              ratio * min(w, h))]))
    elif kind == "custom":
        out.append(("poly", [place(*p) for p in _rounded_rect(-w / 2, -h / 2, w / 2, h / 2, 0.0)]))
        primitives = _sx_child(pad, "primitives")
        for item in (primitives[1:] if primitives is not None else []):
            if isinstance(item, list) and item and item[0].startswith("gr_"):
                graphic = _kicad_graphic(item, place, 0.1)
                if graphic is not None:
                    if graphic[0] == "stroke" and graphic[2] <= 0:
                        graphic = ("poly", graphic[1])
                    out.append(graphic)
    else:                                       # rect, trapezoid, chamfered rect
        out.append(("poly", [place(*p) for p in _rounded_rect(-w / 2, -h / 2, w / 2, h / 2, 0.0)]))
    return out, w * h


def read_kicad_pcb(path):
    """Board description of a KiCad board file (KiCad 5 … 10), see pcb_from_description."""
    with open(path, encoding="utf-8", errors="replace") as stream:
        root = _sexpr_parse(stream.read())
    if not root or root[0] != "kicad_pcb":
        raise ThermalError("This is not a KiCad board file (.kicad_pcb).")
    messages = []
    general = _sx_child(root, "general")
    thickness = float(_sx_value(general, "thickness", 1.6)) if general is not None else 1.6
    names = []
    layers = _sx_child(root, "layers")
    for item in (layers[1:] if layers is not None else []):
        if isinstance(item, list) and len(item) > 1 and str(item[1]).endswith(".Cu"):
            names.append(str(item[1]))

    def order(name):
        if name == "F.Cu":
            return -1
        if name == "B.Cu":
            return 10 ** 6
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 999
    names = sorted(set(names), key=order) or ["F.Cu", "B.Cu"]
    # stack-up: copper thickness and the dielectric between neighbouring copper layers
    copper_um = {name: 35.0 for name in names}
    dielectric = [None] * max(len(names) - 1, 0)
    setup = _sx_child(root, "setup")
    stackup = _sx_child(setup, "stackup") if setup is not None else None
    if stackup is not None:
        index, pending = -1, 0.0
        for item in _sx_children(stackup, "layer"):
            name = str(item[1]) if len(item) > 1 else ""
            kind = str(_sx_value(item, "type", ""))
            value = _sx_value(item, "thickness")
            if name in copper_um:
                if value is not None:
                    copper_um[name] = float(value) * 1000.0
                if index >= 0 and index < len(dielectric):
                    dielectric[index] = pending
                index += 1
                pending = 0.0
            elif kind in ("core", "prepreg") or "dielectric" in name:
                for sub in [item] + _sx_children(item, "addsublayer"):
                    sub_value = _sx_value(sub, "thickness")
                    if sub_value is not None:
                        pending += float(sub_value)
    known = sum(d for d in dielectric if d)
    unknown = [i for i, d in enumerate(dielectric) if not d]
    if unknown:
        rest = max(thickness - known - sum(copper_um.values()) / 1000.0, 0.1 * len(unknown))
        for i in unknown:
            dielectric[i] = rest / len(unknown)
    copper = set(names)

    def copper_of(layer_names):
        out = set()
        for name in layer_names:
            if name in ("*.Cu", "F&B.Cu"):
                out |= copper if name == "*.Cu" else {"F.Cu", "B.Cu"} & copper
            elif name in copper:
                out.add(name)
        return out

    shapes = {name: [] for name in names}
    edges, holes_round, plated, vias = [], [], [], []
    # board graphics
    for item in root[1:]:
        if not (isinstance(item, list) and item and isinstance(item[0], str)):
            continue
        head = item[0]
        if head.startswith("gr_"):
            layer_names = _sx_layers(item)
            graphic = _kicad_graphic(item)
            if graphic is None:
                continue
            if "Edge.Cuts" in layer_names:
                edges.append(graphic[1] if graphic[0] == "stroke" else graphic[1] + [graphic[1][0]])
            for name in copper_of(layer_names):
                shapes[name].append(graphic)
        elif head in ("segment", "arc"):
            a, b = _sx_xy(item, "start"), _sx_xy(item, "end")
            if not a or not b:
                continue
            mid = _sx_xy(item, "mid") if head == "arc" else None
            pts = _arc3(a, mid, b, 0.1) if mid else [a, b]
            width = float(_sx_value(item, "width", 0.2))
            for name in copper_of(_sx_layers(item)):
                shapes[name].append(("stroke", pts, width))
        elif head == "via":
            at = _sx_xy(item, "at")
            if not at:
                continue
            size = float(_sx_value(item, "size", 0.6))
            drill = float(_sx_value(item, "drill", 0.3))
            vias.append((at[0], at[1], drill))
            span = copper if len(item) > 1 and item[1] not in ("blind", "micro") else \
                copper_of(_sx_layers(item))
            ring = _circle_pts(at[0], at[1], size / 2, 0.05)
            for name in span or copper:
                shapes[name].append(("poly", ring))
        elif head == "zone":
            if _sx_child(item, "keepout") is not None:
                continue
            zone_layers = copper_of(_sx_layers(item))
            fills = _sx_children(item, "filled_polygon")
            if fills:
                for fill in fills:
                    target = copper_of(_sx_layers(fill)) or zone_layers
                    pts = _sx_pts(fill)
                    if len(pts) >= 3:
                        for name in target:
                            shapes[name].append(("poly", pts))
            else:
                outline = _sx_child(item, "polygon")
                pts = _sx_pts(outline) if outline is not None else []
                if len(pts) >= 3 and zone_layers:
                    for name in zone_layers:
                        shapes[name].append(("poly", pts))
                    messages.append("A copper zone is not filled in the file (refill it in KiCad and "
                                    "save); its outline counts as full copper.")
    # footprints (KiCad 5: "module")
    footprints = []
    for item in root[1:]:
        if not (isinstance(item, list) and item and item[0] in ("footprint", "module")):
            continue
        lib = str(item[1]) if len(item) > 1 and not isinstance(item[1], list) else ""
        at = _sx_child(item, "at")
        fx, fy = (float(at[1]), float(at[2])) if at is not None else (0.0, 0.0)
        angle = float(at[3]) if at is not None and len(at) > 3 else 0.0
        side = "bottom" if _sx_value(item, "layer", "F.Cu") == "B.Cu" else "top"
        reference, value = "", ""
        for prop in _sx_children(item, "property"):
            if len(prop) > 2 and prop[1] == "Reference":
                reference = str(prop[2])
            elif len(prop) > 2 and prop[1] == "Value":
                value = str(prop[2])
        for text in _sx_children(item, "fp_text"):
            if len(text) > 2 and text[1] == "reference" and not reference:
                reference = str(text[2])
            elif len(text) > 2 and text[1] == "value" and not value:
                value = str(text[2])

        def to_board(x, y, fx=fx, fy=fy, angle=angle):
            rx, ry = _rotate_kicad(x, y, angle)
            return fx + rx, fy + ry

        pad_boxes, pad_area, largest = [], 0.0, 0.0
        for pad in _sx_children(item, "pad"):
            pat = _sx_child(pad, "at")
            px, py = (float(pat[1]), float(pat[2])) if pat is not None else (0.0, 0.0)
            pad_angle = float(pat[3]) if pat is not None and len(pat) > 3 else 0.0
            cx, cy = to_board(px, py)

            def place(x, y, cx=cx, cy=cy, pad_angle=pad_angle):
                rx, ry = _rotate_kicad(x, y, pad_angle)
                return cx + rx, cy + ry
            pad_kind = pad[2] if len(pad) > 2 else "smd"
            layer_set = copper_of(_sx_layers(pad))
            pad_shapes, area = _pad_shapes(pad, place)
            if layer_set:
                for name in layer_set:
                    shapes[name] += pad_shapes
                pad_area += area
                largest = max(largest, area)
                pts = np.array([p for s in pad_shapes for p in s[1]])
                # pad extent in the footprint's own frame
                local = np.array([_rotate_kicad(x - fx, y - fy, -angle) for x, y in pts])
                pad_boxes.append((local.min(0), local.max(0)))
            drill = _sx_child(pad, "drill")
            if drill is not None and len(drill) > 1:
                values = [float(v) for v in drill[1:] if not isinstance(v, list) and v != "oval"]
                if values:
                    d = min(values) if len(values) > 1 else values[0]
                    if pad_kind == "np_thru_hole":
                        holes_round.append((cx, cy, d))
                    else:
                        plated.append((cx, cy, d))
        for graphic_item in item[1:]:
            if isinstance(graphic_item, list) and graphic_item and graphic_item[0].startswith("fp_") \
                    and graphic_item[0] != "fp_text":
                layer_names = _sx_layers(graphic_item)
                graphic = _kicad_graphic(graphic_item, to_board)
                if graphic is None:
                    continue
                if "Edge.Cuts" in layer_names:
                    edges.append(graphic[1] if graphic[0] == "stroke" else graphic[1] + [graphic[1][0]])
                for name in copper_of(layer_names):
                    shapes[name].append(graphic)
        if pad_boxes:
            low = np.min([b[0] for b in pad_boxes], axis=0)
            high = np.max([b[1] for b in pad_boxes], axis=0)
        else:
            low = high = np.zeros(2)
        footprints.append({"reference": reference, "value": value, "footprint": lib,
                           "x": fx, "y": fy, "angle": angle, "side": side,
                           "pads_size": [float(high[0] - low[0]), float(high[1] - low[1])],
                           "pads_centre": [float(x) for x in to_board(*((low + high) / 2))],
                           "pad_area": pad_area, "largest_pad": largest})
    loops = _chain_loops(edges)
    return {"format": "kicad", "path": path, "thickness_mm": thickness, "copper_layers": names,
            "copper_um": [copper_um[n] for n in names], "dielectric_mm": dielectric,
            "loops": loops, "round_holes": holes_round, "shapes": shapes,
            "plated": plated + [(x, y, d) for x, y, d in vias], "footprints": footprints,
            "y_down": True, "messages": messages}


def _gerber_expression(text, variables):
    """Value of a Gerber aperture-macro expression ($n variables, x = multiply)."""
    import re
    expression = re.sub(r"\$(\d+)", lambda m: repr(float(variables.get(int(m.group(1)), 0.0))),
                        text.replace("x", "*").replace("X", "*"))
    if not re.fullmatch(r"[0-9eE.+\-*/() ]*", expression):
        raise ValueError(f"bad macro expression {text!r}")
    return float(eval(expression, {"__builtins__": {}}, {})) if expression.strip() else 0.0


def _gerber_macro_shapes(body, params):
    """Aperture-local shapes [('poly', pts) | ('circle', x, y, r)] of a macro with parameters."""
    variables = {i + 1: value for i, value in enumerate(params)}
    out = []

    def rotated(points, degrees):
        a = math.radians(degrees)
        c, s = math.cos(a), math.sin(a)
        return [(x * c - y * s, x * s + y * c) for x, y in points]

    for statement in body:
        statement = statement.strip()
        if not statement or statement.startswith("0"):
            continue
        if statement.startswith("$"):
            name, _sep, value = statement.partition("=")
            variables[int(name[1:])] = _gerber_expression(value, variables)
            continue
        values = [_gerber_expression(v, variables) for v in statement.split(",")]
        code = int(values[0])
        if len(values) < 2 or values[1] == 0:          # exposure off: left out
            continue
        if code == 1 and len(values) >= 5:            # circle: d, x, y[, rotation]
            x, y = values[3], values[4]
            if len(values) > 5 and values[5]:
                (x, y), = rotated([(x, y)], values[5])
            out.append(("circle", x, y, values[2] / 2))
        elif code in (2, 20) and len(values) >= 8:    # vector line: width, x1, y1, x2, y2, rot
            w, x1, y1, x2, y2, rot = values[2:8]
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy) or 1e-9
            nx, ny = -dy / length * w / 2, dx / length * w / 2
            out.append(("poly", rotated([(x1 + nx, y1 + ny), (x2 + nx, y2 + ny),
                                         (x2 - nx, y2 - ny), (x1 - nx, y1 - ny)], rot)))
        elif code == 21 and len(values) >= 7:         # centre line: w, h, cx, cy, rot
            w, h, cx, cy, rot = values[2:7]
            out.append(("poly", rotated([(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
                                         (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)], rot)))
        elif code == 22 and len(values) >= 7:         # lower-left line (old)
            w, h, x0, y0, rot = values[2:7]
            out.append(("poly", rotated([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)], rot)))
        elif code == 4 and len(values) >= 3:          # outline: n, x0, y0, …, rot
            n = int(values[2])
            coords = values[3:3 + 2 * (n + 1)]
            rot = values[3 + 2 * (n + 1)] if len(values) > 3 + 2 * (n + 1) else 0.0
            pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]
            out.append(("poly", rotated(pts, rot)))
        elif code == 5 and len(values) >= 6:          # polygon: n, cx, cy, d, rot
            n, cx, cy, d = int(values[2]), values[3], values[4], values[5]
            rot = values[6] if len(values) > 6 else 0.0
            pts = [(cx + d / 2 * math.cos(2 * math.pi * i / n), cy + d / 2 * math.sin(2 * math.pi * i / n))
                   for i in range(n)]
            out.append(("poly", rotated(pts, rot)))
        elif code == 7 and len(values) >= 6:          # thermal: cx, cy, outer, inner, gap — a ring
            cx, cy, outer = values[1], values[2], values[3]
            out.append(("circle", cx, cy, outer / 2))
    return out


def read_gerber(path):
    """(file function or None, shapes [(kind, data…, dark)], outline paths) of a Gerber
    (RS-274X) file in mm. Shapes: ('poly', pts, dark), ('stroke', pts, width, dark),
    ('circle', x, y, r, dark)."""
    import re
    with open(path, encoding="latin-1") as stream:
        text = stream.read()
    function = None
    match = re.search(r"TF\.FileFunction,([^*%]+)", text)
    if match:
        function = match.group(1).strip()
    integer, decimals, omit_trailing = 3, 6, False
    scale = 1.0                                    # to mm
    apertures, macros = {}, {}
    shapes, strokes = [], []
    x = y = 0.0
    aperture = None
    mode = "G01"
    dark = True
    region = None                                   # list of contours while in G36 … G37
    last_d = "D01"

    def number(value):
        negative = value.startswith("-")
        digits = value.lstrip("+-")
        if "." in digits:
            result = float(digits)
        else:
            if omit_trailing:
                digits = digits.ljust(integer + decimals, "0")
            result = int(digits or "0") / 10 ** decimals
        return (-result if negative else result) * scale

    def flash(cx, cy):
        if aperture not in apertures:
            return
        kind, params = apertures[aperture]
        if kind == "C":
            shapes.append(("circle", cx, cy, params[0] / 2 * scale, dark))
        elif kind in ("R", "O"):
            w, h = params[0] * scale, params[1] * scale
            if kind == "O" and abs(w - h) > 1e-9:
                if w > h:
                    shapes.append(("stroke", [(cx - (w - h) / 2, cy), (cx + (w - h) / 2, cy)], h, dark))
                else:
                    shapes.append(("stroke", [(cx, cy - (h - w) / 2), (cx, cy + (h - w) / 2)], w, dark))
            elif kind == "O":
                shapes.append(("circle", cx, cy, w / 2, dark))
            else:
                shapes.append(("poly", [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
                                        (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)], dark))
        elif kind == "P":
            d, n = params[0] * scale, int(params[1]) if len(params) > 1 else 6
            rot = math.radians(params[2]) if len(params) > 2 else 0.0
            shapes.append(("poly", [(cx + d / 2 * math.cos(rot + 2 * math.pi * i / n),
                                     cy + d / 2 * math.sin(rot + 2 * math.pi * i / n))
                                    for i in range(n)], dark))
        elif kind in macros:
            for item in _gerber_macro_shapes(macros[kind], params):
                if item[0] == "circle":
                    shapes.append(("circle", cx + item[1] * scale, cy + item[2] * scale,
                                   item[3] * scale, dark))
                else:
                    shapes.append(("poly", [(cx + px * scale, cy + py * scale) for px, py in item[1]],
                                   dark))

    def stroke_width():
        if aperture not in apertures:
            return 0.0
        kind, params = apertures[aperture]
        if kind in ("C", "P"):
            return params[0] * scale
        if kind in ("R", "O") and len(params) > 1:
            return min(params[0], params[1]) * scale
        return 0.1

    for block in re.findall(r"%[^%]*%|[^%*]+\*", text):
        if block.startswith("%"):
            parts = [p.strip() for p in block[1:-1].split("*") if p.strip()]
            if not parts:
                continue
            head = parts[0]
            if head.startswith("FS"):
                m = re.search(r"X(\d)(\d)", head)
                if m:
                    integer, decimals = int(m.group(1)), int(m.group(2))
                omit_trailing = head[2:3] == "T"
            elif head.startswith("MO"):
                scale = 25.4 if head[2:4] == "IN" else 1.0
            elif head.startswith("AD"):
                m = re.match(r"ADD(\d+)([A-Za-z_.$][^,]*),?(.*)", head)
                if m:
                    params = [float(v) for v in m.group(3).split("X") if v.strip()] if m.group(3) else []
                    apertures[int(m.group(1))] = (m.group(2), params)
            elif head.startswith("AM"):
                macros[head[2:]] = parts[1:]
            elif head.startswith("LP"):
                dark = head[2:3] != "C"
            continue
        word = block[:-1].strip().replace("\n", "").replace("\r", "")
        if not word or word.startswith("G04"):
            continue
        for code in re.findall(r"G0?([0-9]{1,2})", word.split("X")[0].split("Y")[0].split("D")[0]):
            code = int(code)
            if code in (1, 2, 3):
                mode = f"G0{code}"
            elif code == 36:
                region = []
            elif code == 37:
                if region:
                    for contour in region:
                        if len(contour) >= 3:
                            shapes.append(("poly", contour, dark))
                region = None
            elif code == 70:
                scale = 25.4
            elif code == 71:
                scale = 1.0
        if word.startswith("G54"):
            word = word[3:]
        m = re.search(r"D(\d+)$", word)
        d_code = int(m.group(1)) if m else None
        if d_code is not None and d_code >= 10:
            aperture = d_code
            continue
        coords = dict(re.findall(r"([XYIJ])([+-]?[\d.]+)", word))
        if not coords and d_code is None:
            continue
        nx = number(coords["X"]) if "X" in coords else x
        ny = number(coords["Y"]) if "Y" in coords else y
        operation = f"D0{d_code}" if d_code is not None else last_d
        last_d = operation
        if operation == "D01":
            if mode == "G01":
                pts = [(x, y), (nx, ny)]
            else:
                ci, cj = number(coords.get("I", "0")), number(coords.get("J", "0"))
                cx, cy = x + ci, y + cj
                r = math.hypot(x - cx, y - cy)
                a0 = math.atan2(y - cy, x - cx)
                a1 = math.atan2(ny - cy, nx - cx)
                sweep = (a1 - a0) % (2 * math.pi)
                if mode == "G02":
                    sweep -= 2 * math.pi
                if abs(sweep) < 1e-9:
                    sweep = 2 * math.pi if mode == "G03" else -2 * math.pi
                pts = _arc_from(cx, cy, r, a0, sweep, 0.1)
            if region is not None:
                if not region:
                    region.append([(x, y)])
                region[-1] += pts[1:]
            else:
                width = stroke_width()
                strokes.append(pts)
                shapes.append(("stroke", pts, width, dark))
        elif operation == "D02":
            if region is not None:
                region.append([(nx, ny)])
        elif operation == "D03":
            flash(nx, ny)
        x, y = nx, ny
    return function, shapes, strokes


def read_excellon(path):
    """Holes [(x, y, diameter, plated)] in mm of an Excellon drill file."""
    import re
    with open(path, encoding="latin-1") as stream:
        lines = stream.read().splitlines()
    scale, integer, decimals, leading = 1.0, 3, 3, True       # leading zeros kept (LZ)
    tools, holes = {}, []
    tool = None
    plated = not ("npth" in os.path.basename(path).lower() or "non_plated" in path.lower())
    x = y = 0.0
    header = True

    def number(value):
        if "." in value:
            return float(value) * scale
        negative = value.startswith("-")
        digits = value.lstrip("+-")
        if leading:
            digits = digits.ljust(integer + decimals, "0")
        result = int(digits or "0") / 10 ** decimals
        return (-result if negative else result) * scale

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith(";"):
            m = re.search(r"FILE_FORMAT\s*=\s*(\d+):(\d+)", upper)
            if m:
                integer, decimals = int(m.group(1)), int(m.group(2))
            if "TYPE=NON_PLATED" in upper or "NPTH" in upper and "TF.FILEFUNCTION" in upper:
                plated = False
            elif "TYPE=PLATED" in upper:
                plated = True
            continue
        if upper.startswith(("METRIC", "M71")):
            scale = 1.0
            if "TZ" in upper:
                leading = False
            elif "LZ" in upper:
                leading = True
            if "000.000" in upper or "0000.00" in upper:
                integer, decimals = (len(upper.split(",")[-1].split(".")[0]),
                                     len(upper.split(",")[-1].split(".")[1]))
            continue
        if upper.startswith(("INCH", "M72")):
            scale = 25.4
            leading = "TZ" not in upper
            if "LZ" not in upper and "TZ" not in upper:
                leading = True
            integer, decimals = 2, 4
            continue
        if upper == "%" or upper.startswith("M95"):
            header = False
            continue
        m = re.match(r"T(\d+).*C([\d.]+)", upper)
        if m:
            tools[int(m.group(1))] = float(m.group(2)) * scale
            continue
        m = re.match(r"T(\d+)$", upper)
        if m and not header:
            tool = int(m.group(1))
            continue
        if upper.startswith(("X", "Y")) and tool in tools:
            first = upper.split("G85")[0]
            coords = dict(re.findall(r"([XY])([+-]?[\d.]+)", first))
            x = number(coords["X"]) if "X" in coords else x
            y = number(coords["Y"]) if "Y" in coords else y
            holes.append((x, y, tools[tool], plated))
            if "G85" in upper:                       # a slot: its other end, too
                coords = dict(re.findall(r"([XY])([+-]?[\d.]+)", upper.split("G85")[1]))
                holes.append((number(coords["X"]) if "X" in coords else x,
                              number(coords["Y"]) if "Y" in coords else y, tools[tool], plated))
    return holes


_GERBER_EXTENSIONS = {".gtl": ("Copper", 1), ".gbl": ("Copper", 99), ".cmp": ("Copper", 1),
                      ".sol": ("Copper", 99), ".gko": ("Profile", 0), ".gml": ("Profile", 1),
                      ".gm1": ("Profile", 2), ".gm2": ("Profile", 3), ".gm3": ("Profile", 4)}


def _gerber_role(path, function):
    """('copper', order) | ('profile', priority) | None from the file function or the name."""
    import re
    name = os.path.basename(path).lower()
    if function:
        parts = [p.strip() for p in function.split(",")]
        if parts[0] == "Copper":
            m = re.match(r"L(\d+)", parts[1]) if len(parts) > 1 else None
            return ("copper", int(m.group(1)) if m else 50)
        if parts[0] == "Profile":
            return ("profile", 0)
        return None
    extension = os.path.splitext(name)[1]
    if extension in _GERBER_EXTENSIONS:
        kind, order = _GERBER_EXTENSIONS[extension]
        return ("copper", order) if kind == "Copper" else ("profile", order)
    m = re.match(r"\.g(\d+)$", extension) or re.match(r"\.gp(\d+)$", extension)
    if m:
        return ("copper", 1 + int(m.group(1)))
    if "f_cu" in name or "top_copper" in name or "copper_top" in name:
        return ("copper", 1)
    if "b_cu" in name or "bottom_copper" in name or "copper_bottom" in name:
        return ("copper", 99)
    m = re.search(r"in(\d+)_cu", name)
    if m:
        return ("copper", 1 + int(m.group(1)))
    if "edge_cuts" in name or "outline" in name or "profile" in name:
        return ("profile", 0)
    return None


def read_gerber_set(path):
    """Board description (see read_kicad_pcb) of a Gerber set: a .gbrjob file, a folder, or a
    single Gerber file (its folder is read)."""
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    job = None
    if path.lower().endswith(".gbrjob"):
        with open(path, encoding="utf-8") as stream:
            job = json.load(stream)
    else:
        candidates = [f for f in os.listdir(folder) if f.lower().endswith(".gbrjob")]
        if candidates:
            with open(os.path.join(folder, candidates[0]), encoding="utf-8") as stream:
                job = json.load(stream)
    files = []
    if job and job.get("FilesAttributes"):
        for entry in job["FilesAttributes"]:
            files.append((os.path.join(folder, entry.get("Path", "")), entry.get("FileFunction")))
    else:
        for name in sorted(os.listdir(folder)):
            full = os.path.join(folder, name)
            if os.path.isfile(full) and os.path.getsize(full) > 0:
                files.append((full, None))
    copper, profiles, holes, messages = [], [], [], []
    for full, function in files:
        if not os.path.isfile(full):
            continue
        lower = full.lower()
        try:
            with open(full, encoding="latin-1") as stream:
                head = stream.read(4000)
        except OSError:
            continue
        if head.lstrip().startswith("M48") or lower.endswith((".drl", ".xln")) and "M48" in head:
            holes += read_excellon(full)
            continue
        if "%FS" not in head and "%MO" not in head and "FSLA" not in head:
            continue
        found_function = function
        if found_function is None:
            import re
            m = re.search(r"TF\.FileFunction,([^*%]+)", head)
            found_function = m.group(1).strip() if m else None
        role = _gerber_role(full, found_function)
        if role is None:
            continue
        func, shapes, strokes = read_gerber(full)
        if role[0] == "copper":
            copper.append((role[1], os.path.basename(full), shapes))
        else:
            profiles.append((role[1], strokes, shapes))
    if not copper:
        raise ThermalError("No copper layer found among the Gerber files (top / bottom copper, e.g. "
                           "*.GTL / *.GBL or *-F_Cu.gbr).")
    copper.sort(key=lambda item: item[0])
    names = [name for _order, name, _shapes in copper]
    loops = []
    for _priority, strokes, shapes in sorted(profiles, key=lambda item: item[0]):
        loops = _chain_loops(strokes + [s[1] for s in shapes if s[0] == "poly"])
        if loops:
            break
    if not loops:
        messages.append("No closed board outline among the Gerber files: the board is the box around "
                        "the copper.")
    thickness, copper_um, dielectric = 1.6, [35.0] * len(names), None
    if job:
        specs = job.get("GeneralSpecs", {})
        thickness = float(specs.get("BoardThickness", thickness))
        stack = job.get("MaterialStackup") or []
        copper_entries = [e for e in stack if str(e.get("Type", "")).lower() == "copper"]
        if len(copper_entries) == len(names):
            copper_um = [float(e.get("Thickness", 0.035)) * 1000.0 for e in copper_entries]
        dielectric, pending, seen = [], 0.0, 0
        for entry in stack:
            kind = str(entry.get("Type", "")).lower()
            if kind == "copper":
                if seen:
                    dielectric.append(pending)
                seen += 1
                pending = 0.0
            elif kind == "dielectric":
                pending += float(entry.get("Thickness", 0.0))
        if len(dielectric) != len(names) - 1 or not all(dielectric):
            dielectric = None
    if dielectric is None:
        rest = max(thickness - sum(copper_um) / 1000.0, 0.2)
        dielectric = [rest / max(len(names) - 1, 1)] * max(len(names) - 1, 0)
    return {"format": "gerber", "path": path, "thickness_mm": thickness, "copper_layers": names,
            "copper_um": copper_um, "dielectric_mm": dielectric, "loops": loops,
            "round_holes": [(x, y, d) for x, y, d, plated in holes if not plated],
            "shapes": {name: shapes for (_o, name, shapes) in copper},
            "plated": [(x, y, d) for x, y, d, plated in holes if plated], "footprints": [],
            "y_down": False, "messages": messages}


def read_board_file(path):
    """Board description of a KiCad board, a Gerber job, a Gerber / drill file or a folder."""
    if os.path.isfile(path) and path.lower().endswith(".kicad_pcb"):
        return read_kicad_pcb(path)
    return read_gerber_set(path)


def _render_layer(shapes, x0, y0, pixel, nx, ny):
    """Copper share per pixel (uint8, row 0 at y0, i.e. bottom-up) of the shapes (board mm)."""
    from PyQt5.QtGui import QImage, QPainterPath, QBrush
    image = QImage(nx, ny, QImage.Format_Grayscale8)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    height = ny * pixel

    def point(p):
        return QPointF((p[0] - x0) / pixel, (y0 + height - p[1]) / pixel)

    for shape in shapes:
        dark = shape[-1] if isinstance(shape[-1], bool) else True
        color = QColor(255, 255, 255) if dark else QColor(0, 0, 0)
        if shape[0] == "poly":
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawPolygon(QPolygonF([point(p) for p in shape[1]]), Qt.WindingFill)
        elif shape[0] == "circle":
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(point((shape[1], shape[2])), shape[3] / pixel, shape[3] / pixel)
        elif shape[0] == "stroke":
            pts, width = shape[1], shape[2]
            if len(pts) < 2 or width <= 0:
                continue
            pen = QPen(color)
            pen.setWidthF(max(width / pixel, 0.5))
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            path = QPainterPath(point(pts[0]))
            for p in pts[1:]:
                path.lineTo(point(p))
            painter.drawPath(path)
    painter.end()
    bits = image.constBits()
    bits.setsize(image.byteCount())
    array = np.frombuffer(bits, np.uint8).reshape(ny, image.bytesPerLine())[:, :nx]
    return np.ascontiguousarray(array[::-1])


def _point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1:
            inside = not inside
    return inside


def pcb_from_description(desc, core="FR-4", copper_um=None, dielectric_mm=None,
                         plating_um=PCB_DEFAULT_PLATING_UM, ims_dielectric_mm=0.1, ims_k=2.0):
    """Study data of a board ("pcb" of a component): outline and holes in board coordinates
    (lower-left corner of the outline at 0, 0; Y up), layer stack, copper coverage rasters per
    copper layer and the plated-hole copper per pixel. Also returns the transform of the
    description's coordinates to the board's (for placing parts)."""
    flip = -1.0 if desc.get("y_down") else 1.0

    def fix(p):
        return (float(p[0]), flip * float(p[1]))

    messages = list(desc.get("messages", []))
    loops = [[fix(p) for p in loop] for loop in desc.get("loops", []) if len(loop) >= 3]
    loops = [loop for loop in loops if abs(_poly_area(loop)) > 1e-6]
    points = [fix(p) for shapes in desc["shapes"].values() for s in shapes
              for p in (s[1] if s[0] in ("poly", "stroke") else [(s[1], s[2])])]
    copper_box = (np.min(points, axis=0) - 1.0, np.max(points, axis=0) + 1.0) if points else None
    outline = max(loops, key=lambda loop: abs(_poly_area(loop))) if loops else None
    if outline is not None and copper_box is not None and \
            abs(_poly_area(outline)) < 0.25 * float(np.prod(copper_box[1] - copper_box[0])):
        messages.append("The closed outline found is much smaller than the copper (a detail, not "
                        "the board edge): the board is the box around the copper.")
        outline = None
    if outline is not None:
        holes = [loop for loop in loops if loop is not outline and
                 _point_in_polygon(loop[0], outline)]
    else:
        if copper_box is None:
            raise ThermalError("The board has neither an outline nor copper.")
        if not loops:
            messages.append("The file has no closed board outline (Edge.Cuts / profile): the board "
                            "is the box around the copper.")
        low, high = copper_box
        outline = [(low[0], low[1]), (high[0], low[1]), (high[0], high[1]), (low[0], high[1])]
        holes = []
    for x, y, d in desc.get("round_holes", []):
        if d >= 0.8:
            holes.append(_circle_pts(*fix((x, y)), d / 2, 0.3)[::-1])
    if _poly_area(outline) < 0:
        outline = outline[::-1]
    holes = [h if _poly_area(h) < 0 else h[::-1] for h in holes]
    low = np.min(outline, axis=0)
    high = np.max(outline, axis=0)
    shift = -low

    def place(p):
        q = fix(p)
        return (q[0] + shift[0], q[1] + shift[1])

    outline = [(x + shift[0], y + shift[1]) for x, y in outline]
    holes = [[(x + shift[0], y + shift[1]) for x, y in hole] for hole in holes]
    size = high - low
    pixel = max(0.05, float(max(size)) / PCB_MAX_PIXELS)
    nx, ny = int(math.ceil(size[0] / pixel)) + 2, int(math.ceil(size[1] / pixel)) + 2
    names = list(desc["copper_layers"])
    copper_um = list(copper_um or desc.get("copper_um") or [35.0] * len(names))
    dielectric_mm = list(dielectric_mm or desc.get("dielectric_mm") or [])
    layers = []
    for index, name in enumerate(names):
        shapes = []
        for shape in desc["shapes"].get(name, []):
            if shape[0] == "circle":
                cx, cy = place((shape[1], shape[2]))
                shapes.append(("circle", cx, cy, shape[3]) + tuple(shape[4:]))
            else:
                shapes.append((shape[0], [place(p) for p in shape[1]]) + tuple(shape[2:]))
        coverage = _render_layer(shapes, -pixel, -pixel, pixel, nx, ny)
        layers.append({"name": name, "copper_um": float(copper_um[index]), "coverage": coverage})
    # plated holes: copper of the barrel (annulus of the plating) per pixel, mm²
    via_area = np.zeros((ny, nx), np.float32)
    t = plating_um / 1000.0
    count = 0
    for x, y, d in desc.get("plated", []):
        px, py = place((x, y))
        ix, iy = int((px + pixel) / pixel), int((py + pixel) / pixel)
        if 0 <= ix < nx and 0 <= iy < ny:
            via_area[iy, ix] += math.pi * ((d / 2 + t) ** 2 - (d / 2) ** 2)
            count += 1
    if not desc.get("plated"):
        messages.append("No plated holes (vias) found.")
    thickness = (sum(c / 1000.0 for c in copper_um) + sum(dielectric_mm)) if core != "IMS" else \
        float(desc.get("thickness_mm", 1.6))
    pcb = {"source": os.path.basename(desc.get("path", "")), "format": desc["format"],
           "outline": [[float(x), float(y)] for x, y in outline],
           "holes": [[[float(x), float(y)] for x, y in hole] for hole in holes],
           "core": core if core in PCB_CORES else "FR-4", "layers": layers,
           "dielectric_mm": [float(d) for d in dielectric_mm], "plating_um": float(plating_um),
           "ims_dielectric_mm": float(ims_dielectric_mm), "ims_k": float(ims_k),
           "thickness_mm": float(thickness), "pixel_mm": pixel, "origin": [-pixel, -pixel],
           "via_area": via_area, "plated_holes": count, "size_mm": [float(size[0]), float(size[1])],
           "background_w": 0.0}
    return pcb, place, messages


def pcb_board_mesh(pcb):
    """(vertices, triangles, faces) of the board: outline and holes extruded from Z 0 to the
    board thickness; faces: top, bottom, outer edge, each hole wall."""
    outer = [tuple(p) for p in pcb["outline"]]
    holes = [[tuple(p) for p in hole] for hole in pcb["holes"]]
    tris, pts = cad.triangulate_with_holes(outer, holes)
    n = len(pts)
    thickness = pcb["thickness_mm"]
    vertices = [(x, y, 0.0) for x, y in pts] + [(x, y, thickness) for x, y in pts]
    triangles, faces = [], []

    def add(group):
        faces.append({"planar": True, "tris": list(range(len(triangles), len(triangles) + len(group)))})
        triangles.extend(group)

    add([(a + n, b + n, c + n) for a, b, c in tris])                     # top (CCW from above)
    add([(a, c, b) for a, b, c in tris])                                 # bottom
    start = 0
    for loop in [outer] + holes:
        m = len(loop)
        walls = []
        for i in range(m):
            a, b = start + i, start + (i + 1) % m
            walls += [(a, b, b + n), (a, b + n, a + n)]
        start += m
        faces.append({"planar": False, "tris": list(range(len(triangles), len(triangles) + len(walls)))})
        triangles.extend(walls)
    return vertices, triangles, faces


def pcb_stack(pcb):
    """Layers of the board from its top: [(thickness mm, copper layer index or None, k in plane,
    k through)] (copper layers get their conductivity from the coverage)."""
    core = PCB_CORES.get(pcb.get("core"), PCB_CORES["FR-4"])
    layers = pcb["layers"]
    if pcb.get("core") == "IMS":
        top = layers[0]["copper_um"] / 1000.0 if layers else 0.035
        dielectric = pcb.get("ims_dielectric_mm", 0.1)
        base = max(pcb["thickness_mm"] - top - dielectric, 0.1)
        k = pcb.get("ims_k", 2.0)
        return [(top, 0, None, None), (dielectric, None, k, k), (base, None, core["base_k"], core["base_k"])]
    slabs = []
    for index, layer in enumerate(layers):
        slabs.append((layer["copper_um"] / 1000.0, index, None, None))
        if index < len(layers) - 1:
            d = pcb["dielectric_mm"][index] if index < len(pcb["dielectric_mm"]) else 0.2
            slabs.append((d, None, core["k_plane"], core["k_through"]))
    return slabs


_PCB_TABLES = {}                 # summed-area tables of board rasters (by the rasters' identity)


def pack_arrays(value):
    """JSON-ready copy: numpy arrays become {"__array__": base64 of zlib bytes, …}; keys that
    start with '_' are left out."""
    import base64, zlib
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {"__array__": base64.b64encode(zlib.compress(array.tobytes(), 6)).decode("ascii"),
                "dtype": str(array.dtype), "shape": list(array.shape)}
    if isinstance(value, dict):
        return {k: pack_arrays(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [pack_arrays(v) for v in value]
    return value


def unpack_arrays(value):
    import base64, zlib
    if isinstance(value, dict):
        if "__array__" in value:
            raw = zlib.decompress(base64.b64decode(value["__array__"]))
            return np.frombuffer(raw, np.dtype(value["dtype"])).reshape(value["shape"]).copy()
        return {k: unpack_arrays(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack_arrays(v) for v in value]
    return value


def copy_sharing_arrays(value):
    """Deep copy of dicts and lists that shares numpy arrays (never changed in place)."""
    if isinstance(value, dict):
        return {k: copy_sharing_arrays(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [copy_sharing_arrays(v) for v in value]
    if isinstance(value, np.ndarray):
        return value
    return copy.deepcopy(value)


def _summed_area(raster):
    table = np.zeros((raster.shape[0] + 1, raster.shape[1] + 1))
    table[1:, 1:] = np.cumsum(np.cumsum(raster.astype(float), axis=0), axis=1)
    return table


def _box_mean(table, origin, pixel, x, y, dx, dy):
    """Mean of a raster (given by its summed-area table) over boxes around (x, y)."""
    ny, nx = table.shape[0] - 1, table.shape[1] - 1
    x0 = np.clip(np.floor((x - dx / 2 - origin[0]) / pixel).astype(int), 0, nx)
    x1 = np.clip(np.ceil((x + dx / 2 - origin[0]) / pixel).astype(int), 0, nx)
    y0 = np.clip(np.floor((y - dy / 2 - origin[1]) / pixel).astype(int), 0, ny)
    y1 = np.clip(np.ceil((y + dy / 2 - origin[1]) / pixel).astype(int), 0, ny)
    total = table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]
    count = np.maximum((x1 - x0) * (y1 - y0), 1)
    return total / count, count


def pcb_conductivity(pcb, x, y, dx, dy, layer, layers, dz):
    """(n, 3) conductivities W/(m·K) of board cells at board coordinates x, y (mm) with cell
    sizes dx, dy, dz; `layer` = index of the cell from the board top within its column of
    `layers` cells. The column stands for the whole stack: its cells share it evenly, so the
    in-plane sheet conductance and the through resistance of the column are the stack's."""
    slabs = pcb_stack(pcb)
    total = sum(s[0] for s in slabs)
    key = (id(pcb["via_area"]),) + tuple(id(l["coverage"]) for l in pcb["layers"])
    cache = _PCB_TABLES.get(key)
    if cache is None:                       # summed-area tables, kept for the next run
        if len(_PCB_TABLES) > 6:
            _PCB_TABLES.clear()
        cache = _PCB_TABLES[key] = {
            "copper": [_summed_area(l["coverage"] / 255.0) for l in pcb["layers"]],
            "vias": _summed_area(pcb["via_area"]), "arrays": (pcb["via_area"], pcb["layers"])}
    origin, pixel = pcb["origin"], pcb["pixel_mm"]
    lo = layer / layers * total
    hi = (layer + 1) / layers * total
    sheet = np.zeros(len(x))
    resistance = np.zeros(len(x))
    depth = 0.0
    for thickness, copper, k_plane, k_through in slabs:
        overlap = np.clip(np.minimum(hi, depth + thickness) - np.maximum(lo, depth), 0.0, None)
        depth += thickness
        if not overlap.any():
            continue
        if copper is not None:
            share, _count = _box_mean(cache["copper"][copper], origin, pixel, x, y, dx, dy)
            k = share * COPPER_K + (1.0 - share) * PCB_RESIN_K
            sheet += overlap * k
            resistance += overlap / k
        else:
            sheet += overlap * k_plane
            resistance += overlap / k_through
    span = hi - lo
    # plated holes: copper barrels through the board, in parallel to the layers
    via_mm2, count = _box_mean(cache["vias"], origin, pixel, x, y, dx, dy)
    share = np.clip(via_mm2 * count / np.maximum(dx * dy, 1e-12), 0.0, 0.9)
    conductance = (1.0 - share) / np.maximum(resistance, 1e-12) + share * COPPER_K / np.maximum(span, 1e-12)
    k_plane_cell = sheet / np.maximum(dz, 1e-12)
    k_through_cell = conductance * dz
    return np.stack([k_plane_cell, k_plane_cell, k_through_cell], axis=1)


def pcb_summary(pcb):
    """'2 copper layers (35 / 35 µm), FR-4, 1.6 mm, 37 × 25 mm, 64 vias'."""
    copper = " / ".join(f"{layer['copper_um']:g}" for layer in pcb["layers"])
    core = PCB_CORES.get(pcb.get("core"), PCB_CORES["FR-4"])["title"]
    shares = [float(layer["coverage"].mean()) / 255.0 for layer in pcb["layers"]]
    return (f"{len(pcb['layers'])} copper layer{'s' if len(pcb['layers']) != 1 else ''} ({copper} µm, "
            f"covered {' / '.join(f'{s:.0%}' for s in shares)}), {core}, {pcb['thickness_mm']:.3g} mm, "
            f"{pcb['size_mm'][0]:.4g} × {pcb['size_mm'][1]:.4g} mm, {pcb.get('plated_holes', 0)} plated holes")


# ── component packages (two-resistor model) ───────────────────────────────────
# Simple blocks; the die is a node with a resistance to the package top (θJC top) and one to
# the board below (θJB, or θJC of an exposed pad / tab soldered to the board). Typical values
# for orientation; datasheet values are better.
PACKAGES = {      # name: (length, width, height mm, θ top K/W, θ board K/W, Tj max °C, footprint pattern)
    "SOT-23": (2.9, 1.3, 1.0, 120.0, 80.0, 150.0, r"SOT-23(?!-?[5-8])"),
    "SOT-23-6": (2.9, 1.6, 1.1, 100.0, 60.0, 150.0, r"SOT-23-[56]|TSOT-23"),
    "SOT-223": (6.5, 3.5, 1.6, 30.0, 15.0, 150.0, r"SOT-223"),
    "SOT-89": (4.5, 2.5, 1.5, 40.0, 20.0, 150.0, r"SOT-89"),
    "SOD-123": (2.7, 1.6, 1.1, 150.0, 100.0, 150.0, r"SOD-123"),
    "SMA (DO-214AC)": (4.3, 2.6, 2.1, 60.0, 25.0, 150.0, r"D_SMA|DO-214AC"),
    "SMB (DO-214AA)": (4.6, 3.6, 2.2, 40.0, 18.0, 150.0, r"D_SMB|DO-214AA"),
    "SOIC-8": (4.9, 3.9, 1.5, 45.0, 60.0, 150.0, r"SOIC-8|SO-8"),
    "SOIC-8 EP": (4.9, 3.9, 1.5, 40.0, 10.0, 150.0, r"SOIC-8-1EP|SO-8.*EP"),
    "MSOP-8": (3.0, 3.0, 1.0, 60.0, 80.0, 150.0, r"MSOP-8|VSSOP-8"),
    "TSSOP-20 EP": (6.5, 4.4, 1.2, 30.0, 8.0, 150.0, r"TSSOP-\d+.*EP|HTSSOP"),
    "WSON / DFN 2×2 EP": (2.0, 2.0, 0.8, 70.0, 15.0, 150.0, r"WSON-6.*2x2|DFN-6.*2x2|SON.*2x2"),
    "QFN 3×3 EP": (3.0, 3.0, 0.9, 40.0, 6.0, 150.0, r"QFN-\d+.*3x3|DFN-\d+.*3x3"),
    "QFN 4×4 EP": (4.0, 4.0, 0.9, 25.0, 4.0, 150.0, r"QFN-\d+.*4x4"),
    "QFN 5×5 EP": (5.0, 5.0, 0.9, 20.0, 3.0, 150.0, r"QFN-\d+.*5x5|DFN-\d+.*5x6"),
    "DPAK (TO-252)": (6.6, 6.1, 2.3, 25.0, 2.0, 175.0, r"TO-252"),
    "D2PAK (TO-263)": (10.2, 9.2, 4.5, 15.0, 1.0, 175.0, r"TO-263"),
    "TO-220 (lying flat)": (10.2, 15.0, 4.5, 30.0, 1.5, 175.0, r"TO-220"),
    "TO-247 (lying flat)": (16.0, 21.0, 5.0, 25.0, 0.6, 175.0, r"TO-247"),
    "Power LED 3535": (3.5, 3.5, 2.0, 60.0, 8.0, 135.0, r"3535"),
    "LED 5050": (5.0, 5.0, 1.6, 60.0, 12.0, 125.0, r"5050"),
    "LED 1206": (3.2, 1.6, 1.1, 300.0, 150.0, 125.0, r"LED_1206"),
    "Resistor 0603": (1.6, 0.8, 0.45, 600.0, 250.0, 155.0, r"R_0603"),
    "Resistor 0805": (2.0, 1.25, 0.5, 450.0, 180.0, 155.0, r"R_0805"),
    "Resistor 1206": (3.2, 1.6, 0.6, 300.0, 120.0, 155.0, r"R_1206"),
    "Resistor / shunt 2512": (6.4, 3.2, 0.6, 150.0, 30.0, 155.0, r"R_2512|R_2010|Shunt"),
    "Inductor 1812": (4.5, 3.2, 3.0, 100.0, 50.0, 125.0, r"L_1812"),
    "Inductor 10×10": (10.0, 10.0, 4.5, 40.0, 20.0, 125.0, r"L_.*10\.?0?x10|IHLP-4040"),
    "Inductor 6×6": (6.0, 6.0, 3.0, 60.0, 30.0, 125.0, r"L_.*6\.?\d?x6|IHLP-2525"),
    "Custom block": (5.0, 5.0, 1.5, 50.0, 20.0, 150.0, None),
}
PACKAGE_MATERIAL = "Mould compound"


def guess_package(footprint):
    """Package name for a footprint name (KiCad library id), or 'Custom block'."""
    import re
    for name, entry in PACKAGES.items():
        if entry[6] and re.search(entry[6], footprint or "", re.IGNORECASE):
            return name
    return "Custom block"


def package_new(name, reference="", power_w=0.0):
    entry = PACKAGES.get(name, PACKAGES["Custom block"])
    return {"type": name if name in PACKAGES else "Custom block", "reference": reference,
            "theta_top": entry[3], "theta_board": entry[4], "tj_max": entry[5], "side": "top"}


def package_settings(data):
    if not isinstance(data, dict):
        return None
    package = package_new(str(data.get("type", "Custom block")), str(data.get("reference", "")))
    for key in ("theta_top", "theta_board", "tj_max"):
        if key in data:
            package[key] = max(1e-3, float(data[key])) if key != "tj_max" else float(data[key])
    package["side"] = "bottom" if data.get("side") == "bottom" else "top"
    return package


def pcb_frame_of(component):
    """Placement of a board: {"origin": world point of its 0, 0, 0, "axes": its unit X, Y, Z in
    world coordinates}."""
    basis = np.asarray(cad.assembly_world_vertices(component, [[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                                               [0, 0, 1]]), float)
    return {"origin": basis[0].tolist(), "axes": (basis[1:] - basis[0]).tolist()}


def pcb_local(frame, points):
    return (np.asarray(points, float) - np.asarray(frame["origin"], float)) @ \
        np.asarray(frame["axes"], float).T


def pcb_flipped(frame):
    """True when the board's top looks down (placed upside down)."""
    return float(np.asarray(frame["axes"], float)[2][2]) < 0


def _column_layers(i, j, k, flipped=False):
    """(index from the board top, cells in the column) of grid cells (i, j, k)."""
    column = np.asarray(i, np.int64) * 1_000_003 + np.asarray(j, np.int64)
    _unique, inverse, counts = np.unique(column, return_inverse=True, return_counts=True)
    top = np.full(len(_unique), -1, np.int64)
    bottom = np.full(len(_unique), 1 << 40, np.int64)
    np.maximum.at(top, inverse, k)
    np.minimum.at(bottom, inverse, k)
    layer = (np.asarray(k) - bottom[inverse]) if flipped else (top[inverse] - np.asarray(k))
    return layer, counts[inverse]


def junction_lines(results, transient):
    """Summary lines and warnings of the package junctions of a run."""
    lines, warnings = [], []
    for entry in results:
        peak = entry.get("peak", entry["t_j"])
        lines.append(f"Junction of '{entry['name']}' ({entry['package']}, {entry['power']:.3g} W): "
                     f"{entry['t_j']:.1f} °C" + (f" at the end, highest {peak:.1f} °C" if transient else "") +
                     f" (package top {entry['t_top']:.1f} °C, board below {entry['t_board']:.1f} °C; "
                     f"Tj max {entry['tj_max']:g} °C)")
        if peak > entry["tj_max"]:
            warnings.append(f"⚠ The junction of '{entry['name']}' reaches {peak:.1f} °C, above its "
                            f"{entry['tj_max']:g} °C limit.")
    return lines, warnings


def package_text(package):
    return (f"{package['type']}, θ top {package['theta_top']:g} K/W, θ board "
            f"{package['theta_board']:g} K/W, Tj max {package['tj_max']:g} °C")


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
    coolants = [c for c in conditions if c["kind"] == "coolant" and c["targets"]]
    baths = [c for c in conditions if c["kind"] == "bath" and c["targets"]]
    if settings["convection"] == "fixed" and settings["h"] <= 0 and not settings["radiation"] \
            and not has_fixed and not coolants and not any(c["bath_mode"] == "fixed" for c in baths):
        raise ThermalError("Without convection (h = 0) and radiation the heat cannot leave.")
    _check_thermostats(bodies, conditions)
    for condition in coolants + baths:
        if condition.get("liquid_model") == "flow":
            messages.append(f"'{condition['name']}': the liquid flow is solved in Airflow mode; "
                            "Conduction mode uses the correlations.")

    # 2. voxels ---------------------------------------------------------------------------
    grid = _voxel_owner(bodies, settings, step, messages)
    owner, h_mm, origin, shape = grid["owner"], grid["h_mm"], grid["origin"], grid["shape"]
    air_index, cavity_cells = grid["air_index"], grid["cavity_cells"]
    h = h_mm / 1000.0
    # liquids: a closed passage or a lidded container next to liquid faces is opened (its
    # faces then take the liquid condition); a bath that fills its container becomes a body of
    # its own (well mixed), the liquid below the fill level
    container_baths = [c for c in baths if c["bath_mode"] == "container"]
    if coolants or baths:
        grid_faces = [origin[a] + np.arange(shape[a] + 1) * h_mm for a in range(3)]
        triangles = {id(c): _condition_triangles(c, bodies) for c in coolants + baths}
        if cavity_cells:
            labels, _count = ndi.label(owner == air_index)
            opened = set()
            for c in coolants + baths:
                seeds = _liquid_seed_cells(*triangles[id(c)], grid_faces, owner < 0, 0.5 * h_mm)
                opened |= set(np.unique(labels[seeds]).tolist()) - {0}
            if opened:
                owner[np.isin(labels, sorted(opened))] = -1
                cavity_cells = int((owner == air_index).sum())
            del labels
        if container_baths:
            new_air = air_index + len(container_baths)
            owner[owner == air_index] = new_air
            for number, c in enumerate(container_baths):
                tris, normals = triangles[id(c)]
                fill = _bath_fill(c, tris)
                open_cells = (owner == -1) | (owner == new_air)
                seeds = _liquid_seed_cells(tris, normals, grid_faces, open_cells, 0.5 * h_mm)
                region, leaks = _bath_region(open_cells, grid_faces, seeds, fill)
                if leaks:
                    raise ThermalError(f"The liquid of '{c['name']}' runs out of its container below "
                                       f"the fill level (Z = {fill:.4g} mm): close the container, "
                                       "lower the fill level, or choose 'Large volume'.")
                if not region.any():
                    raise ThermalError(f"'{c['name']}' holds no liquid: the fill level (Z = "
                                       f"{fill:.4g} mm) is below the picked faces, or the space "
                                       "is thinner than one cell.")
                cells = np.argwhere(region)
                owner[region] = air_index + number
                bath = _bath_body(c, number, origin + cells.min(0) * h_mm,
                                  origin + (cells.max(0) + 1) * h_mm)
                bath["fill_z"] = fill
                bath["liquid_volume"] = len(cells) * (h_mm / 1000.0) ** 3
                bodies.append(bath)
            air_index = new_air
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
    # printed circuit boards: conductivity per element from the layer stack, the copper coverage
    # and the vias where the element sits
    for index, body in enumerate(bodies):
        pcb = body["job"].get("pcb")
        if pcb is None or not body["cells"]:
            continue
        chosen = np.flatnonzero(element_body == index)
        centre = origin + (np.stack([I[chosen], J[chosen], K[chosen]], axis=1) + 0.5) * h_mm
        local = pcb_local(body["job"]["pcb_frame"], centre)
        layer, layers = _column_layers(I[chosen], J[chosen], K[chosen],
                                       pcb_flipped(body["job"]["pcb_frame"]))
        k_base[chosen] = pcb_conductivity(pcb, local[:, 0], local[:, 1], np.full(len(chosen), h_mm),
                                          np.full(len(chosen), h_mm), layer, layers,
                                          np.full(len(chosen), h_mm))
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
    # bath bodies: the film on the wetted faces (natural convection in the liquid) is a
    # resistance 1/h between the liquid cells and the solid cells they touch, split like a
    # contact resistance; h follows the temperatures (update_film)
    bath_bodies = [index for index, body in enumerate(bodies) if body.get("bath") is not None]
    film_groups = {}      # (bath, solid body, orientation) → [(bath elements, solid elements, axis)]
    film_extra = None
    if bath_bodies:
        cell_id = (I.astype(np.int64) * shape[1] + J) * shape[2] + K
        padded_owner = np.pad(owner, 1, constant_values=-1)
        is_bath = np.zeros(air_index + 2, bool)
        is_bath[bath_bodies] = True
        for direction in range(6):
            axis, side = divmod(direction, 2)
            offset = np.zeros(3, int); offset[axis] = 1 if side else -1
            neighbour = padded_owner[I + 1 + offset[0], J + 1 + offset[1],
                                     K + 1 + offset[2]].astype(np.int64)
            wetted = np.isin(element_body, bath_bodies) & (neighbour >= 0) & \
                (neighbour != air_index) & ~is_bath[np.maximum(neighbour, 0)]
            touching = np.nonzero(wetted)[0]
            if not len(touching):
                continue
            other = np.searchsorted(cell_id, cell_id[touching] + (offset[0] * shape[1] + offset[1])
                                    * shape[2] + offset[2])
            # the solid's face looks towards the liquid: up when the liquid is above it
            orientation = 0 if axis != 2 else (2 if side == 1 else 1)
            for bath in np.unique(element_body[touching]):
                for solid in np.unique(neighbour[touching][element_body[touching] == bath]):
                    chosen = (element_body[touching] == bath) & (neighbour[touching] == solid)
                    film_groups.setdefault((int(bath), int(solid), orientation), []).append(
                        (touching[chosen], other[chosen], axis))
        del padded_owner
        film_extra = np.zeros((n_elements, 3))
        film_h = {key: BATH_FILM_GUESS for key in film_groups}

        def set_film():
            film_extra[:] = 0.0
            for key, parts in film_groups.items():
                half = 0.5 / max(film_h[key], 1e-6)
                for bath_elements, solid_elements, axis in parts:
                    np.add.at(film_extra, (bath_elements, axis), half)
                    np.add.at(film_extra, (solid_elements, axis), half)
        set_film()
        for index in bath_bodies:
            if not any(key[0] == index for key in film_groups):
                messages.append(f"⚠ The liquid of '{bodies[index]['bath']['name']}' touches no part.")

    def effective_k(k, elements=slice(None)):
        """Conductivities with the contact layers (and bath films) of the elements."""
        if contact_extra is None and film_extra is None:
            return k
        extra = (0.0 if contact_extra is None else contact_extra[elements]) + \
            (0.0 if film_extra is None else film_extra[elements])
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
    dynamic = bool(k_tabled) or bool(film_groups)     # elements re-assembled during the run
    if dynamic:
        film_elements = [np.concatenate([np.r_[a, b] for a, b, _axis in parts])
                         for parts in film_groups.values()]
        tab_elements = np.unique(np.concatenate([np.nonzero(np.isin(element_body, k_tabled))[0]]
                                                + film_elements))
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

    # component junctions (two-resistor model): the die is a node joined to the nodes of the
    # package top by 1/θ top and to those of the board below it by 1/θ board, each node by its
    # share c_i. Eliminated, it adds diag(c) to the matrix, a load P·c/S and the rank-one term
    # −c·cᵀ/S (S = Σ c), kept beside the sparse matrix; no node can get hotter than the die
    junctions = []
    junction_unit = {}
    for index, body in enumerate(bodies):
        junction = body["job"].get("junction")
        if junction is None or not body["cells"]:
            continue
        chosen = np.flatnonzero(element_body == index)
        k_of = K[chosen]
        upright = junction.get("side", "top") != "bottom"
        top_layer = chosen[k_of == (k_of.max() if upright else k_of.min())]
        board_layer = chosen[k_of == (k_of.min() if upright else k_of.max())]

        def face_weights(elements, high):
            corners = [c for c in range(8) if _CORNERS[c][2] == (1 if high else 0)]
            nodes, counts = np.unique(conn[elements][:, corners].ravel(), return_counts=True)
            return nodes, counts / counts.sum()
        top_nodes, top_w = face_weights(top_layer, upright)
        board_nodes, board_w = face_weights(board_layer, not upright)
        g_top = 1.0 / max(float(junction["theta_top"]), 1e-6)
        g_board = 1.0 / max(float(junction["theta_board"]), 1e-6)
        nodes, where = np.unique(np.r_[top_nodes, board_nodes], return_inverse=True)
        share = np.bincount(where, np.r_[top_w * g_top, board_w * g_board])
        total = g_top + g_board
        unit = np.zeros(n_nodes)
        unit[nodes] = share / total
        junction_unit[index] = unit
        junctions.append({"index": index, "name": body["name"], "junction": junction,
                          "top": (top_nodes, top_w), "board": (board_nodes, board_w),
                          "g_top": g_top, "g_board": g_board, "nodes": nodes, "share": share,
                          "total": total})
    if junctions:
        rows = np.concatenate([j["nodes"] for j in junctions])
        cols = np.concatenate([np.full(len(j["nodes"]), n) for n, j in enumerate(junctions)])
        junction_U = sp.csc_matrix((np.concatenate([j["share"] for j in junctions]), (rows, cols)),
                                   shape=(n_nodes, len(junctions)))
        junction_G = np.array([1.0 / j["total"] for j in junctions])
        junction_diagonal_slots = rows
        junction_diagonal_values = np.concatenate([j["share"] for j in junctions])
    body_power_now = {index: body["power"] for index, body in enumerate(bodies)}
    # heat sources: the power of a body is spread evenly over its cells; a source with a time
    # profile or a thermostat gets its own load vector of 1 W, scaled by its power in every
    # time step (and by the thermostat's on / off state or share of the power); a package's
    # power goes in at its junction
    element_power = np.zeros(n_elements)
    load_junction = np.zeros(n_nodes)
    profiled = []    # [(nodal load of 1 W, profile, nominal W, "source" | "flux", index, slot)]
    controlled = []  # thermostat slots: (name, thermostat, "source" | "flux", index)
    constant_power = 0.0
    for index, body in enumerate(bodies):
        if not body["cells"] or profile_peak_power(body["profile"], body["power"]) <= 0:
            continue
        if body["profile"] is None and body["thermostat"] is None:
            if index in junction_unit:
                load_junction += body["power"] * junction_unit[index]
            else:
                element_power[element_body == index] = body["power"] / body["cells"]
            constant_power += body["power"]
        else:
            unit = junction_unit[index] if index in junction_unit else \
                np.bincount(conn[element_body == index].ravel(), minlength=n_nodes) / \
                (8.0 * body["cells"])
            slot = None
            if body["thermostat"] is not None:
                slot = len(controlled)
                controlled.append((body["name"], body["thermostat"], "source", index))
            profiled.append((unit, body["profile"], body["power"], "source", index, slot))
    load_source = np.bincount(conn.ravel(), np.repeat(element_power / 8.0, 8), n_nodes) + load_junction
    del element_power, load_junction

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
    if not dynamic:
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
            if condition["kind"] == "bath" and condition["bath_mode"] == "container":
                continue                      # its faces now touch the bath body
            mask = np.zeros(n_faces, bool)
            for b, faces in condition["targets"].items():
                mask |= (face_body == b) & np.isin(mesh_face, faces)
            face_condition[mask] = index
        for index, condition in enumerate(conditions):
            if condition["kind"] == "bath" and condition["bath_mode"] == "container":
                continue
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

    # coolant channels and baths of a fixed temperature: their faces exchange heat with the
    # liquid (h from correlations, liquid temperature along the channel), no radiation
    liquid_records = []
    liquid_faces = np.zeros(n_faces, bool)
    liquid_h = np.zeros(n_faces)
    for index, condition in enumerate(conditions):
        if condition["kind"] not in LIQUID_KINDS or \
                (condition["kind"] == "bath" and condition["bath_mode"] == "container"):
            continue
        chosen = np.flatnonzero(face_condition == index)
        if not len(chosen):
            continue
        record = {"index": index, "condition": condition, "faces": chosen}
        if condition["kind"] == "coolant":
            step(39, f"Analysing the channel '{condition['name']}'...")
            geometry = channel_geometry(_condition_triangles(condition, bodies)[0],
                                        condition["reverse"])
            s = geometry["s"][cKDTree(geometry["points"]).query(face_centre[chosen])[1]]
            n_seg = int(np.clip(round(geometry["length_mm"] / h_mm), 8, 400))
            seg = np.clip((s / max(geometry["length_mm"], 1e-9) * n_seg).astype(int), 0, n_seg - 1)
            # a curved wall: the area of the picked faces, not of the larger cell staircase (a
            # picked face partly covered by another part keeps only its open cell faces)
            face_area[chosen] *= min(1.0, geometry["wall_area_mm2"] * 1e-6 /
                                     max(float(face_area[chosen].sum()), 1e-30))
            record.update(geometry=geometry, seg=seg, n_seg=n_seg,
                          area_seg=np.bincount(seg, face_area[chosen], n_seg), t_face=None)
        liquid_faces[chosen] = True
        liquid_records.append(record)
    channels = _ordered_channels([r for r in liquid_records if r["condition"]["kind"] == "coolant"])
    face_orientation = np.where(face_dir == 5, 1, np.where(face_dir == 4, 2, 0))

    def update_liquids(T, relax=1.0):
        """Liquid temperatures and coefficients of the coolant / fixed-bath faces for the nodal
        temperatures T; returns the largest change of a liquid temperature (K)."""
        change = 0.0
        face_t = T[face_nodes].mean(1)
        for record in channels:
            chosen = record["faces"]
            old = face_ambient[chosen].copy()
            face_ambient[chosen] = _update_channel(record, face_t[chosen], face_area[chosen], relax)
            liquid_h[chosen] = record["h"]
            change = max(change, float(np.abs(face_ambient[chosen] - old).max()))
        for record in liquid_records:
            c = record["condition"]
            if c["kind"] != "bath":
                continue
            chosen = record["faces"]
            values = np.zeros(len(chosen))
            for b in np.unique(face_body[chosen]):
                for o in (0, 1, 2):
                    sel = (face_body[chosen] == b) & (face_orientation[chosen] == o)
                    if not sel.any():
                        continue
                    t_mean = float((face_t[chosen][sel] * face_area[chosen][sel]).sum() /
                                   face_area[chosen][sel].sum())
                    values[sel] = _liquid_natural_h(c["fluid"], t_mean - c["bath_c"],
                                                    face_length[chosen][sel], np.array(o),
                                                    0.5 * (t_mean + c["bath_c"]))
            liquid_h[chosen] = values if relax >= 1.0 or not record.get("started") else \
                (1.0 - relax) * liquid_h[chosen] + relax * values
            record["started"] = True
            face_ambient[chosen] = c["bath_c"]
        return change

    # bath bodies: film coefficients from the mean temperatures of the liquid and of the wetted
    # faces of each part (per orientation); evaporation from the open surface
    bath_elements = {index: np.flatnonzero(element_body == index) for index in bath_bodies}
    evaporation_share = np.zeros(air_index + 1)
    for index in bath_bodies:
        condition = bodies[index]["bath"]
        if condition.get("evaporation", True):
            evaporation_share[index] = (FLUIDS.get(condition["fluid"]) or {}).get("evaporation", 0.0)
    face_evaporation = evaporation_share[face_body]

    def update_film(T, relax=1.0):
        """New film coefficients of the bath bodies; returns their largest relative change."""
        change = 0.0
        t_bath = {index: float(T[conn[elements]].mean()) for index, elements in bath_elements.items()}
        for key, parts in film_groups.items():
            bath, solid, orientation = key
            solid_elements = np.concatenate([b for _a, b, _axis in parts])
            t_solid = float(T[conn[solid_elements]].mean())
            if orientation == 0:
                length = min(bodies[solid]["extent"][2], bodies[bath]["extent"][2])
            else:
                e_s, e_b = bodies[solid]["extent"], bodies[bath]["extent"]
                length = min(e_s[0] * e_s[1] / (2.0 * (e_s[0] + e_s[1])),
                             e_b[0] * e_b[1] / (2.0 * (e_b[0] + e_b[1])))
            fluid = bodies[bath]["bath"]["fluid"]
            new = float(_liquid_natural_h(fluid, t_solid - t_bath[bath], max(length, 2.0) / 1000.0,
                                          np.array(orientation), 0.5 * (t_solid + t_bath[bath])))
            new = (1.0 - relax) * film_h[key] + relax * new
            change = max(change, abs(new - film_h[key]) / max(film_h[key], 1e-9))
            film_h[key] = new
        set_film()
        return change

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
                body_power_now[index] = power
            else:
                flux[index] = power
        return load, source, flux

    _load, total_power, flux_steady = heat_input()
    total_in = total_power + float(flux_steady.sum())
    if not transient and total_in <= 0 and not has_fixed:
        raise ThermalError("The heat sources have no power in steady state (an on / off cycle "
                           "with 0 % on, or a table ending at 0 W). Use a transient run.")

    evaporation_h = np.zeros(n_faces)

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
        evaporating = face_evaporation > 0
        if evaporating.any():              # open surface of a bath: evaporation on top
            extra = np.zeros(len(face_t))
            extra[evaporating] = face_evaporation[evaporating] * _evaporation_h(
                face_t[evaporating], ambient, convection[evaporating], 1.0)
            evaporation_h[:] = extra
            convection = convection + extra
        if conditions:
            convection = np.where(own_convection, own_h, convection)
            convection[no_exchange] = 0.0
            radiation[no_exchange] = 0.0
            if liquid_faces.any():
                convection = np.where(liquid_faces, liquid_h, convection)
                radiation[liquid_faces] = 0.0
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
        if body.get("synthetic"):                     # the liquid of a bath has no own mesh
            continue
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

    def junction_temperatures(T):
        """(junction, package top, board) °C of every junction for the nodal temperatures T."""
        out = []
        for j in junctions:
            t_top = float(T[j["top"][0]] @ j["top"][1])
            t_board = float(T[j["board"][0]] @ j["board"][1])
            power = float(body_power_now.get(j["index"], 0.0))
            out.append(((power + float(T[j["nodes"]] @ j["share"])) / j["total"], t_top, t_board))
        return out

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
        if junctions:                     # the junctions' conductances to the package faces
            np.add.at(data, diagonal_slots[junction_diagonal_slots], junction_diagonal_values)
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

    if junctions:
        junction_free = junction_U.tolil()
        if len(fixed_nodes):
            junction_free[fixed_nodes, :] = 0.0
        junction_free = junction_free.tocsr()
        junction_diag = -np.asarray(junction_free.multiply(junction_free) @ junction_G).ravel()

    def junction_product(v, free=True):
        """The junctions' rank-one couplings −c·cᵀ/S applied to the nodal vector v."""
        U = junction_free if free else junction_U.tocsr()
        return -(U @ (junction_G * (U.T @ v)))

    def constrain(data, rhs):
        if not len(fixed_nodes):
            return data, rhs
        matrix = sp.csr_matrix((data, indices, indptr), shape=(n_nodes, n_nodes))
        rhs = rhs - matrix @ fixed_vector
        if junctions:
            rhs = rhs + junction_free @ (junction_G * (junction_U.T @ fixed_vector))
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
        diagonal = data[diagonal_slots]
        if junctions:                         # the matrix plus the junction couplings
            sparse_matrix = matrix
            diagonal = diagonal + junction_diag
            matrix = spla.LinearOperator(sparse_matrix.shape, dtype=float,
                                         matvec=lambda v: sparse_matrix @ v + junction_product(v))
        inverse_diagonal = 1.0 / diagonal
        if prolongation is not None:
            if refresh or coarse["lu"] is None:
                if junctions:
                    PU = prolongation.T @ junction_free
                    coarse_matrix = (prolongation.T @ sparse_matrix @ prolongation -
                                     PU @ sp.diags(junction_G) @ PU.T).tocsc()
                else:
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

    liquids_present = bool(liquid_records) or bool(film_groups)
    nonlinear = settings["convection"] != "fixed" or settings["radiation"] or bool(k_tabled) or \
        liquids_present
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
        for iteration in range(60 if liquids_present else 40 if nonlinear else 1):
            step(45 + 50 * min(1.0, iteration / 8.0),
                 f"Solving {n_nodes:,} nodes — iteration {iteration + 1}".replace(",", " ") +
                 ("" if not iteration else f", change {change:.3g} K"))
            if change > previous_change:          # oscillating: damp the coefficient update
                relaxation = max(0.3, relaxation * 0.6)
            previous_change = change
            if liquid_records:
                update_liquids(T, relaxation if iteration else 1.0)
            if film_groups:
                update_film(T, relaxation if iteration else 1.0)
            convection, radiation = face_coefficients(T)
            if convection_c is None:
                convection_c, radiation_c = convection, radiation
            else:
                convection_c = (1.0 - relaxation) * convection_c + relaxation * convection
                radiation_c = (1.0 - relaxation) * radiation_c + relaxation * radiation
            if dynamic:
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
        for index, elements in bath_elements.items():      # a bath may start warmer or colder
            condition = bodies[index]["bath"]
            if not condition.get("bath_start_auto", True):
                T[np.unique(conn[elements])] = float(condition["bath_c"])
        T[fixed_nodes] = fixed_value[fixed_nodes]          # held from the start
        node_capacity = capacity_at(T)
        ratio_assembled = None
        film_assembled = None
        coolant_series = np.zeros((len(channels), n_steps + 1))
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

        junction_series = np.zeros((len(junctions), n_steps + 1))

        def record(step_index, T):
            stats[step_index] = body_stats(T)
            if junctions:
                junction_series[:, step_index] = [t[0] for t in junction_temperatures(T)]
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
            film_changed = False
            if film_groups:                      # bath films: re-assembled when h moved by 1 %
                update_film(T)
                film_changed = film_assembled is None or max(
                    abs(film_h[key] - film_assembled[key]) / max(film_assembled[key], 1e-9)
                    for key in film_h) > 1e-2
            if dynamic:                          # re-assemble when k changed by over 1 %; the
                ratio = tabled_ratio(T)          # coarse preconditioner may lag behind
                if ratio_assembled is None or film_changed or \
                        float(np.abs(ratio - ratio_assembled).max()) > 1e-2:
                    assemble_k(ratio)
                    ratio_assembled = ratio
                    if film_groups:
                        film_assembled = dict(film_h)
            if cp_tabled and step_index > 1:
                node_capacity = capacity_at(T)
            if liquid_records:
                update_liquids(T)
            convection, radiation = face_coefficients(T)
            load, total_power, flux_now = heat_input((step_index - 1) * dt, step_index * dt,
                                                     factors)
            power_series[step_index] = total_power + float(flux_now.sum())
            # start from the linear extrapolation of the last two steps (far fewer iterations)
            guess = T if step_index == 1 else T + (T - previous)
            T_old = T
            previous = T
            for liquid_pass in range(3 if channels else 1):
                data, rhs = system(convection * face_area, radiation * face_area, load)
                data[diagonal_slots] += node_capacity
                rhs += node_capacity * T_old
                last_system = (data, rhs)
                data, rhs = constrain(data, rhs)
                # the surface coefficients change slowly: the coarse factorisation is renewed
                # now and then only
                T, info, steps_used = solve(data, rhs, guess, TRANSIENT_TOLERANCE,
                                            refresh=refresh and liquid_pass == 0)
                cg_total += steps_used
                solver_short |= info > 0
                # coolant: the liquid temperatures again from the new wall temperatures
                if not channels or liquid_pass == 2 or update_liquids(T) < 0.02:
                    break
                convection = np.where(liquid_faces, liquid_h, convection)
                guess = T
            for number, channel in enumerate(channels):
                coolant_series[number, step_index] = channel["t_out"]
            t_min, t_max = min(t_min, float(T.min())), max(t_max, float(T.max()))
            record(step_index, T)
        q_stored = float((node_capacity * (T - previous)).sum())       # W, heat still stored
        iterations = n_steps
        total_in = total_power + float(flux_now.sum())        # during the last step
        series = {"times": np.arange(n_steps + 1) * dt, "min": stats[:, 0, :].T,
                  "mean": stats[:, 1, :].T, "max": stats[:, 2, :].T, "power": power_series}
        if thermostats is not None:
            series["thermostats"] = switch_series
        if junctions:
            series["junctions"] = junction_series
            series["junction_names"] = [j["name"] for j in junctions]
    if solver_short:
        messages.append("⚠ The linear solver stopped before full accuracy; the result may be "
                        "slightly off.")

    # 6. results --------------------------------------------------------------------------
    step(97, "Evaluating results...")
    convection, radiation = face_coefficients(T)
    face_t = T[face_nodes].mean(1)
    convection_flow = convection * face_area * (face_t - face_ambient)       # W per face
    radiation_flow = radiation * face_area * (face_t - ambient)
    q_liquid = float(convection_flow[liquid_faces].sum())     # into coolant / fixed baths
    q_convection = float(convection_flow.sum()) - q_liquid
    q_evaporation = float((evaporation_h * face_area * (face_t - ambient)).sum())
    q_radiation = float(radiation_flow.sum())
    # heat put in at fixed-temperature nodes to hold them = residual of their rows in the full
    # system (negative: heat leaves the parts there)
    reaction = np.zeros(n_nodes)
    if len(fixed_nodes):
        full = sp.csr_matrix((last_system[0], indices, indptr), shape=(n_nodes, n_nodes))
        reaction[fixed_nodes] = (full @ T - last_system[1] +
                                 (junction_product(T, free=False) if junctions else 0.0))[fixed_nodes]
    q_fixed = -float(reaction.sum())                    # leaving through fixed-temperature faces
    q_out = q_convection + q_radiation + q_fixed + q_liquid
    scale = max(total_in, abs(q_convection) + abs(q_radiation) + abs(q_liquid) +
                float(np.abs(reaction).sum()), 1e-9)
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
                "coolant": float(convection_flow[mask].sum()),
                "bath": float(convection_flow[mask].sum()),
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
    if abs(q_liquid) > 1e-9:
        fixed_text += f", liquids {q_liquid:.4g} W"
    if abs(q_evaporation) > 1e-9:
        fixed_text += f" (convection includes evaporation {q_evaporation:.4g} W)"
    # liquids: coolant channels, fixed-temperature baths, baths that fill their container
    liquid_results = []
    liquid_lines = []
    for record in channels:
        entry, text = channel_summary(record, float(convection_flow[record["faces"]].sum()))
        liquid_results.append(entry)
        liquid_lines.append(text)
        if record["flow_m3s"] <= 0:
            messages.append(f"⚠ Coolant '{entry['name']}': no flow (the pump cannot overcome the "
                            "channel), so the liquid takes up no heat.")
        limit = (FLUIDS.get(entry["fluid"]) or {}).get("limit_c")
        if entry["wall_max"] is not None and limit is not None and entry["wall_max"] > limit:
            messages.append(f"⚠ Coolant '{entry['name']}': the channel wall reaches "
                            f"{entry['wall_max']:.1f} °C; {entry['fluid']} "
                            f"{FLUIDS[entry['fluid']]['limit_text']}.")
    for record in liquid_records:
        c = record["condition"]
        if c["kind"] != "bath":
            continue
        chosen = record["faces"]
        heat = float(convection_flow[chosen].sum())
        area = float(face_area[chosen].sum())
        wall_mean = float((face_t[chosen] * face_area[chosen]).sum() / max(area, 1e-30))
        wall_max = float(face_t[chosen].max())
        liquid_results.append({"name": c["name"], "kind": "bath", "mode": "fixed",
                               "fluid": c["fluid"], "t_liquid": c["bath_c"], "heat": heat,
                               "h": heat / max(area * (wall_mean - c["bath_c"]), 1e-12),
                               "wall_max": wall_max})
        liquid_lines.append(f"Liquid bath '{c['name']}' ({c['fluid']} at {c['bath_c']:g} °C, large "
                            f"volume): {heat:.4g} W into the liquid over {area * 1e4:.4g} cm², "
                            f"wetted faces up to {wall_max:.1f} °C")
        limit = FLUIDS[c["fluid"]]["limit_c"]
        if wall_max > limit:
            messages.append(f"⚠ Liquid bath '{c['name']}': wetted faces reach {wall_max:.1f} °C; "
                            f"{c['fluid']} {FLUIDS[c['fluid']]['limit_text']}.")
    for index in bath_bodies:
        body, c = bodies[index], bodies[index]["bath"]
        elements = bath_elements[index]
        t_liquid = float(T[conn[elements]].mean())
        mine = face_body == index
        lost_convection = float(convection_flow[mine].sum())
        lost_evaporation = float((evaporation_h[mine] * face_area[mine] *
                                  (face_t[mine] - ambient)).sum())
        lost_radiation = float(radiation_flow[mine].sum())
        stored = 0.0
        if transient:
            stored = float((element_capacity[elements] * h ** 3 *
                            (T[conn[elements]].mean(1) - previous[conn[elements]].mean(1))).sum()) / dt
        wetted_area, wetted_h, wall_max = 0.0, 0.0, -np.inf
        for key, parts in film_groups.items():
            if key[0] != index:
                continue
            count = sum(len(a) for a, _b, _axis in parts)
            wetted_area += count * h * h
            wetted_h += film_h[key] * count * h * h
            solid_elements = np.concatenate([b for _a, b, _axis in parts])
            wall_max = max(wall_max, float(T[conn[solid_elements]].max()))
        heat_in = lost_convection + lost_radiation + stored
        open_area = float(face_area[mine].sum())
        entry = {"name": c["name"], "kind": "bath", "mode": "container", "fluid": c["fluid"],
                 "t_liquid": t_liquid, "volume_l": body["liquid_volume"] * 1000.0,
                 "fill_z": body["fill_z"], "heat": heat_in, "open_area_cm2": open_area * 1e4,
                 "h": wetted_h / max(wetted_area, 1e-30), "wetted_area_cm2": wetted_area * 1e4,
                 "convection": lost_convection - lost_evaporation, "evaporation": lost_evaporation,
                 "radiation": lost_radiation, "stored": stored,
                 "wall_max": wall_max if np.isfinite(wall_max) else None}
        liquid_results.append(entry)
        liquid_lines.append(
            f"Liquid bath '{c['name']}' ({c['fluid']}, {entry['volume_l']:.3g} L up to Z = "
            f"{body['fill_z']:.4g} mm, well mixed): {t_liquid:.2f} °C; net {heat_in:.4g} W from the "
            f"wetted faces (warmer parts give, cooler walls take; {entry['wetted_area_cm2']:.4g} cm², "
            f"film h ≈ {entry['h']:.3g} W/m²K); "
            + (f"open surface {entry['open_area_cm2']:.4g} cm² gives off {entry['convection']:.4g} W "
               f"by convection, {lost_evaporation:.4g} W by evaporation, {lost_radiation:.4g} W "
               "by radiation" if open_area > 0 else "closed container (no open surface)")
            + (f"; {stored:.4g} W still warming the liquid" if transient else ""))
        limit = FLUIDS[c["fluid"]]["limit_c"]
        if entry["wall_max"] is not None and entry["wall_max"] > limit:
            messages.append(f"⚠ Liquid bath '{c['name']}': wetted faces reach "
                            f"{entry['wall_max']:.1f} °C; {c['fluid']} "
                            f"{FLUIDS[c['fluid']]['limit_text']}.")
    if transient and channels:
        for number, channel in enumerate(channels):
            coolant_series[number, 0] = channel["condition"]["inlet_c"]
        series["coolant"] = coolant_series
        series["coolant_names"] = [channel["condition"]["name"] for channel in channels]
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
        if entry["kind"] in LIQUID_KINDS:              # reported in their own lines below
            continue
        title = CONDITION_KINDS[entry["kind"]][0]
        effect = {"fixed": f"{entry['heat']:.4g} W leave" if entry["heat"] >= 0 else
                           f"{-entry['heat']:.4g} W enter",
                  "flux": f"{entry['heat']:.4g} W enter",
                  "convection": f"{entry['heat']:.4g} W leave",
                  "insulated": "no heat crosses"}[entry["kind"]]
        summary.append(f"Face condition '{entry['name']}' ({title}, {entry['area_mm2']:.4g} mm²): "
                       f"{effect}")
    summary += liquid_lines
    junction_results = []
    for n, (j, (t_j, t_top, t_board)) in enumerate(zip(junctions, junction_temperatures(T))):
        junction = j["junction"]
        entry = {"name": j["name"], "package": junction.get("type", "Custom block"), "t_j": t_j,
                 "t_top": t_top, "t_board": t_board, "power": float(body_power_now.get(j["index"], 0.0)),
                 "tj_max": float(junction.get("tj_max", 150.0)),
                 "theta_top": float(junction["theta_top"]), "theta_board": float(junction["theta_board"])}
        if transient:
            entry["peak"] = float(junction_series[n].max())
        junction_results.append(entry)
    lines, warnings = junction_lines(junction_results, transient)
    summary += lines
    messages += warnings
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
    tabled = [i for i in sorted(set(k_tabled) | set(cp_tabled if transient else []))
              if not bodies[i].get("synthetic")]          # liquids always follow their tables
    if tabled:
        summary.append("Temperature-dependent material data: " + ", ".join(
            f"{bodies[i]['name']} ({bodies[i]['material']})" for i in tabled))
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
            "thermostats": thermostat_results, "liquids": liquid_results, "q_liquid": q_liquid,
            "junctions": junction_results,
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
    source = np.arange(len(T))
    # long, thin triangles (a tessellated cylinder or pipe) are first cut into strips across
    # their length, from the corner opposite their shortest edge, so the n² split below stays
    # near the ideal count instead of splitting them n × n by their long edges
    lengths = np.linalg.norm(P[:, [1, 2, 0]] - P, axis=2)      # edge i runs from corner i to i+1
    longest = lengths.max(1)
    area2 = np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    shortest = lengths.min(1)
    thin = (longest > 2.0 * max_edge) & (longest > 3.0 * shortest) & (area2 > 0)
    if thin.any():
        chosen = np.flatnonzero(thin)
        apex = (np.argmin(lengths[chosen], axis=1) + 2) % 3        # corner opposite the shortest
        order = (np.arange(3)[None, :] + apex[:, None]) % 3        # apex first, orientation kept
        Q = P[chosen[:, None], order]
        strips = np.clip(np.ceil(longest[chosen] / max(max_edge, 1e-9)), 2, 4096).astype(np.int64)
        pieces, piece_source = [P[~thin]], [source[~thin]]
        for k in np.unique(strips):
            k = int(k)
            mine = strips == k
            A, B, C = Q[mine, 0], Q[mine, 1], Q[mine, 2]
            f = np.arange(k + 1)[None, :, None] / k
            Pk = A[:, None] + f * (B - A)[:, None]                  # along A → B
            Qk = A[:, None] + f * (C - A)[:, None]                  # along A → C
            first = np.stack([Pk[:, :-1], Pk[:, 1:], Qk[:, 1:]], 2)          # (m, k, 3, 3)
            second = np.stack([Pk[:, 1:-1], Qk[:, 2:], Qk[:, 1:-1]], 2)      # none at the apex
            pieces += [first.reshape(-1, 3, 3), second.reshape(-1, 3, 3)]
            ids = source[chosen[mine]]
            piece_source += [np.repeat(ids, k), np.repeat(ids, k - 1)]
        P = np.concatenate(pieces)
        source = np.concatenate(piece_source)
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
        out_source.append(np.repeat(source[chosen], k))
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
    gid = np.empty((len(u), 8), np.int64)
    weights = np.empty((len(u), 8))
    for c, corner in enumerate(_CORNERS):
        cell = base + corner
        gid[:, c] = (cell[:, 0] * ny1 + cell[:, 1]) * nz1 + cell[:, 2]
        weights[:, c] = np.prod(np.where(corner, frac, 1.0 - frac), axis=1)
    # look up each node id once, in sorted order (far faster than unsorted searches)
    unique, inverse = np.unique(gid.ravel(), return_inverse=True)
    position = np.minimum(np.searchsorted(nodes_global, unique), len(nodes_global) - 1)
    index = position[inverse].reshape(gid.shape)
    weights *= (nodes_global[position] == unique)[inverse].reshape(gid.shape)
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


def _slab(axis, start, stop):
    index = [slice(None)] * 3
    index[axis] = slice(start, stop)
    return tuple(index)


def _face_heat_split(G, T, source, target):
    """(W from the cells `source` into the neighbouring cells `target`, W flowing back) through
    the conductances G of the interior cell faces (airflow grid), each face counted by its sign."""
    forward, back = 0.0, 0.0
    for c in range(3):
        g = G[c][_slab(c, 1, -1)]
        lo_T, hi_T = T[_slab(c, 0, -1)], T[_slab(c, 1, None)]
        down = source[_slab(c, 0, -1)] & target[_slab(c, 1, None)]
        up = target[_slab(c, 0, -1)] & source[_slab(c, 1, None)]
        flow = np.concatenate([(g * (lo_T - hi_T))[down], (g * (hi_T - lo_T))[up]])
        forward += float(flow[flow > 0].sum())
        back -= float(flow[flow < 0].sum())
    return forward, back


def _face_heat(G, T, source, target):
    """W flowing (net) from the cells `source` into the neighbouring cells `target`."""
    forward, back = _face_heat_split(G, T, source, target)
    return forward - back


def _airflow_liquid_results(T, owner, solid, bodies, bath_bodies, regions, region_cells,
                            face_liquids, face_channels, evaporation, G, p, velocity, face_area,
                            volume, shape, transient, unsteady, capacity, messages):
    """Result entries and summary lines of the liquids of an airflow run."""
    entries, lines = [], []
    flat_T = T.ravel()
    air = ~solid
    for region in regions:
        air &= ~region_cells[regions.index(region)]
    for record in face_channels:
        entry, text = channel_summary(record, float(record.get("heat", 0.0)))
        entries.append(entry)
        lines.append(text)
    for record in face_liquids:
        c = record["condition"]
        if c["kind"] != "bath":
            continue
        heat = float(record.get("heat", 0.0))
        area = float(record["area"].sum())
        entries.append({"name": c["name"], "kind": "bath", "mode": "fixed", "fluid": c["fluid"],
                        "t_liquid": c["bath_c"], "heat": heat,
                        "h": float((record["h_faces"] * record["area"]).sum() / max(area, 1e-30)),
                        "wall_max": float(flat_T[record["cells"]].max())})
        lines.append(f"Liquid bath '{c['name']}' ({c['fluid']} at {c['bath_c']:g} °C, large volume): "
                     f"{heat:.4g} W into the liquid over {area * 1e4:.4g} cm²")
    cell_volume = np.broadcast_to(volume, shape)

    def stored(mask):
        if not transient or unsteady["dt"] is None:
            return 0.0
        return float((capacity * (T - unsteady["T"]))[mask].sum()) / unsteady["dt"]

    for b in bath_bodies:
        body, c = bodies[b], bodies[b]["bath"]
        mask = owner == b
        values = T[mask]
        weights = cell_volume[mask]
        t_mean = float((values * weights).sum() / weights.sum())
        to_air = _face_heat(G, T, mask, air) if G is not None else 0.0
        evap = sum(float(item.get("heat", 0.0))
                   for item in evaporation if item.get("owner") == ("body", b))
        heat_in = to_air + evap + stored(mask)
        near = np.zeros(shape, bool)                  # solid cells the liquid wets
        for a in range(3):
            near[_slab(a, 0, -1)] |= mask[_slab(a, 1, None)]
            near[_slab(a, 1, None)] |= mask[_slab(a, 0, -1)]
        near &= solid & ~mask
        entry = {"name": c["name"], "kind": "bath", "mode": "container", "fluid": c["fluid"],
                 "t_liquid": t_mean, "volume_l": float(weights.sum()) * 1000.0,
                 "fill_z": body["fill_z"], "heat": heat_in, "to_air": to_air, "evaporation": evap,
                 "stored": stored(mask), "solved": False,
                 "wall_max": float(T[near].max()) if near.any() else None}
        entries.append(entry)
        gained, given = _face_heat_split(G, T, solid & ~mask, mask) if G is not None else (0.0, 0.0)
        entry.update(gained=gained, given=given)
        lines.append(f"Liquid bath '{c['name']}' ({c['fluid']}, {entry['volume_l']:.3g} L up to Z = "
                     f"{body['fill_z']:.4g} mm, well mixed): {t_mean:.2f} °C; {gained:.4g} W from warmer "
                     f"wetted faces, {given:.4g} W into cooler ones; to the air above {to_air:.4g} W, "
                     f"evaporation {evap:.4g} W" +
                     (f", {entry['stored']:.4g} W still warming it" if transient else ""))
    for r, region in enumerate(regions):
        c = region["condition"]
        mask = region_cells[r]
        values = T[mask]
        weights = cell_volume[mask]
        t_mean = float((values * weights).sum() / weights.sum())
        centre_speed = np.sqrt(sum((0.5 * (velocity[a][_slab(a, 0, -1)] + velocity[a][_slab(a, 1, None)]))
                                   ** 2 for a in range(3)))
        top_speed = float(centre_speed[mask].max())
        if region["kind"] == "bath":
            gained, given = _face_heat_split(G, T, solid, mask) if G is not None else (0.0, 0.0)
            from_walls = gained - given
            to_air = _face_heat(G, T, mask, air) if G is not None else 0.0
            evap = sum(float(item.get("heat", 0.0))
                       for item in evaporation if item.get("owner") == ("region", r))
            entry = {"name": c["name"], "kind": "bath", "mode": "container", "fluid": c["fluid"],
                     "t_liquid": t_mean, "t_min": float(values.min()), "t_max": float(values.max()),
                     "volume_l": float(weights.sum()) * 1000.0, "fill_z": region["fill_z"],
                     "heat": from_walls, "to_air": to_air, "evaporation": evap,
                     "stored": stored(mask), "speed_max": top_speed, "solved": True}
            entries.append(entry)
            lines.append(f"Liquid bath '{c['name']}' ({c['fluid']}, {entry['volume_l']:.3g} L up to Z = "
                         f"{region['fill_z']:.4g} mm, flow solved): mean {t_mean:.2f} °C "
                         f"({entry['t_min']:.2f} … {entry['t_max']:.2f} °C), speeds up to "
                         f"{top_speed * 1000.0:.3g} mm/s; {gained:.4g} W from warmer wetted faces, "
                         f"{given:.4g} W into cooler ones; to the air above {to_air:.4g} W, "
                         f"evaporation {evap:.4g} W" +
                         (f", {entry['stored']:.4g} W still warming it" if transient else ""))
            limit = FLUIDS[c["fluid"]]["limit_c"]
        else:
            flat_p = p.ravel()
            t_out = float(region["outlet_weights"] @ flat_T[region["outlet_cells"]])
            dp = float(region["inlet_weights"] @ flat_p[region["inlet_cells"]] -
                       region["outlet_weights"] @ flat_p[region["outlet_cells"]])
            q = region["flow_m3s"]
            region["t_out"] = t_out
            t_in = region.get("t_in", c["inlet_c"])
            correlation = pipe_flow(region["fluid"], q, region["geometry"], 0.5 * (t_in + t_out))
            entry = {"name": c["name"], "kind": "coolant", "fluid": c["fluid"], "flow_lpm": q * 60000.0,
                     "velocity": correlation["velocity"], "re": correlation["re"],
                     "regime": correlation["regime"], "t_in": t_in, "t_out": t_out,
                     "heat": float(region.get("heat", 0.0)), "dp_pa": dp,
                     "dp_correlation": correlation["dp"], "nu": correlation["nu"],
                     "h": correlation["h"], "geometry": channel_text(region["geometry"]),
                     "speed_max": top_speed, "solved": True, "wall_max": None}
            entries.append(entry)
            lines.append(f"Coolant '{c['name']}' ({c['fluid']}, {q * 60000.0:.3g} L/min, "
                         f"{channel_text(region['geometry'])}, flow solved): Re {correlation['re']:.0f}, "
                         f"speeds up to {top_speed:.3g} m/s; in {t_in:.2f} °C → out {t_out:.2f} °C, "
                         f"{entry['heat']:.4g} W taken up, pressure drop {dp / 1000.0:.3g} kPa "
                         f"(pipe correlation {correlation['dp'] / 1000.0:.3g} kPa)")
            limit = FLUIDS[c["fluid"]]["limit_c"]
        near = np.zeros(shape, bool)                  # solid cells next to the liquid
        for a in range(3):
            near[_slab(a, 0, -1)] |= mask[_slab(a, 1, None)]
            near[_slab(a, 1, None)] |= mask[_slab(a, 0, -1)]
        near &= solid
        if near.any() and float(T[near].max()) > limit:
            messages.append(f"⚠ '{c['name']}': wetted faces reach {float(T[near].max()):.1f} °C; "
                            f"{c['fluid']} {FLUIDS[c['fluid']]['limit_text']}.")
    for entry in entries:
        limit = (FLUIDS.get(entry["fluid"]) or {}).get("limit_c")
        if entry.get("wall_max") is not None and limit is not None and entry["wall_max"] > limit:
            messages.append(f"⚠ '{entry['name']}': wetted faces reach {entry['wall_max']:.1f} °C; "
                            f"{entry['fluid']} {FLUIDS[entry['fluid']]['limit_text']}.")
    return entries, lines


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
    coolants = [c for c in conditions if c["kind"] == "coolant" and c["targets"]]
    baths = [c for c in conditions if c["kind"] == "bath" and c["targets"]]
    liquid_sink = bool(coolants) or any(c["bath_mode"] == "fixed" for c in baths)
    has_fixed = any(c["kind"] == "fixed" and c["targets"] for c in conditions) or \
        any("wall_fixed" in pair for pair in side_kind)
    power = [profile_steady_power(body["profile"], body["power"]) for body in bodies]
    if not any(p > 0 for p in power) and not has_fixed and \
            not any(c["kind"] == "flux" and c["targets"] for c in conditions):
        raise ThermalError("No heat source: give a body a power (W), add a heat source, or put a "
                           "heat flux or fixed temperature on faces.")
    closed_box = not any(kind in ("open", "inlet", "wall_fixed") for pair in side_kind for kind in pair)
    if closed_box and not settings["radiation"] and not liquid_sink and \
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
    # liquids: a bath that fills its container is a well-mixed body (correlations) or a liquid
    # region whose flow is solved; a coolant channel with its flow solved is a liquid region
    # with an inlet (the pump) and an outlet
    liquid_region = np.zeros(shape, np.int16)        # 0 = air / solid, r = liquid region r
    regions = []                                     # the liquid regions (flow solved)
    bath_bodies = []                                 # well-mixed bath bodies
    for c in baths:
        if c["bath_mode"] != "container":
            continue
        step(20, f"Filling '{c['name']}'...")
        tris, normals = _condition_triangles(c, bodies)
        fill = _bath_fill(c, tris)
        open_cells = (owner < 0) & (liquid_region == 0)
        seeds = _liquid_seed_cells(tris, normals, faces_mm, open_cells, 0.5 * h_mm)
        region, leaks = _bath_region(open_cells, faces_mm, seeds, fill)
        if leaks:
            raise ThermalError(f"The liquid of '{c['name']}' runs out of its container below the "
                               f"fill level (Z = {fill:.4g} mm): close the container, lower the "
                               "fill level, or choose 'Large volume'.")
        if not region.any():
            raise ThermalError(f"'{c['name']}' holds no liquid: the fill level (Z = {fill:.4g} mm) "
                               "is below the picked faces, or the space is thinner than one cell.")
        cells = np.argwhere(region)
        low = np.array([faces_mm[a][cells[:, a].min()] for a in range(3)])
        high = np.array([faces_mm[a][cells[:, a].max() + 1] for a in range(3)])
        if c["liquid_model"] == "flow":
            liquid_region[region] = len(regions) + 1
            regions.append({"kind": "bath", "condition": c, "fluid": c["fluid"], "fill_z": fill,
                            "low": low, "high": high})
        else:
            index = len(bodies)
            owner[region] = index
            bath = _bath_body(c, len(bath_bodies), low, high)
            bath["fill_z"] = fill
            bodies.append(bath)
            power.append(0.0)
            bath_bodies.append(index)
    for c in coolants:
        if c["liquid_model"] != "flow":
            continue
        step(20, f"Analysing the channel '{c['name']}'...")
        geometry = channel_geometry(_condition_triangles(c, bodies)[0], c["reverse"])
        cells, inlet, outlet = _channel_cells(geometry, faces_mm, (owner < 0) & (liquid_region == 0),
                                              h_mm)
        across = geometry["dh_mm"] / h_mm
        if across < 3.0 or not inlet.any() or not outlet.any() or (inlet & outlet).any():
            messages.append(f"'{c['name']}': the channel is only {across:.1f} cells wide, too few to "
                            "solve the liquid flow, so it uses the correlations (cells of "
                            f"{geometry['dh_mm'] / 4.0:.2g} mm or less at the parts solve it).")
            c["liquid_model"] = "simple"
            continue
        liquid_region[cells] = len(regions) + 1
        points = np.asarray(geometry["points"], float)
        regions.append({"kind": "coolant", "condition": c, "fluid": c["fluid"],
                        "geometry": geometry, "inlet": inlet, "outlet": outlet,
                        "low": points.min(0), "high": points.max(0)})
    fluid = owner < 0
    solid = ~fluid
    liquid = liquid_region > 0
    air_cells = fluid & ~liquid
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
    # fluid properties per cell: air everywhere, liquids in their regions (density constant at a
    # reference value except in the buoyancy term; cp, viscosity and conductivity follow the
    # temperature); a face takes the mean of its two cells
    start_t = ambient if (not transient or settings["initial_ambient"]) else settings["initial_c"]
    region_cells = [liquid_region == r + 1 for r in range(len(regions))]
    for r, region in enumerate(regions):
        c = region["condition"]
        if region["kind"] == "coolant":
            region["t_ref"] = float(c["inlet_c"])
        else:
            region["t_ref"] = float(c["bath_c"]) if transient and not c.get("bath_start_auto", True) \
                else float(start_t)
        region["rho_ref"] = float(fluid_props(region["fluid"], region["t_ref"])["rho"])
    if regions:
        rho_cell, cp_cell, mu_cell = np.full(shape, rho), np.full(shape, cp), np.full(shape, mu_air)
        for r, region in enumerate(regions):
            rho_cell[region_cells[r]] = region["rho_ref"]
    else:
        rho_cell, cp_cell, mu_cell = rho, cp, mu_air

    def to_faces(values):
        """Cell values → c-face values (mean of the two cells, the cell's own at the sides)."""
        if np.ndim(values) == 0:
            return [values] * 3
        out = []
        for c in range(3):
            pad = [(0, 0)] * 3
            pad[c] = (1, 1)
            padded = np.pad(values, pad, mode="edge")
            out.append(0.5 * (padded[sl(c, 0, -1)] + padded[sl(c, 1, None)]))
        return out

    rho_face = to_faces(rho_cell)
    rhocp_face = to_faces(rho_cell * cp_cell)
    mu_face = to_faces(mu_cell)
    # fluid region of every cell (-1 solid, 0 air, r liquid) and of every face (-1 when its two
    # cells differ); faces between air and a liquid are closed to the flow (slip, no coupling)
    cell_region = np.where(fluid, liquid_region, -1).astype(np.int16)
    region_face = []
    for c in range(3):
        values = np.full(tuple(shape[a] + (1 if a == c else 0) for a in range(3)), -1, np.int16)
        lo_r, hi_r = cell_region[sl(c, 0, -1)], cell_region[sl(c, 1, None)]
        values[sl(c, 1, -1)] = np.where(lo_r == hi_r, lo_r, -1)
        values[sl(c, 0, 1)] = cell_region[sl(c, 0, 1)]
        values[sl(c, -1, None)] = cell_region[sl(c, -1, None)]
        region_face.append(values)

    def update_liquid_properties():
        """cp, viscosity and conductivity of the liquid cells at their temperatures."""
        nonlocal rhocp_face, mu_face
        for r, region in enumerate(regions):
            cells = region_cells[r]
            props = fluid_props(region["fluid"], T[cells])
            cp_cell[cells] = props["cp"]
            mu_cell[cells] = props["mu"]
            for axis in range(3):
                k_cell[axis][cells] = props["k"]
            if region["kind"] == "bath":             # buoyancy relative to the mean liquid
                region["rho_ref"] = float(fluid_density(region["fluid"], float(T[cells].mean())))
        rhocp_face = to_faces(rho_cell * cp_cell)
        mu_face = to_faces(mu_cell)
    # printed circuit boards: conductivity per cell from the layer stack, copper and vias
    faces_D = [np.diff(f) for f in faces_mm]
    for index, body in enumerate(bodies):
        pcb = body["job"].get("pcb")
        if pcb is None or not body["cells"]:
            continue
        cells = np.argwhere(owner == index)
        centre = np.stack([centres_mm[a][cells[:, a]] for a in range(3)], axis=1)
        local = pcb_local(body["job"]["pcb_frame"], centre)
        layer, layers = _column_layers(cells[:, 0], cells[:, 1], cells[:, 2],
                                       pcb_flipped(body["job"]["pcb_frame"]))
        k = pcb_conductivity(pcb, local[:, 0], local[:, 1], faces_D[0][cells[:, 0]],
                             faces_D[1][cells[:, 1]], layer, layers, faces_D[2][cells[:, 2]])
        for axis in range(3):
            k_cell[axis][tuple(cells.T)] = k[:, axis]
    # component junctions (two-resistor model): an unknown each, joined to the cells of the
    # package top by 1/θ top and to those of its bottom (on the board) by 1/θ board
    junctions = []
    for index, body in enumerate(bodies):
        junction = body["job"].get("junction")
        if junction is None or not body["cells"]:
            continue
        cells = np.argwhere(owner == index)
        upright = junction.get("side", "top") != "bottom"
        top_k = cells[:, 2].max() if upright else cells[:, 2].min()
        board_k = cells[:, 2].min() if upright else cells[:, 2].max()
        parts_of = []
        for layer_k, theta in ((top_k, junction["theta_top"]), (board_k, junction["theta_board"])):
            chosen = cells[cells[:, 2] == layer_k]
            area = faces_D[0][chosen[:, 0]] * faces_D[1][chosen[:, 1]]
            parts_of.append((np.ravel_multi_index(tuple(chosen.T), shape),
                             area / area.sum() / max(float(theta), 1e-6)))
        flat = np.concatenate([p[0] for p in parts_of])
        conductance = np.concatenate([p[1] for p in parts_of])
        junctions.append({"index": index, "name": body["name"], "junction": junction,
                          "cells": flat, "G": conductance, "top": parts_of[0], "board": parts_of[1],
                          "power": 0.0, "t_j": float(ambient)})
    junction_of = {j["index"]: j for j in junctions}
    # heat sources in W per cell; a source with a time profile or a thermostat is kept as its
    # distribution of 1 W and scaled in every time step (or by its thermostat); a package's
    # power goes in at its junction
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
        if index in junction_of:
            cells, weights = cells[:0], weights[:0]
        if body["profile"] is None and body["thermostat"] is None:
            constant_source[cells] += body["power"] * weights
            constant_power += body["power"]
            if index in junction_of:
                junction_of[index]["power"] = junction_of[index]["constant"] = body["power"]
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
    labels, _count = ndi.label(air_cells)
    offset = int(labels.max())
    for r in range(len(regions)):
        piece, pieces = ndi.label(region_cells[r])
        labels[piece > 0] = piece[piece > 0] + offset
        offset += pieces
    border = set()
    for axis in range(3):
        for side in (0, 1):
            if side_kind[axis][side] in ("open", "inlet"):
                border |= set(np.unique(np.take(labels, 0 if side == 0 else shape[axis] - 1,
                                                axis=axis)).tolist())
    border.discard(0)
    outer_air = np.isin(labels, list(border)) if border else np.zeros(shape, bool)
    region_labels = labels
    cavities = int((air_cells & ~outer_air).sum())
    if cavities:
        messages.append(f"Closed air spaces ({cavities:,} cells) have their own internal air "
                        "flow; no air is exchanged with the outside.".replace(",", " "))

    # 3. velocity unknowns: active faces, fixed values --------------------------------------
    active, fixed = [], []
    for c in range(3):
        Sc = list(shape); Sc[c] += 1
        act = np.zeros(Sc, bool)
        val = np.zeros(Sc)
        act[sl(c, 1, -1)] = fluid[sl(c, 0, -1)] & fluid[sl(c, 1, None)] & \
            (cell_region[sl(c, 0, -1)] == cell_region[sl(c, 1, None)])
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
                    if not faces or (condition["kind"] == "bath" and
                                     condition["bath_mode"] == "container"):
                        continue
                    hit = np.isin(mesh_face, faces)
                    if hit.any():
                        sel = tuple(arr[hit] for arr in fi)
                        face_condition[c][sel] = index
        for index, condition in enumerate(conditions):
            if condition["kind"] == "bath" and condition["bath_mode"] == "container":
                continue
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
            if kind == "source" and index in junction_of:
                junction_of[index]["power"] = value
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

    # liquid faces (coolant channels with correlations, baths of a fixed temperature): the solid
    # cell at each face exchanges heat with the liquid over half a cell and the film 1/h; the
    # air side of the face is closed
    face_liquids = []
    closed_faces = [np.zeros(active[c].shape, bool) for c in range(3)]
    for index, condition in enumerate(conditions):
        simple_coolant = condition["kind"] == "coolant" and condition["liquid_model"] == "simple"
        fixed_bath = condition["kind"] == "bath" and condition["bath_mode"] == "fixed"
        if not (simple_coolant or fixed_bath):
            continue
        cells, faces, areas, centres, halves, axes, orientation = [], [], [], [], [], [], []
        for c in range(3):
            mask = face_condition[c] == index
            if not mask.any():
                continue
            closed_faces[c] |= mask
            idx = np.nonzero(mask)
            below = list(idx)
            below[c] = np.maximum(idx[c] - 1, 0)
            solid_below = solid[tuple(below)] & (idx[c] > 0)
            cell = list(idx)
            cell[c] = np.where(solid_below, idx[c] - 1, idx[c])
            cell = tuple(cell)
            cells.append(np.ravel_multi_index(cell, shape))
            faces.append((c, idx))
            areas.append(np.broadcast_to(face_area[c], mask.shape)[mask])
            centre = np.stack([centres_mm[a][cell[a]] for a in range(3)], axis=1)
            centre[:, c] = faces_mm[c][idx[c]]
            centres.append(centre)
            halves.append(D[c][cell[c]] / 2.0)
            axes.append(np.full(len(idx[0]), c))
            orientation.append(np.zeros(len(idx[0]), int) if c != 2 else
                               np.where(solid_below, 1, 2))     # the solid face looks up / down
        if not cells:
            continue
        record = {"index": index, "condition": condition, "cells": np.concatenate(cells),
                  "faces": faces, "area": np.concatenate(areas), "centre": np.concatenate(centres),
                  "half": np.concatenate(halves), "axis": np.concatenate(axes),
                  "orientation": np.concatenate(orientation)}
        record["owner"] = owner_flat[record["cells"]].astype(np.int64)
        tris = _condition_triangles(condition, bodies)[0]
        if simple_coolant:
            geometry = channel_geometry(tris, condition["reverse"])
            s = geometry["s"][cKDTree(geometry["points"]).query(record["centre"])[1]]
            n_seg = int(np.clip(round(geometry["length_mm"] / h_mm), 8, 400))
            seg = np.clip((s / max(geometry["length_mm"], 1e-9) * n_seg).astype(int), 0, n_seg - 1)
            record["area"] = record["area"] * min(1.0, geometry["wall_area_mm2"] * 1e-6 /
                                                  max(float(record["area"].sum()), 1e-30))
            record.update(geometry=geometry, seg=seg, n_seg=n_seg,
                          area_seg=np.bincount(seg, record["area"], n_seg), t_face=None)
            record["h_faces"] = np.full(len(record["cells"]), 1000.0)
            record["t_liquid"] = np.full(len(record["cells"]), float(condition["inlet_c"]))
        else:
            wall = float(_mesh_area(tris.reshape(-1, 3), np.arange(3 * len(tris)).reshape(-1, 3)))
            record["area"] = record["area"] * min(3.0, max(0.3, wall * 1e-6 / record["area"].sum()))
            record["h_faces"] = np.full(len(record["cells"]), BATH_FILM_GUESS)
            record["t_liquid"] = np.full(len(record["cells"]), float(condition["bath_c"]))
        record["G"] = np.zeros(len(record["cells"]))
        face_liquids.append(record)
    face_channels = _ordered_channels([r for r in face_liquids if r["condition"]["kind"] == "coolant"])
    # flow-solved coolant: the pump feeds the first cell layer, the last one lets the liquid out
    mass_source = np.zeros(shape)
    for r, region in enumerate(regions):
        if region["kind"] != "coolant":
            continue
        c = region["condition"]
        for key in ("inlet", "outlet"):
            cells = np.flatnonzero(region[key].ravel())
            region[key + "_cells"] = cells
            region[key + "_weights"] = volume_flat[cells] / volume_flat[cells].sum()
        region["flow_m3s"] = c["flow_lpm"] / 60000.0 if c["flow_mode"] == "fixed" else \
            0.5 * float(pump_curve(c)[0][-1])
        region["dp"] = 0.0
    closed_caps = [np.zeros(active[c].shape, bool) for c in range(3)]   # channel ends: no heat
    for c in range(3):
        lo_r, hi_r = cell_region[sl(c, 0, -1)], cell_region[sl(c, 1, None)]
        ends = (lo_r >= 0) & (hi_r >= 0) & (lo_r != hi_r)
        for r, region in enumerate(regions):
            if region["kind"] == "coolant":
                closed_caps[c][sl(c, 1, -1)] |= ends & ((lo_r == r + 1) | (hi_r == r + 1))

    def liquid_inlet_t(name, own):
        """Inlet temperature of a coolant: its own, or the outlet of the channel feeding it."""
        for record in face_channels:
            if name and record["condition"]["name"] == name and "t_out" in record:
                return float(record["t_out"])
        for region in regions:
            if name and region["kind"] == "coolant" and region["condition"]["name"] == name and \
                    "t_out" in region:
                return float(region["t_out"])
        return float(own)

    def set_mass_sources():
        mass_source[...] = 0.0
        flat = mass_source.reshape(-1)
        for region in regions:
            if region["kind"] != "coolant":
                continue
            region["mdot"] = region["flow_m3s"] * region["rho_ref"]
            flat[region["inlet_cells"]] += region["mdot"] * region["inlet_weights"]
            flat[region["outlet_cells"]] -= region["mdot"] * region["outlet_weights"]

    set_mass_sources()
    # well-mixed bath bodies: the film on the wetted faces is a resistance 1/h between the bath
    # cells and the solid cells they touch (natural convection in the liquid)
    film_grid = [np.zeros(active[c].shape) for c in range(3)] if bath_bodies else None
    film_groups = {}
    for c in range(3) if bath_bodies else ():
        lo_o, hi_o = owner_index[sl(c, 0, -1)], owner_index[sl(c, 1, None)]
        for b in bath_bodies:
            for bath_low in (True, False):
                bath_side, other = (lo_o, hi_o) if bath_low else (hi_o, lo_o)
                touching = (bath_side == b) & (other < len(bodies)) & ~np.isin(other, bath_bodies)
                if not touching.any():
                    continue
                idx = np.nonzero(touching)
                face = list(idx)
                face[c] = idx[c] + 1
                solid_cell = list(idx)
                if bath_low:
                    solid_cell[c] = idx[c] + 1
                orientation = 0 if c != 2 else (2 if bath_low else 1)
                for s_body in np.unique(other[touching]):
                    chosen = other[touching] == s_body
                    film_groups.setdefault((b, int(s_body), orientation), []).append(
                        (c, tuple(f[chosen] for f in face), tuple(s[chosen] for s in solid_cell)))
    film_h = {key: BATH_FILM_GUESS for key in film_groups}

    def update_bath_films():
        for key, parts in film_groups.items():
            b, s_body, orientation = key
            t_bath = float(T[owner == b].mean())
            t_solid = float(np.mean(np.concatenate([T[cell] for _c, _f, cell in parts])))
            if orientation == 0:
                length = min(bodies[s_body]["extent"][2], bodies[b]["extent"][2])
            else:
                e_s, e_b = bodies[s_body]["extent"], bodies[b]["extent"]
                length = min(e_s[0] * e_s[1] / (2.0 * (e_s[0] + e_s[1])),
                             e_b[0] * e_b[1] / (2.0 * (e_b[0] + e_b[1])))
            new = float(_liquid_natural_h(bodies[b]["bath"]["fluid"], t_solid - t_bath,
                                          max(length, 2.0) / 1000.0, np.array(orientation),
                                          0.5 * (t_solid + t_bath)))
            film_h[key] = 0.5 * film_h[key] + 0.5 * new
            for c, face, _cell in parts:
                film_grid[c][face] = 1.0 / max(film_h[key], 1e-6)

    if film_groups:
        for key, parts in film_groups.items():
            for c, face, _cell in parts:
                film_grid[c][face] = 1.0 / BATH_FILM_GUESS
    # open liquid surfaces: evaporation (water and its mixtures) as a heat sink of the top cells
    evaporating = []          # (flat cells, area m², share of water evaporation, length m)
    for r, region in enumerate(regions):
        if region["kind"] == "bath" and region["condition"].get("evaporation", True):
            top = np.zeros(shape, bool)
            top[:, :, :-1] = region_cells[r][:, :, :-1] & air_cells[:, :, 1:]
            evaporating.append((top, FLUIDS[region["fluid"]]["evaporation"], region, ("region", r)))
    for b in bath_bodies:
        if bodies[b]["bath"].get("evaporation", True):
            top = np.zeros(shape, bool)
            top[:, :, :-1] = (owner[:, :, :-1] == b) & air_cells[:, :, 1:]
            evaporating.append((top, FLUIDS[bodies[b]["bath"]["fluid"]]["evaporation"], bodies[b],
                                ("body", b)))
    evaporation = []
    for top, share, item, key in evaporating:
        cells = np.flatnonzero(top.ravel())
        if not len(cells) or share <= 0:
            continue
        area = np.broadcast_to(along(D[0], 0) * along(D[1], 1), shape).ravel()[cells]
        extent = np.asarray(item["high"], float) - np.asarray(item["low"], float)
        length = max(extent[0] * extent[1] / (2.0 * (extent[0] + extent[1]) + 1e-9), 2.0) / 1000.0
        evaporation.append({"cells": cells, "area": area, "share": share, "length": length,
                            "h": np.zeros(len(cells)), "owner": key, "ambient": ambient})

    def update_face_liquids():
        """Liquid temperatures and film coefficients of the coolant / fixed-bath faces."""
        flat_T = T.ravel()
        for record in face_channels + [r for r in face_liquids if r["condition"]["kind"] == "bath"]:
            c = record["condition"]
            t_cell = flat_T[record["cells"]]
            k_solid = np.choose(record["axis"], [k_cell[a].ravel()[record["cells"]] for a in range(3)])
            r_half = record["half"] / np.maximum(k_solid, 1e-9)
            film = 1.0 / np.maximum(record["h_faces"], 1e-9)
            wall = record["t_liquid"] + (t_cell - record["t_liquid"]) * film / (r_half + film)
            if c["kind"] == "coolant":            # flow and h; the liquid temperatures along
                c_in = dict(c)                    # the channel are unknowns of the energy solve
                c_in["inlet_c"] = liquid_inlet_t(c.get("inlet_from") or "", c["inlet_c"])
                record["condition"] = c_in
                record["upstream"] = None
                _channel_flow(record, wall, record["area"])
                record["condition"] = c
                record["h_faces"][:] = record["h"]
            else:
                values = np.zeros(len(t_cell))
                for o in (0, 1, 2):
                    for b in np.unique(record["owner"]):
                        sel = (record["orientation"] == o) & (record["owner"] == b)
                        if not sel.any():
                            continue
                        extent = bodies[b]["extent"]
                        length = extent[2] if o == 0 else \
                            extent[0] * extent[1] / (2.0 * (extent[0] + extent[1]))
                        t_mean = float((wall[sel] * record["area"][sel]).sum() / record["area"][sel].sum())
                        values[sel] = _liquid_natural_h(c["fluid"], t_mean - c["bath_c"],
                                                        max(length, 2.0) / 1000.0, np.array(o),
                                                        0.5 * (t_mean + c["bath_c"]))
                record["h_faces"] = 0.5 * record["h_faces"] + 0.5 * values
                record["t_liquid"][:] = c["bath_c"]

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
    # liquid regions: a channel is turbulent above Re 2300 (unless the flow is set laminar), a
    # bath stays laminar
    turbulent_cells = air_cells.copy() if turbulent else np.zeros(shape, bool)
    for r, region in enumerate(regions):
        if region["kind"] == "coolant":
            c = region["condition"]
            q = c["flow_lpm"] / 60000.0 if c["flow_mode"] == "fixed" else 0.5 * pump_curve(c)[0][-1]
            region["re_estimate"] = pipe_flow(region["fluid"], q, region["geometry"],
                                              region["t_ref"])["re"]
            if settings["turbulence"] == "mixing" or (settings["turbulence"] == "auto" and
                                                      region["re_estimate"] > 2300.0):
                turbulent_cells |= region_cells[r]
    any_turbulent = bool(turbulent_cells.any())
    mu_t = np.zeros(shape)
    if any_turbulent:
        spacing = [float(np.median(d)) for d in D]
        if regions:
            wall_distance = np.maximum(ndi.distance_transform_edt(air_cells, sampling=spacing),
                                       ndi.distance_transform_edt(liquid, sampling=spacing))
        else:
            wall_distance = ndi.distance_transform_edt(fluid, sampling=spacing)
        extents = [np.ptp(body["V"], axis=0).max() for body in bodies if len(body["V"])]
        reference = max(float(max(extents or [10.0])) / 1000.0, 0.01)
        mixing = np.minimum(MIXING_KAPPA * wall_distance, 0.09 * reference)
        for r, region in enumerate(regions):
            if region["kind"] == "coolant":
                cells = region_cells[r]
                mixing[cells] = np.minimum(MIXING_KAPPA * wall_distance[cells],
                                           0.07 * region["geometry"]["dh_mm"] / 1000.0)

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
        new = rho_cell * mixing ** 2 * np.sqrt(strain)
        new[~turbulent_cells] = 0.0
        mu_t[:] = 0.5 * mu_t + 0.5 * np.minimum(new, 1000.0 * mu_cell)

    # momentum ---------------------------------------------------------------------------------
    def momentum(c):
        """Assemble and solve the c-momentum equation; returns d (pressure-correction factors)."""
        Sc = active[c].shape
        U = velocity[c]
        mu = mu_cell + mu_t
        aP = np.zeros(Sc)
        b = np.zeros(Sc)
        nb = {}
        # neighbours along c (through the cell centres)
        area_c = face_area[c]
        Fc = rho_cell * 0.5 * (U[sl(c, 0, -1)] + U[sl(c, 1, None)]) * area_c
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
            rho_d = rho_face[c]
            if np.ndim(rho_d):
                rho_d = 0.5 * (rho_d[sl(d, 0, -1)] + rho_d[sl(d, 1, None)])
            Fd = rho_d * vd * area
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
            wall_coefficient = mu_face[c] * area_full / half
            if regions:
                # a neighbour face in another fluid region (air above a liquid): no coupling
                # and no wall, i.e. free slip
                rf = region_face[c]
                cut_plus = np.zeros(Sc, bool)
                cut_minus = np.zeros(Sc, bool)
                lo_f, hi_f = rf[sl(d, 0, -1)], rf[sl(d, 1, None)]
                differ = (lo_f >= 0) & (hi_f >= 0) & (lo_f != hi_f)
                cut_plus[sl(d, 0, -1)] = differ
                cut_minus[sl(d, 1, None)] = differ
                wall_plus &= ~cut_plus
                wall_minus &= ~cut_minus
                north[cut_plus] = 0.0
                south[cut_minus] = 0.0
            aP += np.where(wall_plus, wall_coefficient, 0.0) + np.where(wall_minus, wall_coefficient, 0.0)
            north[wall_plus] = 0.0
            south[wall_minus] = 0.0
            # domain sides along d: no-slip at walls and inlets, free slip at symmetry / open sides
            for side, index in ((0, 0), (1, shape[d] - 1)):
                if side_kind[d][side] in ("wall", "wall_fixed", "inlet"):
                    edge = sl(d, index, index + 1)
                    aP[edge] += (np.broadcast_to(mu_face[c], Sc) * area_full / half)[edge]
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
            buoyancy = rho * GRAVITY * beta * (T_face - ambient)
            for r, region in enumerate(regions):      # liquids: density from the table
                mine = region_face[2] == r + 1
                buoyancy[mine] = GRAVITY * (region["rho_ref"] -
                                            fluid_density(region["fluid"], T_face[mine]))
            b += buoyancy * cv
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
        inertia = rho_face[c] * area_face * velocity_scale[0]
        aP = aP + inertia
        b += inertia * U
        if unsteady["dt"] is not None:              # time step: ρ·V/Δt · (U − U_old)
            mass = rho_face[c] * np.broadcast_to(along(cv_length[c], c) * face_area[c], Sc) / unsteady["dt"]
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
            flow = rho_face[c] * velocity[c] * area_face           # kg/s through every c-face
            imbalance += flow[sl(c, 1, None)] - flow[sl(c, 0, -1)]
            coefficient = rho_face[c] * d[c] * area_face
            aP += coefficient[sl(c, 1, None)] + coefficient[sl(c, 0, -1)]
            east = coefficient[sl(c, 1, None)].copy()
            west = coefficient[sl(c, 0, -1)].copy()
            east[sl(c, -1, None)] = 0.0                       # the open side: ghost p' = 0
            west[sl(c, 0, 1)] = 0.0
            nb[(c, 1)], nb[(c, -1)] = east, west
        if regions:
            imbalance -= mass_source                 # pump inlets and outlets of the liquids
        rhs = -imbalance.ravel()
        diagonal = aP.ravel().copy()
        empty = diagonal <= 0
        diagonal[empty] = 1.0
        rhs[empty] = 0.0
        big = 1e6 * max(float(diagonal.max()), 1e-30)     # once: many pins must not compound
        for cell in pin:
            diagonal[cell] += big
        matrix = _seven_point(sp, shape, diagonal, nb)
        mass_scale = max(float((np.abs(imbalance) / rho_cell).sum()), 1e-300)   # m³/s
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
    wall_like = solid | liquid          # (a liquid surface is a wall for the air above it)
    for c in range(3):
        lo_wall, hi_wall = wall_like[sl(c, 0, -1)], wall_like[sl(c, 1, None)]
        lo_air, hi_air = air_cells[sl(c, 0, -1)], air_cells[sl(c, 1, None)]
        kind = np.zeros(lo_wall.shape, np.int8)
        kind[lo_wall & hi_air] = 1
        kind[lo_air & hi_wall] = 2
        solid_face.append(kind)
    # the liquid side of a wetted solid face: a wall function in baths and turbulent channels
    liquid_face = []
    wall_function_liquid = np.zeros(shape, bool)
    for r, region in enumerate(regions):
        if region["kind"] == "bath" or bool((turbulent_cells & region_cells[r]).any()):
            wall_function_liquid |= region_cells[r]
    for c in range(3):
        lo_s, hi_s = solid[sl(c, 0, -1)], solid[sl(c, 1, None)]
        lo_l, hi_l = wall_function_liquid[sl(c, 0, -1)], wall_function_liquid[sl(c, 1, None)]
        kind = np.zeros(lo_s.shape, np.int8)
        kind[lo_s & hi_l] = 1
        kind[lo_l & hi_s] = 2
        liquid_face.append(kind)
    run_length = np.vstack([run_length] + [
        [[max((float(np.sum(region["high"] - region["low"])) - float(region["high"][a] -
                                                                    region["low"][a])) / 2.0,
              2.0) / 1000.0 for a in range(3)]] for region in regions])
    wall_index = owner_index + liquid_region.astype(np.int64)     # liquid r → row of region r

    def conductances():
        """(G per c-face array, G_half per cell for each side) with turbulent conductivity."""
        nonlocal k_eff
        k_turbulent = cp_cell * mu_t / 0.9
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
            if kind.any() or liquid_face[c].any():
                parallel = np.sqrt(sum(centre[a] ** 2 for a in range(3) if a != c))
            if kind.any():
                speed_air = np.where(kind == 1, parallel[sl(c, 1, None)], parallel[sl(c, 0, -1)])
                body_side = np.where(kind == 1, wall_index[sl(c, 0, -1)], wall_index[sl(c, 1, None)])
                film = 1.0 / np.maximum(_forced_h(speed_air, run_length[body_side, c]), 1e-12)
                lo_half = np.where(kind == 2, np.minimum(lo_half, film), lo_half)
                hi_half = np.where(kind == 1, np.minimum(hi_half, film), hi_half)
            kind = liquid_face[c]                    # 1: solid below / liquid above
            if kind.any():
                pick = lambda values: np.where(kind == 1, values[sl(c, 1, None)], values[sl(c, 0, -1)])
                nu_l = pick(mu_cell / rho_cell)
                pr_l = pick(mu_cell * cp_cell / np.maximum(k_cell[c], 1e-9))
                k_l = pick(k_cell[c])
                body_side = np.where(kind == 1, owner_index[sl(c, 0, -1)], owner_index[sl(c, 1, None)])
                film = 1.0 / np.maximum(_forced_h(pick(parallel), run_length[body_side, c], nu_l,
                                                  pr_l, k_l), 1e-12)
                lo_half = np.where(kind == 2, np.minimum(lo_half, film), lo_half)
                hi_half = np.where(kind == 1, np.minimum(hi_half, film), hi_half)
            extra = contact_resistance_grid[c][sl(c, 1, -1)]
            if film_grid is not None:
                extra = extra + film_grid[c][sl(c, 1, -1)]
            g[sl(c, 1, -1)] = area[sl(c, 1, -1)] / (lo_half + hi_half + extra)
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
            insulated |= closed_faces[c]
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
    for r, region in enumerate(regions):
        capacity[region_cells[r]] = (rho_cell * cp_cell * np.broadcast_to(volume, shape))[region_cells[r]]

    def energy(refresh):
        G = conductances()
        aP = np.zeros(shape)
        b = source.copy()
        nb = {}
        net_out = np.zeros(shape) if regions else None       # net convective outflow, W/K
        fixed_heat = []                 # (cells, G_half, T_fixed, condition index)
        for c in range(3):
            Sc = active[c].shape
            area = np.broadcast_to(face_area[c], Sc)
            flow = rhocp_face[c] * velocity[c] * area                # W/K, + along +c
            if regions:
                net_out += flow[sl(c, 1, None)] - flow[sl(c, 0, -1)]
            g = G[c]
            insulated = np.zeros(Sc, bool)
            for index, condition in enumerate(conditions):
                if condition["kind"] == "insulated":
                    insulated |= face_condition[c] == index
            fixed_faces = np.isfinite(face_fixed_value[c])
            g = np.where(insulated | fixed_faces | closed_faces[c] | closed_caps[c], 0.0, g)
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
        flat_aP, flat_b = aP.reshape(-1), b.reshape(-1)
        # coolant (correlations) and fixed-temperature bath faces: solid cell ↔ liquid
        for record in face_liquids:
            k_solid = np.choose(record["axis"], [k_cell[a].ravel()[record["cells"]] for a in range(3)])
            record["G"] = record["area"] / (record["half"] / np.maximum(k_solid, 1e-9) +
                                            1.0 / np.maximum(record["h_faces"], 1e-9))
            np.add.at(flat_aP, record["cells"], record["G"])
            if record["condition"]["kind"] != "coolant":     # channels: coupled below
                np.add.at(flat_b, record["cells"], record["G"] * record["t_liquid"])
        # flow-solved coolant: the pump brings liquid at the inlet temperature, the outlet takes
        # it out at the cell temperature
        for region in regions:
            if region["kind"] != "coolant":
                continue
            c = region["condition"]
            t_in = liquid_inlet_t(c.get("inlet_from") or "", c["inlet_c"])
            cp_in = float(fluid_props(region["fluid"], t_in)["cp"])
            region["t_in"] = t_in
            flat_b[region["inlet_cells"]] += region["mdot"] * region["inlet_weights"] * cp_in * t_in
            flat_aP[region["inlet_cells"]] += region["mdot"] * region["inlet_weights"] * cp_in
        if regions:
            # liquid cells: a_P = Σ a_nb (+ the inflow of a pump inlet) instead of including the
            # net outflow — the same at convergence, and bounded while the flow and the mass
            # sources do not balance yet (the outlet's outflow is the net inflow it receives)
            flat_aP[liquid.ravel()] -= net_out.ravel()[liquid.ravel()]
        # evaporation from open liquid surfaces: q = h_e·(T − T_room) with h_e from the chord
        # of the steep evaporation curve, moved only part of the way each iteration (the energy
        # equation is solved partly per iteration; a full update makes the bath swing)
        flat_T = T.ravel()
        for item in evaporation:
            t_top = flat_T[item["cells"]]
            q0 = _evaporation_q(t_top, ambient, item["length"], item["share"])
            chord = q0 / np.maximum(t_top - ambient, 0.5)
            item["h"] = chord if not item.get("started") else 0.7 * item["h"] + 0.3 * chord
            item["started"] = True
            np.add.at(flat_aP, item["cells"], item["h"] * item["area"])
            np.add.at(flat_b, item["cells"], item["h"] * item["area"] * ambient)
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
        # a cell cut off from everything (no conductance, no flow) keeps its temperature
        isolated = aP_relaxed <= 1e-20
        if isolated.any():
            aP_relaxed = np.where(isolated, 1.0, aP_relaxed)
            b = np.where(isolated, T, b)
            for key in nb:
                nb[key] = np.where(isolated, 0.0, nb[key])
        aP_relaxed = np.maximum(aP_relaxed, 1e-30)
        # absolute tolerance in watts, tied to the heat that flows: a relative tolerance would
        # be dominated by the stiff metal cells, and their tiny temperature errors are large
        # heat errors
        matrix = _seven_point(sp, shape, aP_relaxed, nb)
        rhs = b.ravel()
        start = T.ravel()
        key = "energy"
        if face_channels or junctions:
            # coolant channels with correlations: the liquid temperature of every segment is an
            # unknown, ṁc·(T_i − T_i−1) = Σ G·(T_cell − T_i), coupled with the solid cells;
            # junctions: S·T_j − Σ G·T_cell = P, with G added to the cells' diagonals
            rows, cols, values = [], [], []
            extra_rhs, extra_start = [], []
            offset = N
            for j in junctions:
                rows += [j["cells"], np.full(len(j["cells"]), offset), j["cells"], [offset]]
                cols += [np.full(len(j["cells"]), offset), j["cells"], j["cells"], [offset]]
                values += [-j["G"], -j["G"], j["G"], [float(j["G"].sum())]]
                extra_rhs.append(np.array([float(j["power"])]))
                extra_start.append(np.array([float(j["t_j"])]))
                j["unknown"] = offset
                offset += 1
            for record in face_channels:
                n, seg, cells, conductance = record["n_seg"], record["seg"], record["cells"], record["G"]
                mdot_cp = max(record.get("mdot_cp", 0.0), 1e-9)
                unknown = offset + seg
                rows += [cells, unknown]
                cols += [unknown, cells]
                values += [-conductance, -conductance]
                diagonal = mdot_cp + np.bincount(seg, conductance, n) + 1e-12
                rows += [offset + np.arange(n), offset + np.arange(1, n)]
                cols += [offset + np.arange(n), offset + np.arange(n - 1)]
                values += [diagonal, np.full(n - 1, -mdot_cp)]
                first = np.zeros(n)
                first[0] = mdot_cp * record.get("t_in", record["condition"]["inlet_c"])
                extra_rhs.append(first)
                extra_start.append(record["t_face"] if record.get("t_face") is not None
                                   else np.full(n, float(record["condition"]["inlet_c"])))
                record["unknowns"] = (offset, offset + n)
                offset += n
            size = offset
            coupling = sp.csr_matrix((np.concatenate(values), (np.concatenate(rows),
                                                               np.concatenate(cols))),
                                     shape=(size, size))
            matrix = (sp.bmat([[matrix, None], [None, sp.csr_matrix((size - N, size - N))]])
                      + coupling).tocsr()
            rhs = np.concatenate([rhs] + extra_rhs)
            start = np.concatenate([start] + extra_start)
            key = "energy_channels"
            if coarse.get("channels_size") != size:
                coarse["channels_size"] = size
                coarse["channels_P"] = sp.block_diag((prolongation, sp.identity(size - N))).tocsr()
                coarse[key] = None
        # inner tolerance: 1 % of the starting residual (the outer iterations change the
        # coefficients anyway), never looser than needed for the final heat balance
        initial = float(np.linalg.norm(rhs - matrix @ start))
        tolerance = max(1e-6 * max(heat_scale[0], 1e-12) / math.sqrt(N), 0.01 * initial)
        if face_channels or junctions:
            P_ext = coarse["channels_P"]
            inverse = 1.0 / matrix.diagonal()
            if refresh or coarse.get(key) is None:
                coarse[key] = spla.splu((P_ext.T @ matrix @ P_ext).tocsc())
            lu = coarse[key]
            preconditioner = spla.LinearOperator(matrix.shape, dtype=float, matvec=lambda v: inverse * v + P_ext @ lu.solve(P_ext.T @ v))
        else:
            preconditioner = two_level(matrix, "energy", refresh)
        solution, _info = spla.bicgstab(matrix, rhs, x0=start, rtol=0.0, atol=tolerance,
                                        maxiter=1000, M=preconditioner)
        if not np.all(np.isfinite(solution)):
            raise ThermalError("The temperatures did not converge.")
        for j in junctions:
            j["t_j"] = float(solution[j["unknown"]])
        for record in face_channels:                  # the liquid along each channel
            lo, hi = record["unknowns"]
            t_seg = solution[lo:hi]
            record["t_face"] = record["t_after"] = t_seg
            record["t_out"] = float(t_seg[-1])
            record["t_liquid"] = t_seg[record["seg"]]
            filled = record["area_seg"] > 0
            if "wall" in record:
                record["mode"] = wall_mode_of(record["area_seg"][filled], record["wall"][filled],
                                              t_seg[filled])
        energy_parts["fixed"] = fixed_heat
        energy_parts["G"] = G
        return solution[:N].reshape(shape)

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
        flat_T = T.ravel()
        liquids = 0.0
        for record in face_liquids:
            heat = record["G"] * (flat_T[record["cells"]] - record["t_liquid"])
            record["heat"] = float(heat.sum())
            liquids += record["heat"]
            gross += float(np.abs(heat).sum())
        for region in regions:
            if region["kind"] == "coolant" and "t_in" in region:
                out_cells = region["outlet_cells"]
                cp_out = np.broadcast_to(cp_cell, shape).ravel()[out_cells]
                cp_in = float(fluid_props(region["fluid"], region["t_in"])["cp"])
                region["heat"] = float(region["mdot"] * ((region["outlet_weights"] * cp_out *
                                                         flat_T[out_cells]).sum() - cp_in * region["t_in"]))
                liquids += region["heat"]
                gross += abs(region["heat"])
        for item in evaporation:
            item["heat"] = float((_evaporation_q(flat_T[item["cells"]], ambient, item["length"],
                                                 item["share"]) * item["area"]).sum())
            liquids += item["heat"]
            gross += abs(item["heat"])
        heat_scale[0] = max(power_now[0], gross, 1e-12)
        return out_air, radiation, walls, fixed_out, per_condition, liquids

    def stored_heat():
        """W going into heat storage during the time step (0 in steady state)."""
        if unsteady["dt"] is None:
            return 0.0
        return float((capacity * (T - unsteady["T"])).sum()) / unsteady["dt"]

    # result surfaces (and the grid of the result)
    step(21, "Preparing the result surfaces...")
    grid = {"kind": "cells", "xc": centres_mm[0], "yc": centres_mm[1], "zc": centres_mm[2],
            "faces": faces_mm, "shape": shape, "solid": solid.ravel(), "h_mm": h_mm,
            "owner": owner.ravel().copy(),             # body per cell, -1 = air
            "liquid": liquid_region.ravel().copy() if regions else None}
    # result surfaces of the bodies: trilinear between solid cells
    total_area = sum(body["area"] for body in bodies)
    edge = max(h_mm, math.sqrt(2.0 * max(total_area, 1e-9) / DISPLAY_TRIANGLE_LIMIT))
    surface = {}
    for body in bodies:
        if body.get("synthetic"):
            continue
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
    for region in regions:
        if region["kind"] == "coolant":
            speeds.append(region["flow_m3s"] / max(region["geometry"]["area_mm2"] * 1e-6, 1e-12))
    length = max(float(np.max(plan["size"])) / 1000.0 * 0.3, 0.01)
    speeds.append(math.sqrt(GRAVITY * beta * 10.0 * length))
    mass_reference = max(speeds) * length ** 2
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
        if regions:
            update_liquid_properties()
        if any_turbulent and number % 3 == 1:
            update_turbulence()
        previous_velocity = [v.copy() for v in velocity]
        d = [momentum(c) for c in range(3)]
        # pumps on a curve: the flow follows the pressure difference from inlet to outlet
        moved = False
        for region in regions:
            if region["kind"] != "coolant" or region["condition"]["flow_mode"] != "pump":
                continue
            flows, pressures = pump_curve(region["condition"])
            flat_p = p.ravel()
            dp = float(region["inlet_weights"] @ flat_p[region["inlet_cells"]] -
                       region["outlet_weights"] @ flat_p[region["outlet_cells"]])
            target = float(np.interp(max(dp, 0.0), pressures[::-1], flows[::-1]))
            region["flow_m3s"] = 0.8 * region["flow_m3s"] + 0.2 * target
            region["dp"] = dp
            moved = True
        if moved:
            set_mass_sources()
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
        reference_flow = max(top_speed, 1e-3 * max(speeds)) * length ** 2
        mass = imbalance / max(min(mass_reference, reference_flow), 1e-300)
        change = max(float(np.abs(v - w).max()) for v, w in zip(velocity, previous_velocity)) / \
            max(top_speed, 1e-9)
        if face_liquids:
            update_face_liquids()
        if film_groups and number % 5 == 1:
            update_bath_films()
        T_new = energy(refresh)
        dT = float(np.abs(T_new - T).max())
        T = T_new
        return mass, change, dT

    def balance():
        out_air, radiation, walls, fixed_out, _per, liquids = heat_balance()
        return (out_air + radiation + walls + fixed_out + liquids + stored_heat() - power_now[0]) / \
            heat_scale[0]

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
        for b in bath_bodies:                   # a bath may start warmer or colder
            if not bodies[b]["bath"].get("bath_start_auto", True):
                T[owner == b] = float(bodies[b]["bath"]["bath_c"])
        for r, region in enumerate(regions):
            if region["kind"] == "bath":
                T[region_cells[r]] = region["t_ref"]
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

        junction_series = np.zeros((len(junctions), n_steps + 1))

        def record(step_index):
            flat = T.ravel()
            for n, j in enumerate(junctions):
                junction_series[n, step_index] = j["t_j"] if step_index else float(start)
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
        if junctions:
            series["junctions"] = junction_series
            series["junction_names"] = [j["name"] for j in junctions]
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
    out_air, q_radiation, q_walls, q_fixed, per_condition, q_liquid = heat_balance()
    out = out_air + q_radiation + q_walls + q_fixed + q_liquid
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
    air_T = T[air_cells] if air_cells.any() else T[fluid]
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
               (f", liquids {q_liquid:.4g} W" if abs(q_liquid) > 1e-9 else "") +
               (f", still stored {q_stored:.4g} W" if transient else "") +
               f" (difference {balance_error:+.2%})" + (" at the end" if transient else ""),
               air_text]
    liquid_results, liquid_lines = _airflow_liquid_results(
        T, owner, solid, bodies, bath_bodies, regions, region_cells, face_liquids, face_channels,
        evaporation, energy_parts.get("G"), p, velocity, face_area, volume, shape, transient,
        unsteady, capacity, messages)
    for entry in condition_results:
        if entry["kind"] in LIQUID_KINDS:
            continue
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
    summary += liquid_lines
    junction_results = []
    for n, j in enumerate(junctions):
        flat_T = T.ravel()
        t_top = float(flat_T[j["top"][0]] @ j["top"][1] / j["top"][1].sum())
        t_board = float(flat_T[j["board"][0]] @ j["board"][1] / j["board"][1].sum())
        entry = {"name": j["name"], "package": j["junction"].get("type", "Custom block"),
                 "t_j": j["t_j"], "t_top": t_top, "t_board": t_board, "power": float(j["power"]),
                 "tj_max": float(j["junction"].get("tj_max", 150.0)),
                 "theta_top": float(j["junction"]["theta_top"]),
                 "theta_board": float(j["junction"]["theta_board"])}
        if transient:
            entry["peak"] = float(junction_series[n].max())
        junction_results.append(entry)
    lines, warnings = junction_lines(junction_results, transient)
    summary += lines
    messages += warnings
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
    tabled = [i for i in sorted(set(k_tabled) | set(cp_tabled)) if not bodies[i].get("synthetic")]
    if tabled:
        summary.append("Temperature-dependent material data: " + ", ".join(
            f"{bodies[i]['name']} ({bodies[i]['material']})" for i in tabled))
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
            "thermostats": thermostat_results, "liquids": liquid_results, "q_liquid": q_liquid,
            "junctions": junction_results,
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
    "coolant_flow": ("Flow rate of a coolant channel", "L/min", "coolant"),
    "coolant_inlet": ("Inlet temperature of a coolant channel", "°C", "coolant"),
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
              "coolant_flow": f"Flow rate of '{name}'", "coolant_inlet": f"Inlet temperature of '{name}'",
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
    elif kind in ("coolant_flow", "coolant_inlet"):
        condition = next((c for c in copy_["conditions"] if c["name"] == target and
                          c["kind"] == "coolant"), None)
        if condition is None:
            raise ThermalError(f"The coolant channel '{target}' is not in the study (any more).")
        if kind == "coolant_flow":
            condition["flow_mode"] = "fixed"
            condition["flow_lpm"] = max(0.0, float(value))
        else:
            condition["inlet_from"] = ""
            condition["inlet_c"] = float(value)
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
    structural = settings["mode"] == "structural"
    # a structural run gets the thermal conditions too (for the temperatures of a Conduction run)
    conditions = study_conditions(study, jobs, None if structural else THERMAL_KINDS)
    contacts = study_contacts(study, jobs)
    flow_items = study_flow_items(study)
    notes = []
    if structural:
        flow_items = []
        jobs = [job for job in jobs if not job.get("zone")]       # heat zones are no parts
    if flow_items and settings["mode"] != "airflow":
        notes.append("Fans and vents act in the Airflow + heat solver only; Conduction leaves "
                     "them out.")
        flow_items = []
    return jobs, settings, conditions, contacts, flow_items, notes


def solve_study(study, progress=None, cancelled=None):
    """Solve a study with the solver chosen in its settings."""
    jobs, settings, conditions, contacts, flow_items, notes = study_inputs(study)
    if settings["mode"] == "structural":
        result = solve_structural_run(jobs, settings, progress, cancelled, conditions, contacts)
    elif settings["mode"] == "airflow":
        result = solve_airflow(jobs, settings, progress, cancelled, conditions, contacts, flow_items)
    else:
        result = solve_thermal(jobs, settings, progress, cancelled, conditions, contacts)
    result["messages"] = notes + result["messages"]
    return result


_STRUCTURAL_MISSING = []


def structural_missing_text():
    """' ⚠ Needs …' when a package of Structural studies is missing, else ''."""
    if not _STRUCTURAL_MISSING:
        try:
            import simplecad_structural as structural
            _STRUCTURAL_MISSING.append(structural.missing_packages())
        except Exception as ex:
            _STRUCTURAL_MISSING.append([f"simplecad_structural.py ({ex})"])
    missing = _STRUCTURAL_MISSING[0]
    return "" if not missing else (" ⚠ Needs the Python package " + ", ".join(missing) +
                                   " (pip install " + " ".join(m for m in missing if " " not in m) + ").")


def structural_auto_size(study):
    """Automatic element size (mm) of a Structural study, or None."""
    jobs = [job for job in study_jobs(study) if not job.get("zone")]
    if not jobs:
        return None
    try:
        import simplecad_structural as structural
        return structural.auto_element_size(jobs, study["settings"])[0]
    except Exception:
        return None


def structural_estimate_text(study):
    settings = study["settings"]
    jobs = [job for job in study_jobs(study) if not job.get("zone")]
    if not jobs:
        return "No visible bodies yet."
    size = settings["struct_mesh_mm"] or structural_auto_size(study) or 0.0
    if size <= 0:
        return ""
    volume = sum(abs(_mesh_volume(job["V"], job["T"])) for job in jobs)
    count = int(1.6 * volume * 6.0 * math.sqrt(2.0) / size ** 3 / 0.75)
    missing = [job["name"] for job in jobs if not (job.get("props") or {}).get("E")]
    text = (f"≈ {count:,} elements of {size:g} mm".replace(",", " ") +
            (" (automatic)" if settings["struct_mesh_mm"] <= 0 else "") +
            "; round faces get finer elements.")
    if missing:
        text += " ⚠ No mechanical data: " + ", ".join(missing[:3]) + ("…" if len(missing) > 3 else "")
    return text


def solve_structural_run(jobs, settings, progress=None, cancelled=None, conditions=(), contacts=()):
    """A Structural study (simplecad_structural): stresses or natural frequencies; with thermal
    expansion from a Conduction run, that steady run comes first and gives the temperatures."""
    import simplecad_structural as structural
    loads = [c for c in conditions if c["kind"] in STRUCTURAL_KINDS]
    temperatures, notes, thermal = None, [], None
    report = progress or (lambda _p, _t: None)
    if settings["struct_analysis"] == "static" and settings["struct_thermal"] == "conduction":
        heat_settings = dict(settings, mode="conduction", analysis="steady")
        heat_conditions = [c for c in conditions if c["kind"] in THERMAL_KINDS]
        thermal = solve_thermal(jobs, heat_settings, lambda p, t: report(int(0.4 * p),
                                                                         "Temperatures: " + t),
                                cancelled, heat_conditions, contacts)
        frame = len(thermal["frame_times"]) - 1

        def temperatures(points):
            index, weights = result_point_weights(thermal, points)
            return result_volume_values(thermal, frame, index, weights)
        notes.append(f"Temperatures from a steady Conduction run of this study: "
                     f"{thermal['t_min']:.1f} … {thermal['t_max']:.1f} °C "
                     f"({thermal['power']:.3g} W of heat sources).")
        stage = lambda p, t: report(40 + int(0.6 * p), t)
    else:
        stage = report
    result = structural.solve_structural(jobs, settings, stage, cancelled, loads, temperatures)
    result["messages"] = notes + result["messages"]
    if thermal is not None:
        result["thermal_range"] = (thermal["t_min"], thermal["t_max"])
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
    if study["settings"]["mode"] == "structural":
        raise ThermalError("Parameter sweeps work for Thermal studies so far (Conduction or "
                           "Airflow solver).")
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


def write_vtu(path, points, connectivity, point_data=(), cell_data=(), comment="", cell_type=12):
    """One VTK XML unstructured grid of hexahedra (or another `cell_type`: 10 linear, 24
    quadratic tetrahedra). point_data / cell_data: [(name, array (n,) or (n, k))]. Binary
    appended data, little endian, 64-bit block headers."""
    n_points, n_cells = len(points), len(connectivity)
    per_cell = int(np.asarray(connectivity).shape[1]) if n_cells else 8
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
    offsets_xml = array_xml("offsets", np.arange(per_cell, per_cell * n_cells + 1, per_cell,
                                                 dtype=index_type))
    types_xml = array_xml("types", np.full(n_cells, cell_type, np.uint8))     # 12 = VTK_HEXAHEDRON
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


STRUCT_FIELDS = {"vm": ("von Mises stress", "σv  [MPa]", " MPa"),
                 "disp": ("Displacement", "|u|  [mm]", " mm"),
                 "safety": ("Safety factor", "Safety", "")}


def structural_values(result, item, field, mode=0):
    """Values of a result surface item for the colours: von Mises MPa, displacement mm (a mode
    shape: relative), or the safety factor (yield / von Mises; inf without stress)."""
    disp = item["modes"][mode] if result["analysis"] == "modal" else item["disp"]
    if field == "disp" or result["analysis"] == "modal":
        return np.linalg.norm(disp, axis=1)
    if field == "safety":
        strength = item.get("yield")
        if not strength:
            return np.full(len(item["vm"]), np.inf)
        return strength / np.maximum(item["vm"], 1e-12)
    return item["vm"]


def structural_point(result, point, mode=0):
    """(von Mises MPa, displacement vector mm) at the result node nearest to a world point."""
    tree = result.get("_node_tree")
    volume = result["volume"]
    if tree is None:
        from scipy.spatial import cKDTree
        tree = result["_node_tree"] = cKDTree(volume["points"])
    _d, index = tree.query(np.asarray(point, float).reshape(3))
    disp = volume["modes"][mode][index] if result["analysis"] == "modal" else volume["disp"][index]
    return float(volume["vm"][index]), np.asarray(disp, float)


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
        self.channel_arrows = []        # [(start, unit direction, length mm, label)] coolant in / out
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

    def _click_select3d(self, pos, ctrl=False, shift=False):
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

    def _draw_channel_arrows(self):
        """Arrows where the liquid of a coolant channel enters and leaves."""
        if not self.channel_arrows:
            return
        gl.glPushAttrib(gl.GL_ENABLE_BIT | gl.GL_CURRENT_BIT | gl.GL_LINE_BIT)
        try:
            gl.glDisable(gl.GL_LIGHTING)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glLineWidth(3.0)
            font = QFont("Arial", 9, QFont.Bold)
            for start, direction, length, label in self.channel_arrows:
                start = np.asarray(start, float)
                axis = np.asarray(direction, float)
                axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
                tip = start + axis * length
                side = np.cross(axis, np.eye(3)[int(np.argmin(np.abs(axis)))])
                side = side / max(float(np.linalg.norm(side)), 1e-12) * length * 0.2
                gl.glColor3f(0.2, 0.7, 1.0)
                gl.glBegin(gl.GL_LINES)
                for a, b in ((start, tip), (tip, tip - axis * length * 0.35 + side),
                             (tip, tip - axis * length * 0.35 - side)):
                    gl.glVertex3f(*[float(x) for x in a]); gl.glVertex3f(*[float(x) for x in b])
                gl.glEnd()
                gl.glColor3f(0.75, 0.9, 1.0)
                self.renderText(float(tip[0]), float(tip[1]), float(tip[2]), " " + label, font)
        finally:
            gl.glPopAttrib()

    def paintGL(self):
        super().paintGL()
        if self._mvp is not None:
            self._draw_face_overlays()
            self._draw_probe_markers()
            self._draw_channel_arrows()
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
            if self._settings.get("mode") == "structural":
                result = solve_structural_run(self._jobs, self._settings, self._progress,
                                              lambda: self._cancel, self._conditions, self._contacts)
            elif self._settings.get("mode") == "airflow":
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
        self.btn_table = QPushButton("Import CSV")
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
        self.btn_import = QPushButton("Import CSV")
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
        self.chk_mech = QCheckBox("Mechanical data (Structural studies)")
        form.addRow("", self.chk_mech)
        self.sp_E = _spin(1e-4, 2000.0, " GPa", 1.0, decimals=4, width=170)
        self.sp_E.setToolTip("Young's modulus")
        form.addRow("E:", self.sp_E)
        self.sp_nu = _spin(-0.99, 0.49, "", 0.01, decimals=3, width=170)
        self.sp_nu.setToolTip("Poisson's ratio (metals ≈ 0.3, plastics ≈ 0.35–0.4)")
        form.addRow("ν:", self.sp_nu)
        self.sp_yield = _spin(0.0, 10000.0, " MPa", 5.0, decimals=1, width=170)
        self.sp_yield.setToolTip("Yield strength (brittle materials: tensile strength); 0 = unknown, "
                                 "no safety factor")
        self.chk_brittle = QCheckBox("Brittle (strength, no yielding)")
        strength = QWidget()
        strength_layout = QHBoxLayout(strength)
        strength_layout.setContentsMargins(0, 0, 0, 0)
        strength_layout.addWidget(self.sp_yield)
        strength_layout.addWidget(self.chk_brittle)
        strength_layout.addStretch(1)
        form.addRow("Yield:", strength)
        self.sp_alpha = _spin(-100.0, 1000.0, " µm/m·K", 0.5, decimals=2, width=170)
        self.sp_alpha.setToolTip("Thermal expansion coefficient")
        form.addRow("α:", self.sp_alpha)
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
                       self.sp_max, self.sp_E, self.sp_nu, self.sp_yield, self.sp_alpha):
            widget.valueChanged.connect(self._edited)
        for widget in (self.chk_plane, self.chk_max, self.chk_mech, self.chk_brittle):
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
            self.chk_mech.setChecked(bool(data.get("E")))
            self.sp_E.setValue(data.get("E") or 1.0)
            self.sp_nu.setValue(data.get("nu") if data.get("nu") is not None else 0.3)
            self.sp_yield.setValue(data.get("yield") or 0.0)
            self.chk_brittle.setChecked(bool(data.get("brittle")))
            self.sp_alpha.setValue(data.get("alpha") or 0.0)
            read_only = kind == "builtin"
            for widget in (self.ed_name, self.sp_k, self.chk_plane, self.sp_k_plane, self.sp_cp,
                           self.sp_density, self.sp_emissivity, self.chk_max, self.sp_max,
                           self.ed_note, self.chk_mech):
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
        for widget in (self.sp_E, self.sp_nu, self.sp_yield, self.chk_brittle, self.sp_alpha):
            widget.setEnabled(kind == "user" and self.chk_mech.isChecked())
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
            "cp_table": self.tbl_cp.points(),
            "E": self.sp_E.value() if self.chk_mech.isChecked() else None,
            "nu": self.sp_nu.value(), "yield": self.sp_yield.value() or None,
            "alpha": self.sp_alpha.value(), "brittle": self.chk_brittle.isChecked(),
            "mech_note": (self._user.get(self._current[1], {}) if self._current else {}).get("mech_note", "")})

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
                                                 "max_c", "note", "k_table", "cp_table") +
                MECHANICAL_KEYS}
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
        self.cb_spacing.addItem("Evenly from start to end", "range")
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
        elif target == "coolant":
            out = [(c["name"], c["name"]) for c in study.get("conditions", []) if c["kind"] == "coolant"]
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
        if kind in ("coolant_flow", "coolant_inlet"):
            condition = next((c for c in study["conditions"] if c["name"] == target), None)
            return None if condition is None else \
                condition["flow_lpm" if kind == "coolant_flow" else "inlet_c"]
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


class RunHistoryDialog(QDialog):
    """The runs of this session with their parameters: a run can be shown again, its setup
    restored and the results exported; several selected runs are compared side by side."""

    def __init__(self, window):
        super().__init__(window)
        self._window = window
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        self.setWindowIcon(_icon("settings2.png"))
        self.setWindowTitle("Run history")
        self.resize(940, 640)
        layout = QVBoxLayout(self)
        self.lbl_title = QLabel()
        self.lbl_title.setWordWrap(True)
        layout.addWidget(self.lbl_title)
        split = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Run", "Time", "Type", "Result", "Duration"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._on_selection)
        self.table.itemChanged.connect(self._on_mark)
        self.table.itemDoubleClicked.connect(lambda _item: self._show())
        split.addWidget(self.table)
        box = QWidget()
        inner = QVBoxLayout(box)
        inner.setContentsMargins(0, 0, 0, 0)
        self.lbl_params = QLabel("Parameters")
        inner.addWidget(self.lbl_params)
        self.params = QTableWidget(0, 0)
        self.params.verticalHeader().hide()
        self.params.setEditTriggers(QAbstractItemView.NoEditTriggers)
        inner.addWidget(self.params)
        split.addWidget(box)
        split.setSizes([260, 340])
        layout.addWidget(split, 1)
        buttons = QHBoxLayout()
        self.btn_show = QPushButton("Show in the view")
        self.btn_show.clicked.connect(self._show)
        self.btn_restore = QPushButton("Restore setup")
        self.btn_restore.setToolTip("Put the study back to the values this run was made with")
        self.btn_restore.clicked.connect(self._restore)
        self.btn_csv = QPushButton("Export results CSV")
        self.btn_csv.setToolTip("The runs ticked in the first column, else the selected ones")
        self.btn_csv.clicked.connect(lambda: self._export("csv"))
        self.btn_vtu = QPushButton("Export for ParaView")
        self.btn_vtu.clicked.connect(lambda: self._export("vtu"))
        self.btn_table = QPushButton("Export comparison CSV")
        self.btn_table.setToolTip("One row per selected run: result and all parameters")
        self.btn_table.clicked.connect(lambda: self._export("summary"))
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self._delete)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        for widget in (self.btn_show, self.btn_restore, self.btn_csv, self.btn_vtu, self.btn_table,
                       self.btn_delete):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.refresh()

    def _on_mark(self, item):
        if item.column() == 0 and item.flags() & Qt.ItemIsUserCheckable:
            self._window._set_run_marked(item.data(Qt.UserRole), item.checkState() == Qt.Checked)
            self._on_selection()

    def targets(self):
        """Runs to export: the ticked ones, else the selected ones."""
        return self._window._marked_ids() or self.selected_ids()

    def selected_ids(self):
        ids = []
        for row in sorted({index.row() for index in self.table.selectedIndexes()}):
            ids.append(self.table.item(row, 0).data(Qt.UserRole))
        return ids

    def refresh(self):
        keep = set(self.selected_ids())
        runs = self._window._runs
        self.lbl_title.setText(f"{len(runs)} run(s) kept in this session (the last "
                               f"{self._window.RUN_LIMIT}); a change of the study does not remove them.")
        self.table.blockSignals(True)
        self.table.clearSelection()
        self.table.setRowCount(len(runs))
        for row, run in enumerate(reversed(runs)):
            for column, text in enumerate((f"#{run['id']}", run["time"], run["title"], run["summary"],
                                           f"{run['seconds']:.1f} s")):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, run["id"])
                if column == 0:
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked if run.get("marked") else Qt.Unchecked)
                self.table.setItem(row, column, item)
        self.table.setSelectionMode(QAbstractItemView.MultiSelection)
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).data(Qt.UserRole) in keep:
                self.table.selectRow(row)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.blockSignals(False)
        self._on_selection()

    def _on_selection(self):
        ids = self.selected_ids()
        runs = [r for r in self._window._runs if r["id"] in ids]
        names = []
        for run in runs:
            for name, _value in run["params"]:
                if name not in names:
                    names.append(name)
        self.params.clear()
        self.params.setColumnCount(1 + len(runs))
        self.params.setRowCount(len(names))
        self.params.setHorizontalHeaderLabels(["Parameter"] + [f"Run #{r['id']}" for r in runs])
        for row, name in enumerate(names):
            values = [dict(r["params"]).get(name, "—") for r in runs]
            differs = len(set(values)) > 1
            self.params.setItem(row, 0, QTableWidgetItem(name))
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if differs:
                    item.setForeground(QColor(0, 220, 255))
                self.params.setItem(row, 1 + column, item)
        self.params.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.params.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.lbl_params.setText("Parameters of the selected run" if len(runs) == 1 else
                                "Parameters of the selected runs (values that differ are cyan)"
                                if runs else "Select a run to see its parameters")
        one = len(ids) == 1
        self.btn_show.setEnabled(one)
        self.btn_restore.setEnabled(one)
        targets = self.targets()
        self.btn_csv.setEnabled(bool(targets))
        self.btn_vtu.setEnabled(len(targets) == 1)
        self.btn_table.setEnabled(bool(targets))
        self.btn_delete.setEnabled(bool(ids))

    def _show(self):
        ids = self.selected_ids()
        if len(ids) == 1:
            self._window._show_run(ids[0])

    def _restore(self):
        ids = self.selected_ids()
        if len(ids) == 1:
            self._window._restore_run(ids[0])

    def _export(self, kind):
        ids = self.targets()
        if ids:
            self._window._export_runs(kind, ids)

    def _delete(self):
        ids = self.selected_ids()
        if ids:
            self._window._delete_runs(ids)


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
        self.btn_csv = QPushButton("Export CSV")
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
def study_add_pcb(study, pcb, name="PCB", position=(0.0, 0.0, 0.0)):
    """A board in the study (component kind "pcb"): its outline extruded, conductivity from the
    stack and the copper (see pcb_conductivity)."""
    vertices, triangles, faces = pcb_board_mesh(pcb)
    material = PCB_CORES.get(pcb.get("core"), PCB_CORES["FR-4"])["material"]
    body = thermal_body(name, vertices, triangles, faces, PCB_COLOR, material=material,
                        power_w=float(pcb.get("background_w", 0.0)))
    component = cad.assembly_add_component(study, name, "pcb", [body],
                                           source=f"Board from {pcb.get('source') or 'a file'}")
    body["name"] = component["name"]
    component["pivot"] = [0.0, 0.0, 0.0]
    component["position"] = [float(v) for v in position]
    component["pcb"] = pcb
    return component


def study_add_package(study, package, name, x, y, board_top, board_bottom, power_w, size=None,
                      angle=0.0):
    """A component package (block with a junction, two-resistor model) on a board side."""
    entry = PACKAGES.get(package["type"], PACKAGES["Custom block"])
    length, width, height = size or entry[:3]
    top_side = package.get("side", "top") != "bottom"
    z = board_top + height / 2.0 if top_side else board_bottom - height / 2.0
    component = study_add_primitive(study, "block", {"size": [length, width, height]}, name,
                                    [x, y, z], power_w, PACKAGE_MATERIAL, rotation=[0.0, 0.0, angle])
    component["bodies"][0]["color"] = (0.22, 0.22, 0.25)
    component["package"] = dict(package)
    component["source"] = f"Package {package['type']}"
    return component


class PcbImportDialog(QDialog):
    """Import a printed circuit board: KiCad board or Gerber files, board set-up, and the power
    components to place (KiCad footprints)."""

    PREFIXES = "Q U D L"

    def __init__(self, parent, start_directory):
        super().__init__(parent)
        self.setWindowTitle("Import PCB")
        self.resize(980, 720)
        self._start = start_directory
        self.desc = None
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.ed_path = QLineEdit()
        self.ed_path.setReadOnly(True)
        self.ed_path.setPlaceholderText("KiCad board (.kicad_pcb), Gerber job (.gbrjob) or a folder "
                                        "of Gerber and drill files")
        row.addWidget(self.ed_path, 1)
        board_button = QPushButton("Board file")
        board_button.clicked.connect(self._choose_file)
        row.addWidget(board_button)
        folder_button = QPushButton("Gerber folder")
        folder_button.clicked.connect(self._choose_folder)
        row.addWidget(folder_button)
        layout.addLayout(row)
        self.lbl_info = _hint("Choose a board to import.")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)
        # board set-up
        layout.addWidget(_header("Board"))
        form = QGridLayout()
        self.cb_core = QComboBox()
        for key, core in PCB_CORES.items():
            self.cb_core.addItem(core["title"], key)
        self.cb_core.currentIndexChanged.connect(self._core_changed)
        form.addWidget(QLabel("Core:"), 0, 0)
        form.addWidget(self.cb_core, 0, 1)
        self.sp_plating = _spin(0.0, 100.0, " µm", 5.0, decimals=0)
        self.sp_plating.setValue(PCB_DEFAULT_PLATING_UM)
        self.sp_plating.setToolTip("Copper plating of the holes (vias and through-hole pads)")
        form.addWidget(QLabel("Hole plating:"), 0, 2)
        form.addWidget(self.sp_plating, 0, 3)
        self.sp_ims_mm = _spin(0.02, 1.0, " mm", 0.01, decimals=3)
        self.sp_ims_mm.setValue(0.1)
        self.sp_ims_k = _spin(0.1, 20.0, " W/m·K", 0.1, decimals=2)
        self.sp_ims_k.setValue(2.0)
        self.lbl_ims = QLabel("IMS dielectric:")
        form.addWidget(self.lbl_ims, 1, 0)
        form.addWidget(self.sp_ims_mm, 1, 1)
        form.addWidget(self.sp_ims_k, 1, 2)
        self.sp_background = _spin(0.0, 1000.0, " W", 0.1, decimals=2)
        self.sp_background.setToolTip("Heat of all components that are not placed, spread evenly "
                                      "over the board")
        form.addWidget(QLabel("Other parts:"), 1, 3)
        form.addWidget(self.sp_background, 1, 4)
        form.setColumnStretch(5, 1)
        layout.addLayout(form)
        self.tbl_stack = QTableWidget(0, 2)
        self.tbl_stack.setHorizontalHeaderLabels(["Layer", "Thickness"])
        self.tbl_stack.verticalHeader().hide()
        self.tbl_stack.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_stack.setFixedHeight(150)
        layout.addWidget(self.tbl_stack)
        # components
        layout.addWidget(_header("Components to place (power parts; the rest can go into "
                                 "'Other parts')"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Select by reference:"))
        self.ed_prefix = QLineEdit(self.PREFIXES)
        self.ed_prefix.setFixedWidth(160)
        row.addWidget(self.ed_prefix)
        select = QPushButton("Select these")
        select.clicked.connect(self._select_prefixes)
        row.addWidget(select)
        row.addStretch(1)
        layout.addLayout(row)
        self.tbl_parts = QTableWidget(0, 9)
        self.tbl_parts.setHorizontalHeaderLabels(["", "Ref", "Value", "Footprint", "Package", "Power W",
                                                  "θ top K/W", "θ board K/W", "Tj max °C"])
        self.tbl_parts.verticalHeader().hide()
        header = self.tbl_parts.horizontalHeader()
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        for column in (0, 1, 2, 4, 5, 6, 7, 8):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        layout.addWidget(self.tbl_parts, 1)
        layout.addWidget(_hint("θ top: junction to the package top (θJC top); θ board: junction to the "
                               "board (θJB, or θJC of an exposed pad or tab). Typical values are "
                               "filled in — datasheet values are better."))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        self.btn_ok = buttons.button(QDialogButtonBox.Ok)
        self.btn_ok.setEnabled(False)
        layout.addWidget(buttons)
        self._core_changed()

    # loading ---------------------------------------------------------------------------------
    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import a PCB", self._start,
            "PCB files (*.kicad_pcb *.gbrjob *.gbr *.gtl *.gbl *.cmp *.sol);;All files (*)")
        if path:
            self.load(path)

    def _choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Folder with Gerber and drill files", self._start)
        if path:
            self.load(path)

    def load(self, path):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.desc = read_board_file(path)
        except Exception as ex:
            QApplication.restoreOverrideCursor()
            self.desc = None
            self.btn_ok.setEnabled(False)
            QMessageBox.critical(self, "Import PCB", f"Could not read\n{path}\n\n{ex}")
            return
        QApplication.restoreOverrideCursor()
        desc = self.desc
        self.ed_path.setText(path)
        kind = "KiCad board" if desc["format"] == "kicad" else "Gerber files"
        text = (f"{kind}: {len(desc['copper_layers'])} copper layer"
                f"{'s' if len(desc['copper_layers']) != 1 else ''}, "
                f"{'outline found' if desc['loops'] else 'no outline (the box around the copper)'}, "
                f"{len(desc['plated'])} plated holes, {len(desc['footprints'])} footprints.")
        self.lbl_info.setText(text + ("\n" + "\n".join(desc["messages"]) if desc["messages"] else ""))
        # stack: copper and dielectric rows
        self.tbl_stack.setRowCount(0)
        for index, name in enumerate(desc["copper_layers"]):
            self._stack_row(f"Copper {name}", desc["copper_um"][index], " µm", "copper")
            if index < len(desc["copper_layers"]) - 1:
                self._stack_row("Dielectric", desc["dielectric_mm"][index], " mm", "dielectric")
        # components
        self.tbl_parts.setRowCount(len(desc["footprints"]))
        for row, fp in enumerate(desc["footprints"]):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check.setCheckState(Qt.Unchecked)
            self.tbl_parts.setItem(row, 0, check)
            for column, text in ((1, fp["reference"]), (2, fp["value"]), (3, fp["footprint"])):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.tbl_parts.setItem(row, column, item)
            combo = QComboBox()
            for name in PACKAGES:
                combo.addItem(name, name)
            guess = guess_package(fp["footprint"])
            combo.setCurrentIndex(max(0, combo.findData(guess)))
            combo.currentIndexChanged.connect(lambda _i, r=row: self._package_changed(r))
            self.tbl_parts.setCellWidget(row, 4, combo)
            self.tbl_parts.setItem(row, 5, QTableWidgetItem("0"))
            self._package_changed(row)
        self._select_prefixes()
        self.btn_ok.setEnabled(True)

    def _stack_row(self, title, value, suffix, kind):
        row = self.tbl_stack.rowCount()
        self.tbl_stack.insertRow(row)
        item = QTableWidgetItem(title)
        item.setFlags(Qt.ItemIsEnabled)
        item.setData(Qt.UserRole, kind)
        self.tbl_stack.setItem(row, 0, item)
        spin = _spin(0.001, 10000.0, suffix, 1.0 if kind == "copper" else 0.05,
                     decimals=0 if kind == "copper" else 3)
        spin.setValue(float(value))
        if kind == "copper":
            spin.setToolTip("18 µm = ½ oz, 35 µm = 1 oz, 70 µm = 2 oz, 105 µm = 3 oz")
        self.tbl_stack.setCellWidget(row, 1, spin)

    def _package_changed(self, row):
        name = self.tbl_parts.cellWidget(row, 4).currentData()
        entry = PACKAGES.get(name, PACKAGES["Custom block"])
        for column, value in ((6, entry[3]), (7, entry[4]), (8, entry[5])):
            self.tbl_parts.setItem(row, column, QTableWidgetItem(f"{value:g}"))

    def _select_prefixes(self):
        prefixes = [p.strip().upper() for p in self.ed_prefix.text().replace(",", " ").split() if p.strip()]
        for row in range(self.tbl_parts.rowCount()):
            reference = self.tbl_parts.item(row, 1).text().upper()
            letters = "".join(ch for ch in reference if ch.isalpha())
            chosen = letters in prefixes or any(reference.startswith(p) for p in prefixes if p.endswith("*"))
            self.tbl_parts.item(row, 0).setCheckState(Qt.Checked if chosen else Qt.Unchecked)

    def _core_changed(self, *_args):
        ims = self.cb_core.currentData() == "IMS"
        for widget in (self.lbl_ims, self.sp_ims_mm, self.sp_ims_k):
            widget.setVisible(ims)

    # result ----------------------------------------------------------------------------------
    def _number(self, row, column, default):
        item = self.tbl_parts.item(row, column)
        try:
            return float(item.text().replace(",", ".")) if item is not None else default
        except ValueError:
            return default

    def _accept(self):
        if self.desc is None:
            return
        try:
            self.values()
        except Exception as ex:
            QMessageBox.critical(self, "Import PCB", str(ex))
            return
        self.accept()

    def values(self):
        """(pcb, [(name, package, power W, x, y, angle, size)], messages)."""
        if hasattr(self, "_values"):
            return self._values
        copper, dielectric = [], []
        for row in range(self.tbl_stack.rowCount()):
            value = self.tbl_stack.cellWidget(row, 1).value()
            (copper if self.tbl_stack.item(row, 0).data(Qt.UserRole) == "copper" else dielectric).append(value)
        pcb, place, messages = pcb_from_description(
            self.desc, core=self.cb_core.currentData(), copper_um=copper, dielectric_mm=dielectric,
            plating_um=self.sp_plating.value(), ims_dielectric_mm=self.sp_ims_mm.value(),
            ims_k=self.sp_ims_k.value())
        pcb["background_w"] = float(self.sp_background.value())
        parts = []
        for row, fp in enumerate(self.desc["footprints"]):
            if self.tbl_parts.item(row, 0).checkState() != Qt.Checked:
                continue
            name = self.tbl_parts.cellWidget(row, 4).currentData()
            package = package_new(name, fp["reference"])
            package["theta_top"] = max(1e-3, self._number(row, 6, package["theta_top"]))
            package["theta_board"] = max(1e-3, self._number(row, 7, package["theta_board"]))
            package["tj_max"] = self._number(row, 8, package["tj_max"])
            package["side"] = fp["side"]
            entry = PACKAGES[name]
            length, width, height = entry[:3]
            pads_w, pads_h = fp["pads_size"]
            if name == "Custom block":
                length, width = max(pads_w, 1.0), max(pads_h, 1.0)
            elif (length >= width) != (pads_w >= pads_h):     # along the pads' long side
                length, width = width, length
            x, y = place(fp["pads_centre"])
            angle = fp["angle"] if fp["side"] == "top" else -fp["angle"]
            parts.append((fp["reference"] or name, package, max(0.0, self._number(row, 5, 0.0)),
                          x, y, angle, (length, width, height)))
        self._values = (pcb, parts, messages)
        return self._values


class PackageDialog(QDialog):
    """Place a component package (two-resistor model) by hand."""

    def __init__(self, parent, name):
        super().__init__(parent)
        self.setWindowTitle("Component package")
        form = QFormLayout(self)
        self.ed_name = QLineEdit(name)
        form.addRow("Name:", self.ed_name)
        self.cb_type = QComboBox()
        for package in PACKAGES:
            self.cb_type.addItem(package, package)
        self.cb_type.currentIndexChanged.connect(self._type_changed)
        form.addRow("Package:", self.cb_type)
        self.sp_power = _spin(0.0, 10000.0, " W", 0.1)
        self.sp_power.setValue(1.0)
        form.addRow("Power:", self.sp_power)
        self.sp_top = _spin(0.001, 10000.0, " K/W", 1.0, decimals=2)
        form.addRow("θ junction → top:", self.sp_top)
        self.sp_board = _spin(0.001, 10000.0, " K/W", 1.0, decimals=2)
        form.addRow("θ junction → board:", self.sp_board)
        self.sp_tj = _spin(-50.0, 400.0, " °C", 5.0, decimals=0)
        form.addRow("Tj max:", self.sp_tj)
        self.cb_side = QComboBox()
        self.cb_side.addItem("On the top of the board", "top")
        self.cb_side.addItem("Under the board", "bottom")
        form.addRow("Side:", self.cb_side)
        form.addRow(_hint("Placed on the selected board (else at the origin); move it with the "
                          "arrows. Typical θ values are filled in — datasheet values are better."))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.cb_type.setCurrentIndex(max(0, self.cb_type.findData("DPAK (TO-252)")))
        self._type_changed()

    def _type_changed(self, *_args):
        entry = PACKAGES[self.cb_type.currentData()]
        self.sp_top.setValue(entry[3])
        self.sp_board.setValue(entry[4])
        self.sp_tj.setValue(entry[5])

    def values(self):
        package = package_new(self.cb_type.currentData(), self.ed_name.text().strip())
        package.update(theta_top=self.sp_top.value(), theta_board=self.sp_board.value(),
                       tj_max=self.sp_tj.value(), side=self.cb_side.currentData())
        return self.ed_name.text().strip() or package["type"], package, self.sp_power.value()


_VIEWS = (("view_isometric.png", "Isometric", -54.735610317245346, -135.0),
          ("view_top.png", "Top (XY)", 0.0, 0.0),
          ("view_bottom.png", "Bottom (XY)", 180.0, 0.0),
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
        self._runs = []                      # finished runs of this session (run history)
        self._run_counter = 0
        self._run_dialog = None
        self._run_state = None
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
        self.act_open = action("open.png", "Open study", f"Open a thermal study (*{THERMAL_EXTENSION})",
                               self._open_file, shortcut=QKeySequence.Open)
        self.act_save = action("save.png", "Save study", "Save the thermal study", self._save,
                               shortcut=QKeySequence.Save)
        self.act_save_as = action("save_as.png", "Save study as",
                                  "Save the thermal study under a new name",
                                  lambda: self._save(True), shortcut="Ctrl+Shift+S")
        self.import_menu = QMenu("&Import", self)
        self.import_menu.setIcon(_icon("import.png"))
        self.import_menu.addAction("STEP file", lambda: self._import_files("step"))
        self.import_menu.addAction("STL file", lambda: self._import_files("stl"))
        self.import_menu.addAction("OBJ file / SimpleCAD project", lambda: self._import_files("obj"))
        self.import_menu.addAction("Assembly file (.asby)", self._import_assembly_file)
        self.import_menu.addSeparator()
        self.import_menu.addAction("Assembly of the open project", self._import_project_assembly)
        self.import_menu.addAction("Parts of the open project", self._import_project_parts)
        self.import_menu.addSeparator()
        self.import_menu.addAction("PCB (KiCad board or Gerber files)", self._import_pcb)
        self.act_import = action(
            "import.png", "Import", "Import — STEP, STL or OBJ files, a SimpleCAD project, an "
            "assembly, or parts of the open project", lambda: None)
        self.act_import.setMenu(self.import_menu)
        self.export_menu = QMenu("&Export", self)
        self.export_menu.setIcon(_icon("export.png"))
        self.act_report = self.export_menu.addAction("PDF report", self._export_pdf)
        self.export_menu.addSeparator()
        self.act_export_csv = self.export_menu.addAction("Results table (CSV)", self._export_csv)
        self.act_export_series = self.export_menu.addAction("Values over time (CSV)",
                                                            self._export_series_csv)
        self.act_export_vtu = self.export_menu.addAction("Result for ParaView (VTU)",
                                                         self._export_vtu)
        self.act_export_vtu.setToolTip("Temperatures (and air velocities) of the whole mesh as a "
                                       "VTK file for ParaView; a transient run also as a series")
        self.act_export_png = self.export_menu.addAction("Image of the view (PNG)", self._export_png)
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
        self.act_runs = QAction("Run history", self,
                                toolTip="All runs of this session with their parameters: show, "
                                        "restore the setup, compare and export")
        self.act_runs.triggered.connect(lambda _checked=False: self._open_run_history())
        self.act_runs.setEnabled(False)
        self.act_undo = action("undo.png", "Undo", "Undo the last study change", self._undo,
                               shortcut=QKeySequence.Undo)
        self.act_redo = action("redo.png", "Redo", "Redo the last undone study change", self._redo,
                               shortcut="Ctrl+Y")
        self.act_remove = action("delete_selected.png", "Remove",
                                 "Remove the selected part or heat source from the study",
                                 self._remove_selected)
        self.heat_menu = QMenu("&Heat Source", self)
        self.heat_menu.setIcon(_icon("heat_source_extrusion.png"))
        self.heat_menu.addAction("Block", lambda: self._add_heat_source("block"))
        self.heat_menu.addAction("Cylinder", lambda: self._add_heat_source("cylinder"))
        self.heat_menu.addAction("Extrusion on a face", self._start_extrusion)
        self.heat_menu.addAction("Component package (junction model)", self._add_package)
        self.heat_menu.addSeparator()
        self.heat_menu.addAction("Power of the selected body", self._focus_power)
        self.act_heat = action(
            "heat_source_extrusion.png", "Heat Source",
            "Heat Source — place a block or cylinder heat source (on top of the selected part), "
            "or give the selected body a power", lambda: None)
        self.act_heat.setMenu(self.heat_menu)
        self.condition_menu = QMenu("&Face Condition", self)
        self.condition_menu.setIcon(_icon("heat_convection.png"))
        self.condition_menu.aboutToShow.connect(self._fill_condition_menu)
        self._fill_condition_menu()
        self.act_condition = action(
            "heat_convection.png", "Face Condition",
            "Face Condition — fixed temperature, heat flux, own convection, insulation, a coolant "
            "channel or a liquid bath on faces you click in the view", lambda: None)
        self.act_condition.setMenu(self.condition_menu)
        self.contact_menu = QMenu("&Contacts", self)
        self.contact_menu.setIcon(_icon("fit_together.png"))
        self.contact_menu.addAction("Detect touching parts", self._detect_contacts)
        self.contact_menu.addAction("Contact between two bodies", self._add_contact)
        self.act_contact = action(
            "heat_contact.png", "Contacts",
            "Contacts — touching bodies are bonded", lambda: None)
        self.act_contact.setMenu(self.contact_menu)
        self.act_section = action(
            "view_section.png", "Section view",
            "Section view — cut the view at a movable plane to see the temperatures inside the "
            "parts (and in the air of an airflow result)", self._on_section_toggled, checkable=True)
        self.flow_menu = QMenu("&Airflow", self)
        self.flow_menu.setIcon(_icon("air_flow_on.png"))
        self.flow_menu.addAction("Fan", lambda: self._add_flow_item("fan"))
        self.flow_menu.addAction("Vent / opening", lambda: self._add_flow_item("vent"))
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
            "temperature_probe.png", "Probe",
            "Probe — click on a part to place a temperature probe", self._on_probe_toggled,
            checkable=True)
        self.act_start = action(
            "start_thermal_simulation.png", "Start Simulation",
            "Start Simulation (F5) — mesh the visible bodies and solve", self._start_simulation,
            shortcut="F5")
        self.act_results = action(
            "temperature_map.png", "Show temperatures",
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
        edit_menu.addAction("Rename", self._rename_selected)
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
        for item in (self.act_start, self.act_sweep, self.act_sweep_results, self.act_runs, self.act_cancel,
                     self.act_results):
            simulation_menu.addAction(item)

        # left: study explorer and properties --------------------------------------
        self.tree = cad.ExplorerTree()
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
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
        # a component package: the junction between the package top and the board
        self.w_package = QWidget()
        package_grid = QGridLayout(self.w_package)
        package_grid.setContentsMargins(0, 0, 0, 0)
        package_grid.setColumnMinimumWidth(0, label_width)
        self.lbl_package = value_label()
        self.sp_theta_top = _spin(0.001, 10000.0, " K/W", 1.0, decimals=2)
        self.sp_theta_top.setToolTip("Junction to the top of the package (θJC top)")
        self.sp_theta_board = _spin(0.001, 10000.0, " K/W", 0.5, decimals=2)
        self.sp_theta_board.setToolTip("Junction to the board below (θJB, or θJC of an exposed pad "
                                       "or tab soldered to the board)")
        self.sp_tj_max = _spin(-50.0, 400.0, " °C", 5.0, decimals=0)
        for r, (caption, widget) in enumerate((("Package:", self.lbl_package),
                                               ("θ to top:", self.sp_theta_top),
                                               ("θ to board:", self.sp_theta_board),
                                               ("Tj max:", self.sp_tj_max))):
            label = QLabel(caption)
            label.setFixedWidth(label_width)
            package_grid.addWidget(label, r, 0)
            package_grid.addWidget(widget, r, 1, Qt.AlignLeft)
        package_hint = _hint("The power above goes in at the junction (two-resistor model); the "
                             "results give the junction temperature.")
        package_hint.setFixedWidth(span_width)
        package_grid.addWidget(package_hint, 4, 0, 1, 2)
        for spin in (self.sp_theta_top, self.sp_theta_board, self.sp_tj_max):
            spin.valueChanged.connect(self._on_package_edited)
        body_add(self.w_package, span=True)
        self.lbl_board = _hint("")
        self.lbl_board.setFixedWidth(span_width)
        body_add(self.lbl_board, span=True)
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
        # coolant channel and liquid bath
        def combo(items):
            box = QComboBox()
            for key, title in items:
                box.addItem(title, key)
            box.setFixedWidth(value_width)
            return box
        self.cb_liquid_fluid = combo([(name, name) for name in FLUIDS])
        self.cb_bath_mode = combo(list(BATH_MODES.items()))
        self.cb_liquid_model = combo(list(LIQUID_MODELS.items()))
        self.cb_liquid_model.setToolTip(
            "Correlations: heat transfer from pipe-flow / natural-convection formulas (fast, both "
            "solvers).\nLiquid flow solved: in Airflow mode the liquid itself is simulated (needs "
            "about 4 or more cells across the channel).")
        self.cb_coolant_flow = combo([("fixed", "Fixed flow rate"), ("pump", "Pump curve")])
        self.cb_coolant_pump = QComboBox()
        self.cb_coolant_pump.setFixedWidth(value_width)
        self.cb_coolant_pump.setToolTip("Straight line: from the shut-off pressure to the free flow below.\n"
                                        "A pump of the project: the pump tool's estimated curve at its motor "
                                        "speed, for water, scaled to the liquid's density.")
        self.lbl_pump_source = value_label()
        self.lbl_pump_source.setWordWrap(True)
        self.lbl_pump_source.setFixedWidth(value_width)
        self.sp_flow_lpm = _spin(0.0, 1000.0, " L/min", 0.1, decimals=3)
        self.sp_pump_max_lpm = _spin(0.001, 1000.0, " L/min", 0.5, decimals=2)
        self.sp_pump_max_lpm.setToolTip("Flow of the pump without any resistance")
        self.sp_pump_max_kpa = _spin(0.0, 10000.0, " kPa", 1.0, decimals=1)
        self.sp_pump_max_kpa.setToolTip("Pressure of the pump at zero flow")
        self.cb_inlet_from = QComboBox()
        self.cb_inlet_from.setFixedWidth(value_width)
        self.cb_inlet_from.setToolTip("The liquid comes in at its own temperature, or from the "
                                      "outlet of another coolant channel (channels in series)")
        self.sp_inlet_c = _spin(-50.0, 200.0, " °C", 1.0, decimals=1)
        self.chk_reverse = QCheckBox("Liquid enters at the other end")
        self.lbl_channel = value_label()
        self.lbl_channel.setWordWrap(True)
        self.lbl_channel.setFixedWidth(value_width)
        self.chk_fill_auto = QCheckBox("Fill level at the top of the picked faces")
        self.sp_fill_z = _spin(-1e5, 1e5, " mm", 1.0, decimals=2, prefix="Z ")
        self.sp_bath_c = _spin(-50.0, 300.0, " °C", 1.0, decimals=1)
        self.chk_bath_start_auto = QCheckBox("Liquid starts at the run's start temperature")
        self.chk_evaporation = QCheckBox("Open surface loses heat by evaporation")
        self._condition_rows["liquid"] = [self.cb_liquid_fluid, put_row(self.cb_liquid_fluid, "Liquid:")]
        self._condition_rows["bath"] = [self.cb_bath_mode, put_row(self.cb_bath_mode, "Volume:")]
        self._cap_liquid_model = put_row(self.cb_liquid_model, "Model:")
        self._condition_rows["liquid"] += [self.cb_liquid_model, self._cap_liquid_model]
        self._cap_flow_mode = put_row(self.cb_coolant_flow, "Flow:")
        self._cap_coolant_pump = put_row(self.cb_coolant_pump, "Pump:")
        self._cap_pump_source = put_row(self.lbl_pump_source, "Curve:")
        self._cap_flow_lpm = put_row(self.sp_flow_lpm, "Flow rate:")
        self._cap_pump_lpm = put_row(self.sp_pump_max_lpm, "Free flow:")
        self._cap_pump_kpa = put_row(self.sp_pump_max_kpa, "Shut-off:")
        self._cap_inlet_from = put_row(self.cb_inlet_from, "Inlet:")
        self._cap_inlet_c = put_row(self.sp_inlet_c, "Inlet at:")
        put_row(self.chk_reverse, span=True)
        self._cap_channel = put_row(self.lbl_channel, "Channel:")
        self._condition_rows["coolant"] = [
            self.cb_coolant_flow, self._cap_flow_mode, self.sp_flow_lpm, self._cap_flow_lpm,
            self.cb_coolant_pump, self._cap_coolant_pump, self.lbl_pump_source, self._cap_pump_source,
            self.sp_pump_max_lpm, self._cap_pump_lpm, self.sp_pump_max_kpa, self._cap_pump_kpa,
            self.cb_inlet_from, self._cap_inlet_from, self.sp_inlet_c, self._cap_inlet_c,
            self.chk_reverse, self.lbl_channel, self._cap_channel]
        put_row(self.chk_fill_auto, span=True)
        self._cap_fill_z = put_row(self.sp_fill_z, "Fill level:")
        self._cap_bath_c = put_row(self.sp_bath_c, "Liquid at:")
        put_row(self.chk_bath_start_auto, span=True)
        put_row(self.chk_evaporation, span=True)
        self._condition_rows["bath"] += [self.chk_fill_auto, self.sp_fill_z, self._cap_fill_z,
                                         self.sp_bath_c, self._cap_bath_c, self.chk_bath_start_auto,
                                         self.chk_evaporation]
        # structural: force, pressure, torque, pinned support
        self.sp_force = [_spin(-1e9, 1e9, " N", 10.0, decimals=3) for _axis in range(3)]
        self._condition_rows["force"] = []
        for spin, axis in zip(self.sp_force, "XYZ"):
            spin.setToolTip(f"Force along the world {axis} axis, spread over the faces")
            self._condition_rows["force"] += [spin, put_row(spin, f"Force {axis}:")]
        self.lbl_force = value_label()
        self._condition_rows["force"] += [self.lbl_force, put_row(self.lbl_force, "Total:")]
        self.sp_pressure = _spin(-1e5, 1e5, " MPa", 0.01, decimals=4)
        self.sp_pressure.setToolTip("Positive pushes on the faces, negative pulls; 1 bar = 0.1 MPa")
        self._condition_rows["pressure"] = [self.sp_pressure, put_row(self.sp_pressure, "Pressure:")]
        self.sp_torque = _spin(-1e7, 1e7, " N·m", 0.1, decimals=3)
        self.chk_torque_reverse = QCheckBox("Turn the other way round")
        self._condition_rows["torque"] = [self.sp_torque, put_row(self.sp_torque, "Torque:")]
        put_row(self.chk_torque_reverse, span=True)
        self._condition_rows["torque"].append(self.chk_torque_reverse)
        self.chk_pin_axial = QCheckBox("May also slide along its axis")
        put_row(self.chk_pin_axial, span=True)
        self._condition_rows["support_pin"] = [self.chk_pin_axial]
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
        self.lbl_pick_hint = _hint("")
        self.lbl_pick_hint.setFixedWidth(span_width)
        put_row(self.lbl_pick_hint, span=True)
        finish()
        self.chk_condition_enabled.toggled.connect(self._on_condition_edited)
        self.cb_flux_mode.currentIndexChanged.connect(self._on_condition_edited)
        self.chk_condition_fluid.toggled.connect(self._on_condition_edited)
        for spin in (self.sp_condition_temperature, self.sp_condition_power, self.sp_condition_flux,
                     self.sp_condition_h, self.sp_condition_fluid, self.sp_flow_lpm,
                     self.sp_pump_max_lpm, self.sp_pump_max_kpa, self.sp_inlet_c, self.sp_fill_z,
                     self.sp_bath_c, self.sp_pressure, self.sp_torque, *self.sp_force):
            spin.valueChanged.connect(self._on_condition_edited)
        for check in (self.chk_torque_reverse, self.chk_pin_axial):
            check.toggled.connect(self._on_condition_edited)
        for box in (self.cb_liquid_fluid, self.cb_bath_mode, self.cb_liquid_model, self.cb_coolant_flow,
                    self.cb_coolant_pump, self.cb_inlet_from):
            box.currentIndexChanged.connect(self._on_condition_edited)
        for check in (self.chk_reverse, self.chk_fill_auto, self.chk_bath_start_auto,
                      self.chk_evaporation):
            check.toggled.connect(self._on_condition_edited)

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
        tree_box = QWidget()
        tree_layout = QVBoxLayout(tree_box)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_layout.setSpacing(3)
        tree_layout.addWidget(self.tree, 1)
        self.btn_tree_remove = QPushButton("Remove selected")
        self.btn_tree_remove.setToolTip("Remove the selected results, parts, conditions or probes "
                                        "(Delete key)")
        self.btn_tree_remove.setEnabled(False)
        self.btn_tree_remove.clicked.connect(lambda _c=False: self._remove_selected())
        self.tree.itemSelectionChanged.connect(
            lambda: self.btn_tree_remove.setEnabled(bool(self.tree.selectedItems())))
        tree_layout.addWidget(self.btn_tree_remove)
        left.addWidget(tree_box)
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
        self.cb_mode.addItem("Structural (stress, frequencies)", "structural")
        self.cb_mode.setToolTip(
            "Conduction (FEM): heat conduction in the bodies; air cooling and radiation act on "
            "their surfaces (fast).\nAirflow + heat (CHT): the air around and inside the parts is "
            "simulated as well, with buoyancy, fans and vents (minutes).\nStructural: stresses and "
            "deformation under supports and loads, or the natural frequencies of the parts.")
        put(self.cb_mode, "Solver:")
        self.lbl_mode_hint = _hint("")
        put(self.lbl_mode_hint, span=True)
        self.cb_analysis = QComboBox()
        self.cb_analysis.addItem("Steady state", "steady")
        self.cb_analysis.addItem("Transient (over time)", "transient")
        self.cb_analysis.setToolTip(
            "Steady state: the final temperatures after a long time.\n"
            "Transient: how the temperatures develop after the heat sources are switched on.")
        self._thermal_analysis_rows = [self.cb_analysis, put(self.cb_analysis, "Analysis:")]
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
        # structural studies
        self._structural_rows = []
        self.cb_struct_analysis = QComboBox()
        for key, title in STRUCTURAL_ANALYSES.items():
            self.cb_struct_analysis.addItem(title, key)
        self.cb_struct_analysis.setToolTip(
            "Static: stresses, deformation and safety factor under the supports and loads.\n"
            "Natural frequencies: the lowest frequencies and mode shapes at which the parts "
            "vibrate (loads are not needed).")
        self._structural_rows += [self.cb_struct_analysis, put(self.cb_struct_analysis, "Analysis:")]
        self.sp_struct_modes = QSpinBox()
        self.sp_struct_modes.setRange(1, 30)
        self.sp_struct_modes.setKeyboardTracking(False)
        self.sp_struct_modes.setMinimumWidth(110)
        self.sp_struct_modes.setToolTip("How many of the lowest natural frequencies to find")
        self._struct_modal_rows = [self.sp_struct_modes, put(self.sp_struct_modes, "Frequencies:")]
        self.chk_struct_gravity = QCheckBox("Gravity (the parts' own weight, along −Z)")
        put(self.chk_struct_gravity, span=True)
        self.cb_struct_thermal = QComboBox()
        for key, title in STRUCTURAL_THERMAL.items():
            self.cb_struct_thermal.addItem(title, key)
        self.cb_struct_thermal.setToolTip(
            "Thermal expansion: none; all parts at one temperature; or the temperatures of a "
            "steady Conduction run of this study (its heat sources and thermal face conditions, "
            "run first).")
        self._struct_static_rows = [self.chk_struct_gravity, self.cb_struct_thermal,
                                    put(self.cb_struct_thermal, "Thermal expansion:")]
        self.sp_struct_uniform = _spin(-273.0, 3000.0, " °C", 5.0, decimals=1)
        self._struct_uniform_rows = [self.sp_struct_uniform, put(self.sp_struct_uniform, "Parts at:")]
        self.sp_struct_reference = _spin(-273.0, 3000.0, " °C", 1.0, decimals=1)
        self.sp_struct_reference.setToolTip("Temperature at which the parts are free of thermal stress "
                                            "(usually the assembly temperature)")
        self._struct_thermal_rows = [self.sp_struct_reference,
                                     put(self.sp_struct_reference, "Stress-free at:")]
        self.chk_struct_auto = QCheckBox("Automatic element size")
        put(self.chk_struct_auto, span=True)
        self.sp_struct_size = _spin(0.01, 1000.0, " mm", 0.1, decimals=3)
        self.sp_struct_size.setToolTip("Edge length of the tetrahedra; round faces get finer "
                                       "elements automatically")
        self.sp_struct_max = QSpinBox()
        self.sp_struct_max.setRange(1000, 2000000)
        self.sp_struct_max.setSingleStep(10000)
        self.sp_struct_max.setKeyboardTracking(False)
        self.sp_struct_max.setMinimumWidth(130)
        self.sp_struct_max.setToolTip("Element budget of the automatic size (the run stops above "
                                      "twice this number)")
        self.cb_struct_order = QComboBox()
        self.cb_struct_order.addItem("Quadratic (recommended)", 2)
        self.cb_struct_order.addItem("Linear (fast, too stiff in bending)", 1)
        self.lbl_struct_estimate = _hint("")
        self._structural_rows += [_header("Mesh (tetrahedra)")]
        put(self._structural_rows[-1], span=True)
        self._structural_rows += [self.chk_struct_auto, self.sp_struct_size,
                                  put(self.sp_struct_size, "Element size:"), self.sp_struct_max,
                                  put(self.sp_struct_max, "Max elements:"), self.cb_struct_order,
                                  put(self.cb_struct_order, "Elements:"), self.lbl_struct_estimate]
        put(self.lbl_struct_estimate, span=True)
        for widget in (self.cb_struct_analysis, self.cb_struct_thermal, self.cb_struct_order):
            widget.currentIndexChanged.connect(self._on_settings_edited)
        for widget in (self.sp_struct_modes, self.sp_struct_uniform, self.sp_struct_reference,
                       self.sp_struct_size, self.sp_struct_max):
            widget.valueChanged.connect(self._on_settings_edited)
        for widget in (self.chk_struct_gravity, self.chk_struct_auto):
            widget.toggled.connect(self._on_settings_edited)
        self._environment_rows = [_header("Environment")]
        put(self._environment_rows[0], span=True)
        self.sp_ambient = _spin(-100.0, 500.0, " °C", 1.0, decimals=1)
        self.sp_ambient.setToolTip("Temperature of the surrounding air and surfaces")
        self._environment_rows += [self.sp_ambient, put(self.sp_ambient, "Ambient:")]
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
        self._environment_rows += [self.chk_radiation, self.lbl_environment_hint]

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
                                *self.cb_sides.values(), self.cb_struct_analysis,
                                self.sp_struct_modes, self.chk_struct_gravity, self.cb_struct_thermal,
                                self.sp_struct_uniform, self.sp_struct_reference, self.chk_struct_auto,
                                self.sp_struct_size, self.sp_struct_max, self.cb_struct_order)

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
        self.w_struct_view = QWidget()
        struct_grid = QGridLayout(self.w_struct_view)
        struct_grid.setContentsMargins(0, 0, 0, 0)
        self.cb_struct_field = QComboBox()
        self.cb_struct_field.setToolTip("What the colours show")
        self.cb_struct_mode = QComboBox()
        self.cb_struct_mode.setToolTip("The natural frequency whose mode shape is shown")
        self.lbl_struct_mode = QLabel("Frequency:")
        self.chk_struct_deformed = QCheckBox("Deformed ×")
        self.chk_struct_deformed.setChecked(True)
        self.chk_struct_deformed.setToolTip("Show the deformed shape, exaggerated by this factor")
        self.sp_struct_scale = _spin(0.0, 1e12, "", 1.0, decimals=2, width=110)
        self.chk_struct_scale_auto = QCheckBox("auto")
        self.chk_struct_scale_auto.setChecked(True)
        self.btn_struct_animate = QPushButton("Animate")
        self.btn_struct_animate.setCheckable(True)
        self.btn_struct_animate.setToolTip("Move between the undeformed and the deformed shape (a "
                                           "mode shape swings both ways)")
        struct_grid.addWidget(QLabel("Shows:"), 0, 0)
        struct_grid.addWidget(self.cb_struct_field, 0, 1, 1, 3)
        struct_grid.addWidget(self.lbl_struct_mode, 1, 0)
        struct_grid.addWidget(self.cb_struct_mode, 1, 1, 1, 3)
        struct_grid.addWidget(self.chk_struct_deformed, 2, 0)
        struct_grid.addWidget(self.sp_struct_scale, 2, 1)
        struct_grid.addWidget(self.chk_struct_scale_auto, 2, 2)
        struct_grid.addWidget(self.btn_struct_animate, 2, 3)
        struct_grid.setColumnStretch(4, 1)
        self.cb_struct_field.currentIndexChanged.connect(self._on_struct_view_edited)
        self.cb_struct_mode.currentIndexChanged.connect(self._on_struct_view_edited)
        self.chk_struct_deformed.toggled.connect(self._on_struct_view_edited)
        self.chk_struct_scale_auto.toggled.connect(self._on_struct_view_edited)
        self.sp_struct_scale.valueChanged.connect(self._on_struct_view_edited)
        self.btn_struct_animate.toggled.connect(self._on_struct_animate)
        self._struct_timer = QTimer(self)
        self._struct_timer.setInterval(40)
        self._struct_timer.timeout.connect(self._struct_animation_step)
        self._struct_phase = 1.0
        self.w_struct_view.hide()
        results_layout.addWidget(self.w_struct_view)
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
        self.btn_flow_curve = QPushButton("Import curve CSV")
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
        entry = self._bound_entry()             # its entry in the Objects explorer, else the file
        name = entry["name"] if entry is not None else \
            os.path.basename(path) if path else THERMAL_DEFAULT_NAME
        self.setWindowTitle(f"Simulation — {name}" + (" *" if self.has_unsaved_changes() else ""))

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
            # a study that is not a file of its own appears in the Objects explorer as soon
            # as it has parts (any import, heat source or block), like an Assembly
            if self.study["components"] and not self._window._thermal_file:
                note = self._keep_in_project()
                if note:
                    self._set_status(self.status.text() + note)
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
        arrows = []
        for index, condition in enumerate(self.study["conditions"]):
            if condition["kind"] != "coolant" or not condition["faces"] or \
                    (not condition.get("enabled", True) and index != self._selected_condition):
                continue
            geometry = self._channel_geometry(condition)
            if geometry is None:
                continue
            length = max(3.0 * geometry["dh_mm"], 8.0)
            arrows.append((geometry["inlet"] - geometry["inlet_dir"] * length, geometry["inlet_dir"],
                           length, f"{condition['name']}: in"))
            arrows.append((geometry["outlet"], geometry["outlet_dir"], length, "out"))
        self.view.channel_arrows = arrows
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
        opened_runs = set()
        for index in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(index)
            for k in range(top.childCount()):
                data = top.child(k).data(0, Qt.UserRole + 2)
                if isinstance(data, str) and data.startswith("run:") and top.child(k).isExpanded():
                    opened_runs.add(int(data.split(":")[1]))
        try:
            self.tree.clear()
            # the parts are the top-level rows (no study-name row above them, as in the
            # Assembly window; the name is in the window title)
            root_item = self.tree.invisibleRootItem()
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
                    if condition["kind"] not in condition_kinds(self.study["settings"]["mode"]):
                        item.setForeground(0, QColor("#6b7380"))       # for the other study type
                        item.setToolTip(0, self._condition_text(condition) + "\n(used in " + (
                            "Structural" if condition["kind"] in STRUCTURAL_KINDS else "Thermal") +
                            " studies only)")
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
            if self._runs:
                group = QTreeWidgetItem([f"Results ({len(self._runs)})"])
                group.setFont(0, group_font)
                group.setIcon(0, _icon("fit_together.png"))
                group.setToolTip(0, "Finished runs of this session. Click one to show it again; tick "
                                    "the box to mark it for export; Delete removes the selected ones.")
                root_item.addChild(group)
                for run in reversed(self._runs):
                    item = QTreeWidgetItem([f"#{run['id']}   {run['time']}   {run['summary']}"])
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(0, Qt.Checked if run.get("marked") else Qt.Unchecked)
                    item.setData(0, Qt.UserRole + 2, f"run:{run['id']}")
                    item.setToolTip(0, f"Run #{run['id']}: {run['title']}, {run['seconds']:.1f} s")
                    group.addChild(item)
                    details = [f"Type: {run['title']}", f"Result: {run['summary']}",
                               f"Time: {run['time']}   Duration: {run['seconds']:.1f} s"]
                    for text in details:
                        child = QTreeWidgetItem([text])
                        child.setData(0, Qt.UserRole + 2, f"run:{run['id']}")
                        item.addChild(child)
                    params = QTreeWidgetItem([f"Parameters ({len(run['params'])})"])
                    params.setFont(0, group_font)
                    params.setData(0, Qt.UserRole + 2, f"run:{run['id']}")
                    item.addChild(params)
                    for name, value in run["params"]:
                        child = QTreeWidgetItem([f"{name}: {value}"])
                        child.setToolTip(0, f"{name}: {value}")
                        child.setData(0, Qt.UserRole + 2, f"run:{run['id']}")
                        params.addChild(child)
            self.tree.expandAll()
            if self._runs:                       # runs stay collapsed unless opened before
                for index in range(group.childCount()):
                    run_item = group.child(index)
                    run_id = int(run_item.data(0, Qt.UserRole + 2).split(":")[1])
                    run_item.setExpanded(run_id in opened_runs)
                    for k in range(run_item.childCount()):
                        if run_item.child(k).childCount():
                            run_item.child(k).setExpanded(False)
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
                 "coolant": lambda c: f"{c['fluid']}, " + (
                     f"{c['flow_lpm']:g} L/min" if c["flow_mode"] == "fixed" else "pump") +
                 f", in {c['inlet_from'] or format(c['inlet_c'], 'g') + ' °C'}" +
                 (", flow solved" if c["liquid_model"] == "flow" else ""),
                 "bath": lambda c: f"{c['fluid']}, " + (
                     f"{c['bath_c']:g} °C" if c["bath_mode"] == "fixed" else
                     "fills to the faces" if c["fill_auto"] else f"fill to Z {c['fill_z']:g} mm") +
                 (", flow solved" if c["liquid_model"] == "flow" and c["bath_mode"] != "fixed" else ""),
                 "insulated": lambda c: "",
                 "support_fixed": lambda c: "",
                 "support_roller": lambda c: "",
                 "support_pin": lambda c: "slides along the axis" if c["pin_axial_free"] else "",
                 "force": lambda c: f"({c['fx']:g}, {c['fy']:g}, {c['fz']:g}) N",
                 "pressure": lambda c: f"{c['pressure_mpa']:g} MPa",
                 "torque": lambda c: f"{c['torque_nm']:g} N·m" + (", reversed" if c["torque_reverse"]
                                                                  else "")}[kind](condition)
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
        if kind == "run":
            run = self._run(index)
            if run is not None and self._result is not run["result"] and \
                    len(self.tree.selectedItems()) <= 1:
                self._show_run(index)
            return
        component, body_index = self._item_target(self.tree.currentItem())
        self._select(component, body_index, update_tree=False)

    def _on_tree_item_changed(self, item, _column):
        if self._loading:
            return
        kind, index = self._item_extra(item)
        if kind == "run":
            if item.flags() & Qt.ItemIsUserCheckable:
                self._set_run_marked(index, item.checkState(0) == Qt.Checked, from_tree=True)
            return
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

    def _selected_items(self):
        """Selected tree rows as (kind, index) with kind condition / contact / probe / part."""
        found = []
        for item in self.tree.selectedItems():
            kind, index = self._item_extra(item)
            if kind == "run":
                if ("run", index) not in found:
                    found.append(("run", index))
                continue
            if kind is not None:
                found.append((kind, index))
                continue
            component, body_index = self._item_target(item)
            if component is not None and body_index is None:
                found.append(("part", component))
        return found

    def _remove_many(self, rows):
        if self._worker is not None:
            return
        run_ids = [r[1] for r in rows if r[0] == "run"]
        rows = [r for r in rows if r[0] != "run"]
        if run_ids:
            self._delete_runs(run_ids)
            if not rows:
                return
        parts = [r[1] for r in rows if r[0] == "part"]
        if parts and QMessageBox.question(
                self, "Remove", f"Remove {len(parts)} part(s) from the study?\nProject objects and files "
                "are not changed. Their faces leave the face conditions and their probes are removed.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        if self.view.pick_mode is not None:
            self._set_pick_mode(None)
        for kind, key in (("probe", "probes"), ("condition", "conditions"), ("contact", "contacts")):
            for index in sorted({r[1] for r in rows if r[0] == kind}, reverse=True):
                if 0 <= index < len(self.study[key]):
                    del self.study[key][index]
        self._selected_condition = self._selected_probe = self._selected_contact = None
        if parts:
            for component in parts:
                drop_component_refs(self.study, component["name"])
            self.study["components"] = [c for c in self.study["components"]
                                        if not any(c is q for q in parts)]
            self._selected, self._selected_body = None, None
            self._after_components_changed()
        else:
            self._rebuild_tree()
            self._show_properties()
            self._refresh_overlays()
        self._fill_probe_table()
        self._changed(results=True)
        self._update_plot()
        self._set_status(f"Removed {len(rows)} items")

    def _on_tree_rclick(self, pos):
        item = self.tree.itemAt(pos)
        rows = self._selected_items()
        if item is not None and self._item_extra(item)[0] == "run":
            if not item.isSelected():
                self.tree.setCurrentItem(item)
                rows = self._selected_items()
            run_rows = [r for r in rows if r[0] == "run"]
            menu = QMenu(self)
            show = menu.addAction("Show") if len(run_rows) == 1 else None
            export = menu.addAction("Export results CSV")
            compare = menu.addAction("Export comparison CSV")
            history = menu.addAction("Run history")
            menu.addSeparator()
            remove = menu.addAction(f"Remove {len(run_rows)} selected" if len(run_rows) > 1 else "Remove")
            chosen = menu.exec_(self.tree.viewport().mapToGlobal(pos))
            ids = self._marked_ids() or [r[1] for r in run_rows]
            if chosen is None:
                return
            if chosen is show:
                self._show_run(run_rows[0][1])
            elif chosen is export:
                self._export_runs("csv", ids)
            elif chosen is compare:
                self._export_runs("summary", ids)
            elif chosen is history:
                self._open_run_history()
            elif chosen is remove:
                self._remove_many(rows)
            return
        if item is not None and item.isSelected() and len(self.tree.selectedItems()) > 1 and rows:
            menu = QMenu(self)
            remove = menu.addAction(f"Remove {len(rows)} selected")
            if menu.exec_(self.tree.viewport().mapToGlobal(pos)) is remove:
                self._remove_many(rows)
            return
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
            rename = menu.addAction("Rename")
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
        power = menu.addAction("Set power") if self._body() is not None and \
            component.get("kind") not in FLOW_KINDS else None
        rename = menu.addAction("Rename")
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
                root_item = self.tree.invisibleRootItem()
                item = find(root_item) if component is not None else None
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
            self.w_package.setVisible(False)
            self.lbl_board.setVisible(False)
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
            package = component.get("package") if component is not None else None
            self.w_package.setVisible(package is not None)
            if package is not None:
                self.lbl_package.setText(package["type"] + ("  (under the board)"
                                                            if package.get("side") == "bottom" else ""))
                self.sp_theta_top.setValue(package["theta_top"])
                self.sp_theta_board.setValue(package["theta_board"])
                self.sp_tj_max.setValue(package["tj_max"])
                self.lbl_power.setText("Power at the junction" if power > 0 else "No power")
            pcb = component.get("pcb") if component is not None else None
            self.lbl_board.setVisible(pcb is not None)
            if pcb is not None:
                self.lbl_board.setText("Board: " + pcb_summary(pcb) + ". Its conductivity follows the "
                                       "copper of every layer and the plated holes; the power is "
                                       "the heat of the parts not placed.")
            own = body.get("emissivity") is not None
            self.chk_emissivity.setChecked(own)
            self.sp_emissivity.setEnabled(own)
            props = material_properties(material)
            self.sp_emissivity.setValue(body["emissivity"] if own else
                                        props["emissivity"] if props else 0.9)
        finally:
            self._loading = False

    def _on_package_edited(self, *_args):
        component = self._selected
        if self._loading or component is None or component.get("package") is None:
            return
        component["package"].update(theta_top=float(self.sp_theta_top.value()),
                                    theta_board=float(self.sp_theta_board.value()),
                                    tj_max=float(self.sp_tj_max.value()))
        self._changed()

    def _import_pcb(self):
        if self._worker is not None:
            return
        dialog = PcbImportDialog(self, self._start_directory())
        if dialog.exec_() != QDialog.Accepted:
            return
        pcb, parts, messages = dialog.values()
        name = os.path.splitext(pcb.get("source") or "PCB")[0] or "PCB"
        board = study_add_pcb(self.study, pcb, name)
        top, bottom = pcb["thickness_mm"], 0.0
        added = [board]
        for reference, package, power, x, y, angle, size in parts:
            added.append(study_add_package(self.study, package, reference, x, y, top, bottom, power,
                                           size, angle))
        self._added(added[:1], f"Imported the board {pcb_summary(pcb)}" +
                    (f" and {len(parts)} component{'s' if len(parts) != 1 else ''}" if parts else ""))
        self._after_components_changed(select=board)
        self.view.fit_all()
        for message in messages:
            self._append_note(message)

    def _add_package(self):
        if self._worker is not None:
            return
        number = 1 + sum(1 for c in self.study["components"] if c.get("package") is not None)
        dialog = PackageDialog(self, f"U{number}")
        if dialog.exec_() != QDialog.Accepted:
            return
        name, package, power = dialog.values()
        board = self._selected if self._selected is not None and self._selected.get("pcb") else \
            next((c for c in self.study["components"] if c.get("pcb") is not None), None)
        if board is not None:
            box = cad.assembly_bounds(board)
            x, y = (box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2
            top, bottom = box[1][2], box[0][2]
        else:
            x = y = top = bottom = 0.0
        component = study_add_package(self.study, package, name, x, y, top, bottom, power)
        self._after_components_changed(select=component)
        self._set_status(f"Placed {component['name']} ({package_text(package)}, {power:g} W) — move it "
                         "with the arrows or the position fields")

    def _append_note(self, text):
        notes = self.txt_messages.toPlainText().strip()
        self.txt_messages.setPlainText((notes + "\n" if notes else "") + text)

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
                if result.get("mode") == "structural":
                    caps.append(self._structural_cap(entry, key, vertices, triangles, result, low_high,
                                                      (ci, bi)))
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
    def _fill_condition_menu(self):
        """The Face Condition menu lists the kinds of the current solver (Thermal or Structural)."""
        self.condition_menu.clear()
        mode = self.study["settings"]["mode"] if getattr(self, "study", None) else "conduction"
        for kind in condition_kinds(mode):
            if mode == "structural" and kind == "force":
                self.condition_menu.addSeparator()
            self.condition_menu.addAction(f"{CONDITION_KINDS[kind][0]}", lambda k=kind: self._add_condition(k))

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
        condition["inlet_c"] = float(self.study["settings"]["ambient_c"])
        condition["bath_c"] = float(self.study["settings"]["ambient_c"])
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
                        widget.setVisible(rows_kind == kind or
                                          (rows_kind == "liquid" and kind in LIQUID_KINDS))
            if kind in LIQUID_KINDS:
                self._show_liquid(condition)
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
            for spin, key in zip(self.sp_force, ("fx", "fy", "fz")):
                spin.setValue(condition[key])
            force = math.sqrt(condition["fx"] ** 2 + condition["fy"] ** 2 + condition["fz"] ** 2)
            self.lbl_force.setText(f"{force:.4g} N")
            self.sp_pressure.setValue(condition["pressure_mpa"])
            self.sp_torque.setValue(condition["torque_nm"])
            self.chk_torque_reverse.setChecked(condition["torque_reverse"])
            self.chk_pin_axial.setChecked(condition["pin_axial_free"])
            self.lbl_condition_hint.setText(CONDITION_HINTS[kind])
            mode = self.study["settings"]["mode"]
            other = (kind in STRUCTURAL_KINDS) != (mode == "structural")
            self.lbl_pick_hint.setText(
                ("⚠ This condition belongs to a " + ("Structural" if kind in STRUCTURAL_KINDS else "Thermal") +
                 " study: the current solver leaves it out.\n" if other else "") +
                "Click faces to add them; click a chosen face again to remove it. Esc or the button "
                "ends picking. " + ("Faces without a condition are free (no support, no load)."
                                    if kind in STRUCTURAL_KINDS else
                                    "Faces without a condition exchange heat with the environment "
                                    "(Simulation panel)."))
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
        if condition["kind"] in STRUCTURAL_KINDS:
            for spin, key in zip(self.sp_force, ("fx", "fy", "fz")):
                condition[key] = float(spin.value())
            condition["pressure_mpa"] = float(self.sp_pressure.value())
            condition["torque_nm"] = float(self.sp_torque.value())
            condition["torque_reverse"] = self.chk_torque_reverse.isChecked()
            condition["pin_axial_free"] = self.chk_pin_axial.isChecked()
        if condition["kind"] in LIQUID_KINDS:
            condition["fluid"] = self.cb_liquid_fluid.currentData()
            condition["liquid_model"] = self.cb_liquid_model.currentData()
            if condition["kind"] == "coolant":
                condition["flow_mode"] = self.cb_coolant_flow.currentData()
                condition["flow_lpm"] = float(self.sp_flow_lpm.value())
                self._take_coolant_pump(condition, self.cb_coolant_pump.currentData() or "")
                condition["pump_max_lpm"] = float(self.sp_pump_max_lpm.value())
                condition["pump_max_kpa"] = float(self.sp_pump_max_kpa.value())
                condition["inlet_from"] = self.cb_inlet_from.currentData() or ""
                condition["inlet_c"] = float(self.sp_inlet_c.value())
                condition["reverse"] = self.chk_reverse.isChecked()
            else:
                condition["bath_mode"] = self.cb_bath_mode.currentData()
                condition["fill_auto"] = self.chk_fill_auto.isChecked()
                condition["fill_z"] = float(self.sp_fill_z.value())
                condition["bath_c"] = float(self.sp_bath_c.value())
                condition["bath_start_auto"] = self.chk_bath_start_auto.isChecked()
                condition["evaporation"] = self.chk_evaporation.isChecked()
        self._show_condition()
        self._rebuild_tree()
        self._refresh_overlays()
        self._changed()

    def _take_coolant_pump(self, condition, source, pumps=None, quiet=False):
        """Use the curve of the project pump `source` for a coolant channel ("" = the straight
        line): its estimate at its motor speed is copied into the condition. quiet: only update
        the copy (when the pump was edited) and record that as a change of the study."""
        if not source:
            if condition.get("pump_source"):
                condition.update(pump_source="", pump_curve=[], pump_rpm=0.0, pump_density=0.0)
            return
        pumps = project_pumps(self._window.state["objects"]) if pumps is None else pumps
        if source not in pumps:                         # gone from the project: keep the copy
            condition["pump_source"] = source
            return
        try:
            table, rpm = project_pump_curve(pumps[source])
        except (ValueError, ArithmeticError):
            return
        new = (source, table, rpm, 998.0)
        if (condition.get("pump_source"), condition.get("pump_curve"), condition.get("pump_rpm"),
                condition.get("pump_density")) == new:
            return
        condition.update(pump_source=source, pump_curve=table, pump_rpm=rpm, pump_density=998.0)
        if quiet:
            self._changed()

    def _channel_geometry(self, condition):
        """Shape of a coolant channel from its picked faces (cached), or None."""
        cache = self.__dict__.setdefault("_channel_cache", {})
        components = {c["name"]: c for c in self.study["components"]}
        used = sorted({ref["component"] for ref in condition["faces"]})
        key = repr(([(r["component"], r["body"], r["face"]) for r in condition["faces"]],
                    bool(condition.get("reverse")),
                    [(name, components[name].get("position"), components[name].get("rotation"),
                      components[name].get("pivot")) for name in used if name in components]))
        if key not in cache:
            if len(cache) > 32:
                cache.clear()
            try:
                tris = condition_world_triangles(self.study, condition)
                cache[key] = channel_geometry(tris, condition.get("reverse", False)) if len(tris) else None
            except Exception as ex:                       # odd faces: say so in the panel
                cache[key] = str(ex)
        value = cache[key]
        return value if isinstance(value, dict) else None

    def _show_liquid(self, condition):
        """The coolant / bath rows of the condition page (inside _show_condition)."""
        kind = condition["kind"]
        self.cb_liquid_fluid.setCurrentIndex(max(0, self.cb_liquid_fluid.findData(condition["fluid"])))
        self.cb_liquid_model.setCurrentIndex(max(0, self.cb_liquid_model.findData(condition["liquid_model"])))
        if kind == "coolant":
            pump = condition["flow_mode"] == "pump"
            self.cb_coolant_flow.setCurrentIndex(1 if pump else 0)
            self.sp_flow_lpm.setValue(condition["flow_lpm"])
            self.sp_pump_max_lpm.setValue(condition["pump_max_lpm"])
            self.sp_pump_max_kpa.setValue(condition["pump_max_kpa"])
            pumps = project_pumps(self._window.state["objects"])
            source = condition.get("pump_source") or ""
            if source and source in pumps:            # follow the pump if it was edited since
                self._take_coolant_pump(condition, source, pumps, quiet=True)
            self.cb_coolant_pump.clear()
            self.cb_coolant_pump.addItem("Straight line", "")
            for name in sorted(pumps):
                self.cb_coolant_pump.addItem(name, name)
            if source and source not in pumps:
                self.cb_coolant_pump.addItem(f"{source} (no longer in the project)", source)
            self.cb_coolant_pump.setCurrentIndex(max(0, self.cb_coolant_pump.findData(source)))
            if source and condition.get("pump_curve"):
                table = condition["pump_curve"]
                self.lbl_pump_source.setText(
                    f"pump tool estimate at {condition['pump_rpm']:g} rpm: {table[-1][0]:.3g} L/min free, "
                    f"{table[0][1]:.3g} kPa shut-off (water; scaled to the liquid's density)")
            for widget in (self.sp_flow_lpm, self._cap_flow_lpm):
                widget.setVisible(not pump)
            for widget in (self.cb_coolant_pump, self._cap_coolant_pump):
                widget.setVisible(pump)
            for widget in (self.lbl_pump_source, self._cap_pump_source):
                widget.setVisible(pump and bool(source))
            for widget in (self.sp_pump_max_lpm, self._cap_pump_lpm, self.sp_pump_max_kpa,
                           self._cap_pump_kpa):
                widget.setVisible(pump and not source)
            self.cb_inlet_from.clear()
            self.cb_inlet_from.addItem("Own temperature", "")
            for other in self.study["conditions"]:
                if other is not condition and other["kind"] == "coolant":
                    self.cb_inlet_from.addItem(f"Outlet of '{other['name']}'", other["name"])
            self.cb_inlet_from.setCurrentIndex(max(0, self.cb_inlet_from.findData(condition["inlet_from"])))
            self.sp_inlet_c.setValue(condition["inlet_c"])
            for widget in (self.sp_inlet_c, self._cap_inlet_c):
                widget.setVisible(not condition["inlet_from"])
            self.chk_reverse.setChecked(condition["reverse"])
            if not condition["faces"]:
                self.lbl_channel.setText("pick the inner faces of the channel")
            else:
                geometry = self._channel_geometry(condition)
                if geometry is None:
                    cache = self.__dict__.get("_channel_cache", {})
                    problem = next((v for v in cache.values() if isinstance(v, str)), "")
                    self.lbl_channel.setText(problem or "the faces do not form a channel")
                else:
                    inlet = geometry["inlet"]
                    self.lbl_channel.setText(f"{channel_text(geometry)}; inlet at ({inlet[0]:.1f}, "
                                             f"{inlet[1]:.1f}, {inlet[2]:.1f}) mm")
        else:
            container = condition["bath_mode"] == "container"
            self.cb_bath_mode.setCurrentIndex(0 if container else 1)
            for widget in (self.cb_liquid_model, self._cap_liquid_model, self.chk_fill_auto,
                           self.sp_fill_z, self._cap_fill_z, self.chk_bath_start_auto,
                           self.chk_evaporation):
                widget.setVisible(container)
            self.chk_fill_auto.setChecked(condition["fill_auto"])
            top = condition_world_triangles(self.study, condition)
            self.sp_fill_z.setValue(float(top[:, :, 2].max()) if condition["fill_auto"] and len(top)
                                    else condition["fill_z"])
            self.sp_fill_z.setEnabled(not condition["fill_auto"])
            self.chk_bath_start_auto.setChecked(condition["bath_start_auto"])
            self.chk_evaporation.setChecked(condition["evaporation"])
            self.sp_bath_c.setValue(condition["bath_c"])
            self._cap_bath_c.setText("Liquid at:" if not container else "Starts at:")
            show_temperature = not container or not condition["bath_start_auto"]
            for widget in (self.sp_bath_c, self._cap_bath_c):
                widget.setVisible(show_temperature)

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

    # ── structural results ────────────────────────────────────────────────────
    def _thermal_result_layout(self):
        """Results table, colour range and controls of a Thermal result."""
        self.w_struct_view.hide()
        self._struct_timer.stop()
        self.tbl_results.setHorizontalHeaderLabels(["Body", "Material", "W", "Min °C", "Mean °C",
                                                    "Max °C"])
        self.tbl_probes.setHorizontalHeaderLabels(["Probe", "On part", "°C"])
        for spin in (self.sp_range_low, self.sp_range_high):
            spin.blockSignals(True)
            spin.setRange(-273.0, 5000.0)
            spin.setSuffix(" °C")
            spin.blockSignals(False)

    def _show_structural_result(self):
        result = self._result
        modal = result["analysis"] == "modal"
        self._frame = 0
        self._struct_phase = 1.0
        for widget in (self.cb_struct_field, self.cb_struct_mode):
            widget.blockSignals(True)
        try:
            field = self.cb_struct_field.currentData()
            self.cb_struct_field.clear()
            if modal:
                self.cb_struct_field.addItem("Mode shape (relative displacement)", "disp")
            else:
                for key, (title, _legend, _unit) in STRUCT_FIELDS.items():
                    self.cb_struct_field.addItem(title, key)
                self.cb_struct_field.setCurrentIndex(max(0, self.cb_struct_field.findData(field)))
            self.cb_struct_mode.clear()
            for k, frequency in enumerate(result["frequencies"] if modal else []):
                self.cb_struct_mode.addItem(f"{k + 1}:  {frequency:.4g} Hz")
        finally:
            for widget in (self.cb_struct_field, self.cb_struct_mode):
                widget.blockSignals(False)
        self.cb_struct_mode.setVisible(modal)
        self.lbl_struct_mode.setVisible(modal)
        self.w_struct_view.show()
        self.tbl_results.blockSignals(True)
        self.tbl_results.setHorizontalHeaderLabels(["Body", "Material", "kg", "Max |u| mm",
                                                    "Max σv MPa", "Safety"])
        self.tbl_probes.setHorizontalHeaderLabels(["Probe", "On part", "|u|" if modal else "σv MPa"])
        self.tbl_results.setRowCount(len(result["bodies"]))
        for row, body in enumerate(result["bodies"]):
            safety = body["safety"]
            texts = [body["name"], body["material"] or "", f"{body['mass_kg']:.4g}",
                     "" if modal else f"{body['disp_max']:.4g}", "" if modal else f"{body['vm_max']:.4g}",
                     "" if modal or safety is None else f"{safety:.3g}"]
            for column, text in enumerate(texts):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, body["key"])
                if column >= 2:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl_results.setItem(row, column, item)
            if safety is not None and not modal:
                colour = "#ff6b6b" if safety < 1.0 else "#ffb454" if safety < 1.5 else "#e2e5e9"
                self.tbl_results.item(row, 5).setForeground(QColor(colour))
                self.tbl_results.item(row, 5).setToolTip(
                    f"{'Strength' if body['brittle'] else 'Yield strength'} {body['yield']:g} MPa / "
                    f"max von Mises {body['vm_max']:.4g} MPa")
        self.tbl_results.blockSignals(False)
        self.w_playback.hide()
        self.lbl_plot.hide()
        self.plot.hide()
        notes = self.txt_messages.toPlainText().strip()
        self.txt_messages.setPlainText("\n".join(result["summary"] + [""] + result["messages"] +
                                                 ([notes] if notes else [])))
        self.act_streamlines.setEnabled(False)
        for spin in (self.sp_range_low, self.sp_range_high):
            spin.blockSignals(True)
            spin.setRange(-1e12, 1e12)
            spin.blockSignals(False)
        self._show_structural_frame()
        self.act_results.setEnabled(True)
        self.act_results.setChecked(True)
        self._on_results_toggled(True)
        self._sync_section_ui()

    def _structural_scale(self):
        """Exaggeration of the deformed shape: automatic = the largest displacement shows as
        about 8 % of the model size (never less than the true shape)."""
        result = self._result
        if not self.chk_struct_deformed.isChecked():
            return 0.0
        if self.chk_struct_scale_auto.isChecked():
            points = result["volume"]["points"]
            size = float(np.ptp(points, axis=0).max()) if len(points) else 1.0
            largest = 1.0 if result["analysis"] == "modal" else result["disp_max"]
            scale = _nice(max(1.0, 0.08 * size / max(largest, 1e-30))) if largest > 0 else 1.0
            if result["analysis"] == "modal":
                scale = 0.08 * size
            self.sp_struct_scale.blockSignals(True)
            self.sp_struct_scale.setValue(scale)
            self.sp_struct_scale.blockSignals(False)
            self.sp_struct_scale.setEnabled(False)
            return scale
        self.sp_struct_scale.setEnabled(True)
        return float(self.sp_struct_scale.value())

    def _structural_range(self):
        result = self._result
        field = self.cb_struct_field.currentData() or "vm"
        if result["analysis"] == "modal":
            field = "disp"
        mode = max(0, self.cb_struct_mode.currentIndex())
        suffix = STRUCT_FIELDS[field][2]
        for spin in (self.sp_range_low, self.sp_range_high):
            if spin.suffix() != suffix:
                spin.blockSignals(True)
                spin.setSuffix(suffix)
                spin.blockSignals(False)
        if self.chk_auto_range.isChecked():
            values = [structural_values(result, item, field, mode) for item in result["surface"].values()]
            values = np.concatenate(values) if values else np.zeros(1)
            finite = values[np.isfinite(values)]
            if field == "safety":
                low, high = 0.0, float(min(5.0, max(2.0, np.max(finite) if len(finite) else 5.0)))
            else:
                low, high = 0.0, float(finite.max()) if len(finite) else 1.0
            if high - low < 1e-12:
                high = low + 1.0
            for spin, value in ((self.sp_range_low, low), (self.sp_range_high, high)):
                spin.blockSignals(True); spin.setValue(value); spin.blockSignals(False)
        else:
            low, high = self.sp_range_low.value(), self.sp_range_high.value()
            if high <= low:
                high = low + 1e-6
        self.sp_range_low.setEnabled(not self.chk_auto_range.isChecked())
        self.sp_range_high.setEnabled(not self.chk_auto_range.isChecked())
        return low, high

    def _structural_colors(self, values, low, high, field):
        if field == "safety":               # small safety factors red
            return temperature_colors(-np.minimum(values, high), -high, -low)
        return temperature_colors(values, low, high)

    def _apply_structural_colors(self):
        result = self._result
        low, high = self._color_range()
        modal = result["analysis"] == "modal"
        field = "disp" if modal else (self.cb_struct_field.currentData() or "vm")
        mode = max(0, self.cb_struct_mode.currentIndex())
        scale = self._structural_scale() * self._struct_phase
        strength = {body["key"]: body["yield"] for body in result["bodies"]}
        for ci, component in enumerate(self.study["components"]):
            for bi, body in enumerate(component["bodies"]):
                entry = body.get("_render")
                if entry is None:
                    continue
                item = result["surface"].get((ci, bi))
                if item is None:
                    self.view.set_entry_colors(entry, None)
                    continue
                item["yield"] = strength.get((ci, bi))
                disp = item["modes"][mode] if modal else item["disp"]
                values = structural_values(result, item, field, mode)
                vertices = item["vertices"] + scale * disp if scale else item["vertices"]
                self.view.set_entry_colors(entry, self._structural_colors(values, low, high, field),
                                           (vertices, item["triangles"]))
        title = "|u|  (relative)" if modal else STRUCT_FIELDS[field][1]
        if field == "safety":
            self.legend.set_range(high, low, "", title)
        else:
            self.legend.set_range(low, high, "", title)
        self.view.update()

    def _structural_cap(self, entry, key, vertices, triangles, result, low_high, body_key):
        """A section cap of a structural result: values and displacement of the nearest node."""
        cached = entry.get("_cap_weights")
        if cached is None or cached[0] != (key, id(result)):
            fine, fine_tris = refine_triangles(vertices, triangles, max(result["element_size"] / 2.0, 1e-3))
            tree = result.get("_node_tree")
            if tree is None:
                from scipy.spatial import cKDTree
                tree = result["_node_tree"] = cKDTree(result["volume"]["points"])
            _d, nearest = tree.query(fine)
            cached = ((key, id(result)), fine, fine_tris, nearest)
            entry["_cap_weights"] = cached
        _k, fine, fine_tris, nearest = cached
        volume = result["volume"]
        modal = result["analysis"] == "modal"
        mode = max(0, self.cb_struct_mode.currentIndex())
        field = "disp" if modal else (self.cb_struct_field.currentData() or "vm")
        disp = (volume["modes"][mode] if modal else volume["disp"])[nearest]
        if field == "disp":
            values = np.linalg.norm(disp, axis=1)
        else:
            values = volume["vm"][nearest]
            if field == "safety":
                owner = next((b for b in result["bodies"] if tuple(b["key"]) == tuple(body_key)), None)
                strength = owner["yield"] if owner else None
                values = strength / np.maximum(values, 1e-12) if strength else np.full(len(values), np.inf)
        scale = self._structural_scale() * self._struct_phase
        return (entry, fine + scale * disp if scale else fine, fine_tris,
                self._structural_colors(values, *low_high, field))

    def _show_structural_frame(self):
        result = self._result
        modal = result["analysis"] == "modal"
        self._apply_result_colors()
        if modal:
            self.view.hot_marker = None
            frequencies = result["frequencies"]
            mode = max(0, self.cb_struct_mode.currentIndex())
            text = (f"Natural frequency {mode + 1}: <b>{frequencies[mode]:.4g} Hz</b>" if len(frequencies)
                    else "No natural frequency found")
            if len(frequencies) > 1:
                text += "<br>All: " + ", ".join(f"{f:.4g}" for f in frequencies) + " Hz"
        else:
            hot = max(result["bodies"], key=lambda b: b["vm_max"])
            self.view.hot_marker = (np.asarray(hot["vm_point"], float), f"max {hot['vm_max']:.4g} MPa")
            text = (f"Max von Mises <b>{hot['vm_max']:.4g} MPa</b> in {hot['name']} · max "
                    f"displacement {result['disp_max']:.4g} mm")
            safety = [b["safety"] for b in result["bodies"] if b["safety"] is not None]
            if safety:
                lowest = min(safety)
                colour = "#ff6b6b" if lowest < 1.0 else "#ffb454" if lowest < 1.5 else "#9be29b"
                text += f" · safety factor <span style='color:{colour}'><b>{lowest:.3g}</b></span>"
            lines = [f"{entry['name']}: {np.linalg.norm(entry['force']):.4g} N"
                     for entry in result["reactions"][:4]]
            if lines:
                text += "<br>Reactions: " + " · ".join(lines)
            text += f"<br>Force balance {result['balance_error']:.1e}"
        if self._result_note:
            text = f"<span style='color:#80c8ff'>{self._result_note}</span><br>" + text
        self.lbl_result.setText(text)
        self._fill_probe_table()
        self._refresh_overlays()
        if self._selected_probe is not None:
            self._show_probe()
        self._schedule_section()
        self.view.update()

    def _on_struct_view_edited(self, *_args):
        if self._result is None or self._result.get("mode") != "structural":
            return
        self._show_structural_frame()

    def _on_struct_animate(self, on):
        if on and self._result is not None and self._result.get("mode") == "structural":
            if not self.act_results.isChecked():
                self.act_results.setChecked(True)
            self._struct_clock = time.perf_counter()
            self._struct_timer.start()
        else:
            self._struct_timer.stop()
            self._struct_phase = 1.0
            if self._result is not None and self._result.get("mode") == "structural":
                self._apply_result_colors()
                self._schedule_section()

    def _struct_animation_step(self):
        if self._result is None or self._result.get("mode") != "structural":
            self._struct_timer.stop()
            return
        elapsed = time.perf_counter() - getattr(self, "_struct_clock", time.perf_counter())
        wave = math.sin(2.0 * math.pi * elapsed / 1.6)
        # a mode shape swings both ways; a static deformation grows from the unloaded shape
        self._struct_phase = wave if self._result["analysis"] == "modal" else 0.5 * (1.0 - math.cos(
            2.0 * math.pi * elapsed / 1.6))
        self._apply_result_colors()

    def _check_structural(self, jobs):
        """What a Structural run needs: the packages, elastic data, supports and loads."""
        missing = structural_missing_text()
        if missing:
            QMessageBox.warning(self, "Structural", missing.replace(" ⚠ ", ""))
            return False
        jobs = [job for job in jobs if not job.get("zone")]
        lacking = [job for job in jobs if not (job["props"] or {}).get("E")]
        if lacking:
            ci, bi = lacking[0]["key"]
            self._select(self.study["components"][ci], bi)
            names = "\n".join("•  " + f"{job['name']} ({job['material']})" for job in lacking[:10])
            QMessageBox.warning(self, "Mechanical data missing",
                                f"These bodies have materials without mechanical data (Young's "
                                f"modulus):\n\n{names}\n\nChoose another material, or add E, ν and "
                                "the yield strength to your material in Materials.")
            return False
        settings = self.study["settings"]
        if settings["struct_analysis"] == "modal":
            return True
        conditions = study_conditions(self.study, jobs, STRUCTURAL_KINDS)
        if not any(c["kind"].startswith("support") and c["targets"] for c in conditions):
            QMessageBox.warning(self, "No support",
                                "Nothing holds the parts. Put a support on faces (Face Condition → "
                                "Fixed support, Sliding support or Pinned support).")
            return False
        if not any(c["kind"] in ("force", "pressure", "torque") and c["targets"] for c in conditions) \
                and not settings["struct_gravity"] and settings["struct_thermal"] == "none":
            QMessageBox.warning(self, "No load",
                                "There is no load. Put a force, pressure or torque on faces (Face "
                                "Condition), or switch on gravity or thermal expansion.")
            return False
        return True

    def _write_structural_csv(self, path):
        result = self._result
        modal = result["analysis"] == "modal"
        rows = [["Structural study", self.study["name"]],
                ["Analysis", "Natural frequencies" if modal else "Linear static"],
                ["Elements", result["elements"]], ["Element size mm", f"{result['element_size']:g}"], []]
        if modal:
            rows.append(["Mode", "Frequency Hz"])
            rows += [[k + 1, f"{f:.6g}"] for k, f in enumerate(result["frequencies"])]
        else:
            rows.append(["Body", "Material", "Mass kg", "Max displacement mm", "Max von Mises MPa",
                         "Yield / strength MPa", "Safety factor"])
            for body in result["bodies"]:
                rows.append([body["name"], body["material"] or "", f"{body['mass_kg']:.5g}",
                             f"{body['disp_max']:.5g}", f"{body['vm_max']:.5g}",
                             "" if not body["yield"] else f"{body['yield']:g}",
                             "" if body["safety"] is None else f"{body['safety']:.4g}"])
            rows += [[], ["Support", "Kind", "Fx N", "Fy N", "Fz N", "Mx N·m", "My N·m", "Mz N·m"]]
            for entry in result["reactions"]:
                rows.append([entry["name"], CONDITION_KINDS[entry["kind"]][0]] +
                            [f"{v:.5g}" for v in entry["force"]] + [f"{v:.5g}" for v in entry["moment"]])
            rows += [[], ["Load", "Kind", "Fx N", "Fy N", "Fz N"]]
            for entry in result["loads"]:
                rows.append([entry["name"], CONDITION_KINDS[entry["kind"]][0]] +
                            [f"{v:.5g}" for v in entry["resultant"]])
            rows += [[], ["Force balance", f"{result['balance_error']:.3e}"]]
        rows += [[], ["Probe", "On part", "Von Mises MPa", "Displacement mm"]]
        for probe in self.study["probes"]:
            point = probe_world_point(self.study, probe)
            if point is None:
                continue
            vm, disp = structural_point(result, point)
            rows.append([probe["name"], probe.get("component") or "", "" if modal else f"{vm:.5g}",
                         f"{np.linalg.norm(disp):.5g}"])
        try:
            write_csv(path, rows)
        except OSError as ex:
            QMessageBox.critical(self, "Export failed", f"Could not write\n{path}\n\n{ex}")
            return
        self._set_status(f"Results saved to {path}")

    def _export_structural_vtu(self):
        path = self._export_path("Export for ParaView", ".vtu", "VTK unstructured grid (*.vtu)")
        if not path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            import simplecad_structural as structural
            structural.export_vtu(self._result, path)
        except (OSError, ValueError, MemoryError) as ex:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Export failed", f"Could not write\n{path}\n\n{ex}")
            return
        QApplication.restoreOverrideCursor()
        self._set_status(f"Saved for ParaView: {path} ({os.path.getsize(path) / 1e6:.1f} MB). In "
                         "ParaView: open it, Apply, colour by VonMises; the Warp By Vector filter "
                         "shows the deformed shape.")

    def _structural_report(self):
        """(HTML, images) of the PDF report of a Structural study."""
        import html as html_module
        import datetime
        esc = html_module.escape
        result = self._result
        settings = self.study["settings"]
        modal = settings["struct_analysis"] == "modal" if result is None else result["analysis"] == "modal"
        out = ["<html><head><style>"
               "body { font-family: Arial; font-size: 9pt; color: #202428; }"
               "h1 { font-size: 17pt; color: #17324d; margin-bottom: 2px; }"
               "h2 { font-size: 12pt; color: #17324d; margin-top: 14px; margin-bottom: 4px; }"
               "th { background-color: #e3e8ee; text-align: left; font-weight: bold; }"
               ".num { text-align: right; } .hot { color: #c62828; font-weight: bold; }"
               ".note { color: #5f6b77; font-size: 8pt; } .warn { color: #b26a00; }"
               "</style></head><body>"]

        def table(header, rows, numeric_from=1):
            parts = ["<table border='1' cellspacing='0' cellpadding='3' width='100%' "
                     "style='border-color: #b8c0c8; border-collapse: collapse;'><tr>"]
            parts += [f"<th>{esc(str(title))}</th>" for title in header]
            parts.append("</tr>")
            for row in rows:
                parts.append("<tr>")
                for index, cell in enumerate(row):
                    text, css = cell if isinstance(cell, tuple) else (str(cell), "")
                    parts.append(f"<td class='{'num' if index >= numeric_from else ''} {css}'>{esc(text)}</td>")
                parts.append("</tr>")
            parts.append("</table>")
            return "".join(parts)

        path = self._window._thermal_file
        out.append(f"<h1>Structural study: {esc(self.study['name'])}</h1>")
        out.append(f"<p class='note'>{esc(path) if path else 'not saved yet'} · "
                   f"{datetime.datetime.now():%Y-%m-%d %H:%M} · SimpleCAD {ADDON_TITLE} {ADDON_VERSION}</p>")
        image = self._view_image()
        if image.width() > 1600:
            image = image.scaledToWidth(1600, Qt.SmoothTransformation)
        images = {"view.png": image}
        out.append("<p align='center'><img src='view.png' width='520'><br><span class='note'>"
                   "As shown in the window" + (" (deformation exaggerated "
                   f"{self.sp_struct_scale.value():g}×)" if result is not None and
                   self.chk_struct_deformed.isChecked() else "") + "</span></p>")
        out.append("<h2>Setup</h2>")
        rows = [("Analysis", STRUCTURAL_ANALYSES[settings["struct_analysis"]])]
        if not modal:
            rows.append(("Gravity", "on (along −Z)" if settings["struct_gravity"] else "off"))
            thermal = settings["struct_thermal"]
            rows.append(("Thermal expansion", STRUCTURAL_THERMAL[thermal] +
                         (f" {settings['struct_uniform_c']:g} °C" if thermal == "uniform" else "") +
                         (f", stress-free at {settings['struct_reference_c']:g} °C" if thermal != "none" else "")))
        if result is not None:
            rows.append(("Mesh", result["summary"][0].split(": ", 1)[-1]))
        out.append(table(["Item", "Value"], rows, numeric_from=9))
        conditions = [c for c in self.study["conditions"] if c["kind"] in STRUCTURAL_KINDS]
        if conditions:
            out.append("<h2>Supports and loads</h2>")
            out.append(table(["Name", "Condition", "Faces"],
                             [(c["name"], self._condition_text(c).split("[", 1)[-1].split("]")[0]
                               + ("" if c.get("enabled", True) else " (off)"), str(len(c["faces"])))
                              for c in conditions], numeric_from=2))
        out.append("<h2>Materials</h2>")
        used = sorted({body.get("material") for component in self.study["components"]
                       for body in component["bodies"] if body.get("material")})
        out.append(table(["Material", "Mechanical data"],
                         [(name, mechanical_text(material_properties(name) or {}) or "—") for name in used],
                         numeric_from=9))
        if result is None:
            out.append("<p>No results yet.</p></body></html>")
            return "".join(out), images
        if modal:
            out.append("<h2>Natural frequencies</h2>")
            out.append(table(["Mode", "Frequency Hz"],
                             [(str(k + 1), f"{f:.5g}") for k, f in enumerate(result["frequencies"])]))
        else:
            out.append("<h2>Results</h2>")
            rows = []
            for body in result["bodies"]:
                safety = body["safety"]
                css = "hot" if safety is not None and safety < 1.0 else "warn" if safety is not None and safety < 1.5 else ""
                rows.append((body["name"], body["material"] or "", f"{body['mass_kg']:.4g}",
                             f"{body['disp_max']:.4g}", f"{body['vm_max']:.4g}",
                             "—" if not body["yield"] else f"{body['yield']:g}",
                             ("—" if safety is None else f"{safety:.3g}", css)))
            out.append(table(["Body", "Material", "kg", "Max |u| mm", "Max σv MPa", "Yield MPa",
                              "Safety factor"], rows, numeric_from=2))
            if result["reactions"]:
                out.append("<h2>Reactions at the supports</h2>")
                out.append(table(["Support", "Fx N", "Fy N", "Fz N", "|F| N", "Moment N·m (about the face centre)"],
                                 [(e["name"], *[f"{v:.4g}" for v in e["force"]],
                                   f"{np.linalg.norm(e['force']):.4g}",
                                   ", ".join(f"{v:.3g}" for v in e["moment"])) for e in result["reactions"]]))
        out.append("<h2>Messages</h2><p>" + "<br>".join(esc(line) for line in result["summary"] +
                                                        result["messages"]) + "</p>")
        out.append("<p class='note'>Linear elastic analysis: small deformations, no contact "
                   "separation, no plasticity. Stresses at sharp inside corners depend on the mesh.</p>")
        out.append("</body></html>")
        return "".join(out), images

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
        if result.get("mode") == "structural":
            vm, disp = structural_point(result, point, max(0, self.cb_struct_mode.currentIndex()))
            return np.array([float(np.linalg.norm(disp)) if result["analysis"] == "modal" else vm])
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
        if value is not None and self._result.get("mode") == "structural":
            vm, disp = structural_point(self._result, point, max(0, self.cb_struct_mode.currentIndex()))
            self.lbl_probe_value.setText(
                f"|u| {np.linalg.norm(disp):.3g} (mode shape, relative)" if self._result["analysis"] == "modal"
                else f"σv {vm:.4g} MPa, |u| {np.linalg.norm(disp):.4g} mm")
        elif value is None:
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
            if items is self.study["conditions"]:       # channels fed by this one follow
                for other in self.study["conditions"]:
                    if other.get("inlet_from") == item["name"]:
                        other["inlet_from"] = name
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
            structural = settings["mode"] == "structural"
            self.cb_mode.setCurrentIndex(max(0, self.cb_mode.findData(settings["mode"])))
            self.lbl_mode_hint.setText(
                "Airflow + heat (CHT): air and solids are solved together on one grid — buoyancy, "
                "fans and vents decide where the heat goes. Takes minutes (transient runs longer)."
                if airflow else
                "Structural: stresses and deformation of the parts under supports and loads (Face "
                "Condition), or their natural frequencies. Touching parts are bonded; parts need "
                "materials with mechanical data." + structural_missing_text() if structural
                else "Conduction (FEM): heat conduction in the bodies; air cooling and radiation act "
                     "on their surfaces. Fast.")
            modal = settings["struct_analysis"] == "modal"
            self.cb_struct_analysis.setCurrentIndex(max(0, self.cb_struct_analysis.findData(
                settings["struct_analysis"])))
            self.sp_struct_modes.setValue(int(settings["struct_modes"]))
            self.chk_struct_gravity.setChecked(settings["struct_gravity"])
            self.cb_struct_thermal.setCurrentIndex(max(0, self.cb_struct_thermal.findData(
                settings["struct_thermal"])))
            self.sp_struct_uniform.setValue(settings["struct_uniform_c"])
            self.sp_struct_reference.setValue(settings["struct_reference_c"])
            struct_auto = settings["struct_mesh_mm"] <= 0
            self.chk_struct_auto.setChecked(struct_auto)
            self.sp_struct_size.setEnabled(not struct_auto)
            if not struct_auto:
                self.sp_struct_size.setValue(settings["struct_mesh_mm"])
            self.sp_struct_max.setValue(int(settings["struct_max_elements"]))
            self.cb_struct_order.setCurrentIndex(0 if settings["struct_order"] == 2 else 1)
            heat_run = structural and not modal and settings["struct_thermal"] == "conduction"
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

            thermal = not structural or heat_run
            show(self._structural_rows, structural)
            show(self._struct_modal_rows, structural and modal)
            show(self._struct_static_rows, structural and not modal)
            show(self._struct_uniform_rows, structural and not modal and
                 settings["struct_thermal"] == "uniform")
            show(self._struct_thermal_rows, structural and not modal and
                 settings["struct_thermal"] != "none")
            show(self._thermal_analysis_rows, not structural)
            show(self._environment_rows, thermal)
            show(self._conduction_rows, not airflow and thermal)
            show(self._transient_rows, transient and not structural)
            show(self._speed_rows, not airflow and thermal and settings["convection"] == "forced")
            show(self._h_rows, not airflow and thermal and settings["convection"] == "fixed")
            show(self._mesh_rows, not airflow and thermal)
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
        settings["struct_analysis"] = self.cb_struct_analysis.currentData()
        settings["struct_modes"] = int(self.sp_struct_modes.value())
        settings["struct_gravity"] = self.chk_struct_gravity.isChecked()
        settings["struct_thermal"] = self.cb_struct_thermal.currentData()
        settings["struct_uniform_c"] = float(self.sp_struct_uniform.value())
        settings["struct_reference_c"] = float(self.sp_struct_reference.value())
        settings["struct_mesh_mm"] = 0.0 if self.chk_struct_auto.isChecked() else \
            float(self.sp_struct_size.value())
        settings["struct_max_elements"] = int(self.sp_struct_max.value())
        settings["struct_order"] = int(self.cb_struct_order.currentData())
        mode_changed = getattr(self, "_shown_mode", settings["mode"]) != settings["mode"]
        self._shown_mode = settings["mode"]
        if mode_changed:
            self._fill_condition_menu()
        self._show_settings()
        self._changed()
        if mode_changed:                         # conditions of the other study type grey out
            self._rebuild_tree()
            if self._condition() is not None:
                self._show_condition()

    def _update_estimate(self):
        settings = self.study["settings"]
        if settings["mode"] == "structural":
            self.lbl_struct_estimate.setText(structural_estimate_text(self.study))
            if settings["struct_mesh_mm"] <= 0:
                size = structural_auto_size(self.study)
                if size:
                    self.sp_struct_size.blockSignals(True)
                    self.sp_struct_size.setValue(size)
                    self.sp_struct_size.blockSignals(False)
            if not (settings["struct_analysis"] == "static" and settings["struct_thermal"] == "conduction"):
                return
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

    def _added(self, components, text, keep=False):
        """keep: parts of the open project were added - the study is kept in the project (a
        "Thermal" item of the Objects explorer, _keep_in_project)."""
        if not components:
            return
        self._after_components_changed(select=components[-1])
        self.view.fit_all()
        missing = sum(1 for component in components for body in component["bodies"]
                      if not body.get("material") and not body.get("fused"))
        if missing:
            text += f" — {missing} bod{'y needs' if missing == 1 else 'ies need'} a material (⚠)"
        if keep:
            text += self._keep_in_project()
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
        self._added(added, f"Added {', '.join(c['name'] for c in added)}", keep=True)

    def _import_project_assembly(self):
        assembly = getattr(self._window, "_assembly", None)
        if not assembly or not assembly.get("components"):
            QMessageBox.information(self, "Assembly of the open project",
                                    "The open project has no assembly yet (Assembly window).")
            return
        self._import_assembly(assembly, "of the project", objects=self._window.state["objects"],
                              keep=True)

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

    def _import_assembly(self, assembly, label, objects=None, keep=False):
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
        self._added(added, f"Imported {len(added)} assembly part{'s' if len(added) != 1 else ''}",
                    keep=keep)

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
        rows = self._selected_items()
        if len(rows) > 1 or (rows and rows[0][0] == "run"):
            self._remove_many(rows)
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
    def _bound_entry(self):
        """The "Thermal" entry of the Objects explorer this study is kept in, or None."""
        finder = getattr(self._window, "_bound_thermal_entry", None)
        return finder() if finder is not None else None

    def _keep_in_project(self):
        """The study is kept in the project: it gets a "Thermal" item of its own in the Objects
        explorer (like an Assembly). Called when parts of the open project are imported (also
        into a study opened from a file - Save then stores it in the item, Save As still
        writes files) and when a study without a file gets its first parts. Returns the status
        note."""
        if self._bound_entry() is not None or \
                getattr(self._window, "_create_thermal_entry", None) is None:
            return ""
        entry = self._window._create_thermal_entry(study_to_data(self.study))
        self._mark_saved()
        return f" — stored as {entry['name']} in the Objects explorer"

    def _save(self, save_as=False):
        """Save: a study kept in a "Thermal" entry of the Objects explorer is stored there
        (written to disk with the project); otherwise in its .thermal file (the first time a
        file dialog asks for it). Save As always writes a file (and keeps the entry up to
        date)."""
        entry = self._bound_entry()
        if entry is None and not save_as and self._window._thermal_file:
            # opened from a file: Save keeps it in the open project (a "Thermal" item);
            # Save As still writes files
            self._keep_in_project()
            entry = self._bound_entry()
        if entry is not None and not save_as:
            self._window._store_thermal_in_entry(entry, study_to_data(self.study))
            self._mark_saved()
            self._set_status(f"Thermal study saved in {entry['name']} — it is written to disk "
                             "with the project (File → Save)")
            return True
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
        if entry is None:
            self.study["name"] = os.path.splitext(os.path.basename(path))[0]
        try:
            study_save_file(self.study, path)
        except (OSError, ValueError) as ex:
            QMessageBox.critical(self, "Save failed", f"Could not save\n{path}\n\n{ex}")
            return False
        if entry is not None:                     # Save As of a kept study: the entry too
            self._window._store_thermal_in_entry(entry, study_to_data(self.study))
            self._mark_saved()
            self._set_status(f"Thermal study written to {path} (and kept in {entry['name']})")
            return True
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
        self._window._thermal_entry = None      # a new study gets its own entry on import
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
        self._window._thermal_file = path
        self._window._thermal_entry = None      # an opened file belongs to its file
        self.load_study(loaded, f"Opened {path}")

    def load_study(self, loaded, status=""):
        """Show another study in this window (an opened file, or a "Thermal" entry of the
        Objects explorer opened by double-click / Edit)."""
        self.study.clear()
        self.study.update(loaded)
        set_active_study(self.study)
        self._forget_sweep()
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
        if status:
            self._set_status(status)

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
        self._run_state = study_state(self.study)
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
        if settings["mode"] == "structural":
            return self._check_structural(jobs)
        airflow = settings["mode"] == "airflow"
        conditions = study_conditions(self.study, jobs, THERMAL_KINDS)
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
                    self.act_sweep_results, self.act_runs):
            act.setEnabled(not running)
        self.act_cancel.setEnabled(running)
        self.act_results.setEnabled(not running and self._result is not None)
        self.act_sweep_results.setEnabled(not running and self._sweep_result is not None)
        self.act_runs.setEnabled(not running and bool(self._runs))
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

    # ── run history ───────────────────────────────────────────────────────────
    RUN_LIMIT = 20

    def _run_parameters(self):
        """[(name, text)] of everything a run depends on: settings, parts, conditions, contacts."""
        def text(value):
            if isinstance(value, dict):
                return ", ".join(f"{k} = {text(v)}" for k, v in value.items())
            if isinstance(value, (list, tuple)):
                return ", ".join(text(v) for v in value)
            if isinstance(value, float):
                return f"{value:g}"
            return str(value)
        rows = []
        settings = self.study["settings"]
        for key in sorted(settings):
            rows.append((f"Setting: {key}", text(settings[key])))
        for component in self.study["components"]:
            if component.get("kind") in FLOW_KINDS:
                rows.append((f"Part: {component['name']}", f"{component['kind']}: "
                             f"{flow_text(component['kind'], flow_settings(component.get('flow'), component['kind']))}"))
                continue
            for body in component["bodies"]:
                name = component["name"] if len(component["bodies"]) == 1 else \
                    f"{component['name']} / {body['name']}"
                parts = [body.get("material") or "no material"]
                if body_has_power(body):
                    parts.append(f"{body['power_w']:g} W")
                    if body.get("profile"):
                        parts.append(profile_text(body["profile"], body["power_w"]))
                    if body.get("thermostat"):
                        parts.append(thermostat_text(body["thermostat"]))
                rows.append((f"Part: {name}", ", ".join(parts)))
        for condition in self.study.get("conditions", []):
            if condition.get("enabled", True):
                rows.append((f"Condition: {condition['name']}", self._condition_text(condition)))
        for contact in self.study.get("contacts", []):
            if contact.get("enabled", True):
                rows.append((f"Contact: {contact['name']}", self._contact_item_text(contact)))
        return rows

    @staticmethod
    def _run_summary(result):
        if result.get("mode") == "structural":
            if result["analysis"] == "modal":
                frequencies = result["frequencies"]
                return f"lowest natural frequency {frequencies[0]:.4g} Hz" if len(frequencies) else "modal"
            hot = max(result["bodies"], key=lambda b: b["vm_max"])
            return f"von Mises {hot['vm_max']:.4g} MPa max, displacement {result['disp_max']:.4g} mm"
        return f"max {result['hot_value']:.1f} °C, heat balance {result['balance_error']:+.2%}"

    def _record_run(self, result, state):
        import datetime
        self._run_counter += 1
        mode = result.get("mode")
        kind = "Structural" if mode == "structural" else "Airflow + heat" if mode == "airflow" else "Thermal"
        try:
            summary = self._run_summary(result)
        except (KeyError, IndexError, ValueError, TypeError):
            summary = "-"
        if state is None:
            state = study_state(self.study)
        self._runs.append({"id": self._run_counter, "marked": False, "time": f"{datetime.datetime.now():%H:%M:%S}",
                           "title": f"{kind}, {result.get('analysis', '')}", "summary": summary,
                           "seconds": float(result.get("seconds", 0.0)), "params": self._run_parameters(),
                           "state": state, "key": state_key(state), "result": result})
        del self._runs[:-self.RUN_LIMIT]
        self.act_runs.setEnabled(True)
        self._rebuild_tree()
        if self._run_dialog is not None:
            self._run_dialog.refresh()

    def _marked_ids(self):
        return [r["id"] for r in self._runs if r.get("marked")]

    def _set_run_marked(self, run_id, marked, from_tree=False):
        run = self._run(run_id)
        if run is None:
            return
        run["marked"] = bool(marked)
        if not from_tree:
            self._loading = True
            try:
                for index in range(self.tree.topLevelItemCount()):
                    top = self.tree.topLevelItem(index)
                    for k in range(top.childCount()):
                        item = top.child(k)
                        if item.data(0, Qt.UserRole + 2) == f"run:{run_id}" and \
                                item.flags() & Qt.ItemIsUserCheckable:
                            item.setCheckState(0, Qt.Checked if marked else Qt.Unchecked)
            finally:
                self._loading = False
        if self._run_dialog is not None and from_tree:
            self._run_dialog.refresh()

    def _run(self, run_id):
        return next((r for r in self._runs if r["id"] == run_id), None)

    def _open_run_history(self):
        if not self._runs:
            return
        if self._run_dialog is None:
            self._run_dialog = RunHistoryDialog(self)
            cad._set_dark_title_bar(self._run_dialog)
        self._run_dialog.refresh()
        self._run_dialog.show()
        self._run_dialog.raise_()
        self._run_dialog.activateWindow()

    def _show_run(self, run_id):
        run = self._run(run_id)
        if run is None or self._worker is not None:
            return
        self._discard_results()
        self.txt_messages.setPlainText("\n".join(run["result"].get("messages") or []))
        self._result = run["result"]
        self._result_note = f"Run #{run['id']} of {run['time']}"
        self._show_result()
        changed = state_key(study_state(self.study)) != run["key"]
        self._set_status(f"Showing run #{run['id']} ({run['time']})" +
                         (" - the study has changed since; 'Restore setup' in the run history brings "
                          "its values back" if changed else ""))

    def _restore_run(self, run_id):
        run = self._run(run_id)
        if run is None or self._worker is not None:
            return
        del self._history[self._history_index + 1:]
        self._history.append((run["state"], run["key"]))
        self._history_index += 1
        self._restore_history(self._history_index, f"Restored the setup of run #{run['id']} -")
        self._show_run(run_id)
        self._set_status(f"Setup of run #{run['id']} restored (undo brings your last values back)")

    def _delete_runs(self, ids):
        shown = any(r["id"] in ids and r["result"] is self._result for r in self._runs)
        self._runs = [r for r in self._runs if r["id"] not in ids]
        self.act_runs.setEnabled(bool(self._runs) and self._worker is None)
        if shown:
            self._discard_results()
            self._refresh_status()
        self._rebuild_tree()
        self._set_status(f"Removed {len(ids)} result(s)")
        if self._run_dialog is not None:
            self._run_dialog.refresh()

    def _with_run(self, run, function, *args):
        saved = (self._result, self._result_note, self._frame)
        result = run["result"]
        frames = result.get("frame_times")
        self._result, self._result_note = result, ""
        self._frame = len(frames) - 1 if frames is not None and len(frames) and \
            result.get("analysis") == "transient" else 0
        try:
            return function(*args)
        finally:
            self._result, self._result_note, self._frame = saved

    def _export_runs(self, kind, ids):
        import csv
        runs = [r for r in self._runs if r["id"] in ids]
        if not runs:
            return
        if kind == "vtu":
            self._with_run(runs[0], self._export_vtu)
            return
        if kind == "csv":
            if len(runs) == 1:
                self._with_run(runs[0], self._export_csv)
                return
            folder = QFileDialog.getExistingDirectory(self, "Folder for the result files",
                                                      self._start_directory())
            if not folder:
                return
            name = self.study["name"] or THERMAL_DEFAULT_NAME
            for run in runs:
                self._with_run(run, self._export_csv, os.path.join(folder, f"{name}_run{run['id']}_results.csv"))
            self._set_status(f"{len(runs)} result files saved in {folder}")
            return
        path = self._export_path("Export the run comparison", "_runs.csv", "CSV files (*.csv)")
        if not path:
            return
        names = []
        for run in runs:
            for name, _value in run["params"]:
                if name not in names:
                    names.append(name)
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(["Run", "Time", "Type", "Result", "Duration s"] + names)
                for run in runs:
                    values = dict(run["params"])
                    writer.writerow([f"#{run['id']}", run["time"], run["title"], run["summary"],
                                     f"{run['seconds']:.2f}"] + [values.get(n, "") for n in names])
        except OSError as ex:
            QMessageBox.critical(self, "Export failed", f"Could not write\n{path}\n\n{ex}")
            return
        self._set_status(f"Run comparison saved to {path}")

    def _on_simulation_done(self, result):
        self._record_run(result, self._run_state)
        self._result = result
        self._result_note = ""
        self._show_result()
        if result.get("mode") == "structural":
            if result["analysis"] == "modal":
                frequencies = result["frequencies"]
                self._set_status(f"Simulation finished in {result['seconds']:.1f} s — lowest natural "
                                 f"frequency {frequencies[0]:.4g} Hz" if len(frequencies) else
                                 f"Simulation finished in {result['seconds']:.1f} s")
            else:
                hot = max(result["bodies"], key=lambda b: b["vm_max"])
                self._set_status(f"Simulation finished in {result['seconds']:.1f} s — maximum von "
                                 f"Mises {hot['vm_max']:.4g} MPa in {hot['name']}, maximum "
                                 f"displacement {result['disp_max']:.4g} mm")
            return
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
        if result.get("mode") == "structural":
            self._export_structural_vtu()
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

    def _export_csv(self, path=None):
        result = self._result
        if result is None:
            return
        path = path or self._export_path("Export results", "_results.csv", "CSV files (*.csv)")
        if not path:
            return
        if result.get("mode") == "structural":
            self._write_structural_csv(path)
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
        if result.get("liquids"):
            header, liquid = liquid_rows(result, number)
            rows += [[], header] + liquid
        if result.get("junctions"):
            rows += [[], ["Junction of", "Package", "Power W", "Tj °C", "Highest Tj °C", "Tj max °C",
                          "Package top °C", "Board below °C", "θ top K/W", "θ board K/W"]]
            rows += [[j["name"], j["package"], number(j["power"]), number(j["t_j"]),
                      number(j.get("peak", j["t_j"])), number(j["tj_max"]), number(j["t_top"]),
                      number(j["t_board"]), number(j["theta_top"]), number(j["theta_board"])]
                     for j in result["junctions"]]
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
        outlets = series.get("coolant")
        if outlets is not None:
            header += [f"Coolant {name} outlet °C" for name in series.get("coolant_names", [])]
        junctions = series.get("junctions")
        if junctions is not None:
            header += [f"Junction {name} °C" for name in series.get("junction_names", [])]
        rows = [header]
        for k, t in enumerate(times):
            row = [f"{t:.6g}", f"{series['power'][k]:.6g}"]
            row += [f"{series['max'][i, k]:.5g}" for i in range(len(result["bodies"]))]
            row += [f"{series['mean'][i, k]:.5g}" for i in range(len(result["bodies"]))]
            row += [f"{values[k]:.5g}" for values in probes]
            if outlets is not None:
                row += [f"{outlets[i, k]:.5g}" for i in range(len(outlets))]
            if junctions is not None:
                row += [f"{junctions[i, k]:.5g}" for i in range(len(junctions))]
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
            write_pdf(path, html, images, ("Structural" if self.study["settings"]["mode"] == "structural" else "Thermal") + f" study {self.study['name']}")
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
        if settings["mode"] == "structural" or (result is not None and result.get("mode") == "structural"):
            return self._structural_report()
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
            if result.get("liquids"):
                header, liquid = liquid_rows(
                    result, lambda v: "" if v is None or not np.isfinite(v) else f"{v:.4g}")
                out.append("<p><b>Coolant channels and liquid baths</b></p>")
                out.append(table(header, liquid, numeric_from=3))
            if result.get("junctions"):
                out.append("<p><b>Component junctions</b> (two-resistor model)</p>")
                rows_j = []
                for j in result["junctions"]:
                    peak = j.get("peak", j["t_j"])
                    over = peak > j["tj_max"]
                    rows_j.append([j["name"], j["package"], f"{j['power']:.3g}", c(j["t_j"]),
                                   (c(peak), "hot" if over else ""), f"{j['tj_max']:g}",
                                   ("over the limit" if over else "ok", "hot" if over else "")])
                out.append(table(["Junction of", "Package", "W", "Tj °C", "Highest Tj °C", "Tj max °C",
                                  "Status"], rows_j, numeric_from=2))
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
        self._struct_timer.stop()
        self.btn_struct_animate.blockSignals(True)
        self.btn_struct_animate.setChecked(False)
        self.btn_struct_animate.blockSignals(False)
        self._struct_phase = 1.0
        self.w_struct_view.hide()
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
        if result.get("mode") == "structural":
            return self._structural_range()
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
        if result.get("mode") == "structural":
            self._apply_structural_colors()
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
        if result.get("mode") == "structural":
            self._show_structural_result()
            return
        self._thermal_result_layout()
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
            junction = next((j for j in result.get("junctions") or [] if j["name"] == body["name"]), None)
            self.tbl_results.item(row, 0).setText(body["name"] + (
                f"  — junction {junction['t_j']:.1f} °C" if junction is not None else ""))
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
        if result.get("mode") == "structural":
            self._show_structural_frame()
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
                if self._result.get("mode") == "structural":
                    self._set_status(f"{body['name']}: max von Mises {body['vm_max']:.4g} MPa, max "
                                     f"displacement {body['disp_max']:.4g} mm, {body['mass_kg']:.4g} kg")
                    break
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
