#!/usr/bin/env python3
"""
build.py — Build SimpleCAD for Windows or Linux.

Usage:
    python build.py [--venv] [--exe] [--shortcut] [--platform windows|linux]

Flags:
    --venv        Create/recreate the .venv and install dependencies
    --exe         Run PyInstaller to produce a single-file executable
    --shortcut    Create a desktop shortcut/launcher with icon
    (no flags)    Creates .venv only (Windows)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV = ROOT / ".venv"
ICONS = ROOT / "icons"
ENTRY = ROOT / "SimpleCAD.py"

# pip requirement -> (import name, what it is used for). The minimum versions are the ones
# SimpleCAD was tested with (Python 3.13); every package SimpleCAD imports must be listed.
PACKAGES = {
    "pyinstaller>=6.22": ("PyInstaller", "building the executable"),
    "numpy>=2.3": ("numpy", "geometry and meshes"),
    "PyQt5>=5.15.11": ("PyQt5", "user interface (QtWidgets, QtOpenGL, QtSvg, QtGui)"),
    "PyOpenGL>=3.1.10": ("OpenGL", "3-D view"),
    "manifold3d>=3.5": ("manifold3d", "solid booleans"),
    "ezdxf>=1.4": ("ezdxf", "DXF import / export"),
    "qt-modern-style>=2.0.1": ("qt_modern_style", "styling"),
    "cadquery>=2.8": ("cadquery", "STEP import / export"),
    "cadquery-ocp>=7.9": ("OCP", "STEP kernel used by cadquery"),
}
# optional add-ons: bundled when their file sits next to SimpleCAD.py (SimpleCAD loads them
# with importlib, so PyInstaller has to be told about them)
ADDONS = {"simplecad_thermal": {
    "scipy>=1.16": ("scipy", "Thermal add-on: sparse solvers"),
    "gmsh>=4.15": ("gmsh", "Thermal add-on: tetrahedral meshing (also from STEP)"),
    "pyamg>=5.4": ("pyamg", "Thermal add-on: multigrid preconditioner"),
    "scikit-fem>=12.0": ("skfem", "Thermal add-on: FEM assembly"),
}}
for _addon, _packages in ADDONS.items():
    if (ROOT / f"{_addon}.py").exists():
        PACKAGES.update(_packages)
REQUIREMENTS = list(PACKAGES)
# packages PyInstaller cannot trace on its own (compiled parts, imports made inside functions,
# package data): everything of them goes into the executable
COLLECT_ALL = ["cadquery", "OCP", "manifold3d", "ezdxf"]       # (qt_modern_style: --add-data)
# Help -> About reads the package versions from the package data: keep it in the executable
COPY_METADATA = ["numpy", "PyOpenGL", "manifold3d", "ezdxf", "cadquery", "cadquery-ocp",
                 "qt-modern-style"]
# what each add-on needs in the executable, used only when its file is there:
#   hidden      imported by name (gmsh: its PyInstaller hook then adds the gmsh DLL)
#   submodules  all submodules (scipy.sparse / spatial / ndimage are imported lazily; pyamg and
#               skfem have compiled or lazily loaded parts)
#   metadata    package data (pyamg reads its own version from it at import; About shows them)
ADDON_BUNDLE = {"simplecad_thermal": {
    "hidden": ["gmsh"],
    "submodules": ["scipy", "pyamg", "skfem"],
    "metadata": ["scipy", "gmsh", "pyamg", "scikit-fem"],
}}


def detect_platform(override: str | None) -> str:
    if override:
        return override.lower()
    return "windows"


def venv_python(os_name: str) -> Path:
    if os_name == "windows":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def create_venv(os_name: str):
    print(f"[venv] Creating .venv for {os_name}...")
    run(sys.executable, "-m", "venv", str(VENV))
    py = venv_python(os_name)
    run(str(py), "-m", "pip", "install", "--upgrade", "pip")
    run(str(py), "-m", "pip", "install", *REQUIREMENTS)
    check_requirements(os_name)
    print("[venv] Done.")


def check_requirements(os_name: str):
    """Import every package in the venv and print the installed version; stop when one is
    missing, so a build never starts with an incomplete environment."""
    py = venv_python(os_name)
    code = (
        "import importlib, importlib.metadata as md, sys\n"
        "bad = 0\n"
        "for spec, mod, why in %r:\n"
        "    pkg = spec.split('>')[0].split('=')[0].split('<')[0]\n"
        "    try:\n"
        "        importlib.import_module(mod)\n"
        "        print(f'  ok   {pkg:<18} {md.version(pkg):<14} {why}')\n"
        "    except Exception as ex:\n"
        "        bad += 1; print(f'  FAIL {pkg:<18} {type(ex).__name__}: {ex}')\n"
        "sys.exit(1 if bad else 0)\n"
    ) % ([(spec, mod, why) for spec, (mod, why) in PACKAGES.items()],)
    print("[venv] Checking the installed packages...")
    result = subprocess.run([str(py), "-c", code])
    if result.returncode:
        sys.exit("[venv] A required package is missing or cannot be imported (see above).")


def build_exe(os_name: str):
    py = venv_python(os_name)
    icon = ICONS / "cad_app_dark.ico"
    sep = ";" if os_name == "windows" else ":"
    site_packages = VENV / ("Lib/site-packages" if os_name == "windows" else "lib/python*/site-packages")
    qt_modern_style_path = next(VENV.glob(
        "Lib/site-packages/qt_modern_style" if os_name == "windows"
        else "lib/python*/site-packages/qt_modern_style"
    ), None)

    cmd = [
        str(py), "-m", "PyInstaller",
        "--onefile",
        "--noconsole",
        "--name", "SimpleCAD",
        "--add-data", f"{ICONS}{sep}icons",
        "--icon", str(icon),
        str(ENTRY),
    ]

    check_requirements(os_name)       # nothing is built from an incomplete environment
    for addon in ADDONS:
        if (ROOT / f"{addon}.py").exists():
            cmd += ["--hidden-import", addon]
            bundle = ADDON_BUNDLE.get(addon, {})
            for package in bundle.get("hidden", []):
                cmd += ["--hidden-import", package]
            for package in bundle.get("submodules", []):
                cmd += ["--collect-submodules", package]
            for package in bundle.get("metadata", []):
                cmd += ["--copy-metadata", package]
            print(f"[exe] Bundling add-on {addon}")
    for package in COLLECT_ALL:
        cmd += ["--collect-all", package]
    for package in COPY_METADATA:
        cmd += ["--copy-metadata", package]

    if qt_modern_style_path and qt_modern_style_path.exists():
        cmd += ["--add-data", f"{qt_modern_style_path}{sep}qt_modern_style"]
        print(f"[exe] Bundling qt_modern_style from {qt_modern_style_path}")
    else:
        print("[exe] WARNING: qt_modern_style not found, styles.qss may be missing")


    print(f"[exe] Building executable for {os_name}...")
    run(*cmd, cwd=str(ROOT))
    print(f"[exe] Output: {ROOT / 'dist' / 'SimpleCAD'}")


def create_shortcut_windows():
    icon = ICONS / "cad_app_dark.ico"
    ps1 = ROOT / "_make_shortcut.ps1"
    ps1.write_text(
        f'$ws = New-Object -ComObject WScript.Shell\n'
        f'$s = $ws.CreateShortcut("{ROOT / "SimpleCAD.lnk"}")\n'
        f'$s.TargetPath = "{ROOT / "dist" / "SimpleCAD.exe"}"\n'
        f'$s.IconLocation = "{icon}"\n'
        f'$s.WorkingDirectory = "{ROOT}"\n'
        f'$s.Save()\n'
    )
    run("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1))
    ps1.unlink()
    print(f"[shortcut] Created: {ROOT / 'SimpleCAD.lnk'}")


def create_shortcut_linux():
    icon = ICONS / "cad_app_dark.png"
    desktop = Path(os.path.expanduser("~/Desktop/SimpleCAD.desktop"))
    desktop.write_text(
        f"[Desktop Entry]\n"
        f"Name=SimpleCAD\n"
        f"Exec=bash {ROOT / 'run.sh'}\n"
        f"Icon={icon}\n"
        f"Terminal=false\n"
        f"Type=Application\n"
        f"Categories=Graphics;Engineering;\n"
    )
    desktop.chmod(0o755)
    print(f"[shortcut] Created: {desktop}")


def main():
    parser = argparse.ArgumentParser(description="Build SimpleCAD")
    parser.add_argument("--venv", action="store_true", help="Create .venv and install deps")
    parser.add_argument("--exe", action="store_true", help="Build executable with PyInstaller")
    parser.add_argument("--shortcut", action="store_true", help="Create desktop shortcut with icon")
    parser.add_argument("--platform", dest="os_name", choices=["windows", "linux"],
                        help="Target platform (default: windows)")
    args = parser.parse_args()

    os_name = detect_platform(args.os_name)
    do_venv = args.venv or not (args.venv or args.exe or args.shortcut)
    do_exe = args.exe
    do_shortcut = args.shortcut

    if do_venv:
        create_venv(os_name)
    if do_exe:
        build_exe(os_name)
    if do_shortcut:
        create_shortcut_windows() if os_name == "windows" else create_shortcut_linux()


if __name__ == "__main__":
    main()
