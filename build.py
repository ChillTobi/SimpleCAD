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

REQUIREMENTS = [
    "pyinstaller",
    "numpy",
    "PyQt5",
    "PyOpenGL",
    "ezdxf",
    "svgpathtools",
    "qt_modern_style",
]


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
    print("[venv] Done.")


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
