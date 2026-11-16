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

    cmd = [
        str(py), "-m", "PyInstaller",
        "--onefile",
        "--noconsole",
        "--name", "SimpleCAD",
        "--add-data", f"{ICONS}{';' if os_name == 'windows' else ':'}icons",
        "--icon", str(icon),
        str(ENTRY),
    ]

    print(f"[exe] Building executable for {os_name}...")
    run(*cmd, cwd=str(ROOT))
    print(f"[exe] Output: {ROOT / 'dist' / 'SimpleCAD'}")


def create_shortcut_windows():
    icon = ICONS / "cad_app_dark.ico"
    shortcut = Path(os.path.expanduser("~/Desktop/SimpleCAD.lnk"))
    script = (f"$ws = New-Object -ComObject WScript.Shell;"
              f"$s = $ws.CreateShortcut('{shortcut}');"
              f"$s.TargetPath = '{ROOT / 'run.bat'}';"
              f"$s.IconLocation = '{icon}';"
              f"$s.WorkingDirectory = '{ROOT}';"
              f"$s.Save()")
    run("powershell", "-NoProfile", "-Command", script)
    print(f"[shortcut] Created: {shortcut}")


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
