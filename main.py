"""AR-Python-OpenCV launcher: pick a demo from the menu.

Each demo can also be run directly, e.g. python scripts/ar_depth.py --help
"""
import importlib.util
import json
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = {
    "1": ("Generate a marker", "scripts/generate_aruco.py",
          "Print it: arucoMarkers/DICT_5X5_100_1.png (black square 5 cm wide)"),
    "2": ("ArUco detection", "scripts/aruco_detection.py", "Outline and ID of each marker"),
    "3": ("Pose estimation", "scripts/pose_estimation.py", "3D axes and distance of each marker"),
    "4": ("3D model (OpenGL)", "scripts/pose_obj_object.py", "objects/cube.obj standing on the marker"),
    "5": ("Depth-aware AR", "scripts/ar_depth.py",
          "Cube on the marker that real objects in front can hide (MiDaS depth)"),
    "6": ("Object tracking (YOLOv5)", "scripts/object_tracking.py",
          "Optional: needs torch and deep-sort-realtime, see README"),
}
OPTIONAL_MODULES = {"scripts/object_tracking.py": ["torch", "deep_sort_realtime"]}


def missing_modules(script):
    return [m for m in OPTIONAL_MODULES.get(script, []) if importlib.util.find_spec(m) is None]


def main():
    try:
        with open(os.path.join(BASE_DIR, "config.json")) as f:
            app_name = json.load(f)["Config"]["AppName"]
    except (OSError, KeyError, ValueError):
        app_name = "AR-Python-OpenCV"

    while True:
        print(f"\n=== {app_name} ===")
        for key, (name, _, desc) in SCRIPTS.items():
            print(f"[{key}] {name} - {desc}")
        print("[q] Quit")
        try:
            choice = input("Select: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if choice == "q":
            break
        if choice not in SCRIPTS:
            print("Invalid choice.")
            continue

        name, script, _ = SCRIPTS[choice]
        missing = missing_modules(script)
        if missing:
            print(f"{name} needs: {', '.join(missing)}. Install with: "
                  f"{os.path.basename(sys.executable)} -m pip install {' '.join(missing).replace('_', '-')}")
            continue
        print(f"\n>> {name}  (q or Esc in the window to quit)")
        try:
            subprocess.run([sys.executable, os.path.join(BASE_DIR, script)], cwd=BASE_DIR)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
