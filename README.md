[![Twitter: @NorowaretaGemu](https://img.shields.io/badge/X-@NorowaretaGemu-blue.svg?style=flat)](https://x.com/NorowaretaGemu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<br>
<div align="center">
  <a href="https://ko-fi.com/cursedentertainment">
    <img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="ko-fi" style="width: 20%;"/>
  </a>
</div>
  <br>

<div align="center">
  <img alt="Python" src="https://img.shields.io/badge/python%20-%23323330.svg?&style=for-the-badge&logo=python&logoColor=white"/>
</div>

<div align="center">
   <img alt="OpenCV" src="https://img.shields.io/badge/opencv-%23323330.svg?&style=for-the-badge&logo=opencv&logoColor=white"/>
</div>
<div align="center">
    <img alt="Git" src="https://img.shields.io/badge/git%20-%23323330.svg?&style=for-the-badge&logo=git&logoColor=white"/>
  <img alt="PowerShell" src="https://img.shields.io/badge/PowerShell-%23323330.svg?&style=for-the-badge&logo=powershell&logoColor=white"/>
  <img alt="Shell" src="https://img.shields.io/badge/Shell-%23323330.svg?&style=for-the-badge&logo=gnu-bash&logoColor=white"/>
  <img alt="Batch" src="https://img.shields.io/badge/Batch-%23323330.svg?&style=for-the-badge&logo=windows&logoColor=white"/>
  </div>
  <br>

# AR-Python-OpenCV

## Augmented Reality with Python: ArUco Markers, OpenCV & Depth

Point a webcam at a printed ArUco marker and:

- **Detect** it: outline and ID
- **Estimate its pose**: 3D axes and distance
- **Stand a 3D model on it**: `objects/cube.obj`, rendered with OpenGL
- **Depth-aware AR**: a cube on the marker that real things in front of it (your hand, a mug) can hide, using [MiDaS](https://github.com/isl-org/MiDaS) monocular depth

## Quick Start

Needs Python 3.9 or newer and a webcam. Run one script; on the first run it creates a `psdenv` virtual
environment and installs the requirements, then opens the menu:

Windows:
- `.\run.bat`
or
- `.\run.ps1`

Linux/macOS:
- `./run.sh`

Then:

1. Choose **1** to generate a marker and print `arucoMarkers/DICT_5X5_100_1.png` so the black square is
   **5 cm** wide (or pass `--marker-length` with your size). Keep the white border.
2. Choose a demo and hold the marker in front of the camera. Press **q** or **Esc** in the window to quit.

## Menu

| | Demo | Script |
|---|---|---|
| 1 | Generate a printable marker | `scripts/generate_aruco.py` |
| 2 | ArUco detection | `scripts/aruco_detection.py` |
| 3 | Pose estimation: axes and distance | `scripts/pose_estimation.py` |
| 4 | 3D OBJ model on the marker (OpenGL) | `scripts/pose_obj_object.py` |
| 5 | Depth-aware AR with occlusion | `scripts/ar_depth.py` |
| 6 | Object tracking with YOLOv5 (optional) | `scripts/object_tracking.py` |

Every demo can also be run on its own, and accepts an image or video instead of the webcam:

```bash
python scripts/ar_depth.py                        # webcam
python scripts/ar_depth.py --image photo.jpg
python scripts/ar_depth.py --video clip.mp4 --save out.mp4
python scripts/pose_estimation.py --dict DICT_4X4_50 --marker-length 0.08
python scripts/generate_aruco.py --id 7 --size 600
```

Common options: `--image`, `--video`, `--camera N`, `--dict`, `--marker-length` (metres), `--save`, `--no-show`.
Run any script with `--help` for the full list.

## Depth-aware AR

`scripts/ar_depth.py` stands a cube on the marker and hides the parts of it that are behind real objects.

1. The marker's pose gives its true distance in metres.
2. MiDaS estimates a *relative* depth map of the whole frame. Its value at the marker, paired with the marker's
   true distance, scales that map into metres.
3. The cube is drawn with a depth for every pixel; wherever the real scene is closer than the cube, the cube is
   hidden.

Keys: **o** turns occlusion on and off, **d** shows the depth map, **q** / **Esc** quits.

- The MiDaS Small model (~63 MB) downloads to `models/` on first use. `--model large` is more detailed but about
  20× slower on a CPU.
- MiDaS runs on a background thread, so the video stays smooth; the depth used for occlusion lags by a frame or two.
- If your hand covers the marker, the cube stays where it was for a second instead of vanishing.
- Monocular depth is approximate: occlusion edges are soft, and thin things (fingers, leaves) may not hide the
  cube cleanly. `--tolerance` sets how much closer a surface must be to hide the cube.

## Camera calibration

Without calibration the demos assume a typical webcam (about 60° field of view, no lens distortion). The cube
still sits on the marker, but distances are approximate. For accurate poses, save your camera's calibration as
`calibration.npz` in the repo folder:

```python
import numpy as np
np.savez("calibration.npz", camera_matrix=K, dist_coeffs=dist, image_size=[width, height])
```

(`K` and `dist` come from `cv2.calibrateCamera`; see OpenCV's
[calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html).)

## Manual setup

```bash
python -m venv psdenv
psdenv\Scripts\activate          # Linux/macOS: source psdenv/bin/activate
pip install -r requirements.txt
python main.py
```

Install **`opencv-contrib-python` only**, not `opencv-python` as well: the two packages overwrite each other's
`cv2` module. The code uses the ArUco API from OpenCV 4.7 and newer (`ArucoDetector`, `generateImageMarker`,
`solvePnP`); the older `Dictionary_get` / `detectMarkers` / `estimatePoseSingleMarkers` functions no longer exist.

Object tracking (menu 6) needs extra, large packages (about 2.5 GB):

```bash
pip install torch deep-sort-realtime
```

## Repository layout

```
main.py                   Menu
requirements.txt
run.bat, run.ps1, run.sh  Create the venv, install requirements, start the menu
scripts/
  ar_common.py            ArUco detection, pose (solvePnP), camera intrinsics, webcam/image/video loop
  generate_aruco.py       Printable markers
  aruco_detection.py      Detection
  pose_estimation.py      Pose axes and distance
  pose_obj_object.py      OBJ model with OpenGL + pygame
  ar_depth.py             Depth-aware AR with occlusion
  depth.py                MiDaS depth (ONNX) and metric scaling
  object_tracking.py      YOLOv5 + DeepSORT tracking (optional)
  pose_object.py, glyph*.py, webcam.py, constants.py   Older glyph-marker version (not maintained; needs
                                                       cube_0-3.obj files that are not in the repo)
objects/                  cube.obj, material and texture
```

## Documentation

- [OpenCV ArUco detection tutorial](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [MiDaS monocular depth](https://github.com/isl-org/MiDaS)
- [PyOpenGL on PyPI](https://pypi.org/project/PyOpenGL/)

<br>
<div align="center">
© Cursed Entertainment
</div>
<br>
<div align="center">
<a href="https://cursed-entertainment.itch.io/" target="_blank">
    <img src="https://github.com/CursedPrograms/cursedentertainment/raw/main/images/logos/logo-wide-grey.png"
        alt="CursedEntertainment Logo" style="width:250px;">
</a>
</div>
