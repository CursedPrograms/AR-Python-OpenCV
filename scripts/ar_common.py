"""Shared ArUco helpers for the AR scripts (OpenCV 4.7+ API).

OpenCV 4.7 replaced the old functional ArUco API (Dictionary_get, DetectorParameters_create,
detectMarkers, drawMarker, estimatePoseSingleMarkers) with ArucoDetector, generateImageMarker
and plain solvePnP. Everything here uses the new API.
"""
import argparse
import os
import sys

import cv2
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION_FILE = os.path.join(REPO_DIR, "calibration.npz")

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

DEFAULT_DICT = "DICT_5X5_100"
DEFAULT_MARKER_LENGTH = 0.05  # printed marker side in metres (5 cm)


# ==================== DETECTION AND POSE ====================

def make_detector(dict_name=DEFAULT_DICT):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dict_name])
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def marker_object_points(length):
    """The marker's corners in its own frame (metres), in ArUco's corner order:
    top-left, top-right, bottom-right, bottom-left. This is the layout
    SOLVEPNP_IPPE_SQUARE expects."""
    h = length / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)


def estimate_pose(corners, length, camera_matrix, dist_coeffs):
    """rvec, tvec (marker -> camera, metres) for one marker's corners (1x4x2), or None.
    Replaces cv2.aruco.estimatePoseSingleMarkers, removed in OpenCV 4.7."""
    ok, rvec, tvec = cv2.solvePnP(marker_object_points(length), corners.reshape(4, 2).astype(np.float32),
                                  camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (rvec, tvec) if ok else None


def draw_markers(image, corners, ids):
    """Outline, centre dot and ID for each detected marker."""
    if ids is None:
        return image
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        pts = marker_corners.reshape(4, 2).astype(int)
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.circle(image, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.putText(image, str(marker_id), (int(pts[0][0]), int(pts[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


# ==================== CAMERA INTRINSICS ====================

def camera_intrinsics(width, height):
    """(camera_matrix, dist_coeffs) for a frame size.

    Uses calibration.npz in the repo root if present (keys: camera_matrix, dist_coeffs,
    and optionally image_size = [w, h]; the matrix is rescaled if the frame size differs).
    Otherwise approximates a typical webcam (about 60 degrees horizontal field of view,
    no lens distortion), which is good enough for placing objects on a marker."""
    if os.path.exists(CALIBRATION_FILE):
        data = np.load(CALIBRATION_FILE)
        K = data["camera_matrix"].astype(np.float64).copy()
        dist = data["dist_coeffs"].astype(np.float64)
        if "image_size" in data:
            sx, sy = width / float(data["image_size"][0]), height / float(data["image_size"][1])
            K[0, :] *= sx
            K[1, :] *= sy
        return K, dist
    f = 0.87 * width
    K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64)
    return K, np.zeros(5)


# ==================== INPUT / OUTPUT LOOP ====================

def add_source_args(parser):
    parser.add_argument("--image", help="use an image instead of the webcam")
    parser.add_argument("--video", help="use a video file instead of the webcam")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--save", help="also write the result to this image (for --image) or video file")
    parser.add_argument("--no-show", action="store_true", help="don't open a window (use with --save)")
    return parser


def add_aruco_args(parser):
    parser.add_argument("--dict", default=DEFAULT_DICT, choices=sorted(ARUCO_DICT), help="ArUco dictionary")
    parser.add_argument("--marker-length", type=float, default=DEFAULT_MARKER_LENGTH,
                        help="printed marker side in metres (default 0.05)")
    return parser


def open_capture(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        # CAP_DSHOW avoids a long startup timeout on Windows
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(args.camera, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit(f"Could not open {'video ' + args.video if args.video else 'webcam ' + str(args.camera)}")
    return cap


def run(args, process, window="AR"):
    """Feed frames from --image, --video or the webcam through process(frame) -> frame.
    Shows the result (q or Esc quits) and/or writes it with --save."""
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Could not read image {args.image}")
        out = process(frame)
        if args.save:
            cv2.imwrite(args.save, out)
            print(f"Saved {args.save}")
        if not args.no_show:
            cv2.imshow(window, out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    cap = open_capture(args)
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = process(frame)
        if args.save:
            if writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (out.shape[1], out.shape[0]))
            writer.write(out)
        if not args.no_show:
            cv2.imshow(window, out)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved {args.save}")
    cv2.destroyAllWindows()


def parser(description):
    return argparse.ArgumentParser(description=description)
