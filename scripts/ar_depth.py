"""Depth-aware AR: a virtual cube stands on an ArUco marker, and real things in
front of it (your hand, a mug) hide it, using MiDaS monocular depth.

How it works:
  1. The marker's pose gives its true distance in metres.
  2. MiDaS gives a relative depth map of the whole frame. Its value at the marker,
     paired with the marker's true distance, scales that map into metres.
  3. The cube is drawn with a per-pixel depth of its own; wherever the real scene
     is closer than the cube, the cube is hidden (occlusion).

    python scripts/ar_depth.py                          # webcam
    python scripts/ar_depth.py --image photo.jpg --save out.png --no-show
    Keys: o = occlusion on/off, d = show depth map, q / Esc = quit

MiDaS runs on a background thread, so the video stays smooth; the depth used for
occlusion lags the picture by a frame or two.
"""
import threading
import time

import cv2
import numpy as np

import ar_common
import depth

# Cube faces as vertex indices (vertices 0-3 on the marker, 4-7 on top)
CUBE_FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
CUBE_COLOR = np.array([60, 170, 255], np.float32)  # BGR, orange
# Direction to the light in OpenCV camera space (x right, y down, z forward): upper left, behind the camera
TO_LIGHT = np.array([-0.4, -0.7, -0.6]) / np.linalg.norm([-0.4, -0.7, -0.6])
HOLD_SECONDS = 1.0                                  # keep a lost marker's cube this long


def cube_vertices(length, height=None):
    """Cube standing on the marker: base = the marker square, extending out of it
    (+z in the marker frame points from the marker towards the camera)."""
    h = length / 2
    top = height if height is not None else length
    base = [[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]]
    return np.array(base + [[x, y, top] for x, y, _ in base], dtype=np.float64)


def render_cube(shape, K, rvec, tvec, length):
    """Draw the cube into (colour image, per-pixel depth in metres, coverage mask)."""
    h, w = shape[:2]
    color = np.zeros((h, w, 3), np.uint8)
    zbuf = np.full((h, w), np.inf, np.float32)

    R, _ = cv2.Rodrigues(rvec)
    verts_cam = (R @ cube_vertices(length).T + tvec.reshape(3, 1)).T  # camera space, metres
    if (verts_cam[:, 2] <= 0.01).any():
        return color, zbuf, np.zeros((h, w), bool)  # behind or at the camera
    pts2d = (K @ (verts_cam / verts_cam[:, 2:3]).T).T[:, :2]

    Kinv = np.linalg.inv(K)
    ys, xs = np.mgrid[0:h, 0:w]
    cube_centre = verts_cam.mean(axis=0)
    front_faces = []
    for face in CUBE_FACES:
        p3 = verts_cam[list(face)]
        normal = np.cross(p3[1] - p3[0], p3[2] - p3[0])
        normal /= np.linalg.norm(normal)
        centre = p3.mean(axis=0)
        if normal @ (centre - cube_centre) < 0:
            normal = -normal  # make it point out of the cube
        if normal @ centre >= 0:
            continue  # faces away from the camera: hidden behind the front faces
        front_faces.append(face)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, np.round(pts2d[list(face)]).astype(np.int32), 1, cv2.LINE_AA)
        sel = mask.astype(bool)
        if not sel.any():
            continue
        # Exact depth of the face plane along each pixel's viewing ray
        rays = Kinv @ np.stack([xs[sel], ys[sel], np.ones(sel.sum())])
        denom = normal @ rays
        z = np.where(np.abs(denom) > 1e-9, (normal @ p3[0]) / denom, np.inf)
        closer = z < zbuf[sel]
        idx = np.flatnonzero(sel.ravel())[closer]
        zbuf.ravel()[idx] = z[closer]
        shade = 0.4 + 0.6 * max(0.0, float(normal @ TO_LIGHT))  # normal points out of the cube
        color.reshape(-1, 3)[idx] = (CUBE_COLOR * shade).astype(np.uint8)

    # Dark edges on the visible faces make the shape readable
    for face in front_faces:
        cv2.polylines(color, [np.round(pts2d[list(face)]).astype(np.int32)], True, (20, 40, 80), 2, cv2.LINE_AA)
    covered = np.isfinite(zbuf)
    return color, zbuf, covered


class DepthWorker:
    """Runs MiDaS on the newest frame in a background thread."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.lock = threading.Lock()
        self.frame = None
        self.result = None  # (raw depth, frame it was computed from)
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, frame):
        with self.lock:
            self.frame = frame

    def latest(self):
        with self.lock:
            return self.result

    def _loop(self):
        while self.running:
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                time.sleep(0.005)
                continue
            raw = self.estimator.raw(frame)
            with self.lock:
                self.result = (raw, frame)


def main():
    p = ar_common.parser("Depth-aware AR with an ArUco marker and MiDaS")
    ar_common.add_source_args(p)
    ar_common.add_aruco_args(p)
    p.add_argument("--model", choices=["small", "large"], default="small",
                   help="MiDaS model: small is fast, large is better but ~20x slower on a CPU")
    p.add_argument("--tolerance", type=float, default=0.3,
                   help="how much closer (as a fraction of the marker size) a real surface must be to hide the cube")
    args = p.parse_args()

    detector = ar_common.make_detector(args.dict)
    estimator = depth.DepthEstimator(args.model)
    worker = None if args.image else DepthWorker(estimator)
    state = {"occlusion": True, "show_depth": False, "raw": None}
    tracked = {}  # marker id -> (rvec, tvec, depth scale, time last seen)

    def process(frame):
        h, w = frame.shape[:2]
        K, dist = ar_common.camera_intrinsics(w, h)

        # Depth: synchronous for a single image, background thread for video
        if worker is None:
            raw = estimator.raw(frame)
        else:
            worker.submit(frame.copy())
            result = worker.latest()
            raw = result[0] if result is not None else None
        state["raw"] = raw

        corners, ids, _ = detector.detectMarkers(frame)
        now = time.time()
        if ids is not None:
            for c, marker_id in zip(corners, ids.flatten()):
                pose = ar_common.estimate_pose(c, args.marker_length, K, dist)
                if pose is None:
                    continue
                rvec, tvec = pose
                # Metric scale for the depth map, measured while the marker is visible
                scale = None
                if raw is not None:
                    marker_mask = np.zeros((h, w), np.uint8)
                    cv2.fillConvexPoly(marker_mask, c.reshape(4, 2).astype(np.int32), 1)
                    scale = depth.metric_scale(raw, marker_mask.astype(bool), float(tvec.ravel()[2]))
                if scale is None and int(marker_id) in tracked:
                    scale = tracked[int(marker_id)][2]
                tracked[int(marker_id)] = (rvec, tvec, scale, now)
        # Keep a marker's last pose (and depth scale) briefly after it is lost: a hand
        # passing in front of the cube often covers the marker too, and the cube should
        # stay put behind it. The marker's pixels then show the hand, so the scale from
        # when the marker was last visible is reused.
        for marker_id in [m for m, t in tracked.items() if now - t[3] > HOLD_SECONDS]:
            del tracked[marker_id]

        out = frame.copy()
        for rvec, tvec, scale, _ in tracked.values():
            color, zbuf, covered = render_cube(frame.shape, K, rvec, tvec, args.marker_length)
            visible = covered
            if state["occlusion"] and raw is not None and scale is not None:
                real = depth.to_metres(raw, scale)
                tol = args.tolerance * args.marker_length
                visible = covered & ~(real < zbuf - tol)
                # Clean up speckle in the occlusion mask
                visible = cv2.morphologyEx(visible.astype(np.uint8), cv2.MORPH_OPEN,
                                           np.ones((5, 5), np.uint8)).astype(bool)
            out[visible] = color[visible]

        if state["show_depth"] and raw is not None:
            inset = cv2.resize(depth.colorize(raw), (w // 4, h // 4))
            out[:h // 4, w - w // 4:] = inset

        cv2.putText(out, f"occlusion {'on' if state['occlusion'] else 'off'} (o)  depth map (d)",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    if args.image:
        ar_common.run(args, process, "Depth AR")
        return

    # Video/webcam loop with the o / d keys
    cap = ar_common.open_capture(args)
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = process(frame)
        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         cap.get(cv2.CAP_PROP_FPS) or 30, (out.shape[1], out.shape[0]))
            writer.write(out)
        if not args.no_show:
            cv2.imshow("Depth AR", out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("o"):
                state["occlusion"] = not state["occlusion"]
            if key == ord("d"):
                state["show_depth"] = not state["show_depth"]
    worker.running = False
    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved {args.save}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
