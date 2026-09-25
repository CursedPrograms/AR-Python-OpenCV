"""Estimate each ArUco marker's 3D pose and draw its axes and distance.

    python scripts/pose_estimation.py                   # webcam
    python scripts/pose_estimation.py --image photo.jpg --marker-length 0.05

Distances are only right if --marker-length matches the printed marker. For accurate
poses, put your camera calibration in calibration.npz (see ar_common.camera_intrinsics).
"""
import numpy as np
import cv2

import ar_common


def main():
    p = ar_common.parser("ArUco pose estimation")
    ar_common.add_source_args(p)
    ar_common.add_aruco_args(p)
    args = p.parse_args()

    detector = ar_common.make_detector(args.dict)

    def process(frame):
        K, dist = ar_common.camera_intrinsics(frame.shape[1], frame.shape[0])
        corners, ids, _ = detector.detectMarkers(frame)
        ar_common.draw_markers(frame, corners, ids)
        if ids is None:
            return frame
        for c in corners:
            pose = ar_common.estimate_pose(c, args.marker_length, K, dist)
            if pose is None:
                continue
            rvec, tvec = pose
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, args.marker_length * 0.75)
            x, y = c.reshape(4, 2)[2].astype(int)
            cv2.putText(frame, f"{np.linalg.norm(tvec):.2f} m", (int(x), int(y) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return frame

    ar_common.run(args, process, "Estimated pose")


if __name__ == "__main__":
    main()
