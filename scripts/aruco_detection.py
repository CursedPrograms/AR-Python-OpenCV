"""Detect ArUco markers and draw their outline and ID.

    python scripts/aruco_detection.py                       # webcam
    python scripts/aruco_detection.py --image photo.jpg
    python scripts/aruco_detection.py --dict DICT_4X4_50
"""
import ar_common


def main():
    p = ar_common.parser("Detect ArUco markers")
    ar_common.add_source_args(p)
    ar_common.add_aruco_args(p)
    args = p.parse_args()

    detector = ar_common.make_detector(args.dict)
    seen = set()

    def process(frame):
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            new = set(ids.flatten().tolist()) - seen
            if new:
                print("ArUco marker ID(s):", sorted(new))
                seen.update(new)
        return ar_common.draw_markers(frame, corners, ids)

    ar_common.run(args, process, "ArUco detection")


if __name__ == "__main__":
    main()
