"""Generate a printable ArUco marker PNG into arucoMarkers/.

    python scripts/generate_aruco.py                    # DICT_5X5_100, ID 1
    python scripts/generate_aruco.py --id 7 --size 500 --dict DICT_4X4_50

Print it so the black square is --marker-length wide (5 cm by default) for correct
distances in the pose and depth demos. Keep the white border around the marker.
"""
import argparse
import os

import cv2

import ar_common


def main():
    p = argparse.ArgumentParser(description="Generate an ArUco marker")
    p.add_argument("--dict", default=ar_common.DEFAULT_DICT, choices=sorted(ar_common.ARUCO_DICT))
    p.add_argument("--id", type=int, default=1, help="marker ID within the dictionary")
    p.add_argument("--size", type=int, default=400, help="marker size in pixels, without the border")
    p.add_argument("--no-show", action="store_true", help="don't open a preview window")
    args = p.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(ar_common.ARUCO_DICT[args.dict])
    tag = cv2.aruco.generateImageMarker(dictionary, args.id, args.size)
    # A white quiet zone around the marker is needed for detection
    border = args.size // 8
    tag = cv2.copyMakeBorder(tag, border, border, border, border, cv2.BORDER_CONSTANT, value=255)

    out_dir = os.path.join(ar_common.REPO_DIR, "arucoMarkers")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{args.dict}_{args.id}.png")
    cv2.imwrite(path, tag)
    print(f"ArUco {args.dict} ID {args.id} saved to {path}")

    if not args.no_show:
        cv2.imshow("ArUco marker", tag)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
