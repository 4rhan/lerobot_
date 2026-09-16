"""
Quick sanity check: confirms your printed AprilTag (36h11) is detected live
from a webcam, and estimates the camera-to-tag distance.

Run:
    uv run python src/lerobot/scripts/test_april_tag.py --camera-index 0 --tag-size 0.05

Note: To see the live OpenCV window, you must install the GUI version of OpenCV:
    pip uninstall opencv-python-headless
    pip install opencv-python
"""

import argparse
import time
import sys

import cv2
import numpy as np

from lerobot.utils.apriltag_pose import K_MATRIX, TAG_SIZE_M, load_camera_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-index", type=int, default=0, help="Same index_or_path you use in --robot.cameras.")
    parser.add_argument("--camera-name", type=str, default=None, help="Loads <camera-name>_calibration.npz if given.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--tag-size",
        type=float,
        default=TAG_SIZE_M,
        help="Physical size of the printed tag's black square in meters (e.g. 0.05 for 5cm). CRITICAL for accurate distance.",
    )
    parser.add_argument(
        "--debug-image-path",
        type=str,
        default="test_april_tag_debug.jpg",
        help="Where to save the annotated frame when no GUI window is available.",
    )
    args = parser.parse_args()

    # 1. Load Camera Matrix
    if args.camera_name:
        K = load_camera_matrix(args.camera_name)
    else:
        K = K_MATRIX
        print("\n[WARNING] No --camera-name given. Using placeholder camera matrix.")
        print("[WARNING] The distance estimation WILL BE INCORRECT without real calibration.\n")

    # 2. Setup Camera
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    # 3. Setup AprilTag Detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.01
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # 4. Define 3D Object Points for Pose Estimation
    half_s = args.tag_size / 2.0
    # ArUco standard corner order: top-left, top-right, bottom-right, bottom-left
    obj_points = np.array([
        [-half_s,  half_s, 0],
        [ half_s,  half_s, 0],
        [ half_s, -half_s, 0],
        [-half_s, -half_s, 0]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1)) # Assume minimal distortion for visualization

    window_name = "Live Tag Debugger"
    gui_available = True
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error:
        gui_available = False
        print(
            "*** NO GUI BACKEND DETECTED ***\n"
            "You are likely using opencv-python-headless. To see the live window, run:\n"
            "    pip uninstall opencv-python-headless\n"
            "    pip install opencv-python\n\n"
            f"Falling back to saving annotated frames to '{args.debug_image_path}'. Ctrl+C to quit."
        )

    print(f"Looking for AprilTag (Expecting physical size: {args.tag_size*100:.2f} cm)...")
    last_report_t = 0.0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            corners, ids, _rejected = detector.detectMarkers(gray)

            found = ids is not None and len(ids) > 0
            if found:
                # 5. Draw a thick yellow 2D Bounding Box
                tag_corners = corners[0][0]
                cv2.polylines(frame_bgr, [tag_corners.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=3)

                # 6. Estimate Pose and Distance
                success, rvec, tvec = cv2.solvePnP(obj_points, corners[0], K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                
                if success:
                    # Calculate distance as the magnitude of the translation vector
                    distance_cm = float(np.linalg.norm(tvec)) * 100.0
                    
                    # Draw 3D axes (Red=X, Green=Y, Blue=Z)
                    cv2.drawFrameAxes(frame_bgr, K, dist_coeffs, rvec, tvec, args.tag_size / 2)

                    tag_id = ids.flatten()[0]
                    status = f"TAG ID: {tag_id} | Dist: {distance_cm:.1f}cm"
                    color = (0, 255, 0)
                else:
                    status = "Pose Estimation Failed"
                    color = (0, 165, 255)
            else:
                status = "No Tag Seen"
                color = (0, 0, 255)

            cv2.putText(frame_bgr, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if gui_available:
                cv2.imshow(window_name, frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                now = time.monotonic()
                if now - last_report_t >= 1.0:
                    print(status)
                    cv2.imwrite(args.debug_image_path, frame_bgr)
                    last_report_t = now
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if gui_available:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()