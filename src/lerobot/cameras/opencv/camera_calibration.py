"""
Camera calibration script.

Prints out the real fx, fy, cx, cy (and distortion coefficients) for your
webcam, using a printed checkerboard pattern. Use these values to replace
the guessed K_MATRIX in the record script.

--------------------------------------------------------------------------
STEP 1: Print a checkerboard
--------------------------------------------------------------------------
Print (or display on a second screen) a standard checkerboard pattern.
The easiest source: https://github.com/opencv/opencv/blob/4.x/doc/pattern.png
(9x6 internal corners, i.e. a 10x7 grid of squares). Tape it to something flat
(a clipboard, a book) so it doesn't bend.

Measure ONE SQUARE's side length with a ruler in meters and set
SQUARE_SIZE_M below (e.g. a square that's 2.5cm wide -> 0.025).

--------------------------------------------------------------------------
STEP 2: Set the parameters below to match your setup
--------------------------------------------------------------------------
"""

import glob
import os

import cv2
import numpy as np

CAMERA_INDEX = 2        # same index_or_path you use in --robot.cameras
IMG_WIDTH = 640
IMG_HEIGHT = 480
CHECKERBOARD = (9,6) # (columns, rows) of INTERNAL corners, not squares
SQUARE_SIZE_M = 0.0005     # <-- measure your printed square and set this
NUM_CALIBRATION_SHOTS = 15  # more is better, 15-25 is typical
SAVE_DIR = "calib_images"


def capture_calibration_images():
    """Opens the webcam and lets you capture checkerboard shots by pressing SPACE."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    print("Move the checkerboard around (tilt, rotate, different distances/corners of frame).")
    print("Press SPACE to capture a shot when corners are highlighted green.")
    print("Press ESC when you have enough shots (aim for 15-25).")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame, retrying...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)

        cv2.putText(
            display,
            f"Captured: {count}/{NUM_CALIBRATION_SHOTS}  (SPACE=capture, ESC=finish)",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if found else (0, 0, 255),
            2,
        )
        cv2.imshow("Calibration Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32 and found:  # SPACE
            filename = os.path.join(SAVE_DIR, f"calib_{count:02d}.png")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            count += 1
            if count >= NUM_CALIBRATION_SHOTS:
                print("Reached target number of shots.")
                break

    cap.release()
    cv2.destroyAllWindows()
    return count


def run_calibration():
    """Runs cv2.calibrateCamera on all saved images and prints the results."""
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    images = sorted(glob.glob(os.path.join(SAVE_DIR, "*.png")))
    if len(images) < 5:
        raise RuntimeError(
            f"Only found {len(images)} images in {SAVE_DIR}/. Need at least ~10 for a decent calibration."
        )

    img_shape = None
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
        else:
            print(f"Warning: no corners found in {fname}, skipping.")

    if len(objpoints) < 5:
        raise RuntimeError("Not enough valid checkerboard detections to calibrate.")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    print("\n=== CALIBRATION RESULT ===")
    print(f"RMS reprojection error: {ret:.4f}  (lower is better, <0.5 is good, <1.0 is usable)")
    print("\nCamera matrix K:")
    print(K)
    print("\nDistortion coefficients:")
    print(dist.ravel())

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    print("\n--- Paste this into your record script ---")
    print("K_MATRIX = np.array([")
    print(f"    [{fx:.4f}, 0.0, {cx:.4f}],")
    print(f"    [0.0, {fy:.4f}, {cy:.4f}],")
    print("    [0.0, 0.0, 1.0],")
    print("], dtype=np.float32)")

    np.savez("camera_calibration.npz", K=K, dist=dist, rms_error=ret)
    print("\nAlso saved to camera_calibration.npz")


if __name__ == "__main__":
    n = capture_calibration_images()
    print(f"\nCaptured {n} images. Running calibration...\n")
    run_calibration()