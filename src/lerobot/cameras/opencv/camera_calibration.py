"""
Camera calibration script.

Prints out the real fx, fy, cx, cy (and distortion coefficients) for your
webcam, using a printed checkerboard pattern, and saves them to
``<camera-name>_calibration.npz`` for ``lerobot.utils.apriltag_pose`` to pick
up automatically when recording (see `load_camera_matrix`).

Run this once per physical camera you plan to record with — intrinsics differ
per lens/sensor, so a single global K matrix is wrong for anything but a
single-camera setup.

--------------------------------------------------------------------------
STEP 1: Print a checkerboard
--------------------------------------------------------------------------
Print (or display on a second screen) a standard checkerboard pattern.
The easiest source: https://github.com/opencv/opencv/blob/4.x/doc/pattern.png
(9x6 internal corners, i.e. a 10x7 grid of squares). Tape it to something flat
(a clipboard, a book) so it doesn't bend.

Measure ONE SQUARE's side length with a ruler in meters and pass it as
--square-size-m (e.g. a square that's 2.5cm wide -> 0.025).

--------------------------------------------------------------------------
STEP 2: Run this script
--------------------------------------------------------------------------
uv run python -m lerobot.cameras.opencv.camera_calibration \\
    --camera-name front \\
    --camera-index 0 \\
    --square-size-m 0.025
"""

import argparse
import glob
import os

import cv2
import numpy as np


def capture_calibration_images(
    camera_index: int, img_width: int, img_height: int, checkerboard: tuple[int, int], save_dir: str, num_shots: int
) -> int:
    """Opens the webcam and captures checkerboard shots.

    Uses a live GUI window (SPACE to capture, ESC to finish) when available. `lerobot` pins
    opencv-python-headless, so on most setups there's no GUI backend — in that case this falls
    back to a manual terminal trigger: hold the board still, then press Enter (or type a letter
    and hit Enter) to capture the current frame; type 'q' + Enter to finish early.
    """
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    window_name = "Calibration Capture"
    gui_available = True
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error:
        gui_available = False

    if gui_available:
        print("Move the checkerboard around (tilt, rotate, different distances/corners of frame).")
        print("Press SPACE to capture a shot when corners are highlighted green.")
        print("Press ESC when you have enough shots (aim for 15-25).")
    else:
        print(
            "No GUI backend available (opencv-python-headless is installed) — using terminal capture "
            "instead. Position the checkerboard (tilt/rotate/distance), hold it still, then press "
            "Enter to capture the current frame. Type 'q' + Enter to finish early once you have "
            "enough shots (aim for 15-25)."
        )

    count = 0

    try:
        if gui_available:
            while count < num_shots:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame, retrying...")
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(gray, checkerboard, None)

                display = frame.copy()
                if found:
                    cv2.drawChessboardCorners(display, checkerboard, corners, found)
                cv2.putText(
                    display,
                    f"Captured: {count}/{num_shots}  (SPACE=capture, ESC=finish)",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if found else (0, 0, 255),
                    2,
                )
                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == 32 and found:  # SPACE
                    filename = os.path.join(save_dir, f"calib_{count:02d}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Saved {filename}")
                    count += 1
        else:
            while count < num_shots:
                user_input = input(f"[{count}/{num_shots}] Position the board, then press Enter to capture "
                                    "('q' to finish): ")
                if user_input.strip().lower() == "q":
                    break

                # Grab a couple of frames to flush any stale buffered ones, so what we save/check
                # is the frame as it looks right now, not a moment before you pressed Enter.
                for _ in range(2):
                    ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame, try again.")
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, _corners = cv2.findChessboardCorners(gray, checkerboard, None)
                if not found:
                    print("No checkerboard detected in that frame — not saved, try again.")
                    continue

                filename = os.path.join(save_dir, f"calib_{count:02d}.png")
                cv2.imwrite(filename, frame)
                print(f"Captured {count + 1}/{num_shots}: {filename}")
                count += 1
    except (KeyboardInterrupt, EOFError):
        print(f"\nStopped early with {count} shots captured.")

    if count >= num_shots:
        print("Reached target number of shots.")

    cap.release()
    if gui_available:
        cv2.destroyAllWindows()
    return count


def run_calibration(
    checkerboard: tuple[int, int], square_size_m: float, save_dir: str, out_path: str
) -> None:
    """Runs cv2.calibrateCamera on all saved images and writes K/dist to `out_path`."""
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : checkerboard[0], 0 : checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size_m

    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    images = sorted(glob.glob(os.path.join(save_dir, "*.png")))
    if len(images) < 5:
        raise RuntimeError(
            f"Only found {len(images)} images in {save_dir}/. Need at least ~10 for a decent calibration."
        )

    # Saved so you can visually confirm cv2 is finding the checkerboard where you expect it to —
    # especially useful with no live GUI. If a frame's overlay shows corners in the wrong place, on
    # the wrong sub-grid, or misses squares, that image (or --checkerboard-cols/rows) is the problem.
    annotated_dir = os.path.join(save_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    img_shape = None
    used_filenames = []
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, checkerboard, None)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            used_filenames.append(fname)

            annotated = img.copy()
            cv2.drawChessboardCorners(annotated, checkerboard, corners_refined, found)
            cv2.imwrite(os.path.join(annotated_dir, os.path.basename(fname)), annotated)
        else:
            print(f"Warning: no corners found in {fname}, skipping.")

    if len(objpoints) < 5:
        raise RuntimeError(
            "Not enough valid checkerboard detections to calibrate. If corners were found in most/all "
            "images but the count still seems off, double check --checkerboard-cols/--checkerboard-rows "
            "match your actual printed board's INTERNAL corners (not the number of squares)."
        )
    print(f"\nSaved corner-overlay images to {annotated_dir}/ — open a few to sanity check detection.")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    # Per-image reprojection error: one or two badly-detected/blurred/warped shots can drag the
    # overall RMS way up even if the rest are fine. This tells you which files to throw out.
    print("\nPer-image RMS reprojection error (same units/scale as the overall RMS below):")
    per_image_errors = []
    for i, (objp_i, imgp_i, rvec, tvec) in enumerate(zip(objpoints, imgpoints, rvecs, tvecs)):
        projected, _ = cv2.projectPoints(objp_i, rvec, tvec, K, dist)
        # RMS per point: sqrt(mean squared distance), NOT norm/N (that under-reports by ~sqrt(N)
        # and isn't comparable to `ret` below, which is the true pooled RMS across all images).
        diffs = imgp_i.reshape(-1, 2) - projected.reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))
        per_image_errors.append(error)
        flag = "  <-- high" if error > 1.5 * ret else ""
        print(f"  {os.path.basename(used_filenames[i])}: {error:.4f}{flag}")

    # A high-error frame here usually means findChessboardCorners mis-ordered the grid for that
    # shot (common when a hand occludes part of the board, or blur/glare makes squares near an
    # edge ambiguous) — the 2D points it found are real, but assigned to the wrong 3D grid
    # position, which poisons the joint fit even though every other frame is fine. Automatically
    # drop those and refit once on the clean subset rather than requiring a manual recapture.
    OUTLIER_THRESHOLD_PX = 1.0
    keep_idx = [i for i, e in enumerate(per_image_errors) if e <= OUTLIER_THRESHOLD_PX]
    dropped = [used_filenames[i] for i in range(len(used_filenames)) if i not in keep_idx]

    if dropped and len(keep_idx) >= 5:
        print(
            f"\n{len(dropped)} frame(s) exceeded {OUTLIER_THRESHOLD_PX}px and look mis-detected "
            f"(likely occluded/misordered corners, not bad calibration data): "
            f"{[os.path.basename(f) for f in dropped]}"
        )
        print(f"Refitting on the remaining {len(keep_idx)} clean frames...")
        objpoints = [objpoints[i] for i in keep_idx]
        imgpoints = [imgpoints[i] for i in keep_idx]
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    elif dropped:
        print(
            f"\n{len(dropped)} frame(s) look mis-detected, but dropping them would leave fewer than 5 "
            f"images — keeping the full set. Recommend recapturing a few more clean shots (avoid "
            f"gripping over the pattern, keep the board flat and fully unoccluded) and rerunning."
        )

    print("\n=== CALIBRATION RESULT ===")
    print(f"RMS reprojection error: {ret:.4f}  (lower is better, <0.5 is good, <1.0 is usable)")
    if ret >= 1.0:
        print(
            "WARNING: RMS error is still high after dropping outlier frames. Check the annotated "
            "images — if corners look correctly detected (a clean grid following the physical "
            "checkerboard, not skewed/diagonal) on a flat, rigid, well-lit, fully-unoccluded board, "
            "recapture with more varied tilt/distance and rerun."
        )
    print("\nCamera matrix K:")
    print(K)
    print("\nDistortion coefficients:")
    print(dist.ravel())

    np.savez(out_path, K=K.astype(np.float32), dist=dist, rms_error=ret)
    print(f"\nSaved to {out_path}")
    print("lerobot-record will pick this up automatically for a camera of this name.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--camera-name",
        required=True,
        help="Must match the key used in --robot.cameras. Output saved as <camera-name>_calibration.npz.",
    )
    parser.add_argument("--camera-index", type=int, required=True, help="Same index_or_path you use in --robot.cameras.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--checkerboard-cols", type=int, default=9, help="Internal corners, not squares.")
    parser.add_argument("--checkerboard-rows", type=int, default=6, help="Internal corners, not squares.")
    parser.add_argument(
        "--square-size-m",
        type=float,
        required=True,
        help="Measure one printed checkerboard square's side with a ruler, in meters (e.g. 0.025 for 2.5cm).",
    )
    parser.add_argument("--num-shots", type=int, default=15)
    parser.add_argument(
        "--skip-capture",
        action="store_true",
        help="Re-run calibration on images already in calib_images/<camera-name>/ instead of capturing new ones.",
    )
    args = parser.parse_args()

    checkerboard = (args.checkerboard_cols, args.checkerboard_rows)
    save_dir = os.path.join("calib_images", args.camera_name)
    out_path = f"{args.camera_name}_calibration.npz"

    if not args.skip_capture:
        n = capture_calibration_images(
            args.camera_index, args.width, args.height, checkerboard, save_dir, args.num_shots
        )
        print(f"\nCaptured {n} images. Running calibration...\n")
    run_calibration(checkerboard, args.square_size_m, save_dir, out_path)


if __name__ == "__main__":
    main()
