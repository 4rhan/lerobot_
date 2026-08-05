import cv2
import numpy as np
import torch
from lerobot.datasets import LeRobotDataset

def project_3d_to_2d(points_3d, c2w, K):
    """
    Takes 3D world points and projects them onto the 2D image
    using the camera extrinsics (c2w) and intrinsics (K).
    """
    # 1. Convert c2w (Camera-to-World) into w2c (World-to-Camera)
    w2c = np.linalg.inv(c2w)

    # 2. Add homogeneous coordinate (1.0) to 3D points: (X, Y, Z, 1)
    ones = np.ones((points_3d.shape[0], 1))
    points_3d_homo = np.hstack([points_3d, ones])

    # 3. Transform points from World Space to Camera Space
    points_cam = (w2c @ points_3d_homo.T).T
    points_cam = points_cam[:, :3]  # Drop the 4th coordinate

    # 4. Project points using Camera Intrinsics (K)
    points_2d_homo = (K @ points_cam.T).T

    # 5. Divide by Z to get standard 2D pixel coordinates (u, v)
    u = points_2d_homo[:, 0] / (points_2d_homo[:, 2] + 1e-6)
    v = points_2d_homo[:, 1] / (points_2d_homo[:, 2] + 1e-6)

    return np.stack([u, v], axis=-1).astype(int)


def main():
    # =================================================================
    # SETUP: CHANGE THESE TO MATCH YOUR RECORDING
    # =================================================================
    dataset_repo_id = "Archaive16/record-camOpenSource_condition_test_11_20260804_002033"
    tag_size = 0.052  # physical size of the black square, in meters
    camera_name = "front"
    # =================================================================

    print(f"Loading dataset: {dataset_repo_id}...")
    dataset = LeRobotDataset(dataset_repo_id)
    print("Available keys in dataset:", dataset[0].keys())

    half = tag_size / 2.0
    tag_3d_corners = np.array([
        [-half, half, 0],
        [half, half, 0],
        [half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float32)

    axis_3d = np.array([
        [0, 0, 0],
        [0, 0, tag_size * 1.5],
    ], dtype=np.float32)

    # Full 3-axis gizmo at the tag origin, for checking ROTATION, not just distance.
    # X (red), Y (green), Z (blue) — standard convention. If these point in
    # physically wrong directions (e.g. blue not pointing up off the table),
    # the rotation part of c2w is broken even if distance looks fine.
    gizmo_3d = np.array([
        [0, 0, 0],              # origin
        [tag_size * 1.5, 0, 0],  # X tip
        [0, tag_size * 1.5, 0],  # Y tip
        [0, 0, tag_size * 1.5],  # Z tip
    ], dtype=np.float32)

    print("Playing video... Press 'q' to quit, SPACE to pause, 'p' to print current c2w.")
    print("-" * 60)
    print("VERIFICATION TIPS:")
    print("  1. Distance shown on-screen should roughly match a ruler")
    print("     measurement from your camera lens to the tag center.")
    print("  2. Tilt angle (0deg = straight overhead, 90deg = edge-on at")
    print("     table height) should roughly match how your camera is")
    print("     actually mounted/angled — eyeball your setup and compare.")
    print("  3. The RGB gizmo at the tag origin shows full orientation:")
    print("     blue (Z) should point up off the table, red (X) and")
    print("     green (Y) should lie roughly flat along the table,")
    print("     matching the tag's printed edges.")
    print("  4. While the camera/tag stay still, watch the terminal —")
    print("     distance and tilt should stay stable, not drift/jump,")
    print("     across consecutive frames where the tag is detected.")
    print("-" * 60)

    distances = []  # for a simple stability summary at the end
    tilts = []
    frame_h, frame_w = None, None

    for i in range(len(dataset)):
        frame = dataset[i]

        img_tensor = frame[f"observation.images.{camera_name}"]
        c2w = frame[f"observation.extrinsics.{camera_name}"].numpy()
        K = frame[f"observation.intrinsics.{camera_name}"].numpy()

        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        frame_h, frame_w = img_cv2.shape[:2]

        if np.allclose(c2w, np.eye(4)):
            cv2.putText(img_cv2, "No Tag Extrinsics Saved", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            pixels_corners = project_3d_to_2d(tag_3d_corners, c2w, K)
            pixels_axis = project_3d_to_2d(axis_3d, c2w, K)
            pixels_gizmo = project_3d_to_2d(gizmo_3d, c2w, K)

            cv2.polylines(img_cv2, [pixels_corners], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.line(img_cv2, tuple(pixels_axis[0]), tuple(pixels_axis[1]), color=(0, 0, 255), thickness=3)

            # Full pose gizmo: X=red, Y=green, Z=blue, all from the tag origin.
            origin = tuple(pixels_gizmo[0])
            cv2.line(img_cv2, origin, tuple(pixels_gizmo[1]), color=(0, 0, 255), thickness=3)   # X red
            cv2.line(img_cv2, origin, tuple(pixels_gizmo[2]), color=(0, 255, 0), thickness=3)   # Y green
            cv2.line(img_cv2, origin, tuple(pixels_gizmo[3]), color=(255, 0, 0), thickness=3)   # Z blue

            cv2.putText(img_cv2, "3D GEOMETRY VALID!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # --- Ruler check: camera position in tag/world coords ---
            cam_pos = c2w[:3, 3]
            distance_cm = float(np.linalg.norm(cam_pos)) * 100.0
            distances.append(distance_cm)

            # --- Pose check: tilt angle of the camera relative to straight-overhead. ---
            # 0deg = camera directly above the tag looking straight down.
            # 90deg = camera at table height, looking at the tag edge-on.
            # This is something you can independently eyeball/estimate by looking at
            # your physical setup, then compare against the printed number.
            norm_pos = cam_pos / (np.linalg.norm(cam_pos) + 1e-9)
            tilt_deg = float(np.degrees(np.arccos(np.clip(norm_pos[2], -1.0, 1.0))))
            tilts.append(tilt_deg)

            cv2.putText(img_cv2, f"Est. distance to tag: {distance_cm:.1f} cm", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img_cv2, f"Est. tilt from overhead: {tilt_deg:.1f} deg", (50, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # --- Sanity flag: if the projected box lands way outside the ---
            # frame, that's a strong sign something is still off (wrong  ---
            # matrix direction, bad K, etc.) even if the tag was "found". ---
            margin = 0.5  # allow up to 50% outside frame before flagging
            out_of_bounds = (
                pixels_corners[:, 0].min() < -margin * frame_w
                or pixels_corners[:, 0].max() > frame_w * (1 + margin)
                or pixels_corners[:, 1].min() < -margin * frame_h
                or pixels_corners[:, 1].max() > frame_h * (1 + margin)
            )
            if out_of_bounds:
                cv2.putText(img_cv2, "WARNING: projection far outside frame!", (50, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if i % 15 == 0:  # log every ~0.5s at 30fps instead of flooding the terminal
                print(f"[frame {i}] distance={distance_cm:.1f}cm  tilt={tilt_deg:.1f}deg  "
                      f"cam_pos(xyz)={cam_pos.round(3)}")

        cv2.imshow("Dataset 3D Validation", img_cv2)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
        elif key == ord('p') and not np.allclose(c2w, np.eye(4)):
            print("\nFull c2w matrix:")
            print(c2w)
            print()

    cv2.destroyAllWindows()

    if distances:
        arr = np.array(distances)
        tilt_arr = np.array(tilts)
        print("-" * 60)
        print(f"Distance stats over {len(arr)} tag-visible frames:")
        print(f"  mean={arr.mean():.1f}cm  std={arr.std():.1f}cm  "
              f"min={arr.min():.1f}cm  max={arr.max():.1f}cm")
        print("  Compare 'mean' against a ruler measurement of the real distance.")
        print(f"Tilt stats: mean={tilt_arr.mean():.1f}deg  std={tilt_arr.std():.1f}deg  "
              f"min={tilt_arr.min():.1f}deg  max={tilt_arr.max():.1f}deg")
        print("  Compare 'mean' against your camera's actual mounting angle.")
        print("  A large 'std' on either metric (jumping around a lot) suggests")
        print("  noisy detection/K, even if the mean happens to look roughly right.")
    else:
        print("No frames had the tag detected — nothing to check.")


if __name__ == "__main__":
    main()