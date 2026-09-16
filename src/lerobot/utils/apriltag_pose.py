# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared AprilTag-based camera pose estimation.

Single source of truth for the intrinsics/extrinsics estimation used by
``lerobot-record`` when collecting data for camera-conditioned policies (e.g.
ACT with Plucker-embedding conditioning), and by the rollout pipeline so
inference sees ``observation.extrinsics.<cam>`` / ``observation.intrinsics.<cam>``
computed the exact same way they were at record time. Keeping this in one
place avoids the record/eval estimation logic (and calibration constants)
silently drifting apart.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

TAG_SIZE_M = 0.052  # measure your PRINTED tag's black square with a ruler (meters)

# Placeholder only — used as a last resort when no per-camera calibration file is
# found by `load_camera_matrix`. From cv2.calibrateCamera (RMS reprojection error:
# 3.49 — noticeably high, fx/fy asymmetry suggests the corner detections weren't
# clean). Run `camera_calibration.py` for every camera you actually record with.
K_MATRIX = np.array(
    [
        [229.3946, 0.0, 305.9041],
        [0.0, 202.3217, 226.9424],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def load_camera_matrix(camera_name: str, calib_dir: str | Path = ".") -> np.ndarray:
    """Load the intrinsics matrix `camera_calibration.py` saved for `camera_name`.

    Looks for ``<calib_dir>/<camera_name>_calibration.npz``. Falls back to the
    shared placeholder `K_MATRIX` (with a warning) when no calibration file exists
    for this camera — extrinsics computed from an uncalibrated K will be wrong for
    that specific lens/sensor.
    """
    calib_path = Path(calib_dir) / f"{camera_name}_calibration.npz"
    if calib_path.exists():
        return np.load(calib_path)["K"].astype(np.float32)
    logging.warning(
        f"No calibration file found at '{calib_path}' for camera '{camera_name}'. Falling back to "
        f"the shared placeholder K_MATRIX. Run `camera_calibration.py --camera-name {camera_name} "
        f"...` to get accurate extrinsics for this camera."
    )
    return K_MATRIX


_detector = None


def _get_detector() -> cv2.aruco.ArucoDetector:
    """Module-level singleton: building the dictionary/detector is cheap in isolation but
    this function runs once per frame per camera in the recording hot loop, so avoid
    rebuilding it on every call."""
    global _detector
    if _detector is None:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        # Default minMarkerPerimeterRate (~0.03) rejects tags that look small relative to the
        # frame — easy to hit at 640x480 when the tag is far from the camera. Lower it so
        # smaller/farther tags aren't dropped outright. If you get false positives on noise,
        # raise this back up a bit.
        parameters.minMarkerPerimeterRate = 0.01
        _detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    return _detector


def detect_apriltag_and_get_c2w(
    image, K: np.ndarray = K_MATRIX, tag_size: float = TAG_SIZE_M
) -> np.ndarray:
    """
    Detects an AprilTag (36h11) in an image and returns the camera-to-world (c2w) matrix.

    Returns the identity matrix when the tag isn't detected in the frame.
    """
    # 1. BULLETPROOF TENSOR CONVERSION
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)

    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D image array, got shape {arr.shape}")

    # Detect the layout instead of assuming CHW. Raw obs from robot.get_observation() is
    # typically (H, W, C) straight from the camera (OpenCVCamera returns HWC uint8);
    # dataset-loaded frames are (C, H, W). Assuming the wrong one silently scrambles the
    # image via permute() and detection fails on every frame with no error raised.
    if arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[-1]:
        img_np = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    else:
        img_np = arr  # already HWC

    # Normalize to uint8 regardless of source dtype/range
    if img_np.dtype != np.uint8:
        if np.issubdtype(img_np.dtype, np.floating) and img_np.max() <= 1.0 + 1e-3:
            img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img_np = img_np.clip(0, 255).astype(np.uint8)

    # Ensure memory is contiguous for OpenCV
    img_np = np.ascontiguousarray(img_np)

    # Check if LeRobot passed RGB or BGR (LeRobot defaults to RGB)
    # Convert to Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 2. RUN OPENCV 4.7+ DETECTOR (module-level singleton — this runs once per frame per
    # camera in the recording hot loop, so don't rebuild it every call.)
    corners, ids, _rejected = _get_detector().detectMarkers(gray)

    c2w = np.eye(4, dtype=np.float32)

    # 3. DO THE MATH IF FOUND
    if ids is not None and len(ids) > 0:
        half_size = tag_size / 2.0
        obj_points = np.array(
            [
                [-half_size, half_size, 0],
                [half_size, half_size, 0],
                [half_size, -half_size, 0],
                [-half_size, -half_size, 0],
            ],
            dtype=np.float32,
        )

        dist_coeffs = np.zeros((4, 1))
        # IPPE_SQUARE is the solver built for exactly this case (a single planar square
        # target's 4 coplanar corners) — more robust than the default iterative solver,
        # which can settle on an ambiguous/flipped pose for a flat square marker.
        success, rvec, tvec = cv2.solvePnP(
            obj_points, corners[0][0], K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if success:
            # cv2.solvePnP's (rvec, tvec) define p_camera = R @ p_world + t — i.e. the
            # WORLD-TO-CAMERA transform, not camera-to-world. Invert it (rotation Rᵀ,
            # translation -Rᵀ@t) to get the actual camera-to-world matrix this function
            # promises. Storing R/t directly here previously produced a c2w whose distance
            # to the tag happened to look right (rotation preserves vector norm, so
            # ‖t‖ == ‖camera position‖ by coincidence) while direction/orientation — and
            # therefore any Plucker ray-map or reprojection derived from it — was wrong.
            R_w2c, _ = cv2.Rodrigues(rvec)
            R_c2w = R_w2c.T
            cam_pos_world = -R_c2w @ tvec.flatten()
            c2w[:3, :3] = R_c2w
            c2w[:3, 3] = cam_pos_world

    return c2w
