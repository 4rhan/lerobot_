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

"""
Records a dataset via teleoperation, with an added step that estimates the
camera's extrinsics (pose relative to an AprilTag) and stores both the
extrinsics (4x4 c2w) and intrinsics (3x3 K) alongside every frame.

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Example:

```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --display_data=true
```

IMPORTANT: the camera you want extrinsics/intrinsics for must be named
"laptop" in --robot.cameras, since that key is hardcoded below. If you use a
different camera name, update CAMERA_NAME accordingly.
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import cv2
import numpy as np
import torch

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_openarm_mini,
    bi_rebot_102_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
)
from lerobot.teleoperators.keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)

# --- AprilTag extrinsics/intrinsics config -----------------------------
CAMERA_NAME = "front"          # must match the key in --robot.cameras
TAG_SIZE_M = 0.052              # measure your PRINTED tag's black square with a ruler (meters)

# From cv2.calibrateCamera (RMS reprojection error: 3.49 — noticeably high,
# fx/fy asymmetry suggests the corner detections weren't clean. Recalibrate
# with a steadier/more varied capture when you get the chance for better
# extrinsics accuracy).
K_MATRIX = np.array(
    [
        [229.3946, 0.0, 305.9041],
        [0.0, 202.3217, 226.9424],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def detect_apriltag_and_get_c2w(image, K: np.ndarray, tag_size: float) -> np.ndarray:
    """
    Detects an AprilTag (36h11) in an image and returns the camera-to-world (c2w) matrix.
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

    # --- DEBUG STEP: Save exactly what OpenCV sees to your hard drive! ---
    # (It will just overwrite this same file every frame so it doesn't fill your drive)
    cv2.imwrite("debug_lerobot_vision.jpg", gray)
    # ----------------------------------------------------------------------

    # 2. RUN OPENCV 4.7+ DETECTOR
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    # Default minMarkerPerimeterRate (~0.03) rejects tags that look small relative to the
    # frame — easy to hit at 640x480 when the tag is far from the camera. Lower it so
    # smaller/farther tags aren't dropped outright. If you get false positives on noise,
    # raise this back up a bit.
    parameters.minMarkerPerimeterRate = 0.01
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _rejected = detector.detectMarkers(gray)

    c2w = np.eye(4, dtype=np.float32)

    # 3. DO THE MATH IF FOUND
    if ids is not None and len(ids) > 0:
        half_size = tag_size / 2.0
        obj_points = np.array([
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0],
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1))
        success, rvec, tvec = cv2.solvePnP(obj_points, corners[0][0], K, dist_coeffs)

        if success:
            R, _ = cv2.Rodrigues(rvec)
            c2w[:3, :3] = R
            c2w[:3, 3] = tvec.flatten()

    return c2w


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: TeleoperatorConfig | None = None
    display_data: bool = False
    display_mode: str = "rerun"
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False

    def __post_init__(self):
        if self.teleop is None:
            raise ValueError(
                "A teleoperator is required for recording. "
                "Use --teleop.type=... to specify one. "
                "For policy-based deployment, use lerobot-rollout instead."
            )


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_mode: str = "rerun",
    display_compressed_images: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )
        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm "
                "teleoperator. Currently only supported for LeKiwi robot."
            )

    control_interval = 1 / fps
    no_action_count = 0
    timestamp = 0
    start_episode_t = time.perf_counter()

    # Raw obs dicts from robot.get_observation() key cameras by their bare name
    # (e.g. "front"), NOT the "observation.images.<name>" dataset-schema prefix —
    # that prefix only gets applied later by build_dataset_frame. Using the wrong
    # key here means detect_apriltag_and_get_c2w silently never runs.
    obs_image_key = CAMERA_NAME
    extrinsics_key = f"observation.extrinsics.{CAMERA_NAME}"
    intrinsics_key = f"observation.intrinsics.{CAMERA_NAME}"
    warned_missing_camera_key = False

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Estimate camera pose (extrinsics) from the AprilTag, if visible
        if obs_image_key in obs:
            c2w_matrix = detect_apriltag_and_get_c2w(obs[obs_image_key], K_MATRIX, tag_size=TAG_SIZE_M)
        else:
            if not warned_missing_camera_key:
                logging.warning(
                    f"'{obs_image_key}' not found in raw observation dict — extrinsics/intrinsics "
                    f"will be saved as identity for this whole episode. Actual obs keys: {list(obs.keys())}"
                )
                warned_missing_camera_key = True
            c2w_matrix = np.eye(4, dtype=np.float32)

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            # build_dataset_frame only knows about the robot's native features, so it won't
            # pick up our custom extrinsics/intrinsics keys no matter where we stash them in
            # obs/obs_processed. Add them straight into the frame dict instead — this is what
            # actually gets validated against dataset.features and written to disk.
            observation_frame[extrinsics_key] = torch.from_numpy(c2w_matrix)
            observation_frame[intrinsics_key] = torch.from_numpy(K_MATRIX)

        # Get action from teleop
        if isinstance(teleop, Teleoperator):
            act = teleop.get_action()
            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        elif isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning(
                    "No teleoperator provided, skipping action generation. "
                    "This is likely to happen when resetting the environment without a teleop device. "
                    "The robot won't be at its rest position at the start of the next episode."
                )
            continue

        _sent_action = robot.send_action(robot_action_to_send)

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_visualization_data(
                display_mode,
                observation=obs_processed,
                action=action_values,
                compress_images=display_compressed_images,
            )

        dt_s = time.perf_counter() - start_loop_t
        sleep_time_s: float = control_interval - dt_s
        if sleep_time_s < 0:
            logging.warning(
                f"Record loop is running slower ({1 / dt_s:.1f} Hz) than the target FPS ({fps} Hz). "
                f"Dataset frames might be dropped and robot control might be unstable."
            )
        precise_sleep(max(sleep_time_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(
    cfg: RecordConfig,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_visualization(
            cfg.display_mode, session_name="recording", ip=cfg.display_ip, port=cfg.display_port
        )
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    # Build the base feature dict first...
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    # ...then register the extra extrinsics/intrinsics fields added in record_loop.
    dataset_features[f"observation.extrinsics.{CAMERA_NAME}"] = {
        "dtype": "float32",
        "shape": (4, 4),
        "names": ["row", "col"],
    }
    dataset_features[f"observation.intrinsics.{CAMERA_NAME}"] = {
        "dtype": "float32",
        "shape": (3, 3),
        "names": ["row", "col"],
    }

    dataset = None
    listener = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for policy evaluation. "
                    "lerobot-record is for data collection only. Use lerobot-rollout for policy deployment."
                )
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events = init_keyboard_listener()

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it "
                "for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2"
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_mode=cfg.display_mode,
                    display_compressed_images=display_compressed_images,
                )

                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_mode=cfg.display_mode,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        if dataset:
            dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()
        if listener is not None:
            listener.stop()
        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)
        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")
        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()