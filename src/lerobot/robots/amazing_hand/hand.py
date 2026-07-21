from dataclasses import dataclass

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots import Robot, RobotConfig
from lerobot.motors import MotorCalibration
from typing import Any



@dataclass
class AmazingHandConfig(RobotConfig):
    port: str


class AmazingHand(Robot):
    config_class = AmazingHandConfig
    name = "amazing_hand"

    def __init__(self, config: AmazingHandConfig):
        super().__init__(config)

        self.bus = FeetechMotorsBus(
            port=self.config.port, 
            motors={
                "motor1_finger1": Motor(1, "scs0009", MotorNormMode.RANGE_M100_100),
                "motor2_finger1": Motor(2, "scs0009", MotorNormMode.RANGE_M100_100),
                "motor1_finger2": Motor(3, "scs0009", MotorNormMode.RANGE_M100_100),
                "motor2_finger2": Motor(4, "scs0009", MotorNormMode.RANGE_M100_100),
                "motor1_finger3": Motor(5, "scs0009", MotorNormMode.RANGE_M100_100),
                "motor2_finger3": Motor(6, "scs0009", MotorNormMode.RANGE_M100_100),
                "motor1_finger4": Motor(7, "scs0009", MotorNormMode.RANGE_M100_100),
                "motor2_finger4": Motor(8, "scs0009", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
            protocol_version=1,
        )
        self.cameras = {}   

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "motor1_finger1.pos": float,
            "motor2_finger1.pos": float,
            "motor1_finger2.pos": float,
            "motor2_finger2.pos": float,
            "motor1_finger3.pos": float,
            "motor2_finger3.pos": float,
            "motor1_finger4.pos": float,
            "motor2_finger4.pos": float,
        }

    @property
    def observation_features(self) -> dict:
        return self._motors_ft

    @property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()

        if not self.is_calibrated and calibrate:
            self.calibrate()
        self.configure()

    def disconnect(self) -> None:
        self.bus.disconnect()
    
    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        self.bus.disable_torque()

        input(
            "Move the Amazing Hand to the middle of its range of motion "
            "and press ENTER..."
        )

        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move every joint through its complete range of motion.\n"
            "Press ENTER when finished."
        )

        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}

        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()

        print(f"Calibration saved to {self.calibration_fpath}")


    def configure(self) -> None:
        self.bus.configure_motors()

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        obs_dict = {}

        for motor in self.bus.motors:
            obs_dict[f"{motor}.pos"] = self.bus.read("Present_Position", motor)

        return obs_dict


    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        goal_pos = {
            key.removesuffix(".pos"): value
            for key, value in action.items()
        }

        self.bus.sync_write("Goal_Position", goal_pos)

        return action.matl