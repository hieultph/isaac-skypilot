# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Optional

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.model.transforms import GR00TTransform


class BaseDataConfig(ABC):
    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


#####################################################################################
# helper functions
#####################################################################################


def import_external_data_config(data_config_str: str) -> Optional[BaseDataConfig]:
    """
    Import and instantiate an external data configuration class.

    Format: "module_path:ClassName" (e.g., "my_configs:RobotConfig")
    Supports nested modules like "package.submodule:ClassName"
    """
    if ":" not in data_config_str:
        return None

    import importlib
    import os
    import sys
    from pathlib import Path

    # Add current working directory to Python path
    current_dir = str(Path(os.getcwd()).absolute())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        module_path, class_name = data_config_str.split(":", 1)
        if not module_path or not class_name:
            raise ValueError(f"Invalid format: '{data_config_str}'. Use 'module:ClassName'")

        print(f"Loading external config: {module_path}.{class_name}")

        module = importlib.import_module(module_path)
        if not hasattr(module, class_name):
            available = [
                n
                for n in dir(module)
                if not n.startswith("_") and isinstance(getattr(module, n), type)
            ]
            raise AttributeError(
                f"Class '{class_name}' not found in '{module_path}'. Available: {available}"
            )

        # assert if the class has 'transform' and 'modality_config' methods
        if not hasattr(getattr(module, class_name), "transform"):
            raise AttributeError(f"Class '{class_name}' does not have a 'transform' method")
        if not hasattr(getattr(module, class_name), "modality_config"):
            raise AttributeError(f"Class '{class_name}' does not have a 'modality_config' method")

        return getattr(module, class_name)()

    except (ModuleNotFoundError, AttributeError, ValueError) as e:
        print(f"Config loading failed: {e}")
        print("Example: my_configs:MyConfig, package.submodule:ClassName")
        raise


def load_data_config(data_config_str: str) -> BaseDataConfig:
    """
    Get a data config class from a string.
    >>> load_data_config("so100")
    >>> get_data_config("dir.subdir.my_configs:RobotConfig")
    """
    if data_config_str in DATA_CONFIG_MAP:
        return DATA_CONFIG_MAP[data_config_str]
    data_config_cls = import_external_data_config(data_config_str)
    if data_config_cls is not None:
        return data_config_cls
    # Yellow warning color
    yellow = "\033[93m"
    reset = "\033[0m"
    raise ValueError(
        f"{yellow}Invalid data_config '{data_config_str}'. "
        f"Available options: {list(DATA_CONFIG_MAP.keys())}, "
        f"or use 'module:ClassName' for external configs{reset}"
    )


###########################################################################################


class UAVQuadcopterDataConfig(BaseDataConfig):
    """
    Data configuration for UAV Quadcopter embodiment.
    
    State Space:
    - position: x, y, z (3 dims)
    - orientation: throttle, roll, pitch, yaw (4 dims)  
    - velocity: vx, vy, vz (3 dims)
    - gimbal: roll, pitch, yaw (3)  
    Total: 13 dimensional state
    
    Action Space:
    - flight_control: throttle, roll, pitch, yaw (4 dims)
    - velocity_command: vx, vy, vz (3 dims) 
    - gimbal: roll, pitch, yaw (3)  
    Total: 10 dimensional action
    """
    video_keys = [
        "video.front_camera",
    ]
    state_keys = [
        "state.position",        # x, y, z (3)
        "state.orientation",     # throttle, roll, pitch, yaw (4)
        "state.velocity",        # vx, vy, vz (3)
        "state.gimbal",         # roll, pitch, yaw (3)
    ]
    action_keys = [
        "action.flight_control",  # throttle, roll, pitch, yaw (4)
        "action.velocity_command", # vx, vy, vz (3)
        "action.gimbal",         # roll, pitch, yaw (3)
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(10))  # UAV has 10-dimensional action space

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self):
        from gr00t.data.transform import (
            VideoResize,
            VideoToTensor,
            VideoColorJitter,
            VideoToNumpy,
        )
        from gr00t.model.transforms import GR00TTransform

        video_modality = self.modality_config()["video"]
        state_modality = self.modality_config()["state"]
        action_modality = self.modality_config()["action"]

        transforms = [
            # video transforms
            VideoToTensor(apply_to=video_modality.modality_keys),
            VideoResize(
                apply_to=video_modality.modality_keys,
                height=224,
                width=224,
                antialias=True,
            ),
            VideoColorJitter(
                apply_to=video_modality.modality_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
                backend="torchvision",
            ),
            VideoToNumpy(apply_to=video_modality.modality_keys),
            
            # state transforms
            StateActionToTensor(apply_to=state_modality.modality_keys),
            StateActionTransform(
                apply_to=state_modality.modality_keys,
                normalization_modes={
                    "state.position": "min_max",       # position normalization
                    "state.orientation": "min_max",    # orientation normalization  
                    "state.velocity": "min_max",       # velocity normalization
                    "state.gimbal": "min_max",        # gimbal normalization
                },
                target_rotations={
                    "state.orientation": "euler_angles_rpy",  # Use euler angles for orientation
                },
            ),
            
            # action transforms
            StateActionToTensor(apply_to=action_modality.modality_keys),
            StateActionTransform(
                apply_to=action_modality.modality_keys,
                normalization_modes={
                    "action.flight_control": "min_max",   # flight control normalization
                    "action.velocity_command": "min_max", # velocity command normalization
                    "action.gimbal": "min_max",           # gimbal normalization
                },
            ),
            
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

DATA_CONFIG_MAP = {
    "uav_quadcopter": UAVQuadcopterDataConfig(),
}
