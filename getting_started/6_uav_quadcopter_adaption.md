# UAV Quadcopter Adaptation Tutorial

This tutorial demonstrates how to adapt the GR00T humanoid robot VLA model for UAV quadcopter control. The key insight is to leverage the pretrained Visual Language Model (VLM) and only retrain the diffusion action head for UAV-specific control patterns.

## Architecture Overview

```
GR00T VLA Model for UAV
├── Vision Encoder (FROZEN) ← Pretrained visual understanding
├── Language Model (FROZEN) ← Pretrained language understanding  
├── Shared Representations ← Cross-modal understanding
└── UAV Action Head (TRAINABLE) ← New diffusion model for UAV actions
```

## UAV State and Action Spaces

### State Space (13 dimensions)
- **position**: x, y, z coordinates (3D)
- **orientation**: roll, pitch, yaw angles (3D)  
- **velocity**: vx, vy, vz velocities (3D)
- **battery**: battery level percentage (1D)
- **gps**: latitude, longitude, altitude (3D)

### Action Space (9 dimensions)
- **flight_control**: throttle, roll, pitch, yaw (4D)
- **velocity_command**: vx, vy, vz velocity setpoints (3D)
- **gimbal**: gimbal_pitch, gimbal_yaw (2D)

## Key Adaptations

### 1. New Embodiment Tag
```python
from gr00t.data.embodiment_tags import EmbodimentTag

embodiment_tag = EmbodimentTag.UAV_QUADCOPTER  # New UAV embodiment
```

### 2. UAV Data Configuration
```python
from gr00t.experiment.data_config import UAVQuadcopterDataConfig

uav_config = UAVQuadcopterDataConfig()
modality_configs = uav_config.modality_config()
transforms = uav_config.transform()
```

### 3. Modality Configuration
The `modality.json` file defines how to interpret UAV state and action arrays:

```json
{
    "state": {
        "position": {"start": 0, "end": 3},
        "orientation": {"start": 3, "end": 6, "rotation_type": "euler_angles"},
        "velocity": {"start": 6, "end": 9},
        "battery": {"start": 9, "end": 10},
        "gps": {"start": 10, "end": 13}
    },
    "action": {
        "flight_control": {"start": 0, "end": 4},
        "velocity_command": {"start": 4, "end": 7}, 
        "gimbal": {"start": 7, "end": 9}
    }
}
```

## Training Strategy

### Leverage Pretrained Components
```bash
python scripts/uav_finetune.py \\
    --model_path="nvidia/GR00T-N1.5-3B" \\
    --data_path="./demo_data/uav.Landing" \\
    --embodiment_tag="uav_quadcopter" \\
    --freeze_backbone=true \\          # Keep VLM frozen
    --freeze_language_model=true \\    # Keep language model frozen
    --only_train_action_head=true      # Only train UAV action head
```

### Key Training Parameters
- **freeze_backbone=true**: Preserves visual understanding from humanoid training
- **freeze_language_model=true**: Retains language command comprehension
- **only_train_action_head=true**: Only adapts action generation for UAV control
- **embodiment_tag=uav_quadcopter**: Uses separate action head for UAV

## Step-by-Step Tutorial

### Step 1: Prepare UAV Dataset

1. **Collect UAV Data**: Record UAV flights with:
   - Front camera and gimbal camera feeds
   - UAV telemetry (position, orientation, velocity, battery, GPS)
   - Action commands (flight controls, velocity commands, gimbal)
   - Task descriptions ("Land on platform", "Follow waypoint", etc.)

2. **Convert to LeRobot Format**: Structure your data as:
   ```
   uav_dataset/
   ├── data/
   │   ├── chunk-001/
   │   │   ├── observation.images.front_camera/
   │   │   ├── observation.images.gimbal_camera/
   │   │   ├── observation.state/
   │   │   └── action/
   │   └── ...
   ├── meta/
   │   ├── modality.json    # UAV-specific modality config
   │   ├── info.json
   │   └── stats.json
   └── videos/
   ```

3. **Copy Modality Config**:
   ```bash
   cp getting_started/examples/uav_quadcopter__modality.json /path/to/uav/dataset/meta/modality.json
   ```

### Step 2: Generate Dataset Statistics
```bash
python scripts/load_dataset.py \\
    --data_path /path/to/uav/dataset \\
    --embodiment_tag uav_quadcopter
```

### Step 3: Fine-tune UAV Model
```bash
python scripts/uav_finetune.py \\
    --data_path /path/to/uav/dataset \\
    --output_dir ./checkpoints/uav_model \\
    --batch_size 4 \\
    --learning_rate 1e-4 \\
    --num_epochs 50 \\
    --freeze_backbone \\
    --freeze_language_model \\
    --only_train_action_head
```

### Step 4: Evaluate Trained Model
```bash
python getting_started/examples/eval_uav_quadcopter.py \\
    --model_path ./checkpoints/uav_model \\
    --duration 30 \\
    --frequency 10
```

## Usage Example

### Loading Trained UAV Model
```python
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import UAVQuadcopterDataConfig

# Setup configuration
uav_config = UAVQuadcopterDataConfig()
modality_configs = uav_config.modality_config()
transforms = uav_config.transform()

# Load trained model
uav_policy = Gr00tPolicy(
    model_path="./checkpoints/uav_model",
    modality_config=modality_configs,
    modality_transform=transforms,
    embodiment_tag=EmbodimentTag.UAV_QUADCOPTER,
    device="cuda"
)
```

### Getting Actions
```python
# Prepare observation
observation = {
    "video.front_camera": front_camera_image,    # (H, W, 3)
    "video.gimbal_camera": gimbal_camera_image,  # (H, W, 3)
    "state.position": [x, y, z],                 # 3D position
    "state.orientation": [roll, pitch, yaw],     # 3D orientation
    "state.velocity": [vx, vy, vz],             # 3D velocity
    "state.battery": battery_level,              # Battery %
    "state.gps": [lat, lon, alt],               # GPS coords
    "annotation.human.task_description": "Land on the designated platform"
}

# Get action from policy
action = uav_policy.get_action(observation)

# Extract control commands
flight_control = action["action.flight_control"]      # [throttle, roll, pitch, yaw]
velocity_command = action["action.velocity_command"]  # [vx, vy, vz]
gimbal_control = action["action.gimbal"]             # [gimbal_pitch, gimbal_yaw]
```

## Integration with UAV Systems

### Flight Controller Integration
```python
# Example integration with PX4/ArduPilot
def send_to_flight_controller(flight_control, velocity_command):
    throttle, roll_cmd, pitch_cmd, yaw_cmd = flight_control
    vx_cmd, vy_cmd, vz_cmd = velocity_command
    
    # Send attitude commands
    vehicle.send_attitude_target(
        roll=roll_cmd,
        pitch=pitch_cmd, 
        yaw=yaw_cmd,
        thrust=throttle
    )
    
    # Send velocity commands
    vehicle.send_velocity_ned(vx_cmd, vy_cmd, vz_cmd)
```

### Gimbal Control Integration
```python
def send_to_gimbal(gimbal_control):
    gimbal_pitch, gimbal_yaw = gimbal_control
    
    # Send gimbal commands
    gimbal.set_attitude(pitch=gimbal_pitch, yaw=gimbal_yaw)
```

## Safety Considerations

1. **Action Smoothing**: Apply smoothing filters to prevent abrupt control changes
2. **Safety Bounds**: Implement hard limits on action values
3. **Emergency Stop**: Include manual override capabilities
4. **State Validation**: Verify sensor data before using for control
5. **Fallback Behavior**: Define safe fallback actions for model failures

## Benefits of This Approach

### Leverages Pretrained Capabilities
- **Visual Understanding**: Reuses powerful visual features trained on diverse robot data
- **Language Comprehension**: Inherits natural language command understanding
- **Multimodal Reasoning**: Benefits from cross-modal attention mechanisms

### Efficient Training
- **Minimal Data**: Only UAV action head needs training, reducing data requirements
- **Fast Convergence**: Leverages pretrained representations for quick adaptation
- **Stable Training**: Frozen backbone provides stable feature extraction

### Scalable Architecture
- **Multiple UAVs**: Easy to add different UAV types with separate action heads
- **Task Transfer**: Language understanding transfers across different UAV tasks
- **Embodiment Scaling**: Framework supports adding new robot types

## Expected Performance

The UAV adaptation should achieve:
- **Strong Language Following**: Inherited from GR00T's language capabilities
- **Visual Scene Understanding**: Leveraged from pretrained visual encoder
- **Smooth Control**: Diffusion model generates temporally consistent actions
- **Generalization**: Benefits from diverse pretraining for novel scenarios

## Comparison to Other Approaches

| Approach | Pros | Cons |
|----------|------|------|
| Train from Scratch | Full control | Requires massive UAV dataset |
| Fine-tune Everything | Maximum adaptation | Risk of catastrophic forgetting |
| **Our Approach** | **Leverages pretrained VLM, efficient training** | **Requires careful action space design** |

## Next Steps

1. **Collect UAV Data**: Gather diverse UAV flight data in LeRobot format
2. **Domain Adaptation**: Fine-tune for specific UAV types (racing, delivery, etc.)
3. **Multi-Task Learning**: Train on diverse UAV tasks simultaneously
4. **Real-World Testing**: Validate on actual UAV hardware
5. **Safety Validation**: Extensive testing of safety mechanisms

This approach demonstrates how foundation models trained on one embodiment (humanoid robots) can be efficiently adapted to completely different embodiments (UAVs) by leveraging shared visual-language understanding while only adapting the action generation component.
