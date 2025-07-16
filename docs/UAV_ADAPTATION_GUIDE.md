# UAV GR00T Adaptation Guide

## Overview
This guide shows how to adapt the GR00T architecture for UAV (drone) control. GR00T is a Vision-Language-Action (VLA) model that can process video and text inputs to generate actions using diffusion models.

## Key Advantages for UAV Applications

### ✅ **Perfect Match for UAV Requirements**
- **Multimodal Input**: Handles video (camera feeds) + text (mission commands)
- **Diffusion-based Actions**: Generates smooth, continuous control signals
- **Embodiment-Aware**: Can be adapted for different UAV types
- **Pretrained VLM**: Leverages powerful vision-language understanding

### ✅ **Architecture Benefits**
- **Eagle Backbone**: Processes visual and textual inputs simultaneously
- **Flow Matching**: Advanced diffusion technique for action generation
- **Multi-Camera Support**: Can handle multiple camera feeds
- **Robust Training**: Proven training pipeline with fine-tuning capabilities

## Implementation Steps

### 1. **Data Format**
Your UAV dataset should follow the LeRobot format:
```
dataset/
├── data/
│   └── chunk-000/
│       ├── observation.images.front_camera/
│       ├── observation.images.bottom_camera/
│       ├── observation.state/
│       └── action/
├── meta/
│   ├── episodes.jsonl
│   ├── info.json
│   └── tasks.jsonl
└── videos/
```

### 2. **State Definition**
UAV state includes:
- **Position**: (x, y, z) coordinates
- **Orientation**: Quaternion (w, x, y, z)
- **Velocity**: Linear velocity (vx, vy, vz)
- **Angular Velocity**: (wx, wy, wz)
- **Battery Level**: 0-1 normalized

### 3. **Action Space**
UAV actions are:
- **Thrust**: 0-1 normalized
- **Roll**: -1 to 1
- **Pitch**: -1 to 1
- **Yaw**: -1 to 1

### 4. **Training Process**
```bash
# Fine-tune for UAV
python scripts/uav_gr00t_training.py --mode train

# Test inference
python scripts/uav_gr00t_training.py --mode test
```

## Data Collection Requirements

### **Video Data**
- **Front Camera**: For navigation and obstacle avoidance
- **Bottom Camera**: For landing and ground tracking
- **Resolution**: 224x224 (resized from original)
- **Frame Rate**: 30 FPS recommended

### **State Data**
- **High Frequency**: 100Hz minimum for control
- **Synchronized**: Video and state must be time-aligned
- **Calibrated**: Proper coordinate frame alignment

### **Text Commands**
- **Natural Language**: "fly to the red building"
- **Mission Commands**: "land on the platform"
- **Safety Commands**: "return to home"

## Example Usage

```python
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP

# Load UAV model
data_config = DATA_CONFIG_MAP["uav_quadrotor"]
policy = Gr00tPolicy(
    model_path="./uav_gr00t_checkpoints",
    embodiment_tag="uav_quadrotor",
    modality_config=data_config.modality_config(),
    modality_transform=data_config.transform(),
)

# Get action from observations
action = policy.get_action({
    "video": {
        "front_camera": front_camera_frame,
        "bottom_camera": bottom_camera_frame,
    },
    "state": {
        "position": current_position,
        "orientation": current_orientation,
        "velocity": current_velocity,
        "angular_velocity": current_angular_velocity,
        "battery_level": battery_level,
    },
    "annotation": {
        "human.task_description": "fly to the landing pad"
    }
})

# Extract control commands
thrust = action["action"]["thrust"]
roll = action["action"]["roll"] 
pitch = action["action"]["pitch"]
yaw = action["action"]["yaw"]
```

## Training Tips

### **Transfer Learning Strategy**
1. **Start with pretrained GR00T**: Use `nvidia/GR00T-N1.5-3B`
2. **Freeze backbone**: Keep vision-language model frozen
3. **Fine-tune action head**: Only train the diffusion model for UAV actions
4. **Gradual unfreezing**: Optionally unfreeze visual encoder later

### **Data Requirements**
- **Minimum**: 1000 episodes for basic functionality
- **Recommended**: 10,000+ episodes for robust performance
- **Diverse conditions**: Different weather, lighting, environments
- **Safety data**: Include emergency maneuvers and edge cases

### **Hyperparameters**
- **Learning Rate**: 1e-4 (conservative for fine-tuning)
- **Batch Size**: 16 (adjust based on GPU memory)
- **Action Horizon**: 16 steps (about 0.5 seconds at 30Hz)
- **Denoising Steps**: 50 for inference

## Safety Considerations

### **Model Validation**
- **Simulation First**: Test thoroughly in simulation
- **Hardware-in-the-Loop**: Validate with actual UAV hardware
- **Edge Case Testing**: Test failure modes and recovery

### **Fallback Systems**
- **Traditional Controller**: Keep PID controller as backup
- **Safety Constraints**: Implement hard limits on actions
- **Monitoring**: Real-time performance monitoring

## Performance Expectations

### **Accuracy**
- **Position Control**: ±0.1m accuracy expected
- **Orientation**: ±5° accuracy expected
- **Response Time**: <100ms for action generation

### **Generalization**
- **New Environments**: Good generalization to unseen areas
- **Weather Conditions**: Robust to lighting changes
- **Mission Adaptation**: Flexible to new command types

## Next Steps

1. **Collect UAV Dataset**: Follow LeRobot format
2. **Set up Training**: Use provided scripts
3. **Fine-tune Model**: Start with frozen backbone
4. **Validate in Simulation**: Test before real deployment
5. **Deploy to Hardware**: Integrate with flight controller

The GR00T architecture is very well-suited for UAV applications and should provide excellent performance for vision-language-action tasks in drone control!
