# UAV Quadcopter Adaptation for GR00T

This directory contains the complete UAV quadcopter adaptation of the GR00T humanoid robot VLA model. The key innovation is leveraging the pretrained Visual Language Model (VLM) while only retraining the diffusion action head for UAV-specific control.

## 🚁 Overview

**Architecture Strategy:**
- **Leverage VLM**: Keep visual encoder and language model frozen
- **Retrain Diffusion**: Only fine-tune the action head for UAV control
- **New Embodiment**: Separate action head for UAV-specific control patterns

**State Space (13D):**
- position: x, y, z (3D)
- orientation: roll, pitch, yaw (3D)  
- velocity: vx, vy, vz (3D)
- battery: battery level (1D)
- gps: lat, lon, alt (3D)

**Action Space (9D):**
- flight_control: throttle, roll, pitch, yaw (4D)
- velocity_command: vx, vy, vz (3D)
- gimbal: gimbal_pitch, gimbal_yaw (2D)

## 📁 Files Structure

```
getting_started/uav_test/
├── uav_quadcopter_finetuning.ipynb    # Complete UAV adaptation tutorial
├── test.ipynb                         # Updated test notebook with UAV components
└── README.md                          # This file

getting_started/examples/
├── uav_quadcopter__modality.json      # UAV modality configuration
└── eval_uav_quadcopter.py             # UAV evaluation example

getting_started/
└── 6_uav_quadcopter_adaptation.md     # Comprehensive documentation

scripts/
└── uav_finetune.py                    # UAV fine-tuning script

demo_data/uav.Landing/
└── meta/modality.json                 # UAV dataset modality config
```

## 🚀 Quick Start

### 1. Test UAV Components
```bash
# Open the UAV test notebook
jupyter notebook getting_started/uav_test/test.ipynb

# Or run the comprehensive tutorial
jupyter notebook getting_started/uav_test/uav_quadcopter_finetuning.ipynb
```

### 2. Prepare UAV Dataset
```bash
# Copy modality config to your UAV dataset
cp getting_started/examples/uav_quadcopter__modality.json /path/to/uav/dataset/meta/modality.json

# Generate dataset statistics  
python scripts/load_dataset.py --data_path /path/to/uav/dataset --embodiment_tag uav_quadcopter
```

### 3. Fine-tune for UAV
```bash
python scripts/uav_finetune.py \
    --data_path /path/to/uav/dataset \
    --output_dir ./checkpoints/uav_model \
    --freeze_backbone \
    --freeze_language_model \
    --only_train_action_head
```

### 4. Evaluate UAV Model
```bash
python getting_started/examples/eval_uav_quadcopter.py \
    --model_path ./checkpoints/uav_model \
    --duration 30 \
    --frequency 10
```

## 🔧 Code Components Added

### New Embodiment Tag
```python
# gr00t/data/embodiment_tags.py
class EmbodimentTag(Enum):
    UAV_QUADCOPTER = "uav_quadcopter"

EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.UAV_QUADCOPTER.value: 32,
    # ... other mappings
}
```

### UAV Data Configuration
```python
# gr00t/experiment/data_config.py
class UAVQuadcopterDataConfig(BaseDataConfig):
    video_keys = ["video.front_camera", "video.gimbal_camera"]
    state_keys = ["state.position", "state.orientation", "state.velocity", "state.battery", "state.gps"]
    action_keys = ["action.flight_control", "action.velocity_command", "action.gimbal"]
    # ... configuration details
```

### UAV-Specific Transforms
- Position/orientation normalization with proper ranges
- Euler angle handling for UAV orientation
- Flight control and velocity command normalization
- Video transforms optimized for aerial footage

## 📊 Training Strategy

### Freeze Pretrained Components
```python
# Keep visual understanding from diverse robot training
policy.model.backbone.requires_grad = False

# Retain language command comprehension  
policy.model.language_model.requires_grad = False

# Only train UAV-specific action generation
# Only UAV action head parameters remain trainable
```

### Benefits of This Approach
1. **Data Efficiency**: Requires minimal UAV-specific training data
2. **Fast Convergence**: Leverages pretrained visual-language representations
3. **Stable Training**: Frozen backbone provides consistent features
4. **Transfer Learning**: Visual understanding transfers from ground to aerial
5. **Language Generalization**: Natural command understanding works across embodiments

## 🎯 Expected Performance

The UAV adaptation should achieve:
- **Strong Language Following**: Commands like "land on platform", "follow waypoint"
- **Visual Scene Understanding**: Obstacle detection, landing zone recognition
- **Smooth Control**: Temporally consistent action sequences from diffusion model
- **Generalization**: Benefits from diverse pretraining for novel scenarios

## 🔗 Integration Examples

### Flight Controller Integration
```python
# Send to PX4/ArduPilot
def send_flight_controls(flight_control):
    throttle, roll, pitch, yaw = flight_control
    vehicle.send_attitude_target(roll=roll, pitch=pitch, yaw=yaw, thrust=throttle)
```

### Real-time Control Loop
```python
# 10Hz control loop
while mission_active:
    observation = get_uav_telemetry()
    action = uav_policy.get_action(observation)
    execute_uav_commands(action)
    time.sleep(0.1)
```

## 🛡️ Safety Considerations

1. **Action Bounds**: Implement hard limits on control values
2. **Emergency Stop**: Manual override capabilities
3. **State Validation**: Verify sensor data integrity
4. **Graceful Degradation**: Safe fallback behaviors
5. **Testing**: Extensive simulation before real flights

## 📈 Comparison to Alternatives

| Approach | Data Req. | Training Time | Performance | Generalization |
|----------|-----------|---------------|-------------|----------------|
| Train from Scratch | Very High | Very Long | Custom | Limited |
| Full Fine-tune | High | Long | Good | Moderate |
| **Our Approach** | **Low** | **Short** | **Good** | **High** |

## 🔄 Next Steps

1. **Collect UAV Data**: Gather diverse flight scenarios in LeRobot format
2. **Domain Adaptation**: Adapt for specific UAV types (racing, delivery, etc.)
3. **Multi-Task Training**: Train on various UAV missions simultaneously  
4. **Hardware Testing**: Validate on real UAV platforms
5. **Safety Validation**: Comprehensive safety testing

## 📚 References

- [GR00T N1.5 Paper](https://research.nvidia.com/labs/gear/gr00t-n1_5)
- [LeRobot Data Format](https://github.com/huggingface/lerobot)
- [Getting Started Tutorials](../README.md)

## 💡 Key Insight

> "The same visual-language understanding that enables humanoid robots to manipulate objects can be leveraged for UAVs to navigate and interact with the world - only the action generation needs to be adapted for aerial control."

This approach demonstrates how foundation models can efficiently transfer across completely different embodiments by sharing high-level understanding while adapting low-level control.
