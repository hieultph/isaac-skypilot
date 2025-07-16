# UAV Quadrotor Demo Dataset

This directory contains a demo dataset for UAV quadrotor tracking and landing tasks, designed to work with the GR00T framework.

## Dataset Structure

```
uav_quadrotor.TrackingAndLanding/
├── data/
│   └── chunk-000/
│       ├── episode_000000.npz  # Takeoff and hover
│       ├── episode_000001.npz  # Track moving red target
│       ├── episode_000002.npz  # Track moving blue target
│       ├── episode_000003.npz  # Track stationary target
│       ├── episode_000004.npz  # Landing sequence
│       ├── episode_000005.npz  # Precision landing
│       ├── episode_000006.npz  # Waypoint trajectory
│       └── episode_000007.npz  # Complex mission
├── meta/
│   ├── info.json              # Dataset configuration
│   ├── modality.json          # Data structure definition
│   ├── tasks.jsonl            # Task definitions
│   ├── episodes.jsonl         # Episode metadata
│   └── stats.json             # Statistical information
├── videos/
│   └── chunk-000/
│       ├── front_camera/      # Front-facing camera videos
│       ├── bottom_camera/     # Bottom-facing camera videos
│       └── downward_camera/   # Downward-facing camera videos
├── generate_demo_data.py      # Full data generation script
├── generate_simple_demo.py    # Simple data generation script
└── README.md                  # This file
```

## Dataset Specifications

### Robot Configuration

- **Robot Type**: UAV_Quadrotor
- **State Space**: 18 dimensions
- **Action Space**: 4 dimensions
- **Frame Rate**: 30 FPS
- **Total Episodes**: 8
- **Total Frames**: 3,200

### State Space (18 dimensions)

1. **Position (3D)**: `pos_x, pos_y, pos_z` - UAV position in meters
2. **Orientation (4D)**: `quat_w, quat_x, quat_y, quat_z` - Quaternion orientation
3. **Velocity (3D)**: `vel_x, vel_y, vel_z` - Linear velocity in m/s
4. **Angular Velocity (3D)**: `angular_vel_x, angular_vel_y, angular_vel_z` - Angular velocity in rad/s
5. **Battery Level (1D)**: `battery_level` - Battery level (0-1)
6. **Target Position (3D)**: `target_pos_x, target_pos_y, target_pos_z` - Target position for tracking
7. **Target Distance (1D)**: `target_distance` - Distance to target in meters

### Action Space (4 dimensions)

1. **Thrust**: `thrust` - Thrust command (0-1)
2. **Roll**: `roll` - Roll command (-1 to 1)
3. **Pitch**: `pitch` - Pitch command (-1 to 1)
4. **Yaw**: `yaw` - Yaw command (-1 to 1)

### Camera Views

1. **Front Camera**: 640x480 RGB - Forward-facing view
2. **Bottom Camera**: 640x480 RGB - Downward-facing view
3. **Downward Camera**: 256x256 RGB - High-resolution downward view for landing

## Task Types

### 1. Takeoff and Hover (Episode 0)

- **Duration**: 450 frames (15 seconds)
- **Description**: UAV takes off from ground and hovers at 5m altitude
- **Key Behaviors**: Vertical ascent, altitude stabilization

### 2. Track Moving Red Target (Episode 1)

- **Duration**: 520 frames (~17 seconds)
- **Description**: UAV follows a moving target in a figure-8 pattern
- **Key Behaviors**: Target tracking, dynamic following

### 3. Track Moving Blue Target (Episode 2)

- **Duration**: 480 frames (16 seconds)
- **Description**: UAV tracks a different colored target with varying speed
- **Key Behaviors**: Visual target discrimination, adaptive tracking

### 4. Track Stationary Target (Episode 3)

- **Duration**: 400 frames (~13 seconds)
- **Description**: UAV locates and maintains position relative to stationary target
- **Key Behaviors**: Position holding, precision hovering

### 5. Landing Sequence (Episode 4)

- **Duration**: 350 frames (~12 seconds)
- **Description**: UAV performs controlled descent and landing
- **Key Behaviors**: Altitude control, soft landing

### 6. Precision Landing (Episode 5)

- **Duration**: 380 frames (~13 seconds)
- **Description**: UAV lands on a moving platform with high precision
- **Key Behaviors**: Moving target landing, precision control

### 7. Waypoint Trajectory (Episode 6)

- **Duration**: 420 frames (14 seconds)
- **Description**: UAV follows predefined waypoints and lands
- **Key Behaviors**: Path following, trajectory execution

### 8. Complex Mission (Episode 7)

- **Duration**: 200 frames (~7 seconds)
- **Description**: Combined takeoff, tracking, and landing sequence
- **Key Behaviors**: Multi-task execution, mission planning

## Usage

### Generate Demo Data

```bash
# Run the simple data generator
python generate_simple_demo.py

# Or run the full data generator (requires pandas, opencv)
python generate_demo_data.py
```

### Load Data in Python

```python
import numpy as np

# Load episode data
episode_data = np.load('data/chunk-000/episode_000000.npz')

# Access different data components
observation_state = episode_data['observation_state']  # Shape: (450, 18)
actions = episode_data['action']                       # Shape: (450, 4)
timestamps = episode_data['timestamp']                 # Shape: (450,)

# Extract specific state components
positions = observation_state[:, 0:3]      # x, y, z
orientations = observation_state[:, 3:7]   # quaternions
velocities = observation_state[:, 7:10]    # vx, vy, vz
battery_levels = observation_state[:, 13]  # battery level
```

### Integration with GR00T Framework

The dataset follows the GR00T framework conventions:

- Metadata files define the structure and specifications
- Episode data is stored in chunks for efficient loading
- Video data is synchronized with state/action data
- Statistical information is provided for normalization

## Customization

To adapt this dataset for your specific UAV tasks:

1. **Modify Tasks**: Edit `meta/tasks.jsonl` to define your tasks
2. **Adjust State Space**: Update `meta/info.json` and `meta/modality.json` for your robot configuration
3. **Change Episodes**: Modify the episode configurations in the generation scripts
4. **Add Sensors**: Extend the state space to include additional sensor data (IMU, GPS, etc.)
5. **Update Statistics**: Recalculate `meta/stats.json` based on your real data

## Notes

- The current dataset uses synthetic data for demonstration purposes
- Video files are optional and can be replaced with actual camera footage
- The data format is compatible with the GR00T training pipeline
- For production use, replace synthetic data with real flight data
- Consider adding more diverse weather conditions and environments

## Dependencies

For full functionality:

- `numpy` - Core data processing
- `pandas` - Data manipulation (for parquet format)
- `opencv-python` - Video processing
- `pyarrow` - Parquet file support
