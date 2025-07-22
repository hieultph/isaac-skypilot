#!/usr/bin/env python3
"""
UAV Landing Dataset Generator

This script generates synthetic UAV landing dataset in LeRobot format for testing the GR00T UAV adaptation.
It creates realistic landing scenarios with:
- Dual camera feeds (front camera + gimbal camera)
- UAV telemetry (position, orientation, velocity, battery, GPS)
- Control actions (flight controls, velocity commands, gimbal)
- Task descriptions for different landing scenarios

Usage:
    python scripts/generate_uav_dataset.py --output_dir ./demo_data/uav.Landing --num_episodes 50
"""

import argparse
import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime


class UAVLandingDataGenerator:
    """Generates synthetic UAV landing dataset in LeRobot format."""
    
    def __init__(self, output_dir: str, image_size: tuple = (480, 640, 3)):
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.fps = 20
        self.episode_length = 200  # 10 seconds at 20fps
        
        # Landing scenarios
        self.scenarios = [
            "Land on the designated platform",
            "Perform precision landing on the helipad",
            "Land safely in the marked zone",
            "Execute emergency landing procedure",
            "Land on the moving platform",
            "Navigate to landing zone and land",
            "Approach and land on the target area",
            "Complete autonomous landing sequence",
        ]
        
        # Setup directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories for LeRobot format."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "meta").mkdir(exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)
    
    def generate_landing_trajectory(self) -> Dict[str, np.ndarray]:
        """Generate realistic UAV landing trajectory."""
        timesteps = np.linspace(0, 10, self.episode_length)  # 10 second flight
        
        # Initial conditions (hovering at altitude)
        start_pos = np.array([
            np.random.uniform(-20, 20),    # x: random start position
            np.random.uniform(-20, 20),    # y: random start position  
            np.random.uniform(15, 30)      # z: start altitude
        ])
        
        # Landing target (origin with some noise)
        target_pos = np.array([
            np.random.uniform(-2, 2),      # x: landing target
            np.random.uniform(-2, 2),      # y: landing target
            0.0                            # z: ground level
        ])
        
        # Generate smooth trajectory using polynomial interpolation
        positions = []
        velocities = []
        orientations = []
        
        for i, t in enumerate(timesteps):
            # Smooth approach trajectory
            progress = t / 10.0  # 0 to 1
            smooth_progress = 3 * progress**2 - 2 * progress**3  # Smooth S-curve
            
            # Position interpolation with approach pattern
            if progress < 0.7:  # Approach phase
                pos = start_pos + (target_pos - start_pos) * smooth_progress
                # Add some circular approach pattern
                circle_radius = 5 * (1 - progress)
                angle = progress * 2 * np.pi
                pos[0] += circle_radius * np.cos(angle)
                pos[1] += circle_radius * np.sin(angle)
            else:  # Final descent phase
                descent_progress = (progress - 0.7) / 0.3
                pos = start_pos + (target_pos - start_pos) * (0.7 + 0.3 * descent_progress)
            
            positions.append(pos.copy())
            
            # Calculate velocity (derivative of position)
            if i > 0:
                vel = (positions[i] - positions[i-1]) / (timesteps[1] - timesteps[0])
                # Add some noise and limits
                vel = np.clip(vel + np.random.normal(0, 0.1, 3), -5, 5)
            else:
                vel = np.array([0, 0, 0])
            velocities.append(vel)
            
            # Orientation (heading towards target, with some dynamics)
            to_target = target_pos[:2] - pos[:2]
            target_yaw = np.arctan2(to_target[1], to_target[0])
            
            # Add some roll/pitch for realistic flight dynamics
            roll = np.clip(vel[1] * 0.1 + np.random.normal(0, 0.02), -0.3, 0.3)
            pitch = np.clip(-vel[0] * 0.1 + np.random.normal(0, 0.02), -0.3, 0.3)
            yaw = target_yaw + np.random.normal(0, 0.05)
            
            orientations.append([roll, pitch, yaw])
        
        return {
            'positions': np.array(positions),
            'velocities': np.array(velocities), 
            'orientations': np.array(orientations),
            'timesteps': timesteps
        }
    
    def generate_camera_image(self, position: np.ndarray, orientation: np.ndarray, 
                             camera_type: str = "front") -> np.ndarray:
        """Generate synthetic camera image based on UAV state."""
        img = np.zeros(self.image_size, dtype=np.uint8)
        
        # Create a synthetic landscape
        height, width = self.image_size[:2]
        
        # Sky gradient
        for i in range(height//2):
            intensity = int(135 + (255-135) * (1 - i/(height//2)))
            img[i, :] = [intensity, intensity+20, 255]  # Blue sky gradient
        
        # Ground
        ground_color = [34, 139, 34]  # Green ground
        for i in range(height//2, height):
            img[i, :] = ground_color
        
        # Add landing target based on position
        target_screen_x = int(width//2 - position[0] * 10)
        target_screen_y = int(height - 50 - position[2] * 2)
        
        if 0 < target_screen_x < width and 0 < target_screen_y < height:
            # Draw landing pad
            cv2.circle(img, (target_screen_x, target_screen_y), 20, (255, 255, 255), -1)
            cv2.circle(img, (target_screen_x, target_screen_y), 18, (255, 0, 0), 2)
            cv2.circle(img, (target_screen_x, target_screen_y), 10, (255, 255, 255), 2)
        
        # Add horizon line based on pitch
        pitch = orientation[1]
        horizon_y = int(height//2 + pitch * 100)
        if 0 < horizon_y < height:
            cv2.line(img, (0, horizon_y), (width, horizon_y), (255, 255, 255), 1)
        
        # Add some noise and camera effects
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Camera-specific modifications
        if camera_type == "gimbal":
            # Gimbal camera is more zoomed in
            center_crop = img[height//4:3*height//4, width//4:3*width//4]
            img = cv2.resize(center_crop, (width, height))
        
        return img
    
    def generate_actions(self, trajectory: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate control actions for the landing trajectory."""
        positions = trajectory['positions']
        velocities = trajectory['velocities']
        orientations = trajectory['orientations']
        
        flight_controls = []
        velocity_commands = []
        gimbal_commands = []
        
        for i in range(len(positions)):
            pos = positions[i]
            vel = velocities[i]
            orientation = orientations[i]
            
            # Flight control (throttle, roll, pitch, yaw)
            # Throttle based on altitude and vertical velocity
            altitude_error = pos[2] - 0  # Want to be at ground level
            throttle = 0.5 + altitude_error * 0.02 - vel[2] * 0.1
            throttle = np.clip(throttle, 0.0, 1.0)
            
            # Attitude controls (simplified)
            roll_cmd = orientation[0] + np.random.normal(0, 0.01)
            pitch_cmd = orientation[1] + np.random.normal(0, 0.01)
            yaw_cmd = orientation[2] + np.random.normal(0, 0.01)
            
            flight_controls.append([throttle, roll_cmd, pitch_cmd, yaw_cmd])
            
            # Velocity commands (desired velocities)
            if i < len(positions) - 1:
                desired_vel = (positions[i+1] - positions[i]) * 20  # Scale for 20fps
            else:
                desired_vel = vel
            
            velocity_commands.append(desired_vel)
            
            # Gimbal commands (point camera towards landing target)
            gimbal_pitch = np.clip(-pos[2] * 0.05, -0.5, 0.1)  # Look down more as altitude decreases
            gimbal_yaw = np.random.normal(0, 0.02)  # Small random movements
            
            gimbal_commands.append([gimbal_pitch, gimbal_yaw])
        
        return {
            'flight_control': np.array(flight_controls),
            'velocity_command': np.array(velocity_commands),
            'gimbal': np.array(gimbal_commands)
        }
    
    def generate_episode(self, episode_idx: int) -> Dict[str, Any]:
        """Generate a complete episode."""
        print(f"Generating episode {episode_idx}...")
        
        # Generate trajectory
        trajectory = self.generate_landing_trajectory()
        
        # Generate actions
        actions = self.generate_actions(trajectory)
        
        # Generate images
        front_images = []
        gimbal_images = []
        
        for i in range(self.episode_length):
            pos = trajectory['positions'][i]
            orientation = trajectory['orientations'][i]
            
            front_img = self.generate_camera_image(pos, orientation, "front")
            gimbal_img = self.generate_camera_image(pos, orientation, "gimbal")
            
            front_images.append(front_img)
            gimbal_images.append(gimbal_img)
        
        # Battery simulation (decreases over time)
        battery_levels = 100 - trajectory['timesteps'] * 2  # 2% per second
        
        # GPS simulation (convert position to lat/lon/alt)
        base_lat, base_lon = 37.7749, -122.4194  # San Francisco
        positions = trajectory['positions']
        gps_coords = []
        for pos in positions:
            lat = base_lat + pos[1] / 111000  # Rough conversion
            lon = base_lon + pos[0] / (111000 * np.cos(np.radians(base_lat)))
            alt = pos[2]
            gps_coords.append([lat, lon, alt])
        
        return {
            'episode_idx': episode_idx,
            'length': self.episode_length,
            'timestamp': datetime.now().isoformat(),
            'task_description': np.random.choice(self.scenarios),
            
            # State data
            'positions': positions,
            'orientations': trajectory['orientations'],
            'velocities': trajectory['velocities'],
            'battery': battery_levels,
            'gps': np.array(gps_coords),
            
            # Action data  
            'flight_control': actions['flight_control'],
            'velocity_command': actions['velocity_command'],
            'gimbal': actions['gimbal'],
            
            # Video data
            'front_camera': front_images,
            'gimbal_camera': gimbal_images,
        }
    
    def save_episode(self, episode_data: Dict[str, Any]):
        """Save episode data in LeRobot format."""
        episode_idx = episode_data['episode_idx']
        chunk_dir = self.output_dir / "data" / f"chunk-{episode_idx:03d}"
        chunk_dir.mkdir(exist_ok=True)
        
        # Create observation and action arrays with all required LeRobot fields
        episode_length = episode_data['length']
        
        # Prepare all data arrays
        observations = []
        actions = []
        timestamps = []
        task_indices = []
        episode_indices = []
        indices = []
        rewards = []
        dones = []
        
        for i in range(episode_length):
            # Observation state (13D: position(3) + orientation(3) + velocity(3) + battery(1) + gps(3))
            obs_state = np.concatenate([
                episode_data['positions'][i],      # 3D
                episode_data['orientations'][i],   # 3D  
                episode_data['velocities'][i],     # 3D
                [episode_data['battery'][i]],      # 1D
                episode_data['gps'][i],           # 3D
            ])
            observations.append(obs_state)
            
            # Action (9D: flight_control(4) + velocity_command(3) + gimbal(2))
            action = np.concatenate([
                episode_data['flight_control'][i],    # 4D
                episode_data['velocity_command'][i],  # 3D
                episode_data['gimbal'][i],            # 2D
            ])
            actions.append(action)
            
            # Standard LeRobot metadata fields
            timestamps.append(i / self.fps)  # Time in seconds
            task_indices.append(0)  # All episodes same task for now
            episode_indices.append(episode_idx)
            indices.append(i)
            
            # Reward: higher for successful landing (lower altitude + stable)
            altitude = episode_data['positions'][i][2]
            velocity_magnitude = np.linalg.norm(episode_data['velocities'][i])
            reward = max(0, (100 - altitude) / 100.0 - velocity_magnitude * 0.1)
            rewards.append(reward)
            
            # Done: True only on last frame
            dones.append(i == episode_length - 1)
        
        # Create comprehensive episode dataframe
        episode_df = pd.DataFrame({
            'observation.state': [obs.tolist() for obs in observations],
            'action': [act.tolist() for act in actions], 
            'timestamp': timestamps,
            'task_index': task_indices,
            'episode_index': episode_indices,
            'index': indices,
            'next.reward': rewards,
            'next.done': dones
        })
        
        # Save complete episode data
        episode_df.to_parquet(chunk_dir / f"episode_{episode_idx:06d}.parquet")
        
        # Save videos
        video_dir = self.output_dir / "videos" / f"chunk-{episode_idx:03d}"
        
        # Front camera
        front_video_dir = video_dir / "observation.images.front_camera"
        front_video_dir.mkdir(parents=True, exist_ok=True)
        front_video_path = front_video_dir / f"episode_{episode_idx:06d}.mp4"
        self.save_video(episode_data['front_camera'], front_video_path)
        
        # Gimbal camera
        gimbal_video_dir = video_dir / "observation.images.gimbal_camera"
        gimbal_video_dir.mkdir(parents=True, exist_ok=True)
        gimbal_video_path = gimbal_video_dir / f"episode_{episode_idx:06d}.mp4"
        self.save_video(episode_data['gimbal_camera'], gimbal_video_path)
        
        print(f"✓ Episode {episode_idx} saved")
    
    def save_video(self, frames: List[np.ndarray], output_path: Path):
        """Save video frames as MP4."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].shape[:2]
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
    
    def create_metadata_files(self, num_episodes: int):
        """Create LeRobot metadata files."""
        
        # Calculate total frames
        total_frames = num_episodes * self.episode_length
        
        # info.json with complete features schema
        info = {
            "codebase_version": "v2.0",
            "robot_type": "UAVQuadcopter",
            "total_episodes": num_episodes,
            "total_frames": total_frames,
            "total_tasks": len(self.scenarios),
            "total_videos": 2,  # front_camera and gimbal_camera
            "total_chunks": 0,
            "chunks_size": 1000,
            "fps": float(self.fps),
            "splits": {
                "train": "0:100"
            },
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                # Video features
                "observation.images.front_camera": {
                    "dtype": "video",
                    "shape": list(self.image_size),
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": float(self.fps),
                        "video.codec": "h264", 
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                },
                "observation.images.gimbal_camera": {
                    "dtype": "video",
                    "shape": list(self.image_size),
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": float(self.fps),
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p", 
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                },
                # State features (13D)
                "observation.state": {
                    "dtype": "float64",
                    "shape": [13],
                    "names": [
                        "position_x", "position_y", "position_z",           # 0-2
                        "orientation_roll", "orientation_pitch", "orientation_yaw",  # 3-5
                        "velocity_x", "velocity_y", "velocity_z",           # 6-8
                        "battery_level",                                    # 9
                        "gps_lat", "gps_lon", "gps_alt"                    # 10-12
                    ]
                },
                # Action features (9D)
                "action": {
                    "dtype": "float64",
                    "shape": [9],
                    "names": [
                        "flight_throttle", "flight_roll", "flight_pitch", "flight_yaw",  # 0-3
                        "velocity_cmd_x", "velocity_cmd_y", "velocity_cmd_z",            # 4-6
                        "gimbal_pitch", "gimbal_yaw"                                     # 7-8
                    ]
                },
                # Standard LeRobot fields
                "timestamp": {
                    "dtype": "float64",
                    "shape": [1]
                },
                "task_index": {
                    "dtype": "int64",
                    "shape": [1]
                },
                "episode_index": {
                    "dtype": "int64", 
                    "shape": [1]
                },
                "index": {
                    "dtype": "int64",
                    "shape": [1]
                },
                "next.reward": {
                    "dtype": "float64",
                    "shape": [1]
                },
                "next.done": {
                    "dtype": "bool",
                    "shape": [1]
                }
            },
            "encoding": {
                "video": {
                    "pix_fmt": "yuv420p",
                    "vcodec": "libx264"
                }
            }
        }
        
        with open(self.output_dir / "meta" / "info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        # Copy modality.json with correct rotation type
        modality_path = self.output_dir / "meta" / "modality.json"
        if not modality_path.exists():
            modality = {
                "state": {
                    "position": {"start": 0, "end": 3, "dtype": "float32", "range": [-100.0, 100.0]},
                    "orientation": {"start": 3, "end": 6, "rotation_type": "euler_angles_rpy", "dtype": "float32", "range": [-3.14159, 3.14159]},
                    "velocity": {"start": 6, "end": 9, "dtype": "float32", "range": [-50.0, 50.0]},
                    "battery": {"start": 9, "end": 10, "dtype": "float32", "range": [0.0, 100.0]},
                    "gps": {"start": 10, "end": 13, "dtype": "float32", "range": [-180.0, 180.0]}
                },
                "action": {
                    "flight_control": {"start": 0, "end": 4, "absolute": True, "dtype": "float32", "range": [-1.0, 1.0]},
                    "velocity_command": {"start": 4, "end": 7, "absolute": False, "dtype": "float32", "range": [-10.0, 10.0]},
                    "gimbal": {"start": 7, "end": 9, "absolute": True, "rotation_type": "euler_angles_rpy", "dtype": "float32", "range": [-1.57, 1.57]}
                },
                "video": {
                    "front_camera": {"original_key": "observation.images.front_camera"},
                    "gimbal_camera": {"original_key": "observation.images.gimbal_camera"}
                },
                "annotation": {
                    "human.task_description": {"original_key": "task_description"}
                }
            }
            
            with open(modality_path, 'w') as f:
                json.dump(modality, f, indent=2)
        
        # tasks.jsonl (task descriptions)
        tasks_path = self.output_dir / "meta" / "tasks.jsonl"
        with open(tasks_path, 'w') as f:
            for i in range(num_episodes):
                task = {"task_index": i, "task_description": np.random.choice(self.scenarios)}
                f.write(json.dumps(task) + '\n')
        
        # episodes.jsonl (episode metadata)
        episodes_path = self.output_dir / "meta" / "episodes.jsonl"
        with open(episodes_path, 'w') as f:
            for i in range(num_episodes):
                episode_meta = {
                    "episode_index": i,
                    "tasks": [np.random.choice(self.scenarios), "valid"],
                    "length": self.episode_length
                }
                f.write(json.dumps(episode_meta) + '\n')
        
        # stats.json (dataset statistics)
        stats = {
            "observation.state": {
                "mean": [
                    0.0, 0.0, 50.0,        # position: x, y, z (start at 50m altitude)
                    0.0, 0.0, 0.0,         # orientation: roll, pitch, yaw
                    0.0, 0.0, -2.0,        # velocity: vx, vy, vz (descending)
                    75.0,                  # battery: 75% average
                    37.7749, -122.4194, 50.0  # gps: lat, lon, alt
                ],
                "std": [
                    30.0, 30.0, 25.0,     # position variation
                    0.3, 0.3, 0.3,        # orientation variation (radians)
                    5.0, 5.0, 3.0,        # velocity variation
                    15.0,                  # battery variation
                    0.01, 0.01, 25.0      # gps variation
                ],
                "min": [
                    -100.0, -100.0, 0.0,  # position limits
                    -3.14159, -3.14159, -3.14159,  # orientation limits
                    -50.0, -50.0, -50.0,  # velocity limits
                    0.0,                   # battery minimum
                    -90.0, -180.0, 0.0    # gps limits
                ],
                "max": [
                    100.0, 100.0, 100.0,  # position limits
                    3.14159, 3.14159, 3.14159,  # orientation limits
                    50.0, 50.0, 50.0,     # velocity limits
                    100.0,                 # battery maximum
                    90.0, 180.0, 100.0    # gps limits
                ]
            },
            "action": {
                "mean": [
                    0.5, 0.0, 0.0, 0.0,   # flight_control: throttle up, level
                    0.0, 0.0, -1.0,        # velocity_command: descending
                    -0.1, 0.0              # gimbal: slightly down
                ],
                "std": [
                    0.2, 0.3, 0.3, 0.3,   # flight_control variation
                    2.0, 2.0, 2.0,        # velocity_command variation
                    0.3, 0.2               # gimbal variation
                ],
                "min": [
                    -1.0, -1.0, -1.0, -1.0,  # flight_control limits
                    -10.0, -10.0, -10.0,     # velocity_command limits
                    -1.57, -1.57             # gimbal limits (radians)
                ],
                "max": [
                    1.0, 1.0, 1.0, 1.0,     # flight_control limits
                    10.0, 10.0, 10.0,       # velocity_command limits
                    1.57, 1.57               # gimbal limits (radians)
                ]
            }
        }
        
        stats_path = self.output_dir / "meta" / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("✓ Metadata files created")
    
    def generate_dataset(self, num_episodes: int = 50):
        """Generate complete UAV landing dataset."""
        print(f"Generating UAV Landing Dataset with {num_episodes} episodes...")
        print(f"Output directory: {self.output_dir}")
        
        # Generate episodes
        for i in range(num_episodes):
            episode_data = self.generate_episode(i)
            self.save_episode(episode_data)
        
        # Create metadata
        self.create_metadata_files(num_episodes)
        
        print(f"\n✓ Dataset generation completed!")
        print(f"Generated {num_episodes} episodes in LeRobot format")
        print(f"Dataset ready for UAV training at: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate UAV Landing Dataset")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./demo_data/uav.Landing",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--num_episodes", 
        type=int, 
        default=5,
        help="Number of episodes to generate"
    )
    parser.add_argument(
        "--episode_length", 
        type=int, 
        default=200,
        help="Length of each episode (frames)"
    )
    parser.add_argument(
        "--image_size", 
        type=str, 
        default="480,640,3",
        help="Image size (height,width,channels)"
    )
    
    args = parser.parse_args()
    
    # Parse image size
    image_size = tuple(map(int, args.image_size.split(',')))
    
    # Create generator
    generator = UAVLandingDataGenerator(
        output_dir=args.output_dir,
        image_size=image_size
    )
    generator.episode_length = args.episode_length
    
    # Generate dataset
    generator.generate_dataset(num_episodes=args.num_episodes)
    
    print(f"\nTo use this dataset:")
    print(f"  python scripts/load_dataset.py --data_path {args.output_dir} --embodiment_tag uav_quadcopter")
    print(f"  python scripts/uav_finetune.py --data_path {args.output_dir}")


if __name__ == "__main__":
    main()
