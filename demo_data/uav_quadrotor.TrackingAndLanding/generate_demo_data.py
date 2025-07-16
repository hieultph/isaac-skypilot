#!/usr/bin/env python3
"""
Generate sample UAV quadrotor demo data for tracking and landing tasks.
This script creates synthetic data in the format expected by the GR00T framework.
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import cv2
from datetime import datetime

def generate_uav_trajectory(task_type, episode_length, fps=30.0):
    """Generate realistic UAV trajectory data for different tasks."""
    dt = 1.0 / fps
    time_steps = episode_length
    
    # Initialize state arrays
    positions = np.zeros((time_steps, 3))
    orientations = np.zeros((time_steps, 4))  # quaternions
    velocities = np.zeros((time_steps, 3))
    angular_velocities = np.zeros((time_steps, 3))
    battery_levels = np.zeros((time_steps, 1))
    target_positions = np.zeros((time_steps, 3))
    target_distances = np.zeros((time_steps, 1))
    
    # Initialize actions
    actions = np.zeros((time_steps, 4))  # thrust, roll, pitch, yaw
    
    # Task-specific trajectories
    if task_type == "takeoff_hover":
        # Takeoff and hover
        for i in range(time_steps):
            t = i * dt
            # Takeoff phase (0-3 seconds)
            if t < 3.0:
                positions[i] = [0, 0, (t/3.0) * 5.0]
                actions[i] = [0.8, 0, 0, 0]  # High thrust
            else:
                # Hover phase
                positions[i] = [0, 0, 5.0 + 0.1 * np.sin(t)]  # Small oscillation
                actions[i] = [0.5, 0, 0, 0]  # Hover thrust
            
            orientations[i] = [1, 0, 0, 0]  # No rotation
            velocities[i] = [0, 0, 0] if t > 3.0 else [0, 0, 5.0/3.0]
            battery_levels[i] = 1.0 - (t / (time_steps * dt)) * 0.15
            target_positions[i] = [0, 0, 5.0]
            target_distances[i] = np.linalg.norm(positions[i] - target_positions[i])
    
    elif task_type == "track_moving_target":
        # Track moving target
        for i in range(time_steps):
            t = i * dt
            # Moving target in figure-8 pattern
            target_x = 10 * np.cos(t * 0.2)
            target_y = 5 * np.sin(t * 0.4)
            target_z = 3.0
            target_positions[i] = [target_x, target_y, target_z]
            
            # UAV follows with some lag and maintains distance
            if i > 0:
                # Proportional controller
                error = target_positions[i] - positions[i-1]
                desired_offset = np.array([0, 0, 2.0])  # Stay 2m above target
                desired_pos = target_positions[i] + desired_offset
                
                # Velocity control
                vel_cmd = 0.5 * (desired_pos - positions[i-1])
                vel_cmd = np.clip(vel_cmd, -3.0, 3.0)
                
                positions[i] = positions[i-1] + vel_cmd * dt
                velocities[i] = vel_cmd
                
                # Actions based on desired velocity
                actions[i, 0] = 0.5 + 0.1 * vel_cmd[2]  # Thrust
                actions[i, 1] = np.clip(vel_cmd[1] * 0.1, -0.5, 0.5)  # Roll
                actions[i, 2] = np.clip(vel_cmd[0] * 0.1, -0.5, 0.5)  # Pitch
                actions[i, 3] = np.clip(np.arctan2(vel_cmd[1], vel_cmd[0]) * 0.1, -0.5, 0.5)  # Yaw
            else:
                positions[i] = [0, 0, 5.0]
                actions[i] = [0.5, 0, 0, 0]
            
            orientations[i] = [1, 0, 0, 0]
            battery_levels[i] = 1.0 - (t / (time_steps * dt)) * 0.2
            target_distances[i] = np.linalg.norm(positions[i] - target_positions[i])
    
    elif task_type == "landing":
        # Landing sequence
        for i in range(time_steps):
            t = i * dt
            # Descend from 5m to 0m
            start_height = 5.0
            positions[i] = [0, 0, max(0, start_height - (t/10.0) * start_height)]
            
            if positions[i, 2] > 0.1:
                actions[i] = [0.3, 0, 0, 0]  # Reduced thrust for descent
                velocities[i] = [0, 0, -0.5]
            else:
                actions[i] = [0.0, 0, 0, 0]  # No thrust on ground
                velocities[i] = [0, 0, 0]
            
            orientations[i] = [1, 0, 0, 0]
            battery_levels[i] = 1.0 - (t / (time_steps * dt)) * 0.1
            target_positions[i] = [0, 0, 0]
            target_distances[i] = positions[i, 2]
    
    # Add noise to make data more realistic
    positions += np.random.normal(0, 0.05, positions.shape)
    orientations[:, 1:] += np.random.normal(0, 0.01, orientations[:, 1:].shape)
    velocities += np.random.normal(0, 0.1, velocities.shape)
    angular_velocities += np.random.normal(0, 0.02, angular_velocities.shape)
    actions += np.random.normal(0, 0.01, actions.shape)
    
    # Normalize quaternions
    orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)
    
    return {
        'positions': positions,
        'orientations': orientations,
        'velocities': velocities,
        'angular_velocities': angular_velocities,
        'battery_levels': battery_levels,
        'target_positions': target_positions,
        'target_distances': target_distances,
        'actions': actions
    }

def create_sample_episode(episode_index, task_type, episode_length):
    """Create a sample episode data file."""
    
    # Generate trajectory data
    traj_data = generate_uav_trajectory(task_type, episode_length)
    
    # Prepare data for DataFrame
    data = {
        'timestamp': np.arange(episode_length) / 30.0,  # 30 FPS
        'episode_index': [episode_index] * episode_length,
        'index': np.arange(episode_length),
        'task_index': [get_task_index(task_type)] * episode_length,
        'annotation.human.validity': [1] * episode_length,
        'annotation.human.action.task_description': [get_task_index(task_type)] * episode_length,
        'next.reward': np.random.uniform(-0.1, 1.0, episode_length),
        'next.done': [False] * (episode_length - 1) + [True],
    }
    
    # Add observation state (18 dimensions)
    obs_state = np.column_stack([
        traj_data['positions'],           # 0-2: pos_x, pos_y, pos_z
        traj_data['orientations'],        # 3-6: quat_w, quat_x, quat_y, quat_z
        traj_data['velocities'],          # 7-9: vel_x, vel_y, vel_z
        traj_data['angular_velocities'],  # 10-12: angular_vel_x, angular_vel_y, angular_vel_z
        traj_data['battery_levels'],      # 13: battery_level
        traj_data['target_positions'],    # 14-16: target_pos_x, target_pos_y, target_pos_z
        traj_data['target_distances']     # 17: target_distance
    ])
    
    # Add observation state columns
    for i in range(18):
        data[f'observation.state.{i}'] = obs_state[:, i]
    
    # Add action columns
    for i in range(4):
        data[f'action.{i}'] = traj_data['actions'][:, i]
    
    return pd.DataFrame(data)

def get_task_index(task_type):
    """Get task index for task type."""
    task_map = {
        'takeoff_hover': 0,
        'track_moving_target': 1,
        'track_moving_target_blue': 2,
        'track_stationary_target': 3,
        'landing': 4,
        'precision_landing': 5,
        'waypoint_landing': 6,
        'complex_mission': 7
    }
    return task_map.get(task_type, 0)

def create_sample_video_frame(width=640, height=480, frame_num=0, camera_type='front'):
    """Create a sample video frame for demonstration."""
    # Create a synthetic image
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient background
    for i in range(height):
        frame[i, :, 0] = int(255 * i / height)  # Red gradient
        frame[i, :, 1] = int(128 * (1 - i / height))  # Green gradient
        frame[i, :, 2] = 100  # Blue constant
    
    # Add camera-specific elements
    if camera_type == 'front':
        # Add horizon line
        cv2.line(frame, (0, height//2), (width, height//2), (255, 255, 255), 2)
        # Add target indicator
        cv2.circle(frame, (width//2 + int(50*np.sin(frame_num*0.1)), height//2), 20, (0, 255, 0), 3)
        cv2.putText(frame, "TARGET", (width//2 - 30, height//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    elif camera_type == 'bottom':
        # Add ground pattern
        for i in range(0, width, 50):
            cv2.line(frame, (i, 0), (i, height), (128, 128, 128), 1)
        for i in range(0, height, 50):
            cv2.line(frame, (0, i), (width, i), (128, 128, 128), 1)
        # Add landing pad
        cv2.circle(frame, (width//2, height//2), 30, (255, 255, 0), -1)
        cv2.putText(frame, "LAND", (width//2 - 20, height//2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Add frame number
    cv2.putText(frame, f"Frame {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def generate_demo_data():
    """Generate complete demo dataset."""
    base_path = Path("f:/Work/CT UAV/Projects/isaac_skypilot/demo_data/uav_quadrotor.TrackingAndLanding")
    
    # Episode configurations
    episodes = [
        (0, "takeoff_hover", 450),
        (1, "track_moving_target", 520),
        (2, "track_moving_target_blue", 480),
        (3, "track_stationary_target", 400),
        (4, "landing", 350),
        (5, "precision_landing", 380),
        (6, "waypoint_landing", 420),
        (7, "complex_mission", 200)
    ]
    
    print("Generating UAV demo dataset...")
    
    for episode_idx, task_type, length in episodes:
        print(f"  Creating episode {episode_idx}: {task_type} ({length} frames)")
        
        # Create episode data
        df = create_sample_episode(episode_idx, task_type, length)
        
        # Save as parquet
        output_path = base_path / f"data/chunk-000/episode_{episode_idx:06d}.parquet"
        df.to_parquet(output_path, index=False)
        
        # Create sample video frames (just a few for demo)
        if episode_idx < 3:  # Only create videos for first 3 episodes
            for camera in ['front_camera', 'bottom_camera', 'downward_camera']:
                video_dir = base_path / f"videos/chunk-000/{camera}"
                video_dir.mkdir(parents=True, exist_ok=True)
                
                # Create sample video file (placeholder)
                video_path = video_dir / f"episode_{episode_idx:06d}.mp4"
                
                # Create a short sample video
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                if camera == 'downward_camera':
                    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (256, 256))
                    for frame_num in range(min(60, length)):  # 2 seconds of video
                        frame = create_sample_video_frame(256, 256, frame_num, 'bottom')
                        out.write(frame)
                else:
                    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
                    cam_type = 'front' if camera == 'front_camera' else 'bottom'
                    for frame_num in range(min(60, length)):  # 2 seconds of video
                        frame = create_sample_video_frame(640, 480, frame_num, cam_type)
                        out.write(frame)
                out.release()
    
    print("Demo dataset generation complete!")
    print(f"Dataset location: {base_path}")
    print("\nDataset structure:")
    print("- 8 episodes with different UAV tasks")
    print("- 18-dimensional state space (position, orientation, velocity, etc.)")
    print("- 4-dimensional action space (thrust, roll, pitch, yaw)")
    print("- 3 camera views (front, bottom, downward)")
    print("- Tasks: takeoff/hover, target tracking, landing")

if __name__ == "__main__":
    generate_demo_data()
