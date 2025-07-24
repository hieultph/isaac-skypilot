#!/usr/bin/env python3
"""
UAV Quadcopter Evaluation Example

This script demonstrates how to evaluate a fine-tuned UAV model and provides
an example interface for UAV control integration.

Usage:
    python getting_started/examples/eval_uav_quadcopter.py --model_path /path/to/trained/model
"""

import argparse
import time
import numpy as np
from typing import Dict, Any

from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import UAVQuadcopterDataConfig


class UAVControlInterface:
    """
    Example interface for UAV control integration.
    This class shows how to integrate the GR00T UAV model with actual UAV control systems.
    """
    
    def __init__(self, policy: Gr00tPolicy):
        self.policy = policy
        self.last_observation = None
        
    def get_current_observation(self) -> Dict[str, Any]:
        """
        Get current UAV observation including camera feeds and telemetry.
        In a real implementation, this would interface with your UAV's sensors.
        """
        # Mock observation - replace with actual UAV sensor data
        # Note: Video data needs to be in [T, H, W, C] format where T=1 for single frame
        front_camera_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        gimbal_camera_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        observation = {
            # Camera feeds - add time dimension [T=1, H, W, C] for video format
            "video.front_camera": np.expand_dims(front_camera_frame, axis=0),  # [1, 480, 640, 3]
            "video.gimbal_camera": np.expand_dims(gimbal_camera_frame, axis=0),  # [1, 480, 640, 3]
            
            # UAV state (13D) - ensure all state arrays have batch dimension for transform compatibility
            "state.position": np.array([[10.5, 5.2, 15.0]], dtype=np.float32),  # [1, 3] x, y, z
            "state.orientation": np.array([[0.1, -0.05, 1.57]], dtype=np.float32),  # [1, 3] roll, pitch, yaw
            "state.velocity": np.array([[2.0, 0.5, -0.1]], dtype=np.float32),  # [1, 3] vx, vy, vz
            "state.battery": np.array([[85.5]], dtype=np.float32),  # [1, 1] battery %
            "state.gps": np.array([[37.7749, -122.4194, 100.0]], dtype=np.float32),  # [1, 3] lat, lon, alt
            
            # Task description - must be a list for transform compatibility
            "annotation.human.task_description": ["Land safely on the designated platform while avoiding obstacles"],
        }
        
        self.last_observation = observation
        return observation
    
    def execute_action(self, action: Dict[str, np.ndarray]) -> bool:
        """
        Execute UAV action commands.
        In a real implementation, this would send commands to your UAV's flight controller.
        
        Args:
            action: Dictionary containing UAV control commands
            
        Returns:
            bool: True if commands were executed successfully
        """
        try:
            # Extract action components - actions have shape [action_horizon, action_dim]
            # Take the first action in the sequence for immediate execution
            flight_control = action["action.flight_control"][0]      # [throttle, roll, pitch, yaw]
            velocity_command = action["action.velocity_command"][0]  # [vx, vy, vz]
            gimbal_control = action["action.gimbal"][0]             # [gimbal_pitch, gimbal_yaw]
            
            print(f"Flight Control: throttle={flight_control[0]:.3f}, "
                  f"roll={flight_control[1]:.3f}, pitch={flight_control[2]:.3f}, yaw={flight_control[3]:.3f}")
            print(f"Velocity Command: vx={velocity_command[0]:.3f}, "
                  f"vy={velocity_command[1]:.3f}, vz={velocity_command[2]:.3f}")
            print(f"Gimbal: pitch={gimbal_control[0]:.3f}, yaw={gimbal_control[1]:.3f}")
            
            # In a real implementation, send these commands to your UAV:
            # - flight_control -> flight controller (PX4, ArduPilot, etc.)
            # - velocity_command -> velocity setpoints
            # - gimbal_control -> gimbal controller
            
            return True
            
        except Exception as e:
            print(f"Error executing UAV action: {e}")
            return False
    
    def run_control_loop(self, duration_seconds: float = 30.0, frequency_hz: float = 10.0):
        """
        Run the UAV control loop for a specified duration.
        
        Args:
            duration_seconds: How long to run the control loop
            frequency_hz: Control frequency in Hz
        """
        dt = 1.0 / frequency_hz
        start_time = time.time()
        step = 0
        
        print(f"\\nStarting UAV control loop for {duration_seconds}s at {frequency_hz}Hz...")
        print("Press Ctrl+C to stop early")
        
        try:
            while time.time() - start_time < duration_seconds:
                step_start = time.time()
                
                # Get current observation
                observation = self.get_current_observation()
                
                # Get action from policy
                action = self.policy.get_action(observation)
                
                # Execute action
                success = self.execute_action(action)
                
                if not success:
                    print("Failed to execute action, stopping...")
                    break
                
                # Timing
                step_time = time.time() - step_start
                if step % 20 == 0:  # Print every 2 seconds at 10Hz
                    print(f"Step {step}: {step_time*1000:.1f}ms")
                
                # Sleep to maintain frequency
                sleep_time = dt - step_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                step += 1
                
        except KeyboardInterrupt:
            print("\\nControl loop interrupted by user")
        
        print(f"Control loop completed after {step} steps")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UAV Quadcopter Model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to trained UAV model"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device for inference"
    )
    parser.add_argument(
        "--duration", 
        type=float, 
        default=30.0,
        help="Duration to run control loop (seconds)"
    )
    parser.add_argument(
        "--frequency", 
        type=float, 
        default=10.0,
        help="Control frequency (Hz)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("UAV Quadcopter Evaluation")
    print("=" * 50)
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Duration: {args.duration}s")
    print(f"Frequency: {args.frequency}Hz")
    
    # Setup UAV configuration
    print("\\nSetting up UAV configuration...")
    uav_config = UAVQuadcopterDataConfig()
    modality_configs = uav_config.modality_config()
    transforms = uav_config.transform()
    
    # Load trained UAV model
    print("Loading trained UAV model...")
    
    # Convert relative path to absolute path
    import os
    model_path = os.path.abspath(args.model_path)
    print(f"Absolute model path: {model_path}")
    
    try:
        uav_policy = Gr00tPolicy(
            model_path=model_path,
            modality_config=modality_configs,
            modality_transform=transforms,
            embodiment_tag=EmbodimentTag.UAV_QUADCOPTER,
            device=args.device
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Initialize UAV control interface
    print("Initializing UAV control interface...")
    uav_controller = UAVControlInterface(uav_policy)
    
    # Test single action
    print("\\nTesting single action generation...")
    obs = uav_controller.get_current_observation()
    action = uav_policy.get_action(obs)
    
    print("Generated action:")
    for key, value in action.items():
        print(f"  {key}: {value}")
    
    # Run control loop
    print(f"\\nRunning control loop...")
    uav_controller.run_control_loop(
        duration_seconds=args.duration,
        frequency_hz=args.frequency
    )
    
    print("\\nEvaluation completed!")
    print("\\nIntegration Notes:")
    print("1. Replace mock sensor data with actual UAV telemetry")
    print("2. Implement actual UAV command interface (PX4, ArduPilot, etc.)")
    print("3. Add safety checks and emergency stop mechanisms")
    print("4. Consider adding action smoothing and filtering")
    print("5. Implement proper error handling and recovery")


if __name__ == "__main__":
    main()
