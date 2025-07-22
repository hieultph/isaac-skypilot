#!/usr/bin/env python3
"""
UAV Quadcopter Fine-tuning Script

This script demonstrates how to fine-tune the GR00T VLA model for UAV quadcopter control.
The key approach is to leverage the pretrained VLM and only retrain the diffusion action head.

Usage:
    python scripts/uav_finetune.py --data_path /path/to/uav/dataset --output_dir /path/to/output

State Space (13D):
    - position: x, y, z (3)
    - orientation: roll, pitch, yaw (3)  
    - velocity: vx, vy, vz (3)
    - battery: battery level (1)
    - gps: lat, lon, alt (3)

Action Space (9D):
    - flight_control: throttle, roll, pitch, yaw (4)
    - velocity_command: vx, vy, vz (3)
    - gimbal: gimbal_pitch, gimbal_yaw (2)
"""

import argparse
import os
from pathlib import Path

import torch
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import UAVQuadcopterDataConfig
from gr00t.experiment.runner import ExperimentRunner
from gr00t.model.policy import Gr00tPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GR00T for UAV Quadcopter Control")
    
    # Data arguments
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True,
        help="Path to UAV dataset in LeRobot format"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./checkpoints/uav_quadcopter_finetune",
        help="Output directory for checkpoints"
    )
    
    # Model arguments  
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="nvidia/GR00T-N1.5-3B",
        help="Path to pretrained GR00T model"
    )
    parser.add_argument(
        "--embodiment_tag", 
        type=str, 
        default="uav_quadcopter",
        help="Embodiment tag for UAV"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Training device"
    )
    
    # Freezing arguments - key for UAV adaptation
    parser.add_argument(
        "--freeze_backbone", 
        action="store_true",
        help="Freeze VLM visual encoder (recommended for UAV)"
    )
    parser.add_argument(
        "--freeze_language_model", 
        action="store_true", 
        help="Freeze language model (recommended for UAV)"
    )
    parser.add_argument(
        "--only_train_action_head", 
        action="store_true",
        help="Only train UAV-specific action head (recommended)"
    )
    
    return parser.parse_args()


def setup_uav_dataset(data_path: str):
    """Setup UAV dataset with proper modality configs and transforms."""
    
    # Initialize UAV data configuration
    uav_config = UAVQuadcopterDataConfig()
    modality_configs = uav_config.modality_config()
    transforms = uav_config.transform()
    
    # Create dataset
    dataset = LeRobotSingleDataset(
        dataset_path=data_path,
        modality_configs=modality_configs,
        embodiment_tag=EmbodimentTag.UAV_QUADCOPTER,
        transforms=transforms,
        video_backend="torchvision_av",
    )
    
    return dataset, modality_configs, transforms


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Setting up UAV fine-tuning:")
    print(f"  Data path: {args.data_path}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Model: {args.model_path}")
    print(f"  Embodiment: {args.embodiment_tag}")
    print(f"  Freeze backbone: {args.freeze_backbone}")
    print(f"  Freeze LM: {args.freeze_language_model}")
    print(f"  Only train action head: {args.only_train_action_head}")
    
    # Setup dataset
    print("\\nSetting up UAV dataset...")
    dataset, modality_configs, transforms = setup_uav_dataset(args.data_path)
    print(f"Dataset loaded with {len(dataset)} episodes")
    
    # Load pretrained model
    print("\\nLoading pretrained GR00T model...")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_configs,
        modality_transform=transforms,
        embodiment_tag=EmbodimentTag.UAV_QUADCOPTER,
        device=args.device
    )
    
    # Configure what to freeze for UAV adaptation
    if args.freeze_backbone:
        print("Freezing visual backbone...")
        for param in policy.model.backbone.parameters():
            param.requires_grad = False
            
    if args.freeze_language_model:
        print("Freezing language model...")
        for param in policy.model.language_model.parameters():
            param.requires_grad = False
    
    if args.only_train_action_head:
        print("Only training UAV action head...")
        # Freeze everything except UAV action head
        for name, param in policy.model.named_parameters():
            if "action_head" not in name or "uav_quadcopter" not in name:
                param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in policy.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in policy.model.parameters())
    print(f"\\nTrainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Setup training configuration
    training_config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "output_dir": args.output_dir,
        "device": args.device,
        "embodiment_tag": args.embodiment_tag,
        
        # UAV-specific training settings
        "action_horizon": 16,  # 16-step action prediction
        "state_dim": 13,       # 13D UAV state space
        "action_dim": 9,       # 9D UAV action space
        
        # Logging and checkpointing
        "log_every": 100,
        "save_every": 1000,
        "eval_every": 500,
    }
    
    print(f"\\nTraining configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    
    # Initialize training runner
    print("\\nInitializing training...")
    runner = ExperimentRunner(
        model=policy.model,
        dataset=dataset,
        config=training_config
    )
    
    # Start training
    print("\\nStarting UAV fine-tuning...")
    print("Key insight: Leveraging pretrained VLM and only adapting action generation for UAV control")
    runner.train()
    
    print(f"\\nTraining completed! Model saved to: {args.output_dir}")
    print("\\nTo use the trained model:")
    print(f"  from gr00t.model.policy import Gr00tPolicy")
    print(f"  uav_policy = Gr00tPolicy(model_path='{args.output_dir}', ...)")
    print(f"  action = uav_policy.get_action(observation)")


if __name__ == "__main__":
    main()
