#!/usr/bin/env python3
"""
UAV Training Script for GR00T
This script demonstrates how to train a GR00T model for UAV control
"""

import os
import torch
from pathlib import Path
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy
from transformers import TrainingArguments

def train_uav_model():
    """Train GR00T model for UAV control"""
    
    # Configuration
    MODEL_PATH = "nvidia/GR00T-N1.5-3B"  # Base pretrained model
    DATASET_PATH = "demo_data/uav_quadrotor.TrackingAndLanding"  # Your UAV dataset
    EMBODIMENT_TAG = "uav_quadrotor"
    OUTPUT_DIR = "./uav_gr00t_checkpoints"
    
    # Get data configuration
    data_config = DATA_CONFIG_MAP["uav_quadrotor"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    
    # Create dataset
    dataset = LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=modality_config,
        video_backend="decord",
        transforms=modality_transform,
        embodiment_tag=EMBODIMENT_TAG,
    )
    
    # Load pretrained model
    model = GR00T_N1_5.from_pretrained(
        MODEL_PATH,
        tune_llm=False,      # Keep vision-language backbone frozen
        tune_visual=False,   # Keep vision encoder frozen
        tune_projector=True, # Fine-tune projector for UAV
        tune_diffusion_model=True,  # Fine-tune diffusion model for UAV actions
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        weight_decay=1e-5,
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        fp16=True,
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = TrainRunner(
        model=model,
        training_args=training_args,
        train_dataset=dataset,
        resume_from_checkpoint=False,
    )
    
    # Start training
    trainer.train()
    
    print(f"Training completed! Model saved to {OUTPUT_DIR}")

def test_uav_inference():
    """Test UAV model inference"""
    
    MODEL_PATH = "./uav_gr00t_checkpoints"
    EMBODIMENT_TAG = "uav_quadrotor"
    
    # Get data configuration
    data_config = DATA_CONFIG_MAP["uav_quadrotor"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    
    # Load trained policy
    policy = Gr00tPolicy(
        model_path=MODEL_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Example input (replace with your actual UAV data)
    observations = {
        "video": {
            "front_camera": torch.randn(1, 3, 224, 224),  # Example video frame
            "bottom_camera": torch.randn(1, 3, 224, 224),
        },
        "state": {
            "position": torch.tensor([[0.0, 0.0, 1.0]]),        # x, y, z
            "orientation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]), # w, x, y, z
            "velocity": torch.tensor([[0.0, 0.0, 0.0]]),         # vx, vy, vz
            "angular_velocity": torch.tensor([[0.0, 0.0, 0.0]]), # wx, wy, wz
            "battery_level": torch.tensor([[0.8]]),              # battery level
        },
        "annotation": {
            "human.task_description": "fly to the red building"
        }
    }
    
    # Get action prediction
    with torch.no_grad():
        action = policy.get_action(observations)
    
    print("UAV Action Prediction:")
    print(f"Thrust: {action['action']['thrust']}")
    print(f"Roll: {action['action']['roll']}")
    print(f"Pitch: {action['action']['pitch']}")
    print(f"Yaw: {action['action']['yaw']}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="UAV GR00T Training")
    parser.add_argument("--mode", choices=["train", "test"], default="train", 
                       help="Mode: train or test")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_uav_model()
    else:
        test_uav_inference()
