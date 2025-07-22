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
from transformers import TrainingArguments
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import UAVQuadcopterDataConfig
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5


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
    
    # Data compatibility arguments
    parser.add_argument(
        "--no_language", 
        action="store_true",
        help="Use dataset configuration without language annotations (for compatibility with datasets that don't have task descriptions)"
    )
    
    return parser.parse_args()

def setup_uav_dataset(data_path: str, force_no_language: bool = False):
    """Setup UAV dataset with proper modality configs and transforms."""
    
    # First, check what's actually in the dataset metadata
    import json
    from pathlib import Path
    from gr00t.data.dataset import ModalityConfig
    
    meta_path = Path(data_path) / "meta" / "modality.json"
    has_language_annotation = True
    
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            modality_meta = json.load(f)
        
        # Check if the annotation metadata is correctly configured
        if "annotation" in modality_meta:
            annotation_meta = modality_meta["annotation"]
            if "human.task_description" in annotation_meta:
                original_key = annotation_meta["human.task_description"].get("original_key")
                print(f"Found annotation metadata with original_key: {original_key}")
                
                # Fix the metadata if it's incorrect
                if original_key == "task_description":
                    print("Fixing incorrect original_key in modality.json...")
                    annotation_meta["human.task_description"]["original_key"] = "annotation.human.task_description"
                    
                    # Write back the corrected metadata
                    with open(meta_path, 'w') as f:
                        json.dump(modality_meta, f, indent=2)
                    print("Metadata fixed!")
            else:
                print("No language annotation found in metadata")
                has_language_annotation = False
        else:
            print("No annotation section found in metadata")
            has_language_annotation = False
    else:
        print("No modality.json found - assuming no language annotation")
        has_language_annotation = False
    
    # Check if task_description column exists in the actual data
    try:
        import pandas as pd
        first_episode_path = Path(data_path) / "data" / "episode_0.parquet"
        if first_episode_path.exists():
            df = pd.read_parquet(first_episode_path)
            if "annotation.human.task_description" not in df.columns and "task_description" not in df.columns:
                print("No task description columns found in data - using configuration without language")
                has_language_annotation = False
    except Exception as e:
        print(f"Could not read episode data to check columns: {e}")
    
    # Create a custom UAV config class without language if needed
    if not has_language_annotation or force_no_language:
        if force_no_language:
            print("Forcing UAV configuration without language annotation (--no_language flag)")
        else:
            print("Creating UAV configuration without language annotation...")
        
        class UAVQuadcopterNoLanguageDataConfig(UAVQuadcopterDataConfig):
            """UAV config without language annotation for compatibility."""
            language_keys = []  # No language keys
            
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

                # No language modality
                modality_configs = {
                    "video": video_modality,
                    "state": state_modality,
                    "action": action_modality,
                }

                return modality_configs
        
        uav_config = UAVQuadcopterNoLanguageDataConfig()
    else:
        # Use standard UAV data configuration
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

    print("Dataset loaded successfully!")
    
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
    print(f"  No language: {args.no_language}")
    
    # Setup dataset
    print("\\nSetting up UAV dataset...")
    dataset, modality_configs, transforms = setup_uav_dataset(args.data_path, args.no_language)
    print(f"Dataset loaded with {len(dataset)} episodes")
    
    # Load pretrained model
    print("\\nLoading pretrained GR00T model...")
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        tune_llm=not args.freeze_language_model,
        tune_visual=not args.freeze_backbone,
        tune_projector=not args.only_train_action_head,
        tune_diffusion_model=True,  # Always tune the diffusion model for UAV
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Get UAV data configuration to determine action horizon
    uav_config = UAVQuadcopterDataConfig()
    data_action_horizon = len(uav_config.action_indices)
    
    # Update action_horizon to match UAV data config
    if data_action_horizon != model.action_head.config.action_horizon:
        print(f"Updating action head for UAV: action_horizon {data_action_horizon} (was {model.action_head.config.action_horizon})")
        
        # Update the action head config
        new_action_head_config = model.action_head.config
        new_action_head_config.action_horizon = data_action_horizon
        
        # Import the FlowmatchingActionHead class
        from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead
        
        # Create new action head with updated config
        new_action_head = FlowmatchingActionHead(new_action_head_config)
        
        # Copy weights from old action head (strict=False for size mismatches)
        new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)
        
        # Replace the action head
        model.action_head = new_action_head
        
        # Update model config
        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
    
    # Configure what to freeze for UAV adaptation
    if args.freeze_backbone:
        print("Freezing visual backbone...")
        for param in model.backbone.eagle_model.vision_model.parameters():
            param.requires_grad = False
        for param in model.backbone.eagle_model.mlp1.parameters():
            param.requires_grad = False
            
    if args.freeze_language_model:
        print("Freezing language model...")
        for param in model.backbone.eagle_model.language_model.parameters():
            param.requires_grad = False
    
    if args.only_train_action_head:
        print("Only training UAV action head...")
        # Freeze everything except action head
        for param in model.backbone.parameters():
            param.requires_grad = False
        # Only keep action head trainable
        for param in model.action_head.parameters():
            param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\\nTrainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Setup training arguments (compatible with HuggingFace TrainingArguments)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        report_to=["tensorboard"],
        seed=42,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )
    
    print(f"\\nTraining configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Device: {args.device}")
    print(f"  Mixed precision: {'bf16' if torch.cuda.is_bf16_supported() else 'fp16'}")
    
    # Initialize training runner
    print("\\nInitializing training...")
    runner = TrainRunner(
        model=model,
        training_args=training_args,
        train_dataset=dataset,
        resume_from_checkpoint=False
    )

    
    # Start training
    print("\\nStarting UAV fine-tuning...")
    print("Key insight: Leveraging pretrained VLM and only adapting action generation for UAV control")
    runner.train()
    
    print(f"\\nTraining completed! Model saved to: {args.output_dir}")
    print("\\nTo use the trained model:")
    print(f"  from gr00t.model.gr00t_n1 import GR00T_N1_5")
    print(f"  uav_model = GR00T_N1_5.from_pretrained('{args.output_dir}')")
    print(f"  # Use model for UAV inference")
    
    print("\\n" + "="*60)
    print("TROUBLESHOOTING:")
    print("If you encountered a 'task_description' KeyError:")
    print("1. Use --no_language flag to train without language annotations")
    print("2. Or regenerate your dataset with the fixed generate_uav_dataset.py script")
    print("3. Example: python scripts/uav_finetune.py --data_path /path/to/data --no_language")
    print("="*60)


if __name__ == "__main__":
    main()
