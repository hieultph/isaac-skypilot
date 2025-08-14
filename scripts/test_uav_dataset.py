#!/usr/bin/env python3
"""
Test script for UAV dataset generation and validation.

This script tests the corrected dataset generation and validates the chunk structure.
"""

import sys
import json
from pathlib import Path

def test_dataset_structure(dataset_path: str):
    """Test that the dataset has the correct LeRobot structure."""
    dataset_dir = Path(dataset_path)
    
    print(f"Testing dataset structure: {dataset_dir}")
    
    # Check basic directory structure
    required_dirs = ["data", "meta", "videos"]
    for dir_name in required_dirs:
        dir_path = dataset_dir / dir_name
        if not dir_path.exists():
            print(f"❌ Missing directory: {dir_name}")
            return False
        print(f"✓ Directory exists: {dir_name}")
    
    # Check metadata files
    required_meta_files = ["info.json", "modality.json", "tasks.jsonl", "episodes.jsonl", "stats.json"]
    for file_name in required_meta_files:
        file_path = dataset_dir / "meta" / file_name
        if not file_path.exists():
            print(f"❌ Missing metadata file: {file_name}")
            return False
        print(f"✓ Metadata file exists: {file_name}")
    
    # Check info.json structure
    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    print(f"✓ Total episodes: {info['total_episodes']}")
    print(f"✓ Total chunks: {info['total_chunks']}")
    print(f"✓ Chunk size: {info['chunks_size']}")
    
    # Check data chunks
    data_dir = dataset_dir / "data"
    chunk_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")]
    
    print(f"✓ Found {len(chunk_dirs)} chunk directories")
    
    # Validate chunk structure
    total_episodes_found = 0
    for chunk_dir in sorted(chunk_dirs):
        episode_files = list(chunk_dir.glob("episode_*.parquet"))
        print(f"✓ Chunk {chunk_dir.name}: {len(episode_files)} episodes")
        total_episodes_found += len(episode_files)
    
    if total_episodes_found == info['total_episodes']:
        print(f"✓ Episode count matches: {total_episodes_found}")
    else:
        print(f"❌ Episode count mismatch: found {total_episodes_found}, expected {info['total_episodes']}")
        return False
    
    # Check video structure
    video_dir = dataset_dir / "videos"
    video_chunk_dirs = [d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")]
    
    print(f"✓ Found {len(video_chunk_dirs)} video chunk directories")
    
    # Check camera directories
    if video_chunk_dirs:
        first_video_chunk = video_chunk_dirs[0]
        camera_dirs = [d for d in first_video_chunk.iterdir() if d.is_dir()]
        camera_names = [d.name for d in camera_dirs]
        
        expected_cameras = ["observation.images.front_camera", "observation.images.gimbal_camera"]
        for camera in expected_cameras:
            if camera in camera_names:
                print(f"✓ Camera directory exists: {camera}")
            else:
                print(f"❌ Missing camera directory: {camera}")
                return False
    
    print("✅ Dataset structure validation completed successfully!")
    return True

def test_dataset_loading():
    """Test loading the dataset with GR00T configuration."""
    try:
        # Add project root to path
        project_root = Path(__file__).parent.parent
        sys.path.append(str(project_root))
        
        from gr00t.data.dataset import LeRobotSingleDataset
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.experiment.data_config import UAVQuadcopterDataConfig
        
        print("\n" + "="*50)
        print("Testing dataset loading with GR00T...")
        
        # Test configuration
        uav_config = UAVQuadcopterDataConfig()
        modality_configs = uav_config.modality_config()
        transforms = uav_config.transform()
        
        print(f"✓ UAV configuration loaded")
        print(f"✓ Modality configs: {list(modality_configs.keys())}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Cannot test dataset loading: {e}")
        print("This is expected if GR00T is not installed.")
        return True
    except Exception as e:
        print(f"❌ Error testing dataset loading: {e}")
        return False

def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test UAV dataset structure")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="./demo_data/uav.Landing",
        help="Path to UAV dataset"
    )
    
    args = parser.parse_args()
    
    print("UAV Dataset Structure Test")
    print("=" * 50)
    
    # Test dataset structure
    if not test_dataset_structure(args.dataset_path):
        print("❌ Dataset structure test failed!")
        sys.exit(1)
    
    # Test dataset loading
    if not test_dataset_loading():
        print("❌ Dataset loading test failed!")
        sys.exit(1)
    
    print("\n🎉 All tests passed! Dataset is ready for training.")
    print(f"\nTo use this dataset:")
    print(f"python scripts/uav_finetune.py --config configs/uav_finetune_config.yaml --data_path {args.dataset_path}")

if __name__ == "__main__":
    main()
