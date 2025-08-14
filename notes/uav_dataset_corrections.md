# UAV Dataset Generation - Corrected Structure

## Summary of Changes

The `generate_uav_dataset.py` has been corrected to properly implement the LeRobot dataset format with the correct understanding of **episodes** vs **chunks**.

## Key Corrections

### 1. **Episodes vs Chunks Understanding**

**Before (Incorrect):**

- Each episode was saved in its own chunk directory
- `chunk-000/episode_000000.parquet`
- `chunk-001/episode_000001.parquet`
- etc.

**After (Correct):**

- Multiple episodes are grouped into chunks
- `chunk-000/` contains episodes 0-999
- `chunk-001/` contains episodes 1000-1999
- etc.

### 2. **Chunk Organization**

```
data/
├── chunk-000/
│   ├── episode_000000.parquet
│   ├── episode_000001.parquet
│   ├── ...
│   └── episode_000999.parquet
├── chunk-001/
│   ├── episode_001000.parquet
│   ├── episode_001001.parquet
│   ├── ...
│   └── episode_001999.parquet
└── ...
```

### 3. **Video Organization**

```
videos/
├── chunk-000/
│   ├── observation.images.front_camera/
│   │   ├── episode_000000.mp4
│   │   ├── episode_000001.mp4
│   │   └── ...
│   └── observation.images.gimbal_camera/
│       ├── episode_000000.mp4
│       ├── episode_000001.mp4
│       └── ...
└── chunk-001/
    ├── observation.images.front_camera/
    └── observation.images.gimbal_camera/
```

## Usage Examples

### Generate Small Dataset (Default)

```bash
python scripts/generate_uav_dataset.py --num_episodes 5
```

### Generate Dataset with Custom Chunk Size

```bash
python scripts/generate_uav_dataset.py --num_episodes 2500 --chunk_size 1000
```

This creates:

- `chunk-000`: episodes 0-999
- `chunk-001`: episodes 1000-1999
- `chunk-002`: episodes 2000-2499

### Generate Dataset for Quick Testing

```bash
python scripts/generate_uav_dataset.py \
    --output_dir ./test_data/uav.Landing \
    --num_episodes 10 \
    --chunk_size 5 \
    --episode_length 50
```

## Configuration Integration

The corrected dataset works seamlessly with the new configuration system:

```bash
# Generate data
python scripts/generate_uav_dataset.py --num_episodes 100

# Train with config
python scripts/uav_finetune.py --config configs/uav_quick_experiment.yaml \
    --override data.data_path=./demo_data/uav.Landing
```

## Validation

Test the dataset structure:

```bash
python scripts/test_uav_dataset.py --dataset_path ./demo_data/uav.Landing
```

## Benefits of Corrected Structure

1. **Memory Efficiency**: Large datasets can be processed in chunks
2. **Parallel Loading**: Multiple chunks can be loaded in parallel
3. **Scalability**: Easy to add more episodes without restructuring
4. **Standard Compliance**: Follows LeRobot dataset conventions
5. **GR00T Compatibility**: Works correctly with GR00T's data loading pipeline

## Action Chunking Context

The corrected structure supports GR00T's action chunking mechanism:

- **Episode**: Complete demonstration sequence (e.g., entire landing maneuver)
- **Data Chunk**: Storage unit containing multiple episodes
- **Action Chunk**: Model processing unit (16 actions processed together by DiT)

These are three different concepts:

- **Data chunks**: For efficient storage and loading
- **Action chunks**: For model inference (H=16 actions)
- **Episodes**: For complete demonstrations

## OpenCV Dependency

The script now handles missing OpenCV gracefully:

- Videos are skipped if OpenCV is unavailable
- Core dataset generation continues to work
- Fallback image processing using NumPy/SciPy

Install OpenCV for full functionality:

```bash
pip install opencv-python
```

## Next Steps

1. Generate a test dataset: `python scripts/generate_uav_dataset.py --num_episodes 10`
2. Validate structure: `python scripts/test_uav_dataset.py`
3. Train model: `python scripts/uav_finetune.py --config configs/uav_quick_experiment.yaml`

The corrected dataset generation now properly supports the LeRobot format and GR00T's training pipeline!
