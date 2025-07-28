# UAV-Focused Refactoring Plan

## 🎯 Objective

Streamline the isaac-skypilot codebase to focus exclusively on UAV applications, removing redundant components and simplifying maintenance.

## 📋 Current Analysis

### Core UAV Components (Keep & Optimize)

- ✅ `gr00t/data/embodiment_tags.py` - Only UAV embodiment
- ✅ `gr00t/experiment/data_config.py` - Only UAV config
- ✅ `scripts/uav_finetune.py` - Primary training script
- ✅ `getting_started/examples/eval_uav_quadcopter.py` - UAV evaluation
- ✅ Core GR00T model components (backbone, action head)

### Redundant Components (Remove)

- ❌ All humanoid robot configs (GR1, Unitree G1, etc.)
- ❌ Manipulation robot configs (Panda, SO100, etc.)
- ❌ OxE Droid configurations
- ❌ Getting started notebooks for other embodiments
- ❌ Example configs for non-UAV robots

### New Simplified Structure

```
uav-skypilot/
├── uav_gr00t/                    # Renamed from gr00t
│   ├── data/
│   │   ├── dataset.py           # UAV dataset handling
│   │   ├── embodiment.py        # Only UAV embodiment
│   │   └── transforms/          # UAV-specific transforms
│   ├── model/
│   │   ├── uav_policy.py        # UAV-specific policy
│   │   ├── gr00t_core.py        # Core GR00T model
│   │   └── action_head/         # UAV action head
│   └── training/
│       ├── trainer.py           # UAV training logic
│       └── config.py            # UAV data configuration
├── scripts/
│   ├── train_uav.py             # Main training script
│   ├── eval_uav.py              # Evaluation script
│   └── generate_uav_data.py     # Data generation
├── examples/
│   ├── basic_inference.py       # Simple UAV inference
│   ├── real_uav_interface.py    # Hardware integration
│   └── simulation_test.py       # Simulation testing
├── docs/
│   ├── README.md                # UAV-focused documentation
│   ├── quickstart.md            # Getting started guide
│   └── api_reference.md         # API documentation
└── tests/
    ├── test_uav_model.py
    └── test_data_pipeline.py
```

## 🔧 Refactoring Steps

### Phase 1: Core Simplification

1. Create simplified embodiment system (UAV only)
2. Extract UAV-specific data configuration
3. Create streamlined training pipeline
4. Remove all non-UAV components

### Phase 2: Code Optimization

1. Simplify data transforms for UAV sensors
2. Optimize action space handling
3. Create UAV-specific policy interface
4. Add proper error handling and logging

### Phase 3: Documentation & Examples

1. Create clear UAV-focused documentation
2. Add practical examples for real UAV integration
3. Include troubleshooting guides
4. Add performance optimization tips

## 🚀 Benefits

- **Simpler Maintenance**: Only UAV-relevant code
- **Easier Understanding**: Clear purpose and structure
- **Faster Development**: No need to navigate irrelevant code
- **Better Performance**: Optimized for UAV use cases
- **Cleaner Dependencies**: Remove unused libraries
