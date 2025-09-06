# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Deep Reinforcement Learning (DRL) project that implements Semi-MDP based learning for NPCA (Non-Primary Channel Access) decision-making in wireless STA (Station) systems. STAs learn to choose between staying on the Primary channel or switching to NPCA channels when OBSS (Overlapping BSS) occupies the Primary channel.

## Architecture

### Core Framework (`drl_framework/`)
- **`train.py`**: Contains `SemiMDPLearner` class implementing DQN-based Semi-MDP learning algorithm
- **`network.py`**: DQN neural network architecture and ReplayMemory implementation
- **`random_access.py`**: Core simulation classes - `STA`, `Channel`, and `Simulator`
- **`params.py`**: Hyperparameters (batch size, learning rate, epsilon decay, etc.)
- **`configs.py`**: Simulation configuration settings
- **`custom_env.py`**: Legacy environment implementation (still referenced by test files)

### Main Execution Files
- **`main_semi_mdp_training.py`**: Primary training script for Semi-MDP learning
- **`main_npca_simulation.py`**: Simulation without learning (existing behavior)
- **`test_model.py`**: Model testing and evaluation with different simulation modes

### Environment
- **`npca_semi_mdp_env.py`**: Gymnasium-based Semi-MDP environment for NPCA decisions

## Common Commands

### Training and Simulation
```bash
# Run Semi-MDP training (primary training script)
python main_semi_mdp_training.py

# Run basic simulation without learning
python main_npca_simulation.py

# Run tutorial/demonstration script
python tutorial.py
```

### Testing and Evaluation
```bash
# Test trained model with different simulation modes (requires policy_model.pt)
python test_model.py
```

### Dependencies
```bash
# Install required packages
pip install torch pandas matplotlib gymnasium
```

### Debugging
```bash
# Debug with VS Code (launch.json configured for current file debugging)
# F5 or use "Python: Current File" configuration
```

## Key Concepts

### Semi-MDP Structure
- **States**: Primary channel OBSS occupation time, radio transition time, transmission duration, contention window index
- **Actions**: 0 = StayPrimary, 1 = GoNPCA
- **Rewards**: Positive reward for successful PPDU transmission length, zero for failures
- **Options**: Decision points occur when Primary channel has OBSS occupation

### Model Artifacts and Outputs
- Training results saved to `./semi_mdp_results/`
- Model checkpoint: `semi_mdp_model.pth`
- Training plots: `training_results.png`
- Test results: `*_test_log.csv`, `*_test_rewards.png` (from test_model.py)

## Development Notes

### Core Implementation Details
- **Random seed fixed**: Set to 42 in `random_access.py` for reproducible experiments
- **Multi-mode testing**: `test_model.py` supports three modes: `drl`, `offload_only`, `local_only`
- **Device compatibility**: Automatic selection of CUDA/MPS/CPU with appropriate fallback
- **Hyperparameters**: Centralized in `drl_framework/params.py` with standard DQN settings

### Simulation Configuration
- **Time units**: Slot duration = 9μs (802.11ax standard)
- **Frame sizes**: Configurable via `configs.py` (short: 33 slots, long: 165 slots)
- **Channel setup**: Primary (ID=0, no OBSS) + Secondary (ID=1, OBSS enabled)
- **Training monitoring**: Progress logged every 10 episodes

### Code Organization Notes
- Korean comments for wireless domain terminology
- Legacy `custom_env.py` still referenced by test files
- VS Code debugging configured for current file execution
- Comprehensive `.gitignore` covers typical Python/ML artifacts