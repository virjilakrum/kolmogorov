# Kolmogorov Project State File

## What We're Building
An RLHF (Reinforcement Learning from Human Feedback) system that enables LLMs to learn from real-world human preference data using DPO (Direct Preference Optimization).

## Core Architecture (4 Phases)
1. **Phase 1**: Preference Data Collection Infrastructure
2. **Phase 2**: Reward Model Training (Bradley-Terry)
3. **Phase 3**: Policy Optimization via DPO
4. **Phase 4**: Parameter-Efficient Fine-Tuning (LoRA)

## Technical Stack
- **Base Models**: Mistral-7B, Llama-2, Qwen (7B-13B range)
- **Framework**: TRL (Transformers RL) by HuggingFace
- **PEFT**: LoRA for memory-efficient training
- **Compute**: A100 40GB or 2× RTX 4090 (4-bit quantization)
- **Storage**: PostgreSQL/MongoDB for preference data

## Key Findings from Documentation

### TRL DPO Implementation
- Use `DPOTrainer` with `DPOConfig`
- Dataset format: `{"prompt": "...", "chosen": "...", "rejected": "..."}`
- Key params: `beta=0.1`, `loss_type="sigmoid"`, `max_length=1024`
- Reference model auto-created as frozen copy if not provided

### Reward Model Training
- Use `RewardTrainer` with `RewardConfig`
- Dataset format: `{"chosen": [...], "rejected": [...]}`
- Adds classification head to base model

### SFT (Supervised Fine-Tuning)
- Use `SFTTrainer` with `SFTConfig`
- Dataset format: `{"messages": [{"role": "user", "content": "..."}, ...]}`

### LoRA Configuration
- Typical: `r=16-64`, `alpha=32`, `target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]`
- Use `get_peft_model()` to wrap base model

## Current Status
- [x] Project specification complete
- [x] Documentation gathered from Context7
- [x] Project structure setup
- [x] Phase 1: Data collection infrastructure (PreferenceCollector, PreferenceDataset)
- [x] Phase 2: Reward model training (RewardTrainerWrapper)
- [x] Phase 3: DPO training (DPOTrainerWrapper)
- [x] Phase 4: LoRA fine-tuning (integrated in all trainers)
- [x] Evaluation pipeline (Evaluator, metrics)

## Files Created

### Core Modules
- `src/kolmogorov/` - Main package
  - `data/` - PreferenceDataset, PreferenceCollector, formatters
  - `models/` - Model loader, LoRA config
  - `trainers/` - DPO, Reward, SFT trainer wrappers
  - `evaluation/` - Evaluator, metrics (win-rate, KL-div, etc.)
  - `utils/` - Config loader, logging

### Configuration
- `configs/base_config.yaml` - Base training settings
- `configs/dpo_config.yaml` - DPO-specific settings
- `configs/reward_config.yaml` - Reward model settings
- `configs/sft_config.yaml` - SFT settings

### Scripts
- `scripts/train_dpo.py` - DPO training script
- `scripts/train_reward.py` - Reward model training script
- `scripts/train_sft.py` - SFT training script

## Next Steps
1. Install dependencies and test the implementation
2. Run a small test training to verify everything works
3. Consider adding API for preference collection
4. Add more comprehensive tests

