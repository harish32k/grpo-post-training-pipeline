GRPO Unsloth Example

This repo is a lightweight Python split of a GRPO training notebook using Unsloth, TRL, and the GSM8K dataset. The code loads a Gemma 3 instruct model, adds LoRA adapters, builds reward functions for reasoning-format supervision, trains with `GRPOTrainer`, runs a quick inference example, and saves LoRA weights.

Project Layout

- `run_grpo.py`: entrypoint for the end-to-end workflow
- `src/config.py`: model, prompt, and training configuration dataclasses
- `src/modeling.py`: model loading and LoRA attachment
- `src/data.py`: GSM8K loading and prompt/answer preprocessing
- `src/rewards.py`: GRPO reward functions
- `src/train.py`: GRPO config and trainer construction
- `src/inference.py`: sample generation after training
- `src/save.py`: local save and optional export helpers
- `backup/`: notebook-derived reference files kept for comparison

What The Script Does

1. Imports `unsloth` first so Unsloth patches are applied before `transformers`.
2. Loads `unsloth/gemma-3-1b-it`.
3. Applies LoRA adapters for parameter-efficient fine-tuning.
4. Loads the GSM8K training split from `openai/gsm8k`.
5. Converts each row into a chat prompt with a structured reasoning/solution format.
6. Builds reward functions that score:
   - exact format matches
   - approximate format matches
   - extracted answer correctness
   - numeric answer correctness
7. Trains with TRL's `GRPOTrainer`.
8. Runs one inference example.
9. Saves LoRA adapters locally.

Requirements

You need a Python environment with the relevant training dependencies installed separately. The split code does not include the notebook installation cells. At minimum, the script expects imports from:

- `unsloth`
- `trl`
- `transformers`
- `datasets`
- `torch`

How To Run

Run the main script from the repo root:

```bash
python run_grpo.py
```

Configuration

The main settings live in `src/config.py`.

- `ModelConfig` controls model name, sequence length, quantization flags, and LoRA settings.
- `PromptConfig` defines the reasoning and solution tags used in prompts and reward matching.
- `TrainingConfig` controls GRPO hyperparameters such as:
  - `max_steps`: total number of training update steps
  - `save_steps`: checkpoint save frequency
  - `logging_steps`: metric logging frequency
  - `num_generations`: number of sampled completions per prompt during GRPO

Current Defaults

- model: `unsloth/gemma-3-1b-it`
- max sequence length: `1024`
- max prompt length: `256`
- max training steps: `50`
- save frequency: every `50` steps
- reporting backend: `"none"`

Outputs

By default the script:

- writes trainer outputs/checkpoints under `outputs/`
- saves LoRA adapter weights to `gemma_3_lora/`

The helper functions in `src/save.py` also include optional merged-model and GGUF export paths, but those are disabled by default.

Notes

- The current code logs training metrics through the trainer, but it does not create reward plots.
- The original notebook included extra environment/setup cells in `backup/`; not all notebook runtime flags were copied into the split Python workflow.
- If you want behavior closer to the original notebook, review the files in `backup/` and selectively carry over any needed runtime configuration.
