from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "unsloth/Qwen2.5-1.5B-Instruct" #"unsloth/gemma-3-1b-it"
    max_seq_length: int = 1600
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    random_state: int = 3407


@dataclass(frozen=True)
class PromptConfig:
    reasoning_start: str = "<start_working_out>"
    reasoning_end: str = "<end_working_out>"
    solution_start: str = "<SOLUTION>"
    solution_end: str = "</SOLUTION>"

    @property
    def system_prompt(self) -> str:
        return (
            "You are given a problem.\n"
            "Think about the problem and provide your working out.\n"
            f"Place it between {self.reasoning_start} and {self.reasoning_end}.\n"
            f"Then, provide your solution between {self.solution_start}{self.solution_end}"
        )


@dataclass(frozen=True)
class TrainingConfig:
    max_prompt_length: int = 256
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.001
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch_fused"
    logging_steps: int = 1
    per_device_train_batch_size: int = 6 #1
    gradient_accumulation_steps: int = 1
    num_generations: int = 6 #4
    max_steps: int = 400
    save_steps: int = 400
    max_grad_norm: float = 0.1
    report_to: str = "wandb"
    output_dir: str = "outputs"
