from trl import GRPOConfig, GRPOTrainer

from src.config import ModelConfig, TrainingConfig


def build_training_args(model_config: ModelConfig, training_config: TrainingConfig):
    return GRPOConfig(
        learning_rate=training_config.learning_rate,
        adam_beta1=training_config.adam_beta1,
        adam_beta2=training_config.adam_beta2,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        lr_scheduler_type=training_config.lr_scheduler_type,
        optim=training_config.optim,
        logging_steps=training_config.logging_steps,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        num_generations=training_config.num_generations,
        max_prompt_length=training_config.max_prompt_length,
        max_completion_length=model_config.max_seq_length - training_config.max_prompt_length,
        max_steps=training_config.max_steps,
        save_steps=training_config.save_steps,
        max_grad_norm=training_config.max_grad_norm,
        report_to=training_config.report_to,
        output_dir=training_config.output_dir,
        use_vllm=True,
        log_completions=True,
        # vllm_kwargs={
        #     "enforce_eager": True,          # Disables CUDA graphs to save ~1.5GB VRAM
        #     # "gpu_memory_utilization": 0.3, # Adjust to share VRAM with the trainer
        # },
    )

def build_trainer(model, tokenizer, reward_funcs, dataset, training_args):
    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )
