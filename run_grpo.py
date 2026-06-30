import unsloth

from src.config import ModelConfig, PromptConfig, TrainingConfig
from src.data import build_dataset
from src.modeling import attach_lora_adapters, load_model_and_tokenizer
from src.rewards import build_reward_functions
from src.save import (
    push_gguf_if_enabled,
    push_merged_if_enabled,
    save_gguf_if_enabled,
    save_lora,
    save_merged_if_enabled,
)
from src.train import build_trainer, build_training_args
from src.inference import run_inference


def main():
    model_config = ModelConfig()
    prompt_config = PromptConfig()
    training_config = TrainingConfig()

    model, tokenizer = load_model_and_tokenizer(model_config)
    model = attach_lora_adapters(model, model_config)

    dataset = build_dataset(prompt_config)
    reward_funcs = build_reward_functions(prompt_config)
    training_args = build_training_args(model_config, training_config)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_funcs,
        dataset=dataset,
        training_args=training_args,
    )
    trainer.train()

    run_inference(
        model=model,
        tokenizer=tokenizer,
        system_prompt=prompt_config.system_prompt,
        question="What is the sqrt of 101?",
    )

    save_lora(model, tokenizer)
    save_merged_if_enabled(model, tokenizer, enabled=False)
    push_merged_if_enabled(model, tokenizer, enabled=False)
    save_gguf_if_enabled(model, tokenizer, enabled=False)
    push_gguf_if_enabled(model, tokenizer, enabled=False)


if __name__ == "__main__":
    main()
