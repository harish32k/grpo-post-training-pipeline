from unsloth import FastModel

from src.config import ModelConfig


def load_model_and_tokenizer(config: ModelConfig):
    model, tokenizer = FastModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            full_finetuning=config.full_finetuning,
            fast_inference=True, # added later
            )
    return model, tokenizer


def attach_lora_adapters(model, config: ModelConfig):
    return FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            random_state=config.random_state,
            )
