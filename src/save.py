def save_lora(model, tokenizer, output_dir: str = "gemma_3_lora"):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_merged_if_enabled(model, tokenizer, output_dir: str = "gemma-3-finetune", enabled: bool = False):
    if enabled:
        model.save_pretrained_merged(output_dir, tokenizer)


def push_merged_if_enabled(
    model,
    tokenizer,
    repo_id: str = "HF_ACCOUNT/gemma-3-finetune",
    token: str = "YOUR_HF_TOKEN",
    enabled: bool = False,
):
    if enabled:
        model.push_to_hub_merged(repo_id, tokenizer, token=token)


def save_gguf_if_enabled(
    model,
    tokenizer,
    output_dir: str = "gemma_3_finetune",
    quantization_method: str = "Q8_0",
    enabled: bool = False,
):
    if enabled:
        model.save_pretrained_gguf(
            output_dir,
            tokenizer,
            quantization_method=quantization_method,
        )


def push_gguf_if_enabled(
    model,
    tokenizer,
    repo_id: str = "HF_ACCOUNT/gemma_3_finetune",
    quantization_method: str = "Q8_0",
    token: str = "YOUR_HF_TOKEN",
    enabled: bool = False,
):
    if enabled:
        model.push_to_hub_gguf(
            repo_id,
            tokenizer,
            quantization_method=quantization_method,
            token=token,
        )
