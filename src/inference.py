from transformers import TextStreamer


def run_inference(model, tokenizer, system_prompt: str, question: str, max_new_tokens: int = 64):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    return model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )
