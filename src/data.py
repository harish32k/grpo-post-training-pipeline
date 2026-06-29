from datasets import load_dataset

from src.config import PromptConfig


def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def build_dataset(prompt_config: PromptConfig, split: str = "train"):
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    system_prompt = prompt_config.system_prompt
    return dataset.map(
        lambda row: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["question"]},
            ],
            "answer": extract_hash_answer(row["answer"]),
        }
    )
