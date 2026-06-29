import re

from src.config import PromptConfig


def build_reward_functions(prompt_config: PromptConfig):
    match_format = re.compile(
        rf"^[\s]{{0,}}"
        rf"{prompt_config.reasoning_start}.+?{prompt_config.reasoning_end}.*?"
        rf"{prompt_config.solution_start}(.+?){prompt_config.solution_end}"
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL,
    )
    match_numbers = re.compile(
        rf"{prompt_config.solution_start}.*?([\d\.]{{1,}})",
        flags=re.MULTILINE | re.DOTALL,
    )

    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            scores.append(3.0 if match_format.search(response) is not None else 0.0)
        return scores

    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            score = 0.0
            score += 0.5 if response.count(prompt_config.reasoning_start) == 1 else -0.5
            score += 0.5 if response.count(prompt_config.reasoning_end) == 1 else -0.5
            score += 0.5 if response.count(prompt_config.solution_start) == 1 else -0.5
            score += 0.5 if response.count(prompt_config.solution_end) == 1 else -0.5
            scores.append(score)
        return scores

    def check_answer(prompts, completions, answer, **kwargs):
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [
            guess.group(1) if (guess := match_format.search(response)) is not None else None
            for response in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(0.0)
                continue

            score = 0.0
            if guess == true_answer:
                score += 3.0
            elif guess.strip() == true_answer.strip():
                score += 1.5
            else:
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        score += 0.5
                    elif 0.8 <= ratio <= 1.2:
                        score += 0.25
                    else:
                        score -= 1.0
                except Exception:
                    score -= 0.5
            scores.append(score)
        return scores

    def check_numbers(prompts, completions, answer, **kwargs):
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [
            guess.group(1) if (guess := match_numbers.search(response)) is not None else None
            for response in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(0.0)
                continue
            try:
                true_value = float(true_answer.strip())
                guess_value = float(guess.strip())
                scores.append(1.5 if guess_value == true_value else 0.0)
            except Exception:
                scores.append(0.0)
        return scores

    return [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ]
