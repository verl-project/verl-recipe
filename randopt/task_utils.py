"""Task helpers for the RandOpt recipe."""

import json
import re
from typing import Any

DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
COUNTDOWN_USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "Return the final answer in <answer> </answer> tags, for example "
    "<answer> (1 + 2) / 3 </answer>."
)
COUNTDOWN_RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


def load_data(file_path: str | list[str]) -> list[dict[str, Any]]:
    """Load JSON or parquet records."""
    if isinstance(file_path, list):
        file_path = file_path[0]
    if file_path.endswith(".parquet"):
        import pandas as pd

        return pd.read_parquet(file_path).to_dict("records")
    with open(file_path) as f:
        return json.load(f)


def _apply_chat_template(messages: list[dict[str, str]], tokenizer) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    rendered = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "system":
            rendered.append(f"{content}\n\n")
        elif role == "user":
            rendered.append(f"### Input:\n{content}\n\n### Output:\n")
        else:
            rendered.append(content)
    return "".join(rendered)


def _to_tokens_prompt(text: str, tokenizer):
    from vllm import TokensPrompt

    tokenized = tokenizer(text)
    return TokensPrompt(prompt_token_ids=tokenized["input_ids"])


def create_prompt_processor(config: dict[str, Any]):
    """Create a prompt processor for countdown, verl parquet, or custom data."""
    task_type = config.get("task_type", "countdown")
    if task_type == "custom":
        prompt_processor_path = config.get("prompt_processor_path")
        prompt_processor_name = config.get("prompt_processor_name")
        if not prompt_processor_path or not prompt_processor_name:
            return None
        from verl.utils.import_utils import load_extern_object

        return load_extern_object(prompt_processor_path, prompt_processor_name)

    if task_type == "parquet_prompt":

        def process_parquet_prompt(task_data: dict[str, Any], tokenizer):
            prompt = task_data.get("prompt", task_data.get("context"))
            if isinstance(prompt, str):
                return _to_tokens_prompt(prompt, tokenizer)
            messages = [dict(message) for message in list(prompt)]
            return _to_tokens_prompt(_apply_chat_template(messages, tokenizer), tokenizer)

        return process_parquet_prompt

    if task_type != "countdown":
        raise ValueError("RandOpt recipe currently supports task_type=countdown, parquet_prompt, or custom.")

    system_message = config.get("system_message") or DEFAULT_SYSTEM_MESSAGE
    user_template = config.get("user_template") or COUNTDOWN_USER_TEMPLATE

    def process_countdown(task_data: dict[str, Any], tokenizer):
        numbers, target = get_countdown_ground_truth(task_data)
        user_content = user_template.format(numbers=numbers, target=target)
        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_content}]
        return _to_tokens_prompt(_apply_chat_template(messages, tokenizer), tokenizer)

    return process_countdown


def create_reward_fn(config: dict[str, Any]):
    """Create a reward function for countdown or a user-provided custom task."""
    task_type = config.get("task_type", "countdown")
    if task_type == "custom":
        reward_fn_path = config.get("reward_fn_path")
        reward_fn_name = config.get("reward_fn_name")
        if not reward_fn_path or not reward_fn_name:
            raise ValueError("custom task_type requires reward_fn_path and reward_fn_name.")
        from verl.utils.import_utils import load_extern_object

        return load_extern_object(reward_fn_path, reward_fn_name)
    if task_type == "parquet_prompt":
        raise ValueError("parquet_prompt requires a custom reward_fn_path/reward_fn_name.")
    if task_type != "countdown":
        raise ValueError("RandOpt recipe currently supports task_type=countdown or custom.")
    return countdown_reward


def create_vote_fns(config: dict[str, Any]):
    """Create answer extraction and correctness functions for ensemble voting."""
    task_type = config.get("task_type", "countdown")
    if task_type == "countdown":
        return countdown_vote_answer, countdown_is_voted_answer_correct
    return None, None


def get_countdown_ground_truth(task_data: dict[str, Any]) -> tuple[list[int], int]:
    if "reward_model" in task_data and "ground_truth" in task_data["reward_model"]:
        ground_truth = task_data["reward_model"]["ground_truth"]
        if isinstance(ground_truth, str):
            ground_truth = json.loads(ground_truth)
        return list(ground_truth["numbers"]), int(ground_truth["target"])
    return list(task_data["numbers"]), int(task_data["target"])


def countdown_reward(
    response: str,
    task_data: dict[str, Any],
) -> dict[str, Any]:
    numbers, target = get_countdown_ground_truth(task_data)
    full_response = f"<think>{response}"
    format_reward = _countdown_format_reward(full_response)
    answer_reward, answer_info = _countdown_answer_reward(full_response, numbers, target)
    return {
        "reward": 0.1 * format_reward + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
            **answer_info,
        },
    }


def _countdown_format_reward(response: str) -> float:
    think_match = re.search(r"<think>.*?</think>", response, re.DOTALL)
    answer_match = re.search(r"<answer>.*?</answer>", response, re.DOTALL)
    full_match = re.match(r"^<think>.*?</think>\n<answer>.*?</answer>$", response, re.DOTALL)
    if full_match:
        return 1.0
    return (0.1 if think_match else 0.0) + (0.5 if answer_match else 0.0)


def _countdown_answer_reward(response: str, numbers: list[int], target: int) -> tuple[float, dict[str, Any]]:
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not matches:
        return 0.0, {"reject_reason": "no_answer", "pred_answer": ""}
    expression = matches[-1].strip()
    if not re.match(r"^[0-9+\-*/() ]+$", expression):
        return 0.0, {"reject_reason": "invalid_chars", "pred_answer": expression}
    used_numbers = [int(number) for number in re.findall(r"\d+", expression)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0, {
            "reject_reason": "wrong_numbers",
            "pred_answer": expression,
            "used_numbers": used_numbers,
            "expected_numbers": numbers,
        }
    try:
        result = eval(expression, {"__builtins__": None}, {})
    except Exception:
        return 0.0, {"reject_reason": "eval_error", "pred_answer": expression}
    if abs(float(result) - float(target)) < 1e-5:
        return 1.0, {"reject_reason": None, "pred_answer": expression, "result": float(result)}
    return 0.0, {
        "reject_reason": "wrong_result",
        "pred_answer": expression,
        "result": float(result),
        "target": target,
    }


def countdown_vote_answer(response: str, task_data: dict[str, Any]) -> tuple[str, bool, dict[str, Any] | None]:
    """Extract a numeric answer for majority voting.

    Countdown votes on the evaluated numeric result of valid formulas, matching
    the original RandOpt implementation.
    """
    numbers, _ = get_countdown_ground_truth(task_data)
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not matches:
        return "", False, {"reject_reason": "no_answer", "pred_answer": ""}

    expression = matches[-1].strip()
    if not re.match(r"^[0-9+\-*/() ]+$", expression):
        return "", False, {"reject_reason": "invalid_chars", "pred_answer": expression}

    used_numbers = [int(number) for number in re.findall(r"\d+", expression)]
    if sorted(used_numbers) != sorted(numbers):
        return (
            "",
            False,
            {
                "reject_reason": "wrong_numbers",
                "pred_answer": expression,
                "used_numbers": used_numbers,
                "expected_numbers": numbers,
            },
        )

    try:
        result = eval(expression, {"__builtins__": None}, {})
    except (SyntaxError, ZeroDivisionError, TypeError, ValueError, OverflowError) as exc:
        return "", False, {"reject_reason": "eval_error", "pred_answer": expression, "error": str(exc)}

    if abs(float(result) - round(float(result))) < 1e-9:
        return str(int(round(float(result)))), True, None
    return str(float(result)), True, None


def countdown_is_voted_answer_correct(voted_answer: str, task_data: dict[str, Any]) -> bool:
    if not voted_answer:
        return False
    _, target = get_countdown_ground_truth(task_data)
    try:
        return abs(float(voted_answer) - float(target)) < 1e-5
    except ValueError:
        return False
