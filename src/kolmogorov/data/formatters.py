"""Data formatting utilities for different training pipelines."""

from typing import Any


def format_for_dpo(example: dict[str, Any]) -> dict[str, Any]:
    """Format an example for DPO training.
    
    Expected input formats:
    1. {"prompt": "...", "chosen": "...", "rejected": "..."}
    2. {"chosen": [...messages...], "rejected": [...messages...]}
    """
    if "prompt" in example:
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }
    
    # Conversational format without explicit prompt
    return {
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def format_for_reward_model(example: dict[str, Any]) -> dict[str, Any]:
    """Format an example for reward model training.
    
    Expected input: {"chosen": "...", "rejected": "..."}
    Optionally with "prompt" for explicit prompt format.
    """
    result = {
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }
    
    if "prompt" in example:
        result["prompt"] = example["prompt"]
    
    return result


def format_for_sft(example: dict[str, Any]) -> dict[str, Any]:
    """Format an example for SFT training.
    
    Converts various formats to {"messages": [...]} format.
    """
    if "messages" in example:
        return {"messages": example["messages"]}
    
    if "text" in example:
        return {"text": example["text"]}
    
    # Convert prompt/response format to messages
    if "prompt" in example and "response" in example:
        return {
            "messages": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["response"]},
            ]
        }
    
    # If we have chosen/rejected, use chosen for SFT
    if "prompt" in example and "chosen" in example:
        chosen = example["chosen"]
        if isinstance(chosen, str):
            return {
                "messages": [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": chosen},
                ]
            }
        return {"messages": example.get("prompt", []) + chosen}
    
    raise ValueError(f"Cannot format example for SFT: {example.keys()}")


def convert_winner_format(example: dict[str, Any]) -> dict[str, Any]:
    """Convert 'winner' format to chosen/rejected format.
    
    For datasets like arena-human-preference with winner_model_a/winner_model_b columns.
    """
    if example.get("winner_model_a", 0) == 1:
        chosen = example["response_a"]
        rejected = example["response_b"]
    elif example.get("winner_model_b", 0) == 1:
        chosen = example["response_b"]
        rejected = example["response_a"]
    else:
        # Tie - skip or use response_a as chosen
        chosen = example.get("response_a", "")
        rejected = example.get("response_b", "")
    
    result = {"chosen": chosen, "rejected": rejected}
    
    if "prompt" in example:
        result["prompt"] = example["prompt"]
    
    return result


def prepare_conversation(
    messages: list[dict[str, str]],
    tokenizer,
    add_generation_prompt: bool = False,
) -> str:
    """Format conversation messages using tokenizer's chat template."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    
    # Fallback for tokenizers without chat template
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|{role}|>\n{content}\n"
    
    return formatted

