from typing import List, Dict
import json

# Existing functions (kept for backward compatibility)
def format_example_for_training(input: str,
                                output: str,
                                instructions: str) -> List[Dict]:
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": input},
        {"role": "assistant", "content": output},
    ]
    return messages

def format_example_for_inference(prompt: str,
                                 instructions: str) -> List[Dict]:
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": prompt},
    ]
    return messages

def load_database(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# NEW: For JSONL datasets that already contain role-tagged messages (user/assistant)
def format_jsonl_example_for_training(messages: List[Dict], system_instructions: str) -> List[Dict]:
    """
    Prepend the global system prompt and return messages for training.
    Expects messages like: [{"role":"user","content":...}, {"role":"assistant","content":...}]
    """
    out = [{"role": "system", "content": system_instructions}]
    # Ensure only allowed roles pass through and content is str
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant") and isinstance(content, str):
            out.append({"role": role, "content": content})
    return out
