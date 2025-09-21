from typing import List, Dict
import json


# formats the texts into a dict that the tokenizer.apply_chat_template undesrtands
def format_example_for_training(input: str,
                                output: str,
                                instructions: str) -> List[Dict]:
    
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": input},
        {"role": "assistant", "content": output},
    ]

    return messages


# formats the texts into a dict that the tokenizer.apply_chat_template undesrtands
def format_example_for_inference(prompt: str,
                                 instructions: str) -> List[Dict]:
    
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": prompt},
    ]

    return messages


# Load database
def load_database(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)