# test_specific_model.py
import sys
import os
sys.path.append(os.path.abspath('..'))

from utils.modeling import load_model_for_inference
import torch

# Specify which model you want to test
model_path = "../saved_models/krikri 2025-09-16 18:19:06"  # Choose your model
base_model_id = "ilsp/Llama-Krikri-8B-Instruct"

# Load instructions
with open('../instructions/v1 LLM instructions.txt', 'r', encoding='utf-8') as f:
    system_instructions = f.read()

# Load the model
print(f"Loading model from: {model_path}")
model_inference, tokenizer_inference = load_model_for_inference(model_path, base_model_id)

# Test loop
print("Ready for testing. Type 'exit' to quit.")
while True:
    user_input = input("\nΗ ερώτησή σας: ")
    if user_input.lower() in ['έξοδος', 'exit', 'quit']:
        break
    
    # Format messages
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_input}
    ]
    
    # Apply chat template
    formatted_prompt = tokenizer_inference.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Generate response
    device = next(model_inference.parameters()).device
    
    with torch.no_grad():
        inputs = tokenizer_inference(
            formatted_prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        outputs = model_inference.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer_inference.pad_token_id,
            eos_token_id=tokenizer_inference.eos_token_id,
            repetition_penalty=1.1,
        )
        
        # Get only the generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        response = tokenizer_inference.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        print("\n" + response.strip(), flush=True)