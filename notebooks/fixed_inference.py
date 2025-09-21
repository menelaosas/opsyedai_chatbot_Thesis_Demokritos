# inference_fixed.py
import sys
import os
sys.path.append(os.path.abspath('..'))

from utils.modeling import load_model_for_inference
import torch

def format_chat_prompt_manual(system_instructions, user_input, tokenizer):
    """
    Manual chat formatting as fallback when apply_chat_template fails
    This follows the standard instruction format for fine-tuned models
    """
    # Check if tokenizer has special tokens
    if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token:
        bos = tokenizer.bos_token
    else:
        bos = ""
    
    if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
        eos = tokenizer.eos_token
    else:
        eos = ""
    
    # Format using common instruction template
    formatted_prompt = f"{bos}[INST] {system_instructions}\n\nUser: {user_input} [/INST]"
    
    return formatted_prompt

def debug_tokenizer_template(tokenizer):
    """
    Debug function to understand tokenizer capabilities
    """
    print("=== TOKENIZER DEBUG INFO ===")
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Has chat_template: {hasattr(tokenizer, 'chat_template')}")
    if hasattr(tokenizer, 'chat_template'):
        print(f"Chat template: {tokenizer.chat_template}")
    print(f"BOS token: {tokenizer.bos_token}")
    print(f"EOS token: {tokenizer.eos_token}")
    print(f"PAD token: {tokenizer.pad_token}")
    print("============================")

# Specify which model you want to test
model_path = "../saved_models/krikri 2025-09-16 18:19:06"  # Your trained model
base_model_id = "ilsp/Meltemi-7B-Instruct-v1.5"

# Load instructions
try:
    with open('../instructions/v1 LLM instructions.txt', 'r', encoding='utf-8') as f:
        system_instructions = f.read()
except FileNotFoundError:
    print("Warning: Instructions file not found. Using default instructions.")
    system_instructions = "Είστε ένας χρήσιμος βοηθός για το σύστημα ΗΚΕΛΥ. Απαντήστε στις ερωτήσεις των χρηστών με σαφήνεια και ακρίβεια."

# Load the model
print(f"Loading model from: {model_path}")
try:
    model_inference, tokenizer_inference = load_model_for_inference(model_path, base_model_id)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Debug tokenizer
debug_tokenizer_template(tokenizer_inference)

# Ensure pad token is set
if tokenizer_inference.pad_token is None:
    if tokenizer_inference.eos_token:
        tokenizer_inference.pad_token = tokenizer_inference.eos_token
        print("Set pad_token to eos_token")
    else:
        tokenizer_inference.add_special_tokens({'pad_token': '[PAD]'})
        print("Added new pad_token")

# Test loop
print("\n🚀 Ready for testing. Type 'έξοδος', 'exit', or 'quit' to quit.")
print("🔍 Type 'debug' to see tokenizer information again.")

while True:
    user_input = input("\nΗ ερώτησή σας: ")
    
    if user_input.lower() in ['έξοδος', 'exit', 'quit']:
        print("Αντίο! 👋")
        break
    
    if user_input.lower() == 'debug':
        debug_tokenizer_template(tokenizer_inference)
        continue
    
    try:
        # Method 1: Try apply_chat_template first
        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_input}
        ]
        
        formatted_prompt = None
        
        # Try apply_chat_template
        try:
            formatted_prompt = tokenizer_inference.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            print("✅ Using apply_chat_template method")
            
            # Debug: Check what we got
            print(f"🔍 Formatted prompt type: {type(formatted_prompt)}")
            if isinstance(formatted_prompt, str):
                print(f"🔍 Prompt length: {len(formatted_prompt)} characters")
            else:
                print(f"⚠️  Unexpected format: {formatted_prompt}")
                formatted_prompt = None
                
        except Exception as e:
            print(f"⚠️  apply_chat_template failed: {e}")
            formatted_prompt = None
        
        # Method 2: Fallback to manual formatting
        if formatted_prompt is None or not isinstance(formatted_prompt, str):
            print("🔄 Using manual formatting fallback")
            formatted_prompt = format_chat_prompt_manual(
                system_instructions, 
                user_input, 
                tokenizer_inference
            )
        
        # Ensure we have a valid string
        if not isinstance(formatted_prompt, str):
            print(f"❌ Critical error: formatted_prompt is {type(formatted_prompt)}, expected str")
            continue
            
        # Debug: Show first 200 chars of prompt
        print(f"📝 Prompt preview: {formatted_prompt[:200]}...")
        
        # Generate response
        device = next(model_inference.parameters()).device
        print(f"🔧 Using device: {device}")
        
        with torch.no_grad():
            # Tokenize input
            try:
                inputs = tokenizer_inference(
                    formatted_prompt,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                
                print("✅ Tokenization successful")
                print(f"🔍 Input shape: {inputs['input_ids'].shape}")
                
            except Exception as e:
                print(f"❌ Tokenization error: {e}")
                print(f"🔍 Prompt type: {type(formatted_prompt)}")
                print(f"🔍 Prompt content: {repr(formatted_prompt[:100])}")
                continue
            
            # Generate
            try:
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
                
                print("✅ Generation successful")
                
                # Decode response
                generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
                response = tokenizer_inference.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                print(f"\n🤖 Απάντηση: {response.strip()}")
                
            except Exception as e:
                print(f"❌ Generation error: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        continue