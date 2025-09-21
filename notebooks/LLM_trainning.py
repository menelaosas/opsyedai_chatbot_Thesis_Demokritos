# ============================================
# CELL 1: Initialization
# ============================================
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["WANDB_DISABLED"] = "true"
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datetime import datetime
import requests
import sys

# Add root folder into paths
sys.path.append(os.path.abspath('..'))

# Import custom modules
from utils.datahandling import format_example_for_training
from utils.modeling import prepare_model_for_training, load_model_for_inference

# Ignore SSL errors
response = requests.get("https://huggingface.co/api/whoami-v2", verify=False)

# ============================================
# CELL 2: Configuration
# ============================================
instructions_path = '../instructions/v1 LLM instructions.txt'
dataset_path = '../datasets/dataset_v4.json'

# Read the instructions
with open(instructions_path, 'r', encoding='utf-8') as file:
    system_instructions = file.read()

# Set paths based on variables
model_type = 'Krikri'
extra_info = ''

# Current date and time as name
if extra_info == '':
    now = datetime.now()
    extra_info = now.strftime(" %Y-%m-%d %H:%M:%S")

if model_type == 'Meltemi':
    model_id = "ilsp/Meltemi-7B-Instruct-v1.5"
    save_path = "../saved_models/meltemi" + extra_info
    output_dir = "../saved_models/outputs meltemi" + extra_info
elif model_type == 'Llama':
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    save_path = "../saved_models/llama" + extra_info
    output_dir = "../saved_models/outputs llama" + extra_info
elif model_type == 'Krikri':
    model_id = 'ilsp/Llama-Krikri-8B-Instruct'
    save_path = "../saved_models/krikri" + extra_info
    output_dir = "../saved_models/outputs krikri" + extra_info
else:
    raise ValueError("Invalid model type")

# Set up GPU memory optimization
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# ============================================
# CELL 3: Data Preparation
# ============================================
# Load data from JSON file
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Format messages for training
messages = [format_example_for_training(datum['input'], datum['output'], system_instructions) 
            for datum in data]

# Load the tokenizer with proper configuration
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = "left"

# Prepare the dataset
prompts = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=False 
)

IGNORE_TOKEN_ID = -100

train_data = []

for i, prompt in enumerate(prompts):
    # Tokenize the full prompt
    encoding = tokenizer(
        prompt, 
        padding='max_length', 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    )
    
    input_ids = encoding["input_ids"].squeeze()
    attention_mask = encoding["attention_mask"].squeeze()
    
    # Find where the assistant's response starts
    assistant_token = "<|assistant|>"
    prompt_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    
    if assistant_token in prompt_text:
        # Find the position where assistant response starts
        assistant_start_idx = prompt_text.find(assistant_token) + len(assistant_token)
        
        # Tokenize just the part before assistant's response
        prompt_before_response = prompt_text[:assistant_start_idx]
        prompt_tokens = tokenizer(prompt_before_response, return_tensors='pt')["input_ids"].squeeze()
        
        # Create labels: mask everything before assistant's response
        labels = input_ids.clone()
        labels[:len(prompt_tokens)] = IGNORE_TOKEN_ID
        
        # Also mask padding tokens
        labels[input_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    else:
        # If format is different, adjust accordingly
        labels = input_ids.clone()
        labels[input_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    
    train_data.append({
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": labels.tolist(),
    })

# Create Dataset object
train_dataset = Dataset.from_list(train_data)
print(f"Dataset prepared with {len(train_dataset)} examples")

# ============================================
# CELL 4: Model Training
# ============================================
# Load and prepare model for training
model = prepare_model_for_training(model_id)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} / {all_params:,} "
      f"({100 * trainable_params / all_params:.2f}%)")

# Define DataCollator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# Define training arguments with fixes
# Define training arguments (compatible with older transformers versions)
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,  # Slightly higher for LoRA
    num_train_epochs=10,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    fp16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,  # Increased warmup
    weight_decay=0.01,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    save_strategy="steps",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the fine-tuned model and tokenizer
print(f"Saving model to {save_path}")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print("Training completed successfully!")

# ============================================
# CELL 5: Inference Test
# ============================================
# CRITICAL: Properly load model for inference
print("\nLoading model for inference...")
model_inference, tokenizer_inference = load_model_for_inference(save_path, model_id)

print("Καλώς ήρθατε στον βοηθό ιατρικών ραντεβού. Πληκτρολογήστε 'έξοδος' για να τερματίσετε.")
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