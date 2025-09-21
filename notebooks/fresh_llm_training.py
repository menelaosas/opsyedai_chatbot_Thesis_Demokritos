#!/usr/bin/env python3
"""
Fresh LLM Training Script for HKELY Chatbot
Fixed version with proper path handling
"""

import os
import sys
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

def main():
    """Main training function"""
    
    print("🚀 Starting LLM Training Script...")
    
    # ============================================================================
    # INITIALIZATION AND PATH SETUP
    # ============================================================================
    
    # Environment setup
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ["WANDB_DISABLED"] = "true"
    
    # Get the absolute path to the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from notebooks/
    sys.path.insert(0, project_root)
    
    print(f"📁 Script directory: {script_dir}")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Contents of project root: {os.listdir(project_root)}")
    
    # Import custom utilities
    try:
        from utils.datahandling import format_example_for_training
        from utils.modeling import prepare_model_for_training, generate_response
        print("✅ Successfully imported utils modules")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure utils/ directory exists with required modules")
        return
    
    # SSL certificate handling (only if source is trusted)
    try:
        response = requests.get("https://huggingface.co/api/whoami-v2", verify=False)
        print("✅ HuggingFace connection test successful")
    except Exception as e:
        print(f"⚠️ Warning: SSL verification issue: {e}")
    
    # ============================================================================
    # FILE PATH DETECTION
    # ============================================================================
    
    print("\n🔍 Looking for required files...")
    
    # Try multiple possible locations for instructions file
    possible_instructions_paths = [
        os.path.join(project_root, 'instructions', 'v1 LLM instructions.txt'),
        os.path.join(project_root, 'instructions', 'v1_LLM_instructions.txt'),
        os.path.join(project_root, 'instructions', 'LLM_instructions.txt'),
        os.path.join(project_root, 'instructions.txt'),
    ]
    
    # Try multiple possible locations for dataset file
    possible_dataset_paths = [
        os.path.join(project_root, 'datasets', 'dataset_v2.json'),
        os.path.join(project_root, 'datasets', 'dataset.json'),
        os.path.join(project_root, 'data', 'dataset_v2.json'),
        os.path.join(project_root, 'dataset_v2.json'),
    ]
    
    # Find instructions file
    instructions_path = None
    for path in possible_instructions_paths:
        print(f"  Checking: {path}")
        if os.path.exists(path):
            instructions_path = path
            print(f"  ✅ Found instructions file: {path}")
            break
    
    if not instructions_path:
        print(f"\n❌ Instructions file not found! Tried:")
        for path in possible_instructions_paths:
            print(f"    - {path}")
        
        # Show what's actually in the instructions directory
        instructions_dir = os.path.join(project_root, 'instructions')
        if os.path.exists(instructions_dir):
            print(f"\n📂 Files in instructions directory:")
            for file in os.listdir(instructions_dir):
                print(f"    - {file}")
        else:
            print(f"\n📂 Instructions directory doesn't exist: {instructions_dir}")
        return
    
    # Find dataset file
    dataset_path = None
    for path in possible_dataset_paths:
        print(f"  Checking: {path}")
        if os.path.exists(path):
            dataset_path = path
            print(f"  ✅ Found dataset file: {path}")
            break
    
    if not dataset_path:
        print(f"\n❌ Dataset file not found! Tried:")
        for path in possible_dataset_paths:
            print(f"    - {path}")
        
        # Show what's actually in the datasets directory
        datasets_dir = os.path.join(project_root, 'datasets')
        if os.path.exists(datasets_dir):
            print(f"\n📂 Files in datasets directory:")
            for file in os.listdir(datasets_dir):
                print(f"    - {file}")
        else:
            print(f"\n📂 Datasets directory doesn't exist: {datasets_dir}")
        return
    
    # ============================================================================
    # LOAD INSTRUCTIONS
    # ============================================================================
    
    print(f"\n📖 Loading instructions...")
    try:
        with open(instructions_path, 'r', encoding='utf-8') as file:
            system_instructions = file.read()
        print(f"✅ Loaded {len(system_instructions)} characters from instructions file")
    except Exception as e:
        print(f"❌ Error reading instructions: {e}")
        return
    
    # ============================================================================
    # MODEL CONFIGURATION
    # ============================================================================
    
    # Set model type here - change this to 'Llama' or 'Krikri' as needed
    model_type = 'Meltemi'
    extra_info = ''
    
    # Current date and time as name, if no name was given
    if extra_info == '':
        now = datetime.now()
        extra_info = now.strftime("_%Y%m%d_%H%M%S")
    
    # Model configuration mapping
    model_configs = {
        'Meltemi': {
            'model_id': "ilsp/Meltemi-7B-Instruct-v1.5",
            'save_path': os.path.join(project_root, 'saved_models', f"meltemi{extra_info}"),
            'output_dir': os.path.join(project_root, 'saved_models', f"outputs_meltemi{extra_info}")
        },
        'Llama': {
            'model_id': "meta-llama/Llama-3.2-3B-Instruct",
            'save_path': os.path.join(project_root, 'saved_models', f"llama{extra_info}"),
            'output_dir': os.path.join(project_root, 'saved_models', f"outputs_llama{extra_info}")
        },
        'Krikri': {
            'model_id': 'ilsp/Llama-Krikri-8B-Instruct',
            'save_path': os.path.join(project_root, 'saved_models', f"krikri{extra_info}"),
            'output_dir': os.path.join(project_root, 'saved_models', f"outputs_krikri{extra_info}")
        }
    }
    
    if model_type not in model_configs:
        raise ValueError(f"Invalid model type. Choose from: {list(model_configs.keys())}")
    
    config = model_configs[model_type]
    model_id = config['model_id']
    save_path = config['save_path']
    output_dir = config['output_dir']
    
    print(f"\n🤖 Using {model_type} model: {model_id}")
    print(f"💾 Will save to: {save_path}")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # GPU memory optimization
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # ============================================================================
    # DATA PREPARATION
    # ============================================================================
    
    print(f"\n📊 Loading dataset from: {dataset_path}")
    
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} examples from dataset")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Format messages for training
    print("🔧 Formatting data for training...")
    messages = [format_example_for_training(datum['input'], datum['output'], system_instructions) for datum in data]
    
    # Load tokenizer
    print("🔤 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return
    
    # Prepare the dataset for instruction fine-tuning
    print("⚙️ Preparing training dataset...")
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    IGNORE_TOKEN_ID = -100
    train_data = []
    
    for i, prompt in enumerate(prompts):
        if i % 50 == 0:
            print(f"  Processing example {i+1}/{len(prompts)}")
            
        # Encode text to tokens
        encoding = tokenizer(prompt, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Find where the generation (response) should start
        response_start = prompt.find("<|assistant|>") + len("<|assistant|>")
        response_tokens = tokenizer(prompt[response_start:], padding='max_length', truncation=True, max_length=512, return_tensors='pt')["input_ids"].squeeze()
        
        # Build labels: mask everything before the assistant's response
        labels = input_ids.clone()
        num_response_tokens = (response_tokens != tokenizer.pad_token_id).sum()
        num_total_tokens = len(input_ids)
        response_start_token_idx = num_total_tokens - num_response_tokens
        labels[:response_start_token_idx] = IGNORE_TOKEN_ID
        
        train_data.append({
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "labels": labels.tolist(),
        })
    
    train_dataset = Dataset.from_list(train_data)
    print(f"✅ Dataset prepared with {len(train_dataset)} examples")
    
    # ============================================================================
    # MODEL TRAINING
    # ============================================================================
    
    print("\n🤖 Loading model for training...")
    try:
        model = prepare_model_for_training(model_id)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Define training components
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=10,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        fp16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        remove_unused_columns=False,
    )
    
    print(f"\n🏋️ Starting training with:")
    print(f"  📊 Dataset size: {len(train_dataset)} examples")
    print(f"  🔄 Epochs: {training_args.num_train_epochs}")
    print(f"  🎯 Batch size: {training_args.per_device_train_batch_size}")
    print(f"  📈 Learning rate: {training_args.learning_rate}")
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    try:
        print("\n🚀 Training started...")
        trainer.train()
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Save the model
    print(f"\n💾 Saving model to: {save_path}")
    try:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("✅ Model saved successfully!")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return
    
    print(f"\n🎉 Training complete! Model saved to: {save_path}")

if __name__ == "__main__":
    main()
