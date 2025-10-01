# ============================================
# DPO TRAINING - NO CHAT TEMPLATES
# Matches your original simple input/output format
# ============================================

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import DPOTrainer
from peft import PeftModel
from datetime import datetime
import gc

# ============================================
# Configuration
# ============================================
print("=" * 50)
print("DPO TRAINING - SIMPLE FORMAT (NO CHAT TEMPLATES)")
print("=" * 50)

BASE_MODEL_ID = "ilsp/Meltemi-7B-Instruct-v1.5"
FINE_TUNED_MODEL_PATH = "../saved_models/meltemi-2025-09-22-17:43:17"
PREFERENCE_DATA_PATH = "../datasets/prefs_train.jsonl"

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
SAVE_PATH = f"../saved_models/meltemi_dpo_simple_{timestamp}"
OUTPUT_DIR = f"../saved_models/outputs_dpo_simple_{timestamp}"

os.environ["WANDB_DISABLED"] = "true"
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()
gc.collect()

# ============================================
# Load and Format Dataset - CRITICAL CHANGE
# ============================================
print("\n[1/4] Loading preference data...")
preference_data = []
with open(PREFERENCE_DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        preference_data.append(json.loads(line.strip()))

# Format WITHOUT adding chat templates - keep it simple like original training
formatted_data = []
for item in preference_data:
    # CRITICAL: Keep the exact same format as your original training
    # Just prompt → response, no system messages, no chat wrapping
    formatted_data.append({
        "prompt": item['prompt'],  # Simple Greek question
        "chosen": item['chosen'],  # Good answer
        "rejected": item['rejected']  # Bad answer
    })

dataset = Dataset.from_list(formatted_data)

# Use all data (you have 200 examples)
print(f"Using {len(dataset)} examples")

# ============================================
# Load Tokenizer
# ============================================
print("\n[2/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = "left"

# CRITICAL: Remove chat template so DPO doesn't apply it
tokenizer.chat_template = None

print("Tokenizer configured (chat template disabled)")

# ============================================
# Load Model
# ============================================
print("\n[3/4] Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    FINE_TUNED_MODEL_PATH,
    is_trainable=True
)

model.config.use_cache = False
model.enable_input_require_grads()

print("\nTrainable parameters:")
model.print_trainable_parameters()

trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
if trainable_count == 0:
    raise ValueError("No trainable parameters!")

ref_model = None

# ============================================
# Configure Training
# ============================================
print("\n[4/4] Configuring training...")

from trl import DPOConfig

# Use DPOConfig which includes beta parameter
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    
    # Memory optimized
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    
    # Sequence lengths
    max_length=512,
    max_prompt_length=256,
    
    # Training
    learning_rate=2e-5,
    num_train_epochs=1,
    
    # DPO specific
    beta=0.1,
    
    # Logging
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    
    # Performance
    fp16=True,
    optim="adamw_torch",
    warmup_ratio=0.05,
    
    # Other
    gradient_checkpointing=True,
    remove_unused_columns=False,
    save_strategy="steps",
    report_to="none",
)

print("Configuration set")

# ============================================
# Initialize Trainer
# ============================================
print("\nInitializing DPO Trainer...")
torch.cuda.empty_cache()
gc.collect()

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,  # Use processing_class instead of tokenizer
)

print("Trainer initialized")

# ============================================
# Train
# ============================================
print("\n" + "=" * 50)
print("TRAINING")
print("=" * 50)
print(f"Dataset: {len(dataset)} examples")
print(f"Epochs: {training_args.num_train_epochs}")

try:
    trainer.train()
    print("\nTraining completed successfully!")
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\nOUT OF MEMORY - Reduce max_length to 256")
        raise
    else:
        raise

# ============================================
# Save
# ============================================
print("\n" + "=" * 50)
print("SAVING")
print("=" * 50)

trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print(f"Model saved to: {SAVE_PATH}")

# ============================================
# Test - Simple Format Matching Original
# ============================================
print("\n" + "=" * 50)
print("TESTING")
print("=" * 50)

torch.cuda.empty_cache()
model.eval()

test_questions = [
    "Πώς μπορώ να κλείσω ραντεβού στο νοσοκομείο;",
    "Πώς μπορώ να ελέγξω αν ανήκω στις κατηγορίες Ληπτών Υγείας;",
    "Τι χρειάζομαι για εγγραφή στο ΗΚΕΛΥ;",
]

for q in test_questions:
    # Simple format - no chat template, just like original training
    inputs = tokenizer(q, return_tensors='pt').to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\nQ: {q}")
    print(f"A: {response}")
    print("-" * 50)

print("\n" + "=" * 50)
print("COMPLETE")
print("=" * 50)
print(f"\nModel saved to: {SAVE_PATH}")
print("\nIf outputs are still gibberish, the issue is deeper.")
print("If outputs are good Greek text, DPO worked correctly!")