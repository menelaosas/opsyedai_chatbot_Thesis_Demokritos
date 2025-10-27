import sys
import os
import json
import torch
from tqdm import tqdm
import evaluate
import statistics
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.abspath('..'))

# Import custom utilities
from utils.modeling import load_model_for_inference
from utils.datahandling import load_database

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Set offline mode (optional - comment out if you want to download from HuggingFace)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Model configuration
MODEL_PATH = "../saved_models/krikri 2025-10-26 18:53:51"
BASE_MODEL_ID = "ilsp/Llama-Krikri-8B-Instruct"

# Dataset paths - these are the rephrased FAQ datasets used for evaluation
DATASET_PATHS = [
    '../datasets/rephrased_faq_v1.json',
    '../datasets/rephrased_faq_v2.json',
    '../datasets/rephrased_faq_v3.json'
]

# Instructions file
INSTRUCTIONS_PATH = '../instructions/v4_LLM_instructions.txt'

# Generation parameters (controls how the model generates responses)
GENERATION_CONFIG = {
    'max_new_tokens': 256,      
    'temperature': 0.7,         
    'top_p': 0.9,               
    'do_sample': True,          
    'repetition_penalty': 1.1, 
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_system_instructions(instructions_path):
    print(f"📋 Loading system instructions from: {instructions_path}")
    with open(instructions_path, 'r', encoding='utf-8') as f:
        instructions = f.read()
    print(f"   Instructions loaded successfully ({len(instructions)} characters)")
    return instructions


def generate_response(model, tokenizer, user_input, system_instructions, device):
 
    # Format the prompt with system instructions and user input
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_input}
    ]
    
    # Apply the chat template (this formats the messages for the model)
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Generate response
    with torch.no_grad():  # Disable gradient calculation for inference
        # Tokenize the input
        inputs = tokenizer(
            formatted_prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        # Generate output tokens
        outputs = model.generate(
            **inputs,
            max_new_tokens=GENERATION_CONFIG['max_new_tokens'],
            temperature=GENERATION_CONFIG['temperature'],
            top_p=GENERATION_CONFIG['top_p'],
            do_sample=GENERATION_CONFIG['do_sample'],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=GENERATION_CONFIG['repetition_penalty'],
        )
        
        # Extract only the generated part (remove the input tokens)
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        
        # Decode the generated tokens to text
        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
    return response.strip()


def calculate_exact_match(predictions, references):
 
    exact_matches = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
    return 100 * exact_matches / len(predictions)


def evaluate_on_dataset(model, tokenizer, dataset, system_instructions, device, dataset_name):
 
    print(f"\n{'='*70}")
    print(f"🔍 Evaluating on: {dataset_name}")
    print(f"{'='*70}")
    print(f"   Dataset size: {len(dataset)} examples")
    
    predictions = []
    references = []
    
    # Generate predictions for all examples
    print("\n⏳ Generating predictions...")
    for example in tqdm(dataset, desc="Processing"):
        user_input = example['input']
        reference_output = example['output']
        
        # Generate prediction
        prediction = generate_response(
            model, tokenizer, user_input, system_instructions, device
        )
        
        predictions.append(prediction)
        references.append(reference_output)
    
    print(f"✅ Generated {len(predictions)} predictions")
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    results = {}
    
    # 1. Exact Match
    print("   - Calculating Exact Match...")
    results['exact_match'] = calculate_exact_match(predictions, references)
    
    # 2. BLEU Score
    # BLEU measures n-gram overlap (common in machine translation)
    print("   - Calculating BLEU...")
    bleu = evaluate.load("bleu")
    bleu_result = bleu.compute(predictions=predictions, references=references)
    results['bleu'] = bleu_result
    
    # 3. ROUGE Score
    # ROUGE focuses on recall (how much of the reference is in the prediction)
    print("   - Calculating ROUGE...")
    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(predictions=predictions, references=references)
    results['rouge'] = rouge_result
    
    # 4. METEOR Score
    # METEOR considers synonyms and word stems
    print("   - Calculating METEOR...")
    meteor = evaluate.load("meteor")
    meteor_result = meteor.compute(predictions=predictions, references=references)
    results['meteor'] = meteor_result
    
    # 5. BERTScore
    # BERTScore uses contextual embeddings for semantic similarity
    print("   - Calculating BERTScore (this may take a while)...")
    bertscore = evaluate.load("bertscore")
    bertscore_result = bertscore.compute(
        predictions=predictions, 
        references=references, 
        lang="el",
        device="cpu"  # ← Force CPU for BERTScore
    )
    
    # Calculate average BERTScore metrics
    results['bertscore'] = {
        'precision': statistics.mean(bertscore_result['precision']),
        'recall': statistics.mean(bertscore_result['recall']),
        'f1': statistics.mean(bertscore_result['f1'])
    }
    
    return results, predictions, references


def print_results(results, dataset_name):
    print(f"\n{'='*70}")
    print(f"📈 RESULTS FOR: {dataset_name}")
    print(f"{'='*70}")
    
    # Exact Match
    print(f"\n🎯 Exact Match: {results['exact_match']:.2f}%")
    
    # BLEU
    print(f"\n📝 BLEU Scores:")
    print(f"   - BLEU: {results['bleu']['bleu']:.4f}")
    for i in range(1, 5):
        key = f'precisions'
        if key in results['bleu'] and len(results['bleu'][key]) >= i:
            print(f"   - BLEU-{i}: {results['bleu'][key][i-1]:.4f}")
    
    # ROUGE
    print(f"\n🔴 ROUGE Scores:")
    for key, value in results['rouge'].items():
        print(f"   - {key.upper()}: {value:.4f}")
    
    # METEOR
    print(f"\n☄️  METEOR Score: {results['meteor']['meteor']:.4f}")
    
    # BERTScore
    print(f"\n🤖 BERTScore:")
    print(f"   - Precision: {results['bertscore']['precision']:.4f}")
    print(f"   - Recall: {results['bertscore']['recall']:.4f}")
    print(f"   - F1: {results['bertscore']['f1']:.4f}")


def save_results_to_file(all_results, output_dir='evaluation_results'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    return output_file


def save_predictions_to_file(predictions_data, output_dir='evaluation_results'):
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_dir, f"predictions_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Predictions saved to: {output_file}")
    return output_file


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 KRIKRI MODEL EVALUATION SCRIPT")
    print("="*70)
    print("\nThis script evaluates your fine-tuned Krikri model on multiple datasets")
    print("using standard NLP metrics to measure performance.\n")
    
    # Step 1: Load the model
    print("="*70)
    print("STEP 1: Loading Model and Tokenizer")
    print("="*70)
    print(f"📦 Model path: {MODEL_PATH}")
    print(f"📦 Base model: {BASE_MODEL_ID}")
    
    try:
        model, tokenizer = load_model_for_inference(MODEL_PATH, BASE_MODEL_ID)
        device = next(model.parameters()).device
        print(f"✅ Model loaded successfully on device: {device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Step 2: Load system instructions
    print("\n" + "="*70)
    print("STEP 2: Loading System Instructions")
    print("="*70)
    try:
        system_instructions = load_system_instructions(INSTRUCTIONS_PATH)
    except Exception as e:
        print(f"❌ Error loading instructions: {e}")
        return
    
    # Step 3: Load datasets
    print("\n" + "="*70)
    print("STEP 3: Loading Evaluation Datasets")
    print("="*70)
    datasets = {}
    for path in DATASET_PATHS:
        try:
            dataset_name = os.path.basename(path)
            print(f"📂 Loading: {dataset_name}")
            data = load_database(path)
            datasets[dataset_name] = data
            print(f"   ✅ Loaded {len(data)} examples")
        except Exception as e:
            print(f"   ❌ Error loading {path}: {e}")
    
    if not datasets:
        print("❌ No datasets loaded. Exiting.")
        return
    
    # Step 4: Evaluate on each dataset
    print("\n" + "="*70)
    print("STEP 4: Running Evaluation")
    print("="*70)
    
    all_results = {}
    all_predictions = {}
    
    for dataset_name, dataset in datasets.items():
        try:
            results, predictions, references = evaluate_on_dataset(
                model, tokenizer, dataset, system_instructions, device, dataset_name
            )
            all_results[dataset_name] = results
            all_predictions[dataset_name] = {
                'predictions': predictions,
                'references': references
            }
            
            # Print results for this dataset
            print_results(results, dataset_name)
            
        except Exception as e:
            print(f"❌ Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 5: Calculate overall statistics
    print("\n" + "="*70)
    print("STEP 5: Overall Statistics")
    print("="*70)
    
    if all_results:
        # Calculate average metrics across all datasets
        print("\n📊 Average Metrics Across All Datasets:")
        
        avg_exact_match = statistics.mean([r['exact_match'] for r in all_results.values()])
        print(f"\n🎯 Average Exact Match: {avg_exact_match:.2f}%")
        
        avg_bleu = statistics.mean([r['bleu']['bleu'] for r in all_results.values()])
        print(f"📝 Average BLEU: {avg_bleu:.4f}")
        
        avg_meteor = statistics.mean([r['meteor']['meteor'] for r in all_results.values()])
        print(f"☄️  Average METEOR: {avg_meteor:.4f}")
        
        avg_bert_f1 = statistics.mean([r['bertscore']['f1'] for r in all_results.values()])
        print(f"🤖 Average BERTScore F1: {avg_bert_f1:.4f}")
        
        # Add overall statistics to results
        all_results['overall_averages'] = {
            'exact_match': avg_exact_match,
            'bleu': avg_bleu,
            'meteor': avg_meteor,
            'bertscore_f1': avg_bert_f1
        }
    
    # Step 6: Save results
    print("\n" + "="*70)
    print("STEP 6: Saving Results")
    print("="*70)
    
    try:
        results_file = save_results_to_file(all_results)
        predictions_file = save_predictions_to_file(all_predictions)
    except Exception as e:
        print(f"❌ Error saving results: {e}")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print("\nYou can now:")
    print("1. Review the results printed above")
    print("2. Check the saved JSON files for detailed results")
    print("3. Compare metrics across different datasets")
    print("4. Analyze specific predictions where the model performed well or poorly")


if __name__ == "__main__":
    main()
