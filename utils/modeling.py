from utils.datahandling import format_example_for_inference
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from transformers.pipelines.base import Pipeline
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    PeftMixedModel
)
import torch
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*gradient_checkpointing.*")

def load_model(model_id: str) -> AutoModelForCausalLM:
    """Load model with quantization for training"""
    # Define quantization configuration for 4-bit precision
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Determine the correct device map
    if torch.cuda.is_available():
        device_map = {"": torch.cuda.current_device()}
    else:
        device_map = {"": "cpu"}

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        use_cache=False,  # Disable cache for training
    )

    return model


def prepare_model_for_training(model_id: str) -> PeftModel:
    """Load model and apply LoRA for training"""
    # Load model
    model = load_model(model_id)
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # CRITICAL: Set model to training mode
    model.train()
    
    # Define LoRA configuration
    peft_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj",
                       "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Apply LoRA configuration to the model
    model = get_peft_model(model, peft_config)
    
    # The adapters are already enabled after get_peft_model
    # No need to call enable_adapters() here
    
    return model


def load_model_for_inference(adapter_path: str, base_model_id: str):
    """
    Properly load fine-tuned model for inference
    
    Args:
        adapter_path: Path to saved LoRA adapters
        base_model_id: Original base model ID
    
    Returns:
        model and tokenizer ready for inference
    """
    print("Loading tokenizer...")
    
    # Load tokenizer from saved path
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    except:
        print(f"Could not load tokenizer from {adapter_path}, loading from base model")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Ensure tokenizer has proper configuration
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set padding side for generation
    tokenizer.padding_side = "left"
    
    print("Loading base model...")
    
    # Load base model for inference
    if torch.cuda.is_available():
        # For inference, we can use different quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False  # Simpler for inference
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=True,  # Enable cache for inference
            torch_dtype=torch.float16,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="cpu",
            trust_remote_code=True,
            use_cache=True,
            torch_dtype=torch.float32,
        )
    
    print("Loading LoRA adapters...")
    
    # Load the LoRA adapters
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    # CRITICAL: Set model to evaluation mode
    model.eval()
    
    # Optional: Merge LoRA weights for faster inference
    # Uncomment the next line if you want to merge the adapters
    # model = model.merge_and_unload()
    
    return model, tokenizer


def generate_response(
    prompt: str,
    instructions: str,
    tokenizer: AutoTokenizer,
    model: PeftModel
) -> str:
    """
    Fixed generation function with proper inference setup
    
    Args:
        prompt: User input text
        instructions: System instructions
        tokenizer: Tokenizer instance
        model: Fine-tuned model
    
    Returns:
        Generated response text
    """
    # Format prompt
    messages = format_example_for_inference(prompt, instructions)
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Determine device
    device = next(model.parameters()).device
    
    # CRITICAL: Disable gradient computation for inference
    with torch.no_grad():
        # Tokenize the prompt
        inputs = tokenizer(
            formatted_prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        # Generate response with proper parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            min_new_tokens=5,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
        
        # Get only the generated part (exclude input)
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        
        # Decode the generated tokens
        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Clean up any remaining artifacts
        response = response.strip()
        
        # Remove any remaining special tokens
        special_tokens = ['<|im_end|>', '<|im_start|>', '<|endoftext|>', 
                         '<|assistant|>', '<|user|>', '<|system|>']
        for token in special_tokens:
            response = response.replace(token, '')
        
        return response.strip()


# Make pipeline for text generation (keeping for compatibility)
def make_LLM_pipeline(model_id: str) -> Pipeline:
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = load_model(model_id)

    # Make pipeline
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    
    return gen_pipeline