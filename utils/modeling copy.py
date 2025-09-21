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

# Loads an AutoModelForCausalLM with quantization
def load_model(model_id: str) -> AutoModelForCausalLM:
    # Define quantization configuration for 4-bit precision
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Determine the correct device map
    if torch.cuda.is_available():
        # For single GPU, use current device
        device_map = {"": torch.cuda.current_device()}
        # Alternative: device_map = "auto" (let HF decide)
    else:
        device_map = {"": "cpu"}

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,  # Fixed: proper device mapping
        trust_remote_code=True,
    )

    return model


# Make pipeline for text generation
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


# Load model and apply LoRA
def prepare_model_for_training(model_id: str) -> (PeftModel | PeftMixedModel):
    # Load model
    model = load_model(model_id)

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

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

    return model


# Use given model to generate its response to a prompt
def generate_response(prompt: str,
                      instructions: str,
                      tokenizer: AutoTokenizer,
                      model: (PeftModel | PeftMixedModel)) -> str:
    # Format prompt
    messages = format_example_for_inference(prompt, instructions)

    # Determine device (for compatibility)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenize messages and generate response
    prompt = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=False)
    input_prompt = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(
        **input_prompt,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # Un-tokenize the response to show it as text
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only assistant's reply
    if "<|assistant|>" in decoded:
        response = decoded.split("<|assistant|>")[-1].strip()
    else:
        response = decoded.strip()

    return response