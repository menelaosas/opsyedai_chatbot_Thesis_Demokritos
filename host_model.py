from utils.opsyed_pipeline import pipeline_initialize, run_pipeline_with_memory
from utils.conversation_memory import ConversationManager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

from starlette.status import HTTP_403_FORBIDDEN
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import os
import torch
import uuid
import sys

# Force print output to be unbuffered
print("🔄 Starting OpsyedAI Enhanced Chatbot initialization...", flush=True)

# Load environment variables from .env
load_dotenv()

# Get the API key from environment
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "OpsyedAI_token"

# API initialization
app = FastAPI()
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:8000"] if using a local web server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Parameters (same as your original)
QNA_THRESHOLD = 0.2
RELATED_THRESHOLD = 0.5
QNA_K = 1
PDF_THRESHOLD = 0.45
PDF_K = 3
USE_LLM = True
MAX_CONVERSATION_HISTORY = 10

print("⚙️ Configuration loaded:", flush=True)
print(f"   📊 QNA_THRESHOLD: {QNA_THRESHOLD}", flush=True)
print(f"   📊 USE_LLM: {USE_LLM}", flush=True)
print(f"   📊 MAX_HISTORY: {MAX_CONVERSATION_HISTORY}", flush=True)

# File paths (same as your original)
sklearn_model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
qna_data_path = "datasets/combined_dataset.json"
pdf_data_path = "datasets/opsyed_hkely_user_manual.pdf"
local_generator_path = "saved_models/meltemi 2025-09-22 17:43:17"
instructions_path = 'instructions/v4_LLM_instructions.txt'
unrelated_response_path = 'instructions/v1 LLM unrelated response.txt'
pdf_suggestion_path = 'instructions/pdf suggestion.txt'

print("📁 Checking file paths:", flush=True)
files_to_check = [
    (qna_data_path, "QnA dataset"),
    (pdf_data_path, "PDF manual"), 
    (instructions_path, "Instructions"),
    (unrelated_response_path, "Unrelated response"),
    (pdf_suggestion_path, "PDF suggestion")
]

for filepath, description in files_to_check:
    exists = os.path.exists(filepath)
    print(f"   {'✅' if exists else '❌'} {description}: {filepath}", flush=True)

# Read the instructions files (same as your original)
print("📖 Loading instruction files...", flush=True)

try:
    with open(instructions_path, 'r', encoding='utf-8') as file:
        system_instructions = file.read()
    print("   ✅ System instructions loaded", flush=True)
except Exception as e:
    print(f"   ❌ Error loading system instructions: {e}", flush=True)
    sys.exit(1)

try:
    with open(unrelated_response_path, 'r', encoding='utf-8') as file:
        unrelated_response = file.read()
    print("   ✅ Unrelated response loaded", flush=True)
except Exception as e:
    print(f"   ❌ Error loading unrelated response: {e}", flush=True)
    sys.exit(1)

try:
    with open(pdf_suggestion_path, 'r', encoding='utf-8') as file:
        pdf_suggestion = file.read()
    print("   ✅ PDF suggestion loaded", flush=True)
except Exception as e:
    print(f"   ❌ Error loading PDF suggestion: {e}", flush=True)
    sys.exit(1)

# Initialize pipeline (same as your original)
print("🔄 Initializing pipeline components...", flush=True)
try:
    index, model, corpus, answers, pdf_index, pdf_model, pdf_titles, generator = pipeline_initialize(
        qna_data_path, sklearn_model_name, pdf_data_path, sklearn_model_name, 
        local_generator_path, USE_LLM
    )
    print("✅ Pipeline initialized successfully!", flush=True)
    print(f"   📚 QnA corpus size: {len(corpus) if corpus else 0}", flush=True)
    print(f"   📄 PDF sections: {len(pdf_titles) if pdf_titles else 0}", flush=True)
    print(f"   🤖 LLM generator: {'✅ Loaded' if generator else '❌ Not loaded'}", flush=True)
except Exception as e:
    print(f"❌ Pipeline initialization failed: {e}", flush=True)
    sys.exit(1)

# Initialize conversation manager (NEW)
print("🧠 Initializing conversation memory...", flush=True)
try:
    conversation_manager = ConversationManager(
        storage_type="memory",  # Change to "sqlite" for persistence
        max_history=MAX_CONVERSATION_HISTORY,
        db_path="conversations.db",  # Only used if storage_type="sqlite"
        cleanup_after_hours=24
    )
    print("✅ Conversation memory initialized!", flush=True)
except Exception as e:
    print(f"❌ Conversation memory initialization failed: {e}", flush=True)
    sys.exit(1)

def is_gpu_oom():
    try:
        torch.cuda.empty_cache()
        _ = torch.randn((1024, 1024), device='cuda')
        return False
    except RuntimeError as e:
        return "out of memory" in str(e)

# Request models
class Prompt(BaseModel):
    prompt: str
    response_id: str
    conversation_id: Optional[str] = None  # NEW: Auto-generate if not provided
    include_history: Optional[bool] = True  # NEW: Option to disable history

class RatingRequest(BaseModel):
    rating: int
    response_id: str

# Enhanced generate endpoint
@app.post("/generate")
def generate_text(prompt: Prompt, api_key: str = Depends(get_api_key)):
    """
    Generate response with conversation memory support.
    """
    if is_gpu_oom():
        return JSONResponse({"error": "GPU out of memory"}, status_code=503)
    
    # Auto-generate conversation ID if not provided
    if not prompt.conversation_id:
        prompt.conversation_id = str(uuid.uuid4())
    
    try:
        # Get conversation history
        conversation_history = []
        if prompt.include_history:
            conversation_history = conversation_manager.get_conversation_history(prompt.conversation_id)
        
        # Run enhanced pipeline with memory
        prediction, qna_retrieved, is_related, pdf_retrieved, context_info = run_pipeline_with_memory(
            query=prompt.prompt,
            conversation_history=conversation_history,
            index=index,
            model=model,
            answers=answers,
            qna_k=QNA_K,
            qna_threshold=QNA_THRESHOLD,
            related_threshold=RELATED_THRESHOLD,
            instructions=system_instructions,
            unrelated_response=unrelated_response,
            pdf_index=pdf_index,
            pdf_model=pdf_model,
            pdf_titles=pdf_titles,
            pdf_k=PDF_K,
            pdf_threshold=PDF_THRESHOLD,
            generator=generator,
            pdf_suggestion=pdf_suggestion,
            use_LLM=USE_LLM
        )
        
        # Store the exchange in conversation memory
        conversation_manager.add_exchange(
            conversation_id=prompt.conversation_id,
            question=prompt.prompt,
            answer=prediction,
            response_id=prompt.response_id,
            metadata={
                'qna_retrieved': qna_retrieved,
                'is_related': is_related,
                'pdf_retrieved': pdf_retrieved,
                'context_used': context_info.get('is_relevant', False),
                'relevance_score': context_info.get('relevance_score', 0.0)
            }
        )
        
        # Enhanced logging
        print('=' * 60, flush=True)
        print(f'🗣️  Conversation ID: {prompt.conversation_id}', flush=True)
        print(f'❓ Question: {prompt.prompt}', flush=True)
        print(f'💬 Answer: {prediction[:100]}...' if len(prediction) > 100 else f'💬 Answer: {prediction}', flush=True)
        print(f'🆔 Response ID: {prompt.response_id}', flush=True)
        print(f'📚 QnA Retrieved: {"Yes" if qna_retrieved else "No"}', flush=True)
        print(f'🔗 Related: {"Yes" if is_related else "No"}', flush=True)
        print(f'📄 PDF Retrieved: {"Yes" if pdf_retrieved else "No"}', flush=True)
        print(f'🧠 Context Used: {"Yes" if context_info.get("is_relevant", False) else "No"}', flush=True)
        print(f'📊 History Length: {len(conversation_history)}', flush=True)
        print('=' * 60, flush=True)
        
        return {
            'generated_text': prediction,
            'is_qna_retrieved': qna_retrieved,
            'is_related': is_related,
            'is_pdf_retrieved': pdf_retrieved,
            'conversation_id': prompt.conversation_id,
            'context_used': context_info.get('is_relevant', False),
            'relevance_score': context_info.get('relevance_score', 0.0),
            'history_length': len(conversation_history)
        }
        
    except Exception as e:
        print(f"❌ Error in generate_text: {str(e)}", flush=True)
        return JSONResponse(
            {"error": f"Generation failed: {str(e)}"}, 
            status_code=500
        )

# NEW: Conversation management endpoints
@app.get("/conversation/{conversation_id}")
def get_conversation_history(conversation_id: str, api_key: str = Depends(get_api_key)):
    """Get the full conversation history for a specific conversation."""
    try:
        history = conversation_manager.get_conversation_history(conversation_id)
        return {
            'conversation_id': conversation_id,
            'exchanges': history,
            'total_exchanges': len(history)
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/conversation/{conversation_id}")
def clear_conversation(conversation_id: str, api_key: str = Depends(get_api_key)):
    """Clear conversation history for a specific conversation."""
    try:
        success = conversation_manager.clear_conversation(conversation_id)
        if success:
            return {'message': f'Conversation {conversation_id} cleared successfully'}
        else:
            return JSONResponse({"error": "Conversation not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/conversations")
def list_active_conversations(api_key: str = Depends(get_api_key)):
    """List all active conversation IDs."""
    try:
        active_conversations = conversation_manager.list_active_conversations()
        return {
            'active_conversations': active_conversations,
            'total_count': len(active_conversations)
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/memory/stats")
def get_memory_stats(api_key: str = Depends(get_api_key)):
    """Get conversation memory statistics."""
    try:
        stats = conversation_manager.get_conversation_stats()
        return stats
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/memory/cleanup")
def cleanup_old_conversations(api_key: str = Depends(get_api_key)):
    """Manually trigger cleanup of old conversations."""
    try:
        cleaned = conversation_manager.cleanup_old_conversations()
        return {'message': f'Cleaned up {cleaned} old conversations'}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Original rating endpoint (unchanged)
@app.post("/submit_rating")
async def submit_rating(data: RatingRequest, api_key: str = Depends(get_api_key)):
    print("Received rating:", data.rating, 'for response:', data.response_id, flush=True)
    return JSONResponse(content={"status": "success", "message": "Rating received!"})

# Health check endpoint (enhanced)
@app.get("/health")
def health_check():
    try:
        stats = conversation_manager.get_conversation_stats()
        return {
            "status": "healthy",
            "conversations_active": stats.get('total_conversations', 0),
            "total_exchanges": stats.get('total_exchanges', 0),
            "memory_type": stats.get('storage_type', 'unknown')
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }

# Root endpoint with API info
@app.get("/")
def root():
    return {
        "message": "OpsyedAI Chatbot with Memory - API is running!",
        "version": "2.0.0",
        "features": [
            "Conversation memory",
            "Context-aware responses", 
            "Multi-storage backends",
            "QnA retrieval",
            "PDF document search",
            "Fine-tuned LLM generation"
        ],
        "endpoints": {
            "generate": "/generate - Main chat endpoint",
            "history": "/conversation/{id} - Get conversation history", 
            "clear": "/conversation/{id} - Clear conversation",
            "conversations": "/conversations - List active conversations",
            "stats": "/memory/stats - Memory statistics",
            "health": "/health - Health check"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting OpsyedAI Chatbot with Memory...", flush=True)
    print("📊 Features: Conversation Memory, Context Awareness, Enhanced Retrieval", flush=True)
    print("🌐 Server will be available at: http://127.0.0.1:5000", flush=True)
    print("📖 API docs available at: http://127.0.0.1:5000/docs", flush=True)
    print("🎯 Enhanced chatbot ready to handle conversations!", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=5000)