from utils.qna_irm import create_corpus, build_qna_sklearn_index, qna_irm_pipeline
from utils.pdf_irm import split_pdf_by_sections_skip_intro, build_pdf_sklearn_index, pdf_irm_pipeline
from utils.modeling import make_LLM_pipeline, generate_response
from utils.datahandling import load_database 

from typing import List, Tuple
from transformers.pipelines.base import Pipeline
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors



# Create corpus, embedding space, and load model
def pipeline_initialize(qna_data_path: str,
                        qna_model_name: str,
                        pdf_data_path: str,
                        pdf_model_name: str,
                        generator_path: str,
                        use_LLM: bool) -> Tuple[NearestNeighbors,
                                                      SentenceTransformer,
                                                      List[str],
                                                      List[str],
                                                      NearestNeighbors,
                                                      SentenceTransformer,
                                                      List[str],
                                                      Pipeline | None]:
    # QnA IRM preparation
    data = load_database(qna_data_path)
    corpus, answers = create_corpus(data)
    index, model, _ = build_qna_sklearn_index(corpus, qna_model_name)

    # pdf IRM preparation
    sections = split_pdf_by_sections_skip_intro(pdf_data_path)
    chunks = [section['content'] for section in sections]
    titles = [section['section'] for section in sections]

    pdf_index, pdf_model, _ = build_pdf_sklearn_index(chunks, pdf_model_name)

    # LLM preparation
    if use_LLM:
        generator = make_LLM_pipeline(generator_path)
    else:
        generator = None

    return index, model, corpus, answers, pdf_index, pdf_model, titles, generator


# Full pipeline
def run_pipeline(query: str,
                 index: NearestNeighbors,
                 model: SentenceTransformer,
                 answers: List[str],
                 qna_k: int,
                 qna_threshold: float,
                 related_threshold: float,
                 instructions: str,
                 unrelated_response: str,
                 pdf_index,
                 pdf_model,
                 pdf_titles,
                 pdf_k: int,
                 pdf_threshold: float,
                 generator: Pipeline,
                 pdf_suggestion: str,
                 use_LLM: bool) -> Tuple[str, bool, bool, bool]:
    
    # QnA IRM
    answer, qna_retrieved, is_related = qna_irm_pipeline(query,
                                             unrelated_response,
                                             index,
                                             model,
                                             answers,
                                             qna_k,
                                             qna_threshold,
                                             related_threshold)
    
    # LLM triggers if QnA IRM didn't retrieve and caller wants the LLM usage
    if is_related and not qna_retrieved and use_LLM:
        answer = generate_response(query, instructions, generator.tokenizer, generator.model)


    # PDF IRM only triggers for related answers
    if is_related and (use_LLM or qna_retrieved):
        answer, pdf_retrieved = pdf_irm_pipeline(query,
                                                answer,
                                                pdf_index,
                                                pdf_model,
                                                pdf_titles,
                                                pdf_k,
                                                pdf_threshold,
                                                pdf_suggestion)
    else:
        pdf_retrieved = False

    return answer, qna_retrieved, is_related, pdf_retrieved


# Add this function to the END of your existing utils/opsyed_pipeline.py file

from typing import List, Dict
import re

def assess_context_relevance(current_query: str, 
                           conversation_history: List[Dict],
                           relevance_threshold: float = 0.3) -> Dict:
    """
    Assess how relevant the conversation context is to the current query.
    """
    if not conversation_history:
        return {
            'is_relevant': False,
            'relevance_score': 0.0,
            'should_use_context': False,
            'context_summary': ""
        }
    
    # Extract keywords from current query
    current_keywords = set(word.lower() for word in current_query.split() 
                          if len(word) > 2 and word.isalpha())
    
    # Extract keywords from conversation history
    history_text = " ".join([
        f"{ex['question']} {ex['answer']}" 
        for ex in conversation_history[-3:]  # Look at last 3 exchanges
    ])
    history_keywords = set(word.lower() for word in history_text.split() 
                          if len(word) > 2 and word.isalpha())
    
    # Calculate Jaccard similarity
    if current_keywords and history_keywords:
        intersection = current_keywords.intersection(history_keywords)
        union = current_keywords.union(history_keywords)
        jaccard_score = len(intersection) / len(union)
    else:
        jaccard_score = 0.0
    
    # Check for explicit references to previous conversation
    reference_indicators = [
        "you mentioned", "you said", "earlier", "before", "previous", 
        "last time", "we discussed", "as you explained", "like you told me",
        "that", "it", "this", "they", "them"  # Pronouns suggesting reference
    ]
    
    has_explicit_reference = any(indicator in current_query.lower() 
                                for indicator in reference_indicators)
    
    # Determine relevance
    is_relevant = jaccard_score > relevance_threshold or has_explicit_reference
    
    return {
        'is_relevant': is_relevant,
        'relevance_score': jaccard_score,
        'has_explicit_reference': has_explicit_reference,
        'should_use_context': is_relevant,
        'context_summary': _create_context_summary(conversation_history) if is_relevant else ""
    }

def _create_context_summary(conversation_history: List[Dict], max_exchanges: int = 3) -> str:
    """Create a concise summary of recent conversation history."""
    if not conversation_history:
        return ""
    
    # Get the most recent exchanges
    recent_exchanges = conversation_history[-max_exchanges:]
    
    summary_parts = []
    for exchange in recent_exchanges:
        question = exchange['question']
        answer = exchange['answer']
        if len(answer) > 100:
            answer = answer[:97] + "..."
        
        summary_parts.append(f"Human: {question}")
        summary_parts.append(f"Assistant: {answer}")
    
    return "\n".join(summary_parts)

def extract_current_query(full_prompt: str) -> str:
    """Extract the current user query from a prompt that may contain conversation context."""
    # Look for the pattern "Current question: ..." at the end
    current_question_match = re.search(r"Current question:\s*(.+)$", full_prompt, re.MULTILINE | re.DOTALL)
    if current_question_match:
        return current_question_match.group(1).strip()
    
    # If no context pattern found, assume the entire prompt is the query
    return full_prompt.strip()

def create_context_aware_prompt(current_query: str, 
                               conversation_context: str,
                               is_relevant: bool) -> str:
    """Create an enhanced prompt that includes conversation context when relevant."""
    if not is_relevant or not conversation_context:
        return current_query
    
    return f"""Previous conversation context:
{conversation_context}

Current question: {current_query}

Please respond to the current question while considering the previous conversation context when relevant."""

def enhance_system_instructions(base_instructions: str, has_context: bool) -> str:
    """Enhance system instructions when conversation context is available."""
    if not has_context:
        return base_instructions
    
    context_guidance = """

CONVERSATION CONTEXT GUIDANCE:
You have access to previous conversation history. Use this context to:
1. Understand references and pronouns (it, that, this, they, etc.)
2. Build upon previous explanations rather than repeating them
3. Maintain consistency with previous responses
4. Recognize follow-up questions and provide appropriate continuations

When the user refers to something mentioned earlier, use the context to understand what they mean."""

    return base_instructions + context_guidance

def run_pipeline_with_memory(query: str,
                           conversation_history: List[Dict],
                           index,
                           model,
                           answers,
                           qna_k: int,
                           qna_threshold: float,
                           related_threshold: float,
                           instructions: str,
                           unrelated_response: str,
                           pdf_index,
                           pdf_model,
                           pdf_titles,
                           pdf_k: int,
                           pdf_threshold: float,
                           generator,
                           pdf_suggestion: str,
                           use_LLM: bool,
                           max_context_exchanges: int = 3):
    """
    Enhanced pipeline that incorporates conversation memory for context-aware responses.
    """
    
    # Step 1: Assess context relevance
    context_info = assess_context_relevance(query, conversation_history)
    
    # Step 2: Extract clean query for retrieval operations
    clean_query = extract_current_query(query)
    
    # Step 3: Create conversation context if relevant
    conversation_context = ""
    if context_info['should_use_context'] and conversation_history:
        conversation_context = context_info['context_summary']
    
    # Step 4: QnA IRM using clean query for better matching
    answer, qna_retrieved, is_related = qna_irm_pipeline(
        clean_query,  # Use clean query for retrieval
        unrelated_response,
        index,
        model,
        answers,
        qna_k,
        qna_threshold,
        related_threshold
    )
    
    # Step 5: Enhanced LLM generation with context awareness
    if is_related and not qna_retrieved and use_LLM:
        # Enhance system instructions if we have relevant context
        enhanced_instructions = enhance_system_instructions(
            instructions, 
            bool(conversation_context)
        )
        
        # Create context-aware prompt if context is relevant
        if conversation_context:
            llm_query = create_context_aware_prompt(
                clean_query,
                conversation_context,
                context_info['should_use_context']
            )
        else:
            llm_query = clean_query
        
        # Generate response with enhanced context
        answer = generate_response(
            llm_query, 
            enhanced_instructions, 
            generator.tokenizer, 
            generator.model
        )

    # Step 6: PDF IRM for related answers using clean query
    if is_related and (use_LLM or qna_retrieved):
        answer, pdf_retrieved = pdf_irm_pipeline(
            clean_query,  # Use clean query for PDF retrieval
            answer,
            pdf_index,
            pdf_model,
            pdf_titles,
            pdf_k,
            pdf_threshold,
            pdf_suggestion
        )
    else:
        pdf_retrieved = False

    return answer, qna_retrieved, is_related, pdf_retrieved, context_info