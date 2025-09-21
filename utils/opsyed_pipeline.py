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