from utils.datahandling import load_database

from typing import List, Dict, Tuple, Any
import numpy.typing as npt
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


# Create a searchable corpus from descriptions
def create_corpus(data: List[Dict]) -> Tuple[List[str], List[str]]:
    questions = [item["input"] for item in data if "input" in item]
    answers = [item["output"] for item in data if "output" in item]
    return questions, answers


# Build scikit-learn NearestNeighbors index
def build_qna_sklearn_index(corpus: List[str],
                            model_name: str) -> Tuple[NearestNeighbors,
                                                  SentenceTransformer,
                                                  npt.NDArray[Any]]:
    model = SentenceTransformer(model_name, device='cpu')
    embeddings = model.encode(corpus, convert_to_numpy=True)
    index = NearestNeighbors(metric='cosine')
    index.fit(embeddings)
    return index, model, embeddings


# Retrieve top-k descriptions
def retrieve(query: str,
             index: NearestNeighbors,
             model: SentenceTransformer,
             answers: List[str],
             k: int,
             threshold: float,
             related_threshold: float) -> Tuple[List[str], bool]:
    # encode query
    query_vec = model.encode([query], convert_to_numpy=True)

    # find distances of k closest neighbors
    distances, indices = index.kneighbors(query_vec, n_neighbors=k)

    print([(distances[0][i], answers[id]) for i, id in enumerate(indices[0])])

    # find if query is at least somewhat related to any QnA answer
    related = True
    if distances[0][0] > related_threshold:
        related = False

    return [answers[id] for i, id in enumerate(indices[0])
            if distances[0][i]<=threshold], related


# Create corpus, embedding space, and load model
def qna_irm_initialize(data_path: str,
                       model_name: str) -> Tuple[NearestNeighbors,
                                                     SentenceTransformer,
                                                     List[str],
                                                     List[str]]:
    data = load_database(data_path)
    corpus, answers = create_corpus(data)
    index, model, _ = build_qna_sklearn_index(corpus, model_name)
    return index, model, corpus, answers


# Full pipeline
def qna_irm_pipeline(query: str,
                     unrelated_response: str,
                     index: NearestNeighbors,
                     model: SentenceTransformer,
                     answers: List[str],
                     k: int,
                     threshold: float,
                     related_threshold: float) -> Tuple[str, bool, bool]:
    
    contexts, related = retrieve(query, index, model, answers,
                                 k, threshold, related_threshold)
    
    qna_retrieved = False
    qna_related = False

    # Unrelated question response
    if not related:
        return unrelated_response, qna_retrieved, qna_related
    
    # we passed the unrelated check
    qna_related = True
    

    # If we couldn't retrieve but the response wasn't deemed unrelated,
    # return the unrelated response and let caller know qna_related = True
    # The caller will decide if they will use the unrelated response or call the LLM
    if len(contexts) == 0:
        return unrelated_response, qna_retrieved, qna_related
    
    # QnA IRM response
    else:
        qna_retrieved = True
        return contexts[0], qna_retrieved, qna_related
