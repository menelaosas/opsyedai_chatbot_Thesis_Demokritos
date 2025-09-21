import re
from PyPDF2 import PdfReader
from typing import List, Dict, Tuple, Any
import numpy.typing as npt
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import faiss
import numpy as np


# Chunk the pdf into sections based on chapter labeling
def split_pdf_by_sections_skip_intro(pdf_path: str,
                                     skip_pages: int = 4) -> List[Dict[str, str]]:
    reader = PdfReader(pdf_path)
    
    # Skip the first `skip_pages` (e.g., TOC)
    pages = reader.pages[skip_pages:]
    full_text = "\n".join([page.extract_text() for page in pages if page.extract_text()])

    # Match section headers like: "2 Εγγραφή", "2.1 Σύνδεση", "5.3 Τα ραντεβού μου"
    section_pattern = r"\n(?P<header>\d{1,2}(\.\d+)?(\.\d+)?\s+[^\n]+)\n"
    matches = list(re.finditer(section_pattern, full_text))

    sections = []
    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i+1].start() if i+1 < len(matches) else len(full_text)
        header = matches[i].group("header").strip()
        content = full_text[start:end].strip()
        sections.append({
            "section": header,
            "content": content
        })

    return sections


# Build scikit-learn NearestNeighbors index
def build_pdf_sklearn_index(chunks: List[str],
                            model_name: str) -> Tuple[NearestNeighbors,
                                                  SentenceTransformer,
                                                  npt.NDArray[Any]]:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    # normalize all embeddings before indexing
    embeddings = normalize(embeddings, axis=1)

    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings))
    return index, model, embeddings


# Retrieve relevant pdf chapters
def retrieve_relevant_chunks_l2(query: str,
                                index: NearestNeighbors,
                                model: SentenceTransformer,
                                titles: List[str],
                                k: int,
                                threshold: float) -> List[str]:
    # normalize query vector
    query_embedding = normalize(model.encode([query]), axis=1)
    D, I = index.search(np.array(query_embedding), k)

    results = []
    for distance, idx in zip(D[0], I[0]):
        if distance >= threshold:
            results.append(titles[idx])

    return results


# full pipeline
def pdf_irm_pipeline(query: str,
                     answer: str,
                     index: NearestNeighbors,
                     model: SentenceTransformer,
                     titles: List[str],
                     k: int,
                     threshold: float,
                     suggestion: str) -> Tuple[str, bool, bool]:
    relevant_chunks = retrieve_relevant_chunks_l2(query, index, model, titles, k, threshold)
    
    # Does the pdf retrieval add any context from the pdf?
    if len(relevant_chunks) > 0:
        answer = answer + suggestion + ', '.join(relevant_chunks)
        pdf_retrieved = True
    else:
        pdf_retrieved = False

    return answer, pdf_retrieved