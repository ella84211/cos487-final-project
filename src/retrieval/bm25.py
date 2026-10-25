"""
BM25 retrieval system
Contains the retrieve() function, which retrieves documents from an index using the BM25 algorithm.
"""

import math

from src.index import load_index, get_term_frequency, get_num_documents, get_document_length, get_average_document_length, split_words, PREPROCESSING_OPTIONS


indexes = {}


def retrieve(query: str, preprocessing: str, k: int, k_1: float = 1.2, b: float = 0.75) -> dict:
    """
    Retrieves documents that match the given query from the given index
    Returns a dictionary with the top-k retrieval results, sorted descending by score

    Args:
        query (str): The query to search for; assumes the preprocessing has already been applied to the query
        preprocessing (str): The preprocessing option used to create the index
        k (int): The number of results to return
        k_1 (float): The k_1 parameter for BM25
        b (float): The b parameter for BM25
    
    Returns:
        dict: A dictionary containing document IDs as keys with their scores as values
    """

    index = load_index(preprocessing)

    docs = {}
    average_doc_length = get_average_document_length(index)
    terms = split_words(query)
    for term in terms:
        tf_map = get_term_frequency(index, term)
        docs_with_term = len(tf_map)
        idf = math.log((get_num_documents(index) - docs_with_term + 0.5) / (docs_with_term + 0.5) + 1)
        for doc_id, count in tf_map.items():
            term_score = idf * (count * (k_1 + 1)) / (count + k_1 * (1 - b + b * get_document_length(index, doc_id) / average_doc_length))
            if doc_id not in docs:
                docs[doc_id] = 0
            docs[doc_id] += term_score

    return dict(sorted(docs.items(), key=lambda item: item[1], reverse=True)[:k])
