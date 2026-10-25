"""
TF-IDF retrieval system
Contains the retrieve() function, which retrieves documents from an index using the TF-IDF algorithm.
"""

import math

from src.index import load_index, get_term_frequency, get_num_documents, split_words, PREPROCESSING_OPTIONS


indexes = {}


def retrieve(query: str, preprocessing: str, k: int) -> dict:
    """
    Retrieves documents that match the given query from the given index
    Returns a dictionary with the top-k retrieval results, sorted descending by score

    Args:
        query (str): The query to search for; assumes the preprocessing has already been applied to the query
        preprocessing (str): The preprocessing option used to create the index
        k (int): The number of results to return
    
    Returns:
        dict: A dictionary containing document IDs as keys with their scores as values
    """

    index = load_index(preprocessing)

    docs = {}
    terms = split_words(query)
    for term in terms:
        tf_map = get_term_frequency(index, term)
        docs_with_term = len(tf_map)
        idf = math.log(get_num_documents(index) / docs_with_term) if docs_with_term > 0 else 0
        for doc_id, count in tf_map.items():
            term_score = idf * count
            if doc_id not in docs:
                docs[doc_id] = 0
            docs[doc_id] += term_score

    return dict(sorted(docs.items(), key=lambda item: item[1], reverse=True)[:k])
