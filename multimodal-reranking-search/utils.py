import numpy as np

from typing import List, Tuple
from qdrant_client.http.models import ScoredPoint
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("clip-ViT-B-32")

IMAGE_EMBEDDING_DIMS = 512
TEXT_EMBEDDING_DIMS = 512


def generate_embeddings(*args, **kwargs):
    return model.encode(*args, **kwargs)


def rank_list(search_result: List[ScoredPoint]):
    return [(point.id, rank + 1) for rank, point in enumerate(search_result)]


def rrf(rank_lists, alpha=60, default_rank=1000):
    """
    Optimized Reciprocal Rank Fusion (RRF) using NumPy for large rank lists.

    :param rank_lists: A list of rank lists. Each rank list should be a list of (item, rank) tuples.
    :param alpha: The parameter alpha used in the RRF formula. Default is 60.
    :param default_rank: The default rank assigned to items not present in a rank list. Default is 1000.
    :return: Sorted list of items based on their RRF scores.
    """
    # Consolidate all unique items from all rank lists
    all_items = set(item for rank_list in rank_lists for item, _ in rank_list)

    # Create a mapping of items to indices
    item_to_index = {item: idx for idx, item in enumerate(all_items)}

    # Initialize a matrix to hold the ranks, filled with the default rank
    rank_matrix = np.full((len(all_items), len(rank_lists)), default_rank)

    # Fill in the actual ranks from the rank lists
    for list_idx, rank_list in enumerate(rank_lists):
        for item, rank in rank_list:
            rank_matrix[item_to_index[item], list_idx] = rank

    # Calculate RRF scores using NumPy operations
    rrf_scores = np.sum(1.0 / (alpha + rank_matrix), axis=1)

    # Sort items based on RRF scores
    sorted_indices = np.argsort(-rrf_scores)  # Negative for descending order

    # Retrieve sorted items
    sorted_items = [
        (list(item_to_index.keys())[idx], rrf_scores[idx]) for idx in sorted_indices
    ]

    return sorted_items
