from typing import List
from qdrant_client.http.models import (
    VectorParams,
    SearchRequest,
    NamedVector,
    ScoredPoint,
)
from qdrant_client import QdrantClient
from utils import (
    generate_embeddings,
    rrf,
    rank_list,
    IMAGE_EMBEDDING_DIMS,
    TEXT_EMBEDDING_DIMS,
)

QDRANT_COLLECTION = "multimodal-reranking"

client = QdrantClient(mode=":inmemory:")

client.create_collection(
    collection_name=QDRANT_COLLECTION,
    vectors_config={
        "image": VectorParams(
            size=IMAGE_EMBEDDING_DIMS,
        ),
        "text": VectorParams(  # for re-ranking
            size=TEXT_EMBEDDING_DIMS,
        ),
    },
)


def search(query: str) -> List[ScoredPoint]:
    query_vector = generate_embeddings(query)[0]

    # TODO: Can oversample?
    search_results = client.search_batch(
        collection_name=QDRANT_COLLECTION,
        requests=[
            SearchRequest(
                vector=NamedVector(name="image", vector=query_vector),
                limit=10,
            ),
            SearchRequest(
                vector=NamedVector(name="text", vector=query_vector),
                limit=10,
            ),
        ],
    )

    # Reranking:
    image_rank_list, text_rank_list = [rank_list(res.points) for res in search_results]
    reranked_results = rrf(
        [image_rank_list, text_rank_list], alpha=60, default_rank=1000
    )

    # Fetch payload for the points:
    point_ids, point_scores = zip(*reranked_results)
    points = client.retrieve(
        collection_name=QDRANT_COLLECTION, ids=point_ids, with_payload=True
    )

    return [
        ScoredPoint(
            id=point.id,
            payload=point.payload,
            vector=point.vector,
            shard_key=point.shard_key,
            score=point_scores[i],
        )
        for i, point in enumerate(points)
    ]


if __name__ == "__main__":
    pass
