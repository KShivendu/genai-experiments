import uuid
from typing import List
from qdrant_client import QdrantClient, models as qmodels
from llama_index.llms.openai import OpenAI
from fastembed import TextEmbedding

from food_recommender.models import FoodItem
from food_recommender.utils import synthesize_food_item

likes = ["dosa", "fanta", "croissant", "waffles"]
dislikes = ["virgin mohito"]

menu = ["croissant", "mango", "jalebi"]


class RecommendationEngine:
    def __init__(
        self, category: str, qdrant: QdrantClient, fastembed_model: TextEmbedding
    ) -> None:
        self.collection = f"{category}_preferences"
        self.qdrant = qdrant
        self.embedding_model = fastembed_model

        if self.qdrant.collection_exists(self.collection):
            self.counter = self.qdrant.count(self.collection, exact=True).count
        else:
            self.reset()
            self.counter = 0

    def reset(self):
        self.qdrant.recreate_collection(
            self.collection,
            vectors_config=qmodels.VectorParams(
                size=384, distance=qmodels.Distance.COSINE
            ),
        )

    def _generate_vector(self, model_json: dict):
        embedding_txt = ""
        for k, v in model_json.items():
            embedding_txt += f"{k}: {v}"
        return list(self.embedding_model.passage_embed([embedding_txt]))[0]

    def _insert_preference(self, item: FoodItem, *args, **kwargs):
        model_json: dict = item.model_dump()
        embedding = self._generate_vector(model_json)

        model_json.update(kwargs)

        self.qdrant.upsert(
            self.collection,
            points=[
                qmodels.PointStruct(
                    id=self.counter, payload=model_json, vector=embedding
                )
            ],
        )
        self.counter += 1

    def like(self, item: FoodItem):
        self._insert_preference(item, liked=True)

    def dislike(self, item: FoodItem):
        self._insert_preference(item, liked=False)

    def recommend_from_given(self, items: List[FoodItem], limit: int = 3):
        liked_points, _offset = self.qdrant.scroll(
            self.collection,
            scroll_filter={"must": [{"key": "liked", "match": {"value": True}}]},
        )

        disliked_points, _offset = self.qdrant.scroll(
            self.collection,
            scroll_filter={"must": [{"key": "liked", "match": {"value": False}}]},
        )

        # Insert points in DB so they can be recommended:
        # A bit ugly but this is the best possible thing at the moment.
        query_id = str(uuid.uuid1())
        for item in items:
            self._insert_preference(item, query_id=query_id)

        scored_points = self.qdrant.recommend(
            self.collection,
            positive=[p.id for p in liked_points],
            negative=[p.id for p in disliked_points],
            query_filter={"must": [{"key": "query_id", "match": {"value": query_id}}]},
            with_payload=True,
            strategy="best_score",
        )
        self.qdrant.delete(self.collection, [p.id for p in scored_points])

        return [point.payload["name"] for point in scored_points]


if __name__ == "__main__":
    llm = OpenAI(model="gpt-3.5-turbo")
    qdrant = QdrantClient()
    fastembed_model = TextEmbedding()
    rec_engine = RecommendationEngine("food", qdrant, fastembed_model)

    if rec_engine.counter != len(likes) + len(dislikes):
        rec_engine.reset()
        print("Filling with starter data")
        for food_name in likes:
            food_item = synthesize_food_item(food_name, llm)
            rec_engine.like(food_item)

        for food_name in dislikes:
            food_item = synthesize_food_item(food_name, llm)
            rec_engine.dislike(food_item)

    new_items = [synthesize_food_item(food_name, llm) for food_name in menu]
    recommendations = rec_engine.recommend_from_given(items=new_items)

    print(recommendations)
