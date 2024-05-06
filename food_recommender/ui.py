import json
import gradio as gr
from typing import Tuple, Dict
from paddleocr import PaddleOCR

from qdrant_client import QdrantClient
from fastembed import TextEmbedding

from llama_index.llms.openai import OpenAI
from food_recommender.utils import extract_food_items, synthesize_food_item
from food_recommender.main import RecommendationEngine

ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
llm = OpenAI(model="gpt-3.5-turbo")
rec_engine = RecommendationEngine("food", QdrantClient(), TextEmbedding())


def run_ocr(img_path):
    result = ocr.ocr(img_path, cls=True)[0]
    return "\n".join([line[1][0] for line in result])


def recommend(
    likes_str, dislikes_str, img_path
) -> Tuple[str, str, str, Dict[str, float]]:
    likes = [c.strip() for c in likes_str.split(",")]
    dislikes = [c.strip() for c in dislikes_str.split(",")]
    print(likes, dislikes)

    rec_engine.reset()
    for food_name in likes:
        rec_engine.like(synthesize_food_item(food_name, llm))
    for food_name in dislikes:
        rec_engine.dislike(synthesize_food_item(food_name, llm))

    ocr_text = run_ocr(img_path)

    food_names = extract_food_items(ocr_text, llm)
    food_items = [synthesize_food_item(name, llm) for name in food_names]

    print("New food items from menu", food_items)

    recommendations = rec_engine.recommend_from_given(food_items)
    print(recommendations)

    return (
        ocr_text,
        json.dumps(food_names, indent=4),
        json.dumps([item.model_dump() for item in food_items], indent=4),
        recommendations,
    )


title = "Food recommender"
description = "Food recommender by <a href='https://kshivendu.dev/bio'>KShivendu</a> using Qdrant Recommendation API + OpenAI Function calling + FastEmbed embeddings"
article = "<a href='https://github.com/KShivendu/rag-cookbook'>Github Repo</a></p>"
examples = [
    [
        "fanta, waffles, chicken biriyani, most of indian food",
        "virgin mojito, any pork dishes",
        "food_recommender/sf-menu3.jpg",
    ]
]

step1_ocr = gr.Text(label="OCR Output")
step2_extraction = gr.Code(language="json", label="Extracted food items")
step3_enrichment = gr.Code(language="json", label="Enriched food items")
step4_recommend = gr.Label(label="Recommendations")

app = gr.Interface(
    fn=recommend,
    inputs=[
        gr.Textbox(label="Likes (comma seperated)"),
        gr.Textbox(label="Dislikes (comma seperated)"),
        gr.Image(type="filepath", label="Input", width=20),
    ],
    outputs=[step1_ocr, step2_extraction, step3_enrichment, step4_recommend],
    title=title,
    description=description,
    article=article,
    examples=examples,
)
app.queue(max_size=10)
app.launch(debug=True)
