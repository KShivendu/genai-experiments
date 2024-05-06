from typing import List
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core.llms.llm import LLM

from food_recommender.models import FoodItem, ExtractedFoodName


def synthesize_food_item(food_name: str, llm: LLM) -> FoodItem:
    prompt = """Tell me what you know about the food item / dish '{food_name}' and return as a JSON object"""

    program = OpenAIPydanticProgram.from_defaults(
        output_cls=FoodItem,
        llm=llm,
        prompt_template_str=prompt,
        verbose=True,
    )
    result: FoodItem = program(food_name=food_name)
    return result


def extract_food_items(text: str, llm: LLM) -> List[str]:
    prompt = """You're world's best bot for parsing data from noisy OCR output on images. Exact all the food items that you can find in this menu. Please avoid parsing things that are names of the sections. Generally section comes before the food items: '{text}'. Return result as a JSON object"""

    program = OpenAIPydanticProgram.from_defaults(
        output_cls=ExtractedFoodName,
        llm=llm,
        prompt_template_str=prompt,
        verbose=True,
    )
    result: ExtractedFoodName = program(text=text)
    return [item.lower() for item in result.food_names]
