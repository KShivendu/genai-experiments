from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core.llms.llm import LLM

from .models import FoodItem


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
