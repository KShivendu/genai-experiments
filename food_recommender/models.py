from typing import List
from pydantic import BaseModel, Field


class FoodItem(BaseModel):
    """Details of a food item or dish"""

    name: str
    ingredients_and_approach: str = Field(
        description="Short description of the ingredients and the overall approach to cook"
    )
    taste_and_texture: str = Field(
        description="Short description of how does it taste and feel in mouth"
    )
    is_vegetarian: bool  # TODO: Can be ambiguous. LLM could ask more question while asking for preferences?


class ExtractedFoodName(BaseModel):
    """Food items / dishes extracted from output of an OCR"""

    food_names: List[str] = Field(
        description="Each item must be an actual food item because OCR data can have lot of noise. If doubtful, discard. I don't want False positives. You can also fix small typos based on your understanding"
    )
