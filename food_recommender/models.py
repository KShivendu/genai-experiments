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
