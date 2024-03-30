from pydantic import BaseModel, Field
from typing import List, Literal


class NewField(BaseModel):
    """Key and value for new extracted field"""

    k: str
    v: str


class ListOfFields(BaseModel):
    """List of new fields (key + values) extracted from existing fields"""

    fields: List[NewField]


class GenerateField(BaseModel):
    """New field to be generated for extraction logic to answer user's query"""

    name: str = Field(description="Keep the name short and to the point")
    python_type: Literal["bool", "int", "str", "float"]  # TODO: no lists for now
    existing_fields_to_refer: List[str] = Field(
        description="List of existing field names that this field depends on. Remember that you need to process more entries which might have somewhat different edge cases. So the more dependencies you choose, the more reliable the extraction will be. MUST be more than 1.",
        default=[],
    )


class MaybeGenerateFields(BaseModel):
    """Generate new fields if they are required to answer the user's query with SQL operations"""

    chain_of_thought: str = Field(
        description="Think step by step if there's any simple way to answer the query with SQL operations on existing field values. This approach doesn't have to be perfect."
    )

    new_fields: List[GenerateField] = Field(
        description="Only if it cannot be answered with existing fields"
    )
