import logging
import sys
import json

from typing import List
from pathlib import Path
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.program.openai import OpenAIPydanticProgram

from llama_index.core.query_engine import JSONalyzeQueryEngine
from llama_index.core.tools import QueryEngineTool

from utils import JsonList
from models import ListOfFields, NewField, MaybeGenerateFields, GenerateField

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

menu = JsonList(Path("menu.json"))

# TODO: Load from .env
OPENAI_KEY = "INSERT_YOUR_API_KEY"
llm = OpenAI(model="gpt-3.5-turbo")


def add_numbers(x: int, y: int) -> int:
    """Adds the two numbers together and returns the result."""
    return x + y


def suggest_new_fields(json_sample: dict, user_query: str) -> MaybeGenerateFields:
    prompt_template_str = """
    Sample JSON entry from the list: {json_sample}
    User query "{user_query}"

    Generate new field(s) ONLY if required so the developer can run SQL filters/operations to get a sufficiently good answer. The developer can use normal filters as well as in-built text filters in sqlite"""

    program = OpenAIPydanticProgram.from_defaults(
        output_cls=MaybeGenerateFields,
        llm=llm,
        prompt_template_str=prompt_template_str,
        verbose=True,
    )
    output: MaybeGenerateFields = program(
        json_sample=str(json_sample), user_query=user_query
    )

    return output.new_fields


def extract_new_fields(
    record: dict, fields_to_extract: List[GenerateField]
) -> List[NewField]:
    prompt = """Given the JSON entry {record}, extract the following new fields: {fields_to_extract} and return them as a JSON object"""

    program: ListOfFields = OpenAIPydanticProgram.from_defaults(
        output_cls=ListOfFields,
        llm=llm,
        prompt_template_str=prompt,
        verbose=True,
    )
    result: ListOfFields = program(
        record=record, fields_to_extract=str(fields_to_extract)
    )
    res = {}
    for i, field in enumerate(result.fields):
        field_type = fields_to_extract[i].python_type

        if field_type == "int":
            res[field.k] = int(field.v)
        elif field_type == "float":
            res[field.k] = float(field.v)
        elif field_type == "bool":
            res[field.k] = bool(field.v)
        elif field_type == "str":
            res[field.k] = field.v
        else:
            # TODO: list
            raise Exception("Unknown type of field")

    return res


class CustomJSONQueryEngine(JSONalyzeQueryEngine):
    def __init__(self, list_of_dict, llm):
        json_dump = json.dumps(list_of_dict)
        list_of_dict = json.loads(json_dump.lower())
        super().__init__(
            list_of_dict=list_of_dict, llm=llm, synthesize_response=False, verbose=True
        )

    def _query(self, query_bundle):
        json_records = self._list_of_dict
        json_sample = self._list_of_dict[0]

        fields_to_extract = suggest_new_fields(json_sample, query_bundle.query_str)

        if len(fields_to_extract) > 0:
            new_json_records = []
            for record in json_records:
                new_key_values = extract_new_fields(record, fields_to_extract)
                new_json_records.append({**record, **new_key_values})

            self._list_of_dict = new_json_records
            menu.store_variation(new_json_records)

        return super()._query(query_bundle)


# The problem is GPT models are often wrong with JSONPath. (or it's confused because there are multiple versions of it)
# So I'm going to extend JSONalyze engine instead. It uses SQLite
json_query_engine = CustomJSONQueryEngine(
    list_of_dict=menu.data,
    llm=llm,
)

fns = [add_numbers]
func_tools = [FunctionTool.from_defaults(fn) for fn in fns]
query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=json_query_engine,
        name="restaurant_menu_json_query_tool",
    )
]

agent = OpenAIAgent.from_tools(
    tools=(func_tools + query_engine_tools),
    llm=llm,
    verbose=True,
)


agent.chat_repl()  # Type "exit" to exit

# fields_to_extract = suggest_new_fields(menu.data[0], "how many veg items are there?")
# fields_to_extract = [
#     GenerateField(name="is_veg", python_type="bool", dependencies=["name"])
# ]
# new_key_values = extract_new_fields(
#     menu.data[0], fields_to_extract
# )
