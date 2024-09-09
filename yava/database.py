"""

Define report data structures as pydantic models:

1. Tools - list of tools to use with the report
2. Facts - database of known facts to use for content generation

"""

from typing import Any, Dict, List

from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import Tool


class ToolDef(BaseModel):
    """
    Definition of tool
    """

    tool: Tool
    output: str


class Tools(BaseModel):
    """
    All tools for one report
    """

    tools: Dict[str, ToolDef]

    def get_tools(self):
        return [value.tool for key, value in self.tools.items()]

    def invoke(self, model_response):
        response = {}
        for call in model_response.tool_calls:
            name = call["name"]
            if call["args"] == {}:
                response[self.tools[name].output] = self.tools[name].tool.func()
            else:
                response[self.tools[name].output] = self.tools[
                    name
                ].tool.invoke(model_response=call["args"])

        return response


class Fact(BaseModel):
    name: str
    description: str
    value: Any = None


class Facts(BaseModel):
    # TODO: dictionary
    facts: List[Fact]

    @classmethod
    def from_dict(cls, data):
        facts = [
            Fact(name=key, description=value) for key, value in data.items()
        ]
        return cls(facts=facts)
