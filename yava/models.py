"""

Define report data structures as pydantic models:


1. Tools - list of tools to use with the report

"""


from typing import Dict

from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, root_validator, ValidationError

class ToolDef(BaseModel):
    tool: Tool
    output: str

class Tools(BaseModel):
    tools: Dict[str, ToolDef]

    def get_tools(self):
        return [value.tool for key,value in self.tools.items()]
    
    def invoke(self, input):
        response = {}
        for call in input.tool_calls:
            name = call['name']
            if call['args'] == {}:
                response[self.tools[name].output] = self.tools[name].tool.func()
            else:
                response[self.tools[name].output] = self.tools[name].tool.invoke(input = call['args'])

        return response
