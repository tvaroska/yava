from typing import List
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate

from langchain_google_vertexai.model_garden import ChatAnthropicVertex

class Tasks(BaseModel):
    notes: List[str]

template = ChatPromptTemplate([
    ("system", "You are generating synthetic data. Always create at least 25 items"),
    ("user", "Create list of example notes from high school student. His aim is to collect his tasks. Easch note should contain multiple tasks, some for today and tomorow, some for multiple days in advance. His tasks should be Use informal language and you may use some typos") 
])


llm = ChatAnthropicVertex(
    model_name="claude-3-5-sonnet@20240620",
    project='boris001',
    location='us-east5'
).with_structured_output(Tasks)

data = (template | llm).invoke({})

pass