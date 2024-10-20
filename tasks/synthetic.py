from langchain_core.prompts import ChatPromptTemplate

from langchain_google_vertexai.model_garden import ChatAnthropicVertex

template = ChatPromptTemplate([
    ("system", "You are generating synthetic data. Always create at least 25 items"),
    ("user", "Create list of example notes from high school student. His aim is to collect his tasks, use informal language and you may use some typos") 
])


llm = ChatAnthropicVertex(
    model_name="claude-3-5-sonnet@20240620",
    project='boris001',
    location='us-east5'
)

data = (template | llm).invoke({})