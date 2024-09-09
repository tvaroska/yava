"""

    Generate and review plans


"""
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatLiteLLMRouter

from yava.models import router
from yava.tools import xmlfind
from yava.prompts import prompts

planning_prompt = ChatPromptTemplate([
        ("system", prompts["planning/planning"]),
        ("user", "{user_query}")
    ])

planning_model = ChatLiteLLMRouter(router=router, model_name="gemini-1.5-flash")
planning_chain = planning_prompt | planning_model

async def generate_plans(user_query, n=5, chain=planning_chain):

    tasks = [chain.ainvoke({"user_query": user_query}) for _ in range(n)]
    results = await asyncio.gather(*tasks)
    plans = [xmlfind(item.content, "PLAN") for item in results]
    steps = [xmlfind(item.content, "STEPS") for item in results]

    return plans, steps
