"""

    Reflection subgraph

"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables.base import RunnableSequence
from langchain_google_vertexai import ChatVertexAI, VertexAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages

from .prompts import prompts

writer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompts.prompts["reflection/writer"].prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
writer_model = ChatVertexAI(model="gemini-1.5-pro-001")
default_writer = writer_prompt | writer_model

reviewer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompts.prompts["reflection/reviewer"].prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reviewer_model = ChatVertexAI(model="gemini-1.5-pro-001")
default_reviewer = reviewer_prompt | reviewer_model

check_essay_prompt = PromptTemplate.from_template(
    prompts.prompts["reflection/check"].prompt
)
check_essay_model = VertexAI(model_name="gemini-1.5-flash-001")
default_check_essay = check_essay_prompt | check_essay_model


class ReflectionState(TypedDict):
    final: str
    previous: str
    counter: int
    format: str
    topic: str
    context: str
    messages: Annotated[AnyMessage, add_messages]


class Reflection(CompiledStateGraph):
    """

    Agent using reflection pattern to crete writing in specified format
    (eg. essay, homework, ...) on topic with additional context.

    Input:
        topic: str
        format: str = 'essay'
        context: str = None

    Output:
        final -> the best response
        previous -> second last version of the writing
        messages -> whole interaction

    """

    @classmethod
    def create(
        cls,
        writer: RunnableSequence = default_writer,
        reviewer: RunnableSequence = default_reviewer,
        check_essay: RunnableSequence = default_check_essay,
        max_count: int = 5,
    ):

        async def initialize(state: ReflectionState) -> ReflectionState:
            response = {}
            if "messages" in state and state["messages"]:
                return response

            if "format" not in state:
                document_format = "essay"
                response["format"] = "essay"
            else:
                document_format = state["format"]

            if "topic" not in state:
                raise ValueError()
            else:
                topic = state["topic"]

            message = f"Write an {document_format} about {topic}."

            if "context" in state:
                message = (
                    message
                    + f'\nAdditional context for your writing: f{state["context"]}'
                )

            response["messages"] = [HumanMessage(message)]

            return response

        async def generation_node(state: ReflectionState) -> ReflectionState:
            if "counter" in state and state["counter"]:
                idx = state["counter"] + 1
            else:
                idx = 1

            response = {}
            response["counter"] = idx
            response["messages"] = await writer.ainvoke(state["messages"])

            return response

        async def reflection_node(state: ReflectionState) -> ReflectionState:
            # Other messages we need to adjust
            cls_map = {"ai": HumanMessage, "human": AIMessage}
            # First message is the original user request.
            # We hold it the same for all nodes
            translated = [state["messages"][0]] + [
                cls_map[msg.type](content=msg.content)
                for msg in state["messages"][1:]
            ]
            res = await reviewer.ainvoke(translated)
            # We treat the output of this as human feedback for the generator

            response = {}
            response["messages"] = [HumanMessage(content=res.content)]
            if state["final"]:
                response["previous"] = state["final"]
            response["final"] = state["messages"][-1].content
            return response

        def is_essay(state):
            essay = state["messages"][-1].content
            decision = check_essay.invoke(
                {
                    "format": state["format"],
                    "topic": state["topic"],
                    "essay": essay,
                }
            )

            if decision.startswith("Yes"):
                return "reflection"
            else:
                return END

        def check(state: ReflectionState) -> str:
            if "counter" not in state:
                state["counter"] = 1

            if state["counter"] > max_count:
                return END
            else:
                return "generate"

        builder = StateGraph(ReflectionState)
        builder.add_node("initialize", initialize)
        builder.add_node("generate", generation_node)
        builder.add_node("reflection", reflection_node)

        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "generate")

        builder.add_conditional_edges("generate", is_essay)
        builder.add_conditional_edges("reflection", check)
        graph = builder.compile()

        return graph
