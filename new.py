import json
import os
from typing import Any, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from typing_extensions import TypedDict

load_dotenv()

llm = AzureChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    deployment_name=os.getenv("OPENAI_MODEL"),
    temperature=0.5,
)

app = FastAPI(title="LangGraph Chat API", version="1.0.0")


# import json
class State(TypedDict):
    question: str
    answer: Optional[str] = None
    route: Optional[str] = "general"


@tool
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract two numbers"""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


math_tools = [add, subtract, multiply, divide]


math_agent = create_react_agent(
    model=llm,
    tools=math_tools,
    # state_modifier="You are a helpful math assistant. Always show your work.",
    debug=True,
)

# math_executor = AgentExecutor(agent=math_agent, tools=math_tools, verbose=True)


def generalAgent(state: State) -> State:
    question = state["question"]
    response = llm.invoke(question)
    try:
        response_json = response.model_dump()
        print(f"LLM response (parsed as JSON): {response_json}")
    except json.JSONDecodeError:
        print(f"LLM response is not valid JSON: {response.content}")
    # print(f"LLM response: {json.load(response)}")
    return {"answer": response.content}


def supervisorNode(state: State):

    question = state["question"]

    prompt = f"""
You are a router agent.

If the question requires calculation return: math
If it is a general question return: general

Question: {question}

Answer only with: math or general
"""

    response = llm.invoke(prompt)

    route = response.content.strip().lower()

    return {"route": route}


def mathAgentNode(state: State):

    question = state["question"]

    result = math_agent.invoke({"messages": [("user", question)]})
    for message in result["messages"]:
        print(f"\n\n-------------Message {message}")
    print(f"Math agent result: {result}")   
    print(f"Math agent messages: {len(result['messages'])} messages, \n\nlast message: {result['messages'][-2]} \n\nsecond last message content: {result['messages'][-2].content}")
    return {"answer":  result["messages"][-1].content}


def route(state: State):

    if state["route"] == "math":
        return "math_agent"

    return "general_agent"


builder = StateGraph(State)
builder.add_node("supervisor", supervisorNode)
builder.add_node("math_agent", mathAgentNode)
builder.add_node("general_agent", generalAgent)


builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor", route, {"math_agent": "math_agent", "general_agent": "general_agent"}
)

builder.add_edge("math_agent", END)
builder.add_edge("general_agent", END)

graph = builder.compile()

graph = builder.compile()


class ChatRequest(BaseModel):
    id: str
    question: str


@app.post("/chat")
async def chat(request: ChatRequest):
    # body = request

    print(f"Data {request} received at /chat endpoint.")
    # result = graph.invoke(request)
    result = graph.invoke({"question": request.question})

    return {"ok": True, "payload": request, "result": result}


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health endpoint for probes."""

    return {"status": "ok helo dhuvarakesan"}
