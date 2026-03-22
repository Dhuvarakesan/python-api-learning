import functools
import json
import os
from typing import Any, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from IPython.display import Image, display

# LangChain/LangGraph Imports
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from typing_extensions import TypedDict

load_dotenv()

# --- 1. Logging Decorator ---
def LogNodeEntry(func):
    """Decorator to print entry and exit for each Graph Node."""
    @functools.wraps(func)
    def Wrapper(state: Any, *args, **kwargs):
        node_name = func.__name__
        print(f"\n>>> [ENTRY] Node: {node_name}")
        print(f"    Input State: {state}")
        
        result = func(state, *args, **kwargs)
        
        print(f"<<< [EXIT]  Node: {node_name}")
        print(f"    Produced Output: {result}")
        return result
    return Wrapper

# --- 2. LLM Initialization ---
Llm = AzureChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_deployment=os.getenv("OPENAI_MODEL"),
    temperature=0.5,
)

app = FastAPI(title="LangGraph CamelCase API", version="1.0.0")

# --- 3. State Definition ---
class State(TypedDict):
    question: str
    answer: Optional[str]
    route: Optional[str]

# --- 4. Math Tools (All decorated) ---
@tool
def Add(a: float, b: float) -> float:
    """Add two numbers"""
    print(f"Performing Addition: {a} + {b}")
    return a + b

@tool
def Subtract(a: float, b: float) -> float:
    """Subtract two numbers"""
    print(f"Performing Subtraction: {a} - {b}")
    return a - b

@tool
def Multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    print(f"Performing Multiplication: {a} * {b}")
    return a * b

@tool
def Divide(a: float, b: float) -> float:
    """Divide two numbers. Handles zero division."""
    if b == 0: return "Error: Cannot divide by zero."
    return a / b

MathTools = [Add, Subtract, Multiply, Divide]

# --- 5. Internal ReAct Agent ---
MathAgent = create_react_agent(
    model=Llm,
    tools=MathTools,
    debug=False # Set to True for internal tool logs
)

# --- 6. Nodes with CamelCase and Logging Decorator ---

@LogNodeEntry
def SupervisorNode(state: State):
    prompt = f"Route this: '{state['question']}'. Reply only with 'math' or 'general'."
    response = Llm.invoke(prompt)
    route_choice = response.content.strip().lower()
    # Logic to ensure clean routing
    final_route = "math" if "math" in route_choice else "general"
    return {"route": final_route}

@LogNodeEntry
def MathAgentNode(state: State):
    # Pass the question as a message list to the internal graph
    result = MathAgent.invoke({"messages": [("user", state["question"])]})
    # Extract final AI content from the message history
    return {"answer": result["messages"][-1].content}

@LogNodeEntry
def GeneralAgentNode(state: State):
    response = Llm.invoke(state["question"])
    return {"answer": response.content}

def RouterLogic(state: State):
    """Determines which node to visit next based on state['route']"""
    if state["route"] == "math":
        return "MathAgentNode"
    return "GeneralAgentNode"

# --- 7. Graph Construction ---
Builder = StateGraph(State)

Builder.add_node("SupervisorNode", SupervisorNode)
Builder.add_node("MathAgentNode", MathAgentNode)
Builder.add_node("GeneralAgentNode", GeneralAgentNode)

Builder.add_edge(START, "SupervisorNode")

Builder.add_conditional_edges(
    "SupervisorNode",
    RouterLogic,
    {
        "MathAgentNode": "MathAgentNode",
        "GeneralAgentNode": "GeneralAgentNode"
    }
)

Builder.add_edge("MathAgentNode", END)
Builder.add_edge("GeneralAgentNode", END)

Graph = Builder.compile()

# Save the graph visualization as a PNG file
with open("graph.png", "wb") as f:
    f.write(Graph.get_graph().draw_mermaid_png())

# --- 8. FastAPI Endpoints ---
class ChatRequest(BaseModel):
    id: str
    question: str

@app.post("/chat")
async def Chat(request: ChatRequest):
    print(f"\n--- Incoming API Request: {request.id} ---")
    # Invoke the graph
    result = Graph.invoke({"question": request.question})
    return {"ok": True, "payload": request, "result": result}

@app.get("/health")
def Health():
    return {"status": "ok"}