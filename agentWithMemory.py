"""
LangGraph + FastAPI + Postgres Memory (Production Ready)

This application:
✔ Maintains conversation memory using Postgres
✔ Routes queries to math or general agent
✔ Uses full conversation context
✔ Exposes APIs for chat + history retrieval
"""

# -------------------- IMPORTS --------------------

import os  # Used to access environment variables like DB credentials
from typing import Dict, List, Optional  # Type annotations for strong typing
from urllib.parse import quote_plus  # Encodes DB username/password safely

from dotenv import load_dotenv  # Loads .env file variables into environment
from fastapi import FastAPI, HTTPException  # Web framework to expose APIs
from fastapi.encoders import jsonable_encoder
from langchain_core.tools import tool  # Decorator to define tools

# LangGraph / LangChain imports
from langchain_openai import AzureChatOpenAI  # Azure OpenAI LLM wrapper
from langgraph.checkpoint.postgres import PostgresSaver  # Persistent memory
from langgraph.graph import (  # Graph builder components
    END,
    START,
    MessagesState,
    StateGraph,
)
from langgraph.prebuilt import create_react_agent  # Tool-based agent
from pydantic import BaseModel  # Used to validate incoming API request body
from typing_extensions import TypedDict  # Used to define structured state

# -------------------- ENV SETUP --------------------

load_dotenv()  # Load environment variables from .env file

# Read DB credentials from environment
DB_USER = os.getenv("DB_USER")  # DB username
DB_PASSWORD = os.getenv("DB_PASSWORD")  # DB password
DB_HOST = os.getenv("DB_HOST")  # DB host (IP or domain)
DB_PORT = os.getenv("DB_PORT", "5432")  # Default port 5432
DB_NAME = os.getenv("DB_NAME", "postgres")  # Default DB name

# Validate required DB config
if not all([DB_USER, DB_PASSWORD, DB_HOST]):
    raise ValueError("Missing DB config")

# Encode username/password (important if contains @, #, etc.)
DB_URI = f"postgresql://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# -------------------- FASTAPI APP --------------------

# Initialize FastAPI application
app = FastAPI(title="LangGraph Memory API")

# -------------------- LLM SETUP --------------------

"""
AzureChatOpenAI:
- This is your LLM (Large Language Model)
- Used in both math and general agents
"""
Llm = AzureChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # API key
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # Endpoint URL
    api_version=os.getenv("OPENAI_API_VERSION"),  # API version
    azure_deployment=os.getenv("OPENAI_MODEL"),  # Model name
    temperature=0.3,  # Controls randomness (lower = more deterministic)
)

# -------------------- STATE DEFINITION --------------------

class State(MessagesState):
    """
    Graph State (VERY IMPORTANT)

    messages:
        - Stores entire conversation history
        - Auto-accumulates via MessagesState

    route:
        - Determines which agent to call
        - Values: "math" or "general"
    """
    route: Optional[str]

# -------------------- TOOLS --------------------

@tool
def add(a: float, b: float) -> float:
    """
    PURPOSE:
        Adds two numbers

    INPUT:
        a (float), b (float)

    OUTPUT:
        float → sum of a and b
    """
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """
    PURPOSE:
        Subtracts b from a

    INPUT:
        a (float), b (float)

    OUTPUT:
        float → result of subtraction
    """
    return a - b

# List of tools used by math agent
TOOLS = [add, subtract]

# -------------------- AGENT --------------------

"""
MathAgent:
- Uses ReAct pattern (Reason + Act)
- Can decide when to use tools (add, subtract)
"""
MathAgent = create_react_agent(
    model=Llm,
    tools=TOOLS
)

# -------------------- GRAPH NODES --------------------

def SupervisorNode(state: State):
    """
    PURPOSE:
        Decide whether query is math or general

    INPUT:
        state["messages"] → full conversation

    PROCESS:
        - Take last user message
        - Ask LLM to classify

    OUTPUT:
        {"route": "math"} or {"route": "general"}
    """

    # Get latest user message
    last = state["messages"][-1]
    last_msg = last["content"] if isinstance(last, dict) else last.content

    # Ask LLM to classify
    prompt = f"Classify: '{last_msg}'. Reply ONLY 'math' or 'general'."

    response = Llm.invoke(prompt)

    return {
        "route": "math" if "math" in response.content.lower() else "general"
    }


def MathAgentNode(state: State):
    """
    PURPOSE:
        Handle math-related queries

    INPUT:
        Full conversation history

    PROCESS:
        - Add system instruction
        - Pass full history to MathAgent
        - Agent decides tool usage

    OUTPUT:
        Updated messages with assistant response
    """

    messages = state["messages"]

    # System prompt ensures context understanding
    system_prompt = {
        "role": "system",
        "content": (
            "You are a math assistant.\n"
            # "Use previous conversation context.\n"
            # "If the user says 'that', refer to the last answer.\n"
            # "Return only the final answer."
        )
    }

    # Call agent with full context
    result = MathAgent.invoke({
        "messages": [system_prompt] + messages
    })

    # Extract final answer
    answer = result["messages"][-1].content

    # Append assistant response to history (via MessagesState reducer)
    return {
        "messages": [
            {"role": "assistant", "content": answer}
        ]
    }


def GeneralAgentNode(state: State):
    """
    PURPOSE:
        Handle general queries

    INPUT:
        Full conversation

    PROCESS:
        - Add system prompt
        - Send entire history to LLM

    OUTPUT:
        Updated messages
    """

    messages = state["messages"]

    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful assistant.\n"
            "Use full conversation context when answering.\n"
            "If the user refers to earlier answers (e.g., 'that'), use the last answer."
        )
    }

    # Call LLM with context
    response = Llm.invoke([system_prompt] + messages)

    return {
        "messages": [
            {"role": "assistant", "content": response.content}
        ]
    }


def Router(state: State):
    """
    PURPOSE:
        Route execution based on classification

    INPUT:
        state["route"]

    OUTPUT:
        Node name (string)
    """
    return "MathAgentNode" if state["route"] == "math" else "GeneralAgentNode"

# -------------------- GRAPH BUILD --------------------

Builder = StateGraph(State)

# Register nodes
Builder.add_node("SupervisorNode", SupervisorNode)
Builder.add_node("MathAgentNode", MathAgentNode)
Builder.add_node("GeneralAgentNode", GeneralAgentNode)

# Start → Supervisor
Builder.add_edge(START, "SupervisorNode")

# Conditional routing
Builder.add_conditional_edges(
    "SupervisorNode",
    Router,
    {
        "MathAgentNode": "MathAgentNode",
        "GeneralAgentNode": "GeneralAgentNode"
    }
)

# End connections
Builder.add_edge("MathAgentNode", END)
Builder.add_edge("GeneralAgentNode", END)

# -------------------- CHECKPOINTER --------------------

"""
PostgresSaver:
- Stores conversation state in DB
- Enables memory across API calls
"""
CheckpointerCtx = PostgresSaver.from_conn_string(DB_URI)

Checkpointer = None  # DB connection instance
Graph = None  # Compiled graph

@app.on_event("startup")
def startup():
    """
    PURPOSE:
        Initialize DB + Graph

    OUTPUT:
        Graph ready for execution
    """
    global Checkpointer, Graph

    Checkpointer = CheckpointerCtx.__enter__()  # Open DB connection
    Checkpointer.setup()  # Create tables if not exist

    Graph = Builder.compile(checkpointer=Checkpointer)  # Compile graph


@app.on_event("shutdown")
def shutdown():
    """
    PURPOSE:
        Close DB connection safely
    """
    CheckpointerCtx.__exit__(None, None, None)

# -------------------- REQUEST MODEL --------------------

class ChatRequest(BaseModel):
    """
    API Request Schema

    id:
        Conversation ID (thread_id)

    question:
        User input text
    """
    id: str
    question: str

# -------------------- API ENDPOINTS --------------------

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    PURPOSE:
        Continue conversation

    INPUT:
        req.id → conversation id
        req.question → user message

    PROCESS:
        - Append new message
        - Load previous memory
        - Execute graph

    OUTPUT:
        Full updated conversation
    """

    result = Graph.invoke(
        {
            "messages": [
                {"role": "user", "content": req.question}
            ]
        },
        config={"configurable": {"thread_id": req.id}},
    )

    return {
        "ok": True,
        "conversation_id": req.id,
        "messages": result["messages"]
    }


@app.get("/chat/{conversation_id}")
async def get_chat(conversation_id: str):
    """
    PURPOSE:
        Retrieve full conversation

    INPUT:
        conversation_id

    OUTPUT:
        Stored messages from DB
    """

    checkpoint = Checkpointer.get({
        "configurable": {"thread_id": conversation_id}
    })
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Conversation not found")

    state = checkpoint.get("channel_values", {})

    return {
        "conversation_id": conversation_id,
        "messages": state.get("messages", [])
    }


@app.get("/chat/{conversation_id}/state")
async def get_state(conversation_id: str):
    """
    PURPOSE:
        Retrieve entire state for a conversation
    """

    checkpoint = Checkpointer.get({
        "configurable": {"thread_id": conversation_id}
    })
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return jsonable_encoder({
        "conversation_id": conversation_id,
        "checkpoint": checkpoint,
        "state": checkpoint.get("channel_values", {})
    })


@app.get("/chat/{conversation_id}/checkpoints")
async def list_checkpoints(conversation_id: str, limit: int = 10):
    """
    PURPOSE:
        List recent checkpoints for a conversation
    """

    config = {"configurable": {"thread_id": conversation_id}}
    checkpoints = []
    for item in Checkpointer.list(config, limit=limit):
        checkpoints.append(item._asdict())

    return jsonable_encoder({
        "conversation_id": conversation_id,
        "count": len(checkpoints),
        "checkpoints": checkpoints,
    })


@app.get("/health")
def health():
    """
    PURPOSE:
        Health check endpoint
    """
    return {"status": "ok"}