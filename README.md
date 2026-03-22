# LangGraph CamelCase API

## Overview

This project implements a simple yet powerful FastAPI application that uses LangGraph to create a multi-agent system. The core functionality is to receive a question via an API endpoint and intelligently route it to the correct "agent" for processing.

- If the question is math-related, it's sent to a **Math Agent** equipped with tools for addition, subtraction, multiplication, and division.
- If the question is general in nature, it's sent to a **General Agent**.

A **Supervisor Agent** determines where to route the question. The entire workflow is defined and executed by LangGraph, and the application also generates a `graph.png` file to visualize this workflow.

## How to Run

### 1. Prerequisites

- Python 3.8+
- An Azure OpenAI account with a deployed model.

### 2. Installation

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Create a file named `.env` in the root of the project directory. This file will store your Azure OpenAI credentials.

```
OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
AZURE_OPENAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
OPENAI_API_VERSION="YOUR_API_VERSION"
OPENAI_MODEL="YOUR_DEPLOYMENT_NAME"
```

### 4. Running the Application

Once the dependencies are installed and your `.env` file is configured, you can start the FastAPI server using `uvicorn`.

```bash
uvicorn task1:app --reload
```

- `task1`: Refers to the `task1.py` file.
- `app`: Refers to the `FastAPI` instance created in the script (`app = FastAPI(...)`).
- `--reload`: Enables hot-reloading, so the server will restart automatically after you make code changes.

The server will be available at `http://127.0.0.1:8000`.

### 5. Generating the Graph Image

The `graph.png` file, which visualizes the agent workflow, is automatically created in the project's root directory every time you start the application.

## API Endpoints

### Health Check

- **Endpoint:** `GET /health`
- **Description:** A simple endpoint to verify that the server is running.
- **Response:**
  ```json
  {
    "status": "ok"
  }
  ```

### Chat

- **Endpoint:** `POST /chat`
- **Description:** The main endpoint for asking questions. The request body should be a JSON object containing an ID and the question.
- **Request Body:**
  ```json
  {
    "id": "some_unique_id",
    "question": "What is 2 + 2?"
  }
  ```
- **Response:** The response includes the final answer from the appropriate agent.

## Code Explained

Here is a detailed breakdown of the `task1.py` script.

---

### Imports

```python
import functools
import json
import os
from typing import Any, List, Optional
# --- Environment variables
from dotenv import load_dotenv
# --- Web framework
from fastapi import FastAPI
# --- For saving the graph image
from IPython.display import Image, display
# --- LangChain/LangGraph libraries
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
# --- For data validation in API requests
from pydantic import BaseModel
from typing_extensions import TypedDict
```
This section imports all the necessary libraries for the application to function.

### `load_dotenv()`
```python
load_dotenv()
```
This function call loads the environment variables from your `.env` file, making them accessible via `os.getenv()`.

### `LogNodeEntry` Decorator
```python
def LogNodeEntry(func):
    """Decorator to print entry and exit for each Graph Node."""
    @functools.wraps(func)
    def Wrapper(state: Any, *args, **kwargs):
        # ... prints node name and state ...
        return result
    return Wrapper
```
This is a helper function (a decorator) that wraps the main graph nodes. It's used for debugging by printing the name of the node being executed and the data (state) flowing into and out of it. This provides a clear log of the graph's execution path.

### LLM Initialization
```python
Llm = AzureChatOpenAI(...)
```
This line initializes the connection to the Azure OpenAI Large Language Model (LLM). It reads the credentials from the environment variables you set in the `.env` file. `temperature=0.5` makes the model's responses more focused and less random.

### State Definition
```python
class State(TypedDict):
    question: str
    answer: Optional[str]
    route: Optional[str]
```
The `State` class defines the "memory" of our graph. As the graph executes, it passes a `State` object between nodes. This object holds the initial question, the final answer, and the `route` decision made by the supervisor.

### Math Tools
```python
@tool
def Add(a: float, b: float) -> float:
    # ...
@tool
def Subtract(a: float, b: float) -> float:
    # ...
# ... and so on for Multiply and Divide
MathTools = [Add, Subtract, Multiply, Divide]
```
These are the tools given to the Math Agent. The `@tool` decorator automatically makes these Python functions available for an LLM to call. When the Math Agent receives a question like "What is 5 times 3?", it knows it can use the `Multiply` tool to get the answer.

### `MathAgent`
```python
MathAgent = create_react_agent(
    model=Llm,
    tools=MathTools,
)
```
This line creates the specialized Math Agent. It's a "ReAct" agent, which means it can Reason and Act. We pass it our LLM and the `MathTools` so it knows what it can do.

### Graph Nodes
Nodes are the fundamental processing units in the graph.

1.  **`SupervisorNode`**:
    ```python
    @LogNodeEntry
    def SupervisorNode(state: State):
        prompt = f"Route this: '{state['question']}'. Reply only with 'math' or 'general'."
        response = Llm.invoke(prompt)
        # ...
        return {"route": final_route}
    ```
    This is the first node to execute. It takes the user's question and asks the LLM to classify it as either `math` or `general`. The decision is stored in the `state` object.

2.  **`MathAgentNode`**:
    ```python
    @LogNodeEntry
    def MathAgentNode(state: State):
        result = MathAgent.invoke({"messages": [("user", state["question"])]})
        return {"answer": result["messages"][-1].content}
    ```
    This node is only executed if the supervisor chooses the "math" route. It passes the user's question to the `MathAgent`, which uses its tools to solve the problem. The final answer is stored in the `state`.

3.  **`GeneralAgentNode`**:
    ```python
    @LogNodeEntry
    def GeneralAgentNode(state: State):
        response = Llm.invoke(state["question"])
        return {"answer": response.content}
    ```
    This node is executed for any non-math question. It simply sends the question directly to the LLM and stores the answer.

### `RouterLogic`
```python
def RouterLogic(state: State):
    if state["route"] == "math":
        return "MathAgentNode"
    return "GeneralAgentNode"
```
This function acts as a conditional branch. After the `SupervisorNode` runs, this function checks the `route` value in the `state` and tells LangGraph which node to execute next.

### Graph Construction
```python
Builder = StateGraph(State)

# Add nodes
Builder.add_node("SupervisorNode", SupervisorNode)
# ...

# Define edges
Builder.add_edge(START, "SupervisorNode")
Builder.add_conditional_edges("SupervisorNode", RouterLogic, ...)
Builder.add_edge("MathAgentNode", END)
Builder.add_edge("GeneralAgentNode", END)

Graph = Builder.compile()
```
This is where the workflow is assembled:
1.  A new graph is initialized with our `State` definition.
2.  All the nodes are added to the graph.
3.  The edges (connections) are defined:
    - The graph `START`s at the `SupervisorNode`.
    - After the supervisor, the `RouterLogic` function is called to decide the next step.
    - Both the `MathAgentNode` and `GeneralAgentNode` lead to the `END` of the graph execution.
4.  `Builder.compile()` creates the final, runnable graph object.

### Graph Visualization
```python
# Save the graph visualization as a PNG file
with open("graph.png", "wb") as f:
    f.write(Graph.get_graph().draw_mermaid_png())
```
This code takes the compiled graph, generates a visual representation of it as a PNG image, and saves it to the `graph.png` file.

### FastAPI Endpoints
```python
class ChatRequest(BaseModel):
    id: str
    question: str

@app.post("/chat")
async def Chat(request: ChatRequest):
    result = Graph.invoke({"question": request.question})
    return {"ok": True, "payload": request, "result": result}
```
This section defines the API. The `@app.post("/chat")` decorator exposes the `Chat` function as an HTTP POST endpoint. When you send a request to `/chat`, this function:
1.  Receives the user's `question`.
2.  Invokes the `Graph` with the question.
3.  Waits for the full graph to execute.
4.  Returns the final state, which contains the answer.
