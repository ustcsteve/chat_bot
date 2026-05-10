import csv
import os
import operator
from typing import List, Annotated
from typing_extensions import NotRequired, TypedDict

from langchain_aws import BedrockEmbeddings, ChatBedrock
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = BedrockAgentCoreApp()

# Configuration
MEMORY_ID = os.getenv("BEDROCK_AGENTCORE_MEMORY_ID")
REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = "us.amazon.nova-lite-v1:0"

# --- 1. Knowledge Base (FAISS) ---
def load_faq_csv(path: str) -> List[Document]:
    docs = []
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            docs.append(Document(page_content=f"Q: {row['question']}\nA: {row['answer']}"))
    return docs

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name=REGION)
faq_docs = load_faq_csv("./wf_checking.csv")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(faq_docs)
faq_store = FAISS.from_documents(chunks, embeddings)

# --- 2. Banking Tools ---
@tool
def get_balance(account_type: str = "checking") -> str:
    """Retrieve the current balance for checking or savings accounts."""
    mock_balances = {"checking": "$4,250.60", "savings": "$12,100.00"}
    return f"Your {account_type} balance is {mock_balances.get(account_type.lower(), 'not found')}."

@tool
def search_banking_policies(query: str) -> str:
    """Search for policies regarding overdrafts, transfers, or account limits."""
    results = faq_store.similarity_search(query, k=2)
    return "\n\n".join([doc.page_content for doc in results]) if results else "No policy found."

TOOLS = [get_balance, search_banking_policies]
TOOL_MAP = {t.name: t for t in TOOLS}

# --- 3. LangGraph Logic with Memory Logging ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

SYSTEM_PROMPT = "You are a Wells Fargo Assistant. Use the conversation history to answer follow-up questions."

def get_llm():
    return ChatBedrock(model_id=MODEL_ID, region_name=REGION, system=SYSTEM_PROMPT).bind_tools(TOOLS)

def call_model(state: AgentState):    
    response = get_llm().invoke(state["messages"])
    return {"messages": [response]}
# def call_model(state: AgentState, config: dict = None): # Added default None to prevent TypeError
#     """
#     LLM Node: Logs current message history. 
#     The 'config' is automatically injected by LangGraph if provided during invoke().
#     """
#     # Safe retrieval of thread_id
#     config = config or {}
#     thread_id = config.get("configurable", {}).get("thread_id", "no_thread_found")
    
#     print("\n" + "="*50)
#     print(f"DEBUG: MEMORY LOG (Thread: {thread_id})")
#     print(f"Total messages in state: {len(state['messages'])}")
    
#     for i, msg in enumerate(state['messages']):
#         role = "User" if isinstance(msg, HumanMessage) else "Assistant"
#         snippet = msg.content[:60]
#         print(f"  {i}. [{role}]: {snippet}...")
#     print("="*50 + "\n")
    
#     response = get_llm().invoke(state["messages"])
#     return {"messages": [response]}

def execute_tools(state: AgentState):
    last_message = state["messages"][-1]
    tool_messages = []
    for tool_call in last_message.tool_calls:
        output = TOOL_MAP[tool_call["name"]].invoke(tool_call["args"])
        tool_messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
    return {"messages": tool_messages}

def router(state: AgentState):
    if hasattr(state["messages"][-1], "tool_calls") and state["messages"][-1].tool_calls:
        return "tools"
    return "end"

# Persistence layer
memory_checkpointer = MemorySaver()

builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", execute_tools)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", router, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")

AGENT_GRAPH = builder.compile(checkpointer=memory_checkpointer)

# --- 4. Bedrock Entrypoint ---

@app.entrypoint
def invoke(payload, context):
    prompt = payload.get("prompt", "")
    session_id = context.session_id or "default_session"
    
    # We pass the session_id to 'thread_id' so MemorySaver can retrieve the right history
    config = {"configurable": {"thread_id": session_id}}
    
    result = AGENT_GRAPH.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config
    )
    
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return {"response": msg.content}
    return {"response": "I encountered an error retrieving that information."}

if __name__ == "__main__":
    app.run()

# if __name__ == "__main__":
#     prompt = input("Enter your prompt: ")
#     session_id = input("Enter your session ID: ")
    
#     # We pass the session_id to 'thread_id' so MemorySaver can retrieve the right history
#     config = {"configurable": {"thread_id": session_id}}
    
#     result = AGENT_GRAPH.invoke(
#         {"messages": [HumanMessage(content=prompt)]},
#         config=config
#     )
    
#     print(result['messages'][-1].content)