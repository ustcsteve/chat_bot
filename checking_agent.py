"""
Complete Production AI Agent
Memory + Secure Code Execution + Full Observability
Refactored to use langgraph StateGraph orchestration.
"""
import csv
import os
from typing import Any, List

from typing_extensions import NotRequired, TypedDict

from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig,
    RetrievalConfig,
)
from langchain_aws import BedrockEmbeddings
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from strands import Agent

from langgraph.graph import END, START, StateGraph
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from strands_tools.code_interpreter import AgentCoreCodeInterpreter

app = BedrockAgentCoreApp()

MEMORY_ID = os.getenv("BEDROCK_AGENTCORE_MEMORY_ID")
REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = "us.amazon.nova-lite-v1:0"

_agent: Agent | None = None


class AgentState(TypedDict):
    prompt: str
    actor_id: str
    session_id: str
    response: NotRequired[str]
    error: NotRequired[str]


class AgentOutput(TypedDict):
    response: str

def load_faq_csv(path: str) -> List[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"].strip()
            a = row["answer"].strip()
            docs.append(Document(page_content=f"Q: {q}\nA: {a}"))
    return docs


docs = load_faq_csv("./lauki_qna.csv")
emb = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name=REGION
)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
chunks = splitter.split_documents(docs)
faq_store = FAISS.from_documents(chunks, emb)


@tool
def search_faq(query: str) -> str:
    """Search the FAQ knowledge base for relevant information.
    Use this tool when the user asks questions about products, services, or policies.
    
    Args:
        query: The search query to find relevant FAQ entries
        
    Returns:
        Relevant FAQ entries that might answer the question
    """
    results = faq_store.similarity_search(query, k=3)
    
    if not results:
        return "No relevant FAQ entries found."
    
    context = "\n\n---\n\n".join([
        f"FAQ Entry {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(results)
    ])
    
    return f"Found {len(results)} relevant FAQ entries:\n\n{context}"


@tool
def search_detailed_faq(query: str, num_results: int = 5) -> str:
    """Search the FAQ knowledge base with more results for complex queries.
    Use this when the initial search doesn't provide enough information.
    
    Args:
        query: The search query
        num_results: Number of results to retrieve (default: 5)
        
    Returns:
        More comprehensive FAQ entries
    """
    results = faq_store.similarity_search(query, k=num_results)
    
    if not results:
        return "No relevant FAQ entries found."
    
    context = "\n\n---\n\n".join([
        f"FAQ Entry {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(results)
    ])
    
    return f"Found {len(results)} detailed FAQ entries:\n\n{context}"


@tool
def reformulate_query(original_query: str, focus_aspect: str) -> str:
    """Reformulate the query to focus on a specific aspect.
    Use this when you need to search for a different angle of the question.
    
    Args:
        original_query: The original user question
        focus_aspect: The specific aspect to focus on (e.g., "pricing", "activation", "troubleshooting")
        
    Returns:
        A reformulated query focused on the specified aspect
    """
    reformulated = f"{focus_aspect} related to {original_query}"
    results = faq_store.similarity_search(reformulated, k=3)
    
    if not results:
        return f"No results found for aspect: {focus_aspect}"
    
    context = "\n\n---\n\n".join([
        f"Entry {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(results)
    ])
    
    return f"Results for '{focus_aspect}' aspect:\n\n{context}"


tools = [search_faq, search_detailed_faq, reformulate_query]

system_prompt = """You are a helpful FAQ assistant with access to a knowledge base and user memory.

Your goal is to answer user questions accurately using the available tools while remembering user preferences.

Guidelines:
1. Check if you have relevant user preferences or history from previous conversations
2. Use the search_faq tool to find relevant information from the knowledge base
3. If the query is complex, use reformulate_query to search different aspects
4. Personalize responses based on user preferences when relevant
5. Always provide a clear, concise answer based on the retrieved information
6. If you cannot find relevant information, clearly state that

Think step-by-step and use tools strategically to provide the best answer."""

def get_or_create_agent(actor_id: str, session_id: str) -> Agent:
    """
    Create or reuse a global Strands agent configured with AgentCore memory and code execution.
    """
    global _agent

    if _agent is None:
        memory_config = AgentCoreMemoryConfig(
            memory_id=MEMORY_ID,
            session_id=session_id,
            actor_id=actor_id,
            retrieval_config={
                f"/users/{actor_id}/facts": RetrievalConfig(
                    top_k=3, relevance_score=0.5
                ),
                f"/users/{actor_id}/preferences": RetrievalConfig(
                    top_k=3, relevance_score=0.5
                ),
            },
        )

        _agent = Agent(
            model=MODEL_ID,
            session_manager=AgentCoreMemorySessionManager(memory_config, REGION),
            system_prompt=system_prompt,
            tools=tools,
        )

    return _agent


def _validate_environment(state: AgentState) -> dict[str, str]:
    if not MEMORY_ID:
        return {
            "error": "Memory not configured. Set BEDROCK_AGENTCORE_MEMORY_ID environment variable."
        }
    return {}


def _execute_agent(state: AgentState) -> dict[str, str]:
    if state.get("error"):
        return {}

    agent = get_or_create_agent(state["actor_id"], state["session_id"])
    prompt = state["prompt"]

    try:
        result = agent(prompt)
        response = result.message.get("content", [{}])[0].get("text", str(result))
        return {"response": response}
    except Exception as exc:
        return {"error": str(exc)}


def _ensure_response(state: AgentState) -> AgentOutput:
    if state.get("error"):
        return {"response": state["error"]}

    return {
        "response": state.get(
            "response", "Sorry, I could not generate a response."
        )
    }


def build_agent_graph() -> Any:
    graph = StateGraph(state_schema=AgentState, output_schema=AgentOutput)
    graph.add_node("validate_environment", _validate_environment)
    graph.add_node("execute_agent", _execute_agent)
    graph.add_node("ensure_response", _ensure_response)
    graph.add_edge(START, "validate_environment")
    graph.add_edge("validate_environment", "execute_agent")
    graph.add_edge("execute_agent", "ensure_response")
    graph.add_edge("ensure_response", END)
    return graph.compile()


AGENT_GRAPH = build_agent_graph()


@app.entrypoint
def invoke(payload, context):
    actor_id = (
        context.request_headers.get(
            "X-Amzn-Bedrock-AgentCore-Runtime-Custom-Actor-Id", "user"
        )
        if context.request_headers
        else "user"
    )
    session_id = context.session_id or "default_session"
    prompt = payload.get("prompt", "Hello!")

    result = AGENT_GRAPH.invoke(
        {
            "prompt": prompt,
            "actor_id": actor_id,
            "session_id": session_id,
        }
    )

    return {"response": result["response"]}


if __name__ == "__main__":
    app.run()
