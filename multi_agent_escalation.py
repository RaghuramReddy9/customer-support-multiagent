import os
import json    
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import mlflow

# configure MLflow
mlflow.set_experiment("multi-agent-escalation-bot")

# helper function
def log_interaction(log_data:dict, log_file="logs/session_logs.jsonl"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")

# Load Environment Variables
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Define State Structure

class EscalationState(TypedDict):
    query: str        # the user’s input
    response: str     # the agent’s reply
    escalated: bool   # whether query was escalated or not
    route: str        # which department handled the query
    confidence: float   # confidence score (if applicable)
    route_history: list[str]  # history of departments involved
    handoff_summary: str  # short context for next agent
    conversation_history: list[dict[str, str]]  # full chat history(store last few turns)

## Define Agents (Gemini models, explicit per role)

# Router agent (decides which department should handle the query)
router_agent = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001", google_api_key=GOOGLE_API_KEY, temperature=0  # deterministic output (no creativity)
)

# Frontline agent (triage only)
frontline_agent = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001", google_api_key=GOOGLE_API_KEY, temperature=0.3   # slightly more creative for first contact
)

# Billing specialist (must be precise)
billing_agent = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001", google_api_key=GOOGLE_API_KEY, temperature=0.0   # strict, accurate answers
)

# Tech specialist (helpful troubleshooting)
tech_agent = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001", google_api_key=GOOGLE_API_KEY, temperature=0.2   # mostly accurate but allows small flexibility
)

# General FAQ specialist (friendly)
general_agent = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001", google_api_key=GOOGLE_API_KEY, temperature=0.5   # more conversational
)

## Setup RAG Knowledge Bases (Billing, Tech, General)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Billing FAQ
billing_loader = TextLoader("billing_faq.txt")
billing_chunks = splitter.split_documents(billing_loader.load())
billing_vectorstore = Chroma.from_documents(billing_chunks, embedding=embeddings)
billing_retriever = billing_vectorstore.as_retriever(search_kwargs={"k": 2})

# Tech FAQ
tech_loader = TextLoader("tech_faq.txt")
tech_chunks = splitter.split_documents(tech_loader.load())
tech_vectorstore = Chroma.from_documents(tech_chunks, embedding=embeddings)
tech_retriever = tech_vectorstore.as_retriever(search_kwargs={"k": 2})

# General FAQ
general_loader = TextLoader("general_faq.txt")
general_chunks = splitter.split_documents(general_loader.load())
general_vectorstore = Chroma.from_documents(general_chunks, embedding=embeddings)
general_retriever = general_vectorstore.as_retriever(search_kwargs={"k": 2})

# format conversation history
def format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return ""
    formatted = "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in history[-5:]])
    return f"Conversation so far:\n{formatted}\n" 

# define frontline node function
def frontline_node(state: EscalationState):
    user_query = state["query"]
    reply = frontline_agent.invoke(
        f"You are a frontline assistant. Greet the user and prepare to route their issue: {user_query}"
    ).content
    state["response"] = reply
    # Update memory
    state["conversation_history"].append({"role": "user", "content": user_query})
    state["conversation_history"].append({"role": "bot", "content": reply})
    return state

# Router with JSON enforcement + keyword fallback
def router_node(state: EscalationState):
    user_query = state["query"]

    prompt = f"""
    You are a strict routing agent.
    Classify the query into ONE of: billing, tech, general.
    Respond ONLY with JSON in this exact format:

    {{
      "route": "billing|tech|general",
      "confidence": 0.0-1.0
    }}

    Query: {user_query}
    """
    raw = router_agent.invoke(prompt).content

    try:
        data = json.loads(raw)
        route = data.get("route", "").strip().lower()
        conf = float(data.get("confidence", 0.0))
    except Exception:
        # Keyword fallback if JSON parsing fails
        q = user_query.lower()
        if any(word in q for word in ["bill", "charge", "payment", "invoice"]):
            route, conf = "billing", 0.7
        elif any(word in q for word in ["app", "error", "crash", "bug", "technical", "freeze"]):
            route, conf = "tech", 0.7
        else:
            route, conf = "general", 0.6

    if route not in {"billing", "tech", "general"}:
        route = "clarify"

    state["route"] = route
    state["confidence"] = conf
    state["route_history"] = (state.get("route_history") or []) + [route]
    state["handoff_summary"] = f"User asked: '{user_query}'. Routed to {route} with confidence {conf:.2f}."
    return state

# Billing specialist
def billing_node(state: EscalationState):
    history = format_history(state["conversation_history"])
    docs = billing_retriever.invoke(state["query"])
    context = "\n".join([d.page_content for d in docs])
    prompt = f"""
    {state.get("handoff_summary","")}
    {history}

    You are a billing specialist. Use the following FAQs to answer.

    Context:
    {context}

    Question: {state["query"]}
    """
    reply = billing_agent.invoke(prompt).content
    state["response"] = reply
    state["escalated"] = True
    state["conversation_history"].append({"role": "user", "content": state["query"]})
    state["conversation_history"].append({"role": "bot", "content": reply})
    return state

# Tech specialist
def tech_node(state: EscalationState):
    history = format_history(state["conversation_history"])
    docs = tech_retriever.invoke(state["query"])
    context = "\n".join([d.page_content for d in docs])
    prompt = f"""
    {state.get("handoff_summary","")}
    {history}

    You are a technical support specialist.
    Use the following FAQs to answer.

    Context:
    {context}

    Question: {state["query"]}
    """
    reply = tech_agent.invoke(prompt).content
    state["response"] = reply
    state["escalated"] = True
    state["conversation_history"].append({"role": "user", "content": state["query"]})
    state["conversation_history"].append({"role": "bot", "content": reply})
    return state

# General specialist
def general_node(state: EscalationState):
    history = format_history(state["conversation_history"])
    docs = general_retriever.invoke(state["query"])
    context = "\n".join([d.page_content for d in docs])
    prompt = f"""
    {state.get("handoff_summary","")}
    {history}

    You are a general support specialist.
    Use the following FAQs to answer.

    Context:
    {context}

    Question: {state["query"]}
    """
    reply = general_agent.invoke(prompt).content
    state["response"] = reply
    state["escalated"] = True
    state["conversation_history"].append({"role": "user", "content": state["query"]})
    state["conversation_history"].append({"role": "bot", "content": reply})
    return state

# Clarify node (short + consistent)
def clarify_node(state: EscalationState):
    reply = "Could you clarify your request so I can direct you to the right department?"
    state["response"] = reply
    state["escalated"] = True
    state["conversation_history"].append({"role": "user", "content": state["query"]})
    state["conversation_history"].append({"role": "bot", "content": reply})
    return state

## Build LangGraph Workflow
graph = StateGraph(EscalationState)

graph.add_node("frontline", frontline_node)
graph.add_node("router", router_node)
graph.add_node("billing", billing_node)
graph.add_node("tech", tech_node)
graph.add_node("general", general_node)
graph.add_node("clarify", clarify_node)

# Define entry point
graph.set_entry_point("frontline")

# Routing from frontline to router
graph.add_edge("frontline", "router")

# Router → next agent (with confidence filter)
def _router_decision(state: EscalationState):
    route = state["route"]
    conf = state.get("confidence", 0.0)
    if route in {"billing", "tech", "general"} and conf >= 0.6:
        return route
    return "clarify"

graph.add_conditional_edges(
    "router",
    _router_decision,
    {"billing": "billing", "tech": "tech", "general": "general", "clarify": "clarify"},
)

# Teminal edges
graph.add_edge("billing", END)
graph.add_edge("tech", END)
graph.add_edge("general", END)
graph.add_edge("clarify", END)  

# Compile the graph into an executable app
app = graph.compile()

# Run the Bot
if __name__ == "__main__":
    query = input("User: ")
    result = app.invoke({
        "query": query,
        "response": "",
        "escalated": False,
        "route": "",
        "confidence": 0.0,
        "route_history": [],
        "handoff_summary": "",
        "conversation_history": []
    })

    print("Bot:", result["response"])
    print("Route chosen:", result["route"])
    print("Confidence:", result["confidence"])
    print("History:", result["route_history"])
    print("Conversation memory:", result["conversation_history"])

    # Log interaction to MLflow
    import mlflow, json, os

    with mlflow.start_run():
        mlflow.log_param("model_router", "gemini-2.0-flash-001")
        mlflow.log_param("model_specialists", "gemini-2.0-flash-001")
        mlflow.log_metric("confidence", result["confidence"])
        mlflow.log_metric("query_length", len(result["query"]))
        mlflow.log_metric("conversation_turns", len(result["conversation_history"]))

        log_file_path = "logs/latest_conversation.json"
        os.makedirs("logs", exist_ok=True)
        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(result["conversation_history"], f, indent=2)
        mlflow.log_artifact(log_file_path, artifact_path="conversations")

    print("✅ Logged interaction to MLflow")
