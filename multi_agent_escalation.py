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

# define frontline node function
def frontline_node(state: EscalationState):
    user_query = state["query"]
    reply = frontline_agent.invoke(
        f"You are a frontline customer support assistant. Greet the user and route their issue: {user_query}"
    ).content
    state["response"] = reply
    return state

# Define Router Node with confidence + route history
def router_node(state: EscalationState):
    user_query = state["query"]

    prompt = f"""
    You are a strict routing agent. Classify the query into ONE of: billing, tech, general.
    Return strict JSON with keys: route (string), confidence (0.0-1.0 float).
    Only these routes are allowed.

    Query: {user_query}
    JSON:
    """
    raw = router_agent.invoke(prompt).content
    try:
        data = json.loads(raw)
        route = data.get("route", "").strip().lower()
        conf = float(data.get("confidence", 0.0))
    except Exception:
        route, conf = "clarify", 0.0

    if route not in {"billing", "tech", "general"}:
       route = "clarify"

    # Save to state
    state["route"] = route
    state["confidence"] = conf
    state["route_history"] = (state.get("route_history") or []) + [route]
    return state     

# Billing node
def billing_node(state: EscalationState):
    q = state["query"].lower()
    if "charge" in q and "double" not in q:
        return {
            "query": state["query"],
            "response": "Do you mean an overcharge on your monthly bill or a one-time double charge?",
            "escalated": True,
            "route": "clarify",
            "confidence": state.get("confidence", 0.0),
            "route_history": state.get("route_history", []),
        }
    docs = billing_retriever.invoke(state["query"])
    context = "\n".join([d.page_content for d in docs])
    prompt = f"""
    You are a billing specialist. Use the following FAQs to answer:

    Context:
    {context}

    Question: {state["query"]}
    """
    reply = billing_agent.invoke(prompt).content
    state["response"] = reply
    state["escalated"] = True
    return state


# Tech node
def tech_node(state: EscalationState):
    q = state["query"].lower()
    if "app is not working" in q or "issue" in q or "problem" in q:
        return {
            "query": state["query"],
            "response": "Can you clarify the issue? Is the app crashing, freezing, or not opening?",
            "escalated": True,
            "route": "clarify",
            "confidence": state.get("confidence", 0.0),
            "route_history": state.get("route_history", []),
        }
    docs = tech_retriever.invoke(state["query"])
    context = "\n".join([d.page_content for d in docs])
    prompt = f"""
    You are a technical support specialist.
    Use the following FAQs to answer the question.

    Context:
    {context}

    Question: {state["query"]}
    """
    reply = tech_agent.invoke(prompt).content
    state["response"] = reply
    state["escalated"] = True
    return state


# General node
def general_node(state: EscalationState):
    user_query = state["query"]
    docs = general_retriever.invoke(user_query)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"""
    You are a general support specialist.
    Use the following FAQs to answer:

    Context:
    {context}

    Question: {user_query}
    """
    reply = general_agent.invoke(prompt).content
    state["response"] = reply
    state["escalated"] = True
    return state


# Clarify node
def clarify_node(state: EscalationState):
    return {
        "query": state["query"],
        "response": state.get("response", "Could you clarify your issue?"),
        "escalated": True,
        "route": "clarify",
        "confidence": state.get("confidence", 0.0),
        "route_history": state.get("route_history", []),
    }

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
    {"billing": "billing", "tech": "tech", "general": "general", "clarify": "clarify"}
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
        "route_history": []
    })
    print("Bot:", result["response"])
    print("Route chosen:", result["route"])
    print("Confidence:", result["confidence"])
    print("History:", result["route_history"])
