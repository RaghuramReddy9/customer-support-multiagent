import os
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
    query: str       # the user’s input
    response: str    # the agent’s reply
    escalated: bool  # whether query was escalated or not
    route: str      # which department handled the query

## Define Agents (Gemini models, explicit per role)

# Router agent (decides which department should handle the query)
router_agent = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0  # deterministic output (no creativity)
)

# Frontline agent (triage only)
frontline_agent = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3   # slightly more creative for first contact
)

# Billing specialist (must be precise)
billing_agent = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0   # strict, accurate answers
)

# Tech specialist (helpful troubleshooting)
tech_agent = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2   # mostly accurate but allows small flexibility
)

# General FAQ specialist (friendly)
general_agent = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.5   # more conversational
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

# Define Agent Node Functions
def router_node(state: EscalationState):
    user_query = state["query"]

    prompt = f"""
    You are a routing agent for a customer support bot.
    Your job is to classify the query into ONE of three departments:

    - billing → for anything about payments, charges, invoices
    - tech → for anything about app issues, crashes, installation
    - general → for everything else (passwords, account updates, contact info)

    Only respond with exactly one word: billing, tech, or general.

    Query: {user_query}
    """

    decision = router_agent.invoke(prompt).content.strip().lower()
    state["route"] = decision
    return state

def frontline_node(state: EscalationState):
    """Frontline agent greets and triages only, doesn’t solve issues."""
    user_query = state["query"]
    reply = frontline_agent.invoke(
        f"You are a frontline customer support assistant. Greet the user and route their issue: {user_query}"
    ).content
    state["response"] = reply
    return state

def billing_node(state: EscalationState):
    """Billing specialist uses RAG over billing_faq.txt"""
    user_query = state["query"]
    docs = billing_retriever.invoke(user_query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    You are a billing support specialist.
    Use the following billing FAQs to answer the question.

    Context:
    {context}

    Question: {user_query}
    """
    reply = billing_agent.invoke(prompt).content
    state["response"] = reply
    return state

def tech_node(state: EscalationState):
    """Tech specialist uses RAG over tech_faq.txt"""
    user_query = state["query"]
    docs = tech_retriever.invoke(user_query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    You are a technical support specialist.
    Use the following tech FAQs to answer the question.

    Context:
    {context}

    Question: {user_query}
    """
    reply = tech_agent.invoke(prompt).content
    state["response"] = reply
    return state

def general_node(state: EscalationState):
    """General specialist uses RAG over general_faq.txt"""
    user_query = state["query"]
    docs = general_retriever.invoke(user_query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    You are a general customer support specialist.
    Use the following general FAQs to answer the question.

    Context:
    {context}

    Question: {user_query}
    """
    reply = general_agent.invoke(prompt).content
    state["response"] = reply
    return state

## Build LangGraph Workflow
graph = StateGraph(EscalationState)

graph.add_node("frontline", frontline_node)
graph.add_node("router", router_node)
graph.add_node("billing", billing_node)
graph.add_node("tech", tech_node)
graph.add_node("general", general_node)

graph.set_entry_point("frontline")

# Routing from frontline to router
graph.add_edge("frontline", "router")

# Router output → specialist agents
graph.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {"billing": "billing", "tech": "tech", "general": "general"}
)

# Specialist agents → END
graph.add_edge("billing", END)
graph.add_edge("tech", END)
graph.add_edge("general", END)

app = graph.compile()

# Run the Bot
if __name__ == "__main__":
    query = input("User: ")
    result = app.invoke({"query": query, "response": "", "escalated": False})
    print("Bot:", result["response"])
