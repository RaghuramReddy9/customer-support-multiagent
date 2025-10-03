import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from multi_agent_escalation import app  # Import the multi-agent escalation app

# Define request body
class ChatRequest(BaseModel):
    query: str

# Define response schema
class ChatResponse(BaseModel):
    response: str
    route: str
    confidence: float
    route_history: list[str]

# Create FastAPI app
api = FastAPI(title="Customer Support Multi-Agent Bot")

@api.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    # Run the multi-agent bot
    result = app.invoke({
        "query": request.query,
        "response": "",
        "escalated": False,
        "route": "",
        "confidence": 0.0,
        "route_history": [],
        "handoff_summary": "",
        "conversation_history": []
    })
    return ChatResponse(
        response=result["response"],
        route=result["route"],
        confidence=result["confidence"],
        route_history=result["route_history"],
    )

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8000)
