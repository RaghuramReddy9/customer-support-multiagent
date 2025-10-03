import streamlit as st
from multi_agent_escalation import app   # import LangGraph workflow

st.set_page_config(page_title="Multi-Agent Support Bot", page_icon="🤖")

st.title("Customer Support Chatbot")
st.write("Ask me about billing, tech, or general issues.")

# Keep chat history in session
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Show chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Type your question..."):
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run LangGraph workflow
    result = app.invoke({"query": prompt, "response": "", "escalated": False, "route": ""})
    bot_reply = result["response"]

    # Show bot response
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

