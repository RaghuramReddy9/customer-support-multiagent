---
title: Customer Support AI Assistant
emoji: 💬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# 💬 Customer Support AI Assistant

An AI-powered customer support bot built with **Streamlit** and **Gemini 1.5 Flash**, deployed on **Hugging Face Spaces**.

---
##  Live Demo: [Customer Support Bot](https://raghuramreddyt-customer-support.hf.space/docs)

---


## 🚀 Features
- Answer customer support queries in plain English  
- Built with **Google Gemini API**  
- Simple and clean **Streamlit UI**  
- Runs directly on **Hugging Face Spaces**  

---

## 🛠️ Tech Stack
- **Python**
- **Streamlit**
- **Google Generative AI SDK**
- **Hugging Face Spaces**

---

## 📂 Project Structure
```
customer-support-bot/
│── app.py # Main Streamlit app
│── requirements.txt # Python dependencies
│── README.md # Project documentation
│── .env (local only) # API key for Gemini (never pushed)
```
---
## ⚡ Running Locally
Clone the repo and run with Streamlit:
```bash
git clone https://huggingface.co/spaces/RaghuramReddyT/customer-support-bot
cd customer-support-bot
pip install -r requirements.txt
streamlit run app.py
```
---
## Set your API key in .env:

GEMINI_API_KEY=your_api_key_here
---
##  License
```
MIT License
```
---