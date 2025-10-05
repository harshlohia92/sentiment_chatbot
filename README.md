# Sentiment-Aware Chatbot 🤖💬
A Python-based sentiment-aware chatbot that provides motivational and supportive replies.
It uses BERT for sentiment detection and GPT-2 for generating human-like motivational responses,
all wrapped inside a beautiful Streamlit interface.
---

### Tech Stack
BERT → detects positive or negative emotions from user text

GPT-2 → generates natural motivational replies

Streamlit → interactive real-time chatbot UI

PyTorch + Transformers → model backbone

![chatbot screenshot](chatbot1.jpg)

### ✨ Model Details
Base Model: bert-base-uncased

Fine-tuned Dataset: 50,000 labeled text samples

Full Dataset Size: 1.6 million records (16 lakh)

The model was trained on a small subset (50k) for faster experimentation,

so responses may sometimes differ slightly from human expectations.
---

### Install dependencies:
pip install -r requirements.txt

### Run the Streamlit app
streamlit run streamlit/main.py

### Features

Features
✅ Sentiment detection using fine-tuned BERT

✅ Dynamic motivational replies via GPT-2

✅ Beautiful dark/light themes

✅ Interactive Streamlit interface

✅ “Clear Chat” + “Celebrate Positivity” buttons 🎉