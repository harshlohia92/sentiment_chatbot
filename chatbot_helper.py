import os
import torch
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

BERT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME).to(device)

GPT_MODEL_NAME = "gpt2"
gpt_tokenizer = GPT2Tokenizer.from_pretrained(GPT_MODEL_NAME)
gpt_model = GPT2LMHeadModel.from_pretrained(GPT_MODEL_NAME).to(device)


def predict(model, text, max_length=128):
    """Predict sentiment using pretrained sentiment model."""
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()  

    if pred_class <= 2:
        return "negative"
    else:
        return "positive"


def predict_with_keywords(model, text):
    """Directly use BERT sentiment (keywords removed for accuracy)."""
    return predict(model, text)


positive_replies = [
    "That's awesome! Keep up the great work! 😊",
    "Great to hear! Keep shining! ✨",
    "Life is beautiful, enjoy every moment! 🌸",
    "That sounds amazing! Keep spreading positivity! 😄",
    "I love hearing that! Stay happy and keep smiling! 😃",
    "Wonderful! It's great to have such beautiful moments in life! 💖",
]

negative_replies = [
    "I'm really sorry you're feeling this way. 💙",
    "Don't worry, things will get better with time. 🌤️",
    "Stay strong — you’ve got what it takes to overcome this. 💪",
    "Take a deep breath and be kind to yourself today. 🌸",
    "Every storm passes. You’re not alone. 💫",
]


def generate_response(sentiment, text=None):
    """Generate a motivational GPT-based response."""
    sentiment = sentiment.lower()
    reply = random.choice(positive_replies if sentiment == "positive" else negative_replies)

    if text:
        inputs = gpt_tokenizer(reply, return_tensors="pt").to(device)
        outputs = gpt_model.generate(
            **inputs,
            max_length=60,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=gpt_tokenizer.eos_token_id,
            eos_token_id=gpt_tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )
        final_reply = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if reply in final_reply:
            final_reply = reply
    else:
        final_reply = reply

    return final_reply
