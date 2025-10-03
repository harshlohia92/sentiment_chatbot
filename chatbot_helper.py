import os
import torch
import random
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer


device = "mps" if torch.backends.mps.is_available() else 'cpu'


BASE_DIR = os.path.dirname(__file__)

# Use relative paths instead of absolute Mac paths
BERT_PATH = os.path.join(BASE_DIR, "model/bert_model")
GPT_PATH = os.path.join(BASE_DIR, "model/gpt_model")


bert_model = BertForSequenceClassification.from_pretrained(
    BERT_PATH, local_files_only=True
).to(device)

bert_tokenizer = BertTokenizer.from_pretrained(
    BERT_PATH, local_files_only=True
)

gpt_tokenizer = GPT2Tokenizer.from_pretrained(GPT_PATH,local_files_only=True)
gpt_model = GPT2LMHeadModel.from_pretrained(GPT_PATH,local_files_only=True).to(device)


positive_keywords = [
    "got job", "new job", "happy", "love", "excited",
    "awesome", "great", "fantastic", "good news", "achieved"
]
negative_keywords = [
    "lost job", "sad", "angry", "upset", "failed",
    "bad news", "disappointed", "tired", "not working"
]


def predict(model, text, max_length=128):
    encoding = bert_tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits  # shape: [1,2]

        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    return "positive" if pred_class == 1 else "negative"


def predict_with_keywords(model, text):
    text_lower = text.lower().strip()
    for kw in positive_keywords:
        if kw in text_lower:
            return "positive"
    for kw in negative_keywords:
        if kw in text_lower:
            return "negative"
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
    "Don't worry! You can handle this step by step. 💪",
    "Keep going, you'll get it done in time!",
    "Take it easy, do what you can, and things will work out.",
    "Stay positive! Small steps lead to big progress.",
    "You got this! Keep your head up and move forward.",
]

def generate_response(sentiment, text=None):
    sentiment = sentiment.lower()
    reply = random.choice(positive_replies if sentiment == "positive" else negative_replies)

    if text:
        inputs = gpt_tokenizer(reply, return_tensors="pt").to(device)
        outputs = gpt_model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=gpt_tokenizer.eos_token_id,
            eos_token_id=gpt_tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
        final_reply = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if reply in final_reply:
            final_reply = reply
    else:
        final_reply = reply

    return final_reply
