import os
import torch
import random
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ------------------ DEVICE ------------------
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

# ------------------ PATHS ------------------
# Automatically detect project root (parent of current file directory)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BERT_PATH = os.path.join(BASE_DIR, "../model_per/bert")
GPT_PATH = os.path.join(BASE_DIR, "../model_per/gpt")

# your GPT-2 model

print(f"BERT path: {BERT_PATH}")
print(f"GPT path: {GPT_PATH}")

# ------------------ VALIDATION ------------------
if not os.path.exists(BERT_PATH):
    raise FileNotFoundError(f"BERT model not found at {BERT_PATH}. Please ensure the path is correct.")
if not os.path.exists(GPT_PATH):
    raise FileNotFoundError(f"GPT-2 model not found at {GPT_PATH}. Please ensure the path is correct.")

# ------------------ LOAD MODELS ------------------
bert_model = BertForSequenceClassification.from_pretrained(
    BERT_PATH, local_files_only=True
).to(device)
bert_tokenizer = BertTokenizer.from_pretrained(
    BERT_PATH, local_files_only=True
)

gpt_tokenizer = GPT2Tokenizer.from_pretrained(GPT_PATH)
gpt_model = GPT2LMHeadModel.from_pretrained(GPT_PATH).to(device)

# GPT-2 requires explicit padding token
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

# ------------------ RANDOM SEED ------------------
torch.manual_seed(42)
random.seed(42)

# ------------------ KEYWORDS ------------------
positive_keywords = [
    "got job", "new job", "happy", "love", "excited",
    "awesome", "great", "fantastic", "good news", "achieved"
]
negative_keywords = [
    "lost job", "sad", "angry", "upset", "failed",
    "bad news", "disappointed", "tired", "not working"
]

# ------------------ BERT PREDICTION ------------------
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
        logits = output.logits

        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    return "positive" if pred_class == 1 else "negative"

# ------------------ HYBRID PREDICTION ------------------
def predict_with_keywords(model, text):
    text_lower = text.lower().strip()
    for kw in positive_keywords:
        if kw in text_lower:
            return "positive"
    for kw in negative_keywords:
        if kw in text_lower:
            return "negative"
    return predict(model, text)

# ------------------ GPT-2 REPLIES ------------------
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

# ------------------ GPT-2 RESPONSE GENERATION ------------------
def generate_response(sentiment, text=None):
    sentiment = sentiment.lower()
    reply = random.choice(positive_replies if sentiment == "positive" else negative_replies)

    if text:
        inputs = gpt_tokenizer(reply, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

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

# ------------------ COMBINED CHAT FUNCTION ------------------
def analyze_and_respond(text):
    """Predict sentiment and return a generated GPT-2 response."""
    sentiment = predict_with_keywords(bert_model, text)
    reply = generate_response(sentiment, text)
    return sentiment, reply

# ------------------ TEST ------------------
if __name__ == "__main__":
    user_input = input("You: ")
    sentiment, response = analyze_and_respond(user_input)
    print(f"Detected sentiment: {sentiment}")
    print(f"Chatbot: {response}")
