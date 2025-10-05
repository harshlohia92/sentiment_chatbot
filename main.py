import streamlit as st
from chatbot_helper import predict_with_keywords, generate_response, bert_model

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="Motivational Chatbot 💬", page_icon="💬", layout="centered")

# ---------------- Session State ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "theme" not in st.session_state:
    st.session_state.theme = "🌙 Dark"

# ---------------- Theme Styles ----------------
def apply_theme():
    if st.session_state.theme == "🌙 Dark":
        bg_color = "#141e30"
        gradient = "linear-gradient(to right, #141e30, #243b55)"
        user_bubble = "#4cafef"
        bot_bubble = "#2c2c2c"
        text_color = "white"
    else:
        bg_color = "#f0f0f0"
        gradient = "linear-gradient(to right, #f0f0f0, #dfe9f3)"
        user_bubble = "#2575fc"
        bot_bubble = "#e6e6e6"
        text_color = "black"

    st.markdown(
        f"""
        <style>
        body {{
            background: {gradient};
            color: {text_color};
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .stTextInput > div > div > input {{
            background-color: #1c1c1c;
            color: white;
            border-radius: 8px;
            border: 1px solid #4cafef;
            padding: 8px;
        }}
        .stButton>button {{
            background: linear-gradient(45deg, #6a11cb, #2575fc);
            color: white;
            border-radius: 10px;
            padding: 10px 24px;
            font-weight: bold;
        }}
        .stButton>button:hover {{
            background: linear-gradient(45deg, #2575fc, #6a11cb);
            box-shadow: 0px 0px 10px #6a11cb;
        }}
        .chat-bubble-user {{
            background-color: {user_bubble};
            color: white;
            padding: 12px 18px;
            border-radius: 20px;
            margin: 5px 0;
            text-align: right;
            max-width: 70%;
            word-wrap: break-word;
        }}
        .chat-bubble-bot {{
            background-color: {bot_bubble};
            color: {text_color};
            padding: 12px 18px;
            border-radius: 20px;
            margin: 5px 0;
            text-align: left;
            max-width: 70%;
            word-wrap: break-word;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_theme()

# ---------------- Title ----------------
st.markdown(
    "<h1 style='text-align:center; color:#FFD700;'>💬 Motivational Chatbot</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#ffffff;'>Get personalized motivational support powered by BERT & GPT-2</p>",
    unsafe_allow_html=True,
)

# ---------------- User Input ----------------
user_input = st.text_input("Type your message here:")

col1, col2 = st.columns([1, 1])
with col1:
    send_btn = st.button("Send")
with col2:
    clear_btn = st.button("🗑️ Clear Chat")

# ---------------- Handle Send ----------------
if send_btn and user_input.strip():
    sentiment = predict_with_keywords(bert_model, user_input)
    reply = generate_response(sentiment, user_input)

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", reply))

# ---------------- Handle Clear ----------------
if clear_btn:
    st.session_state.chat_history = []
    st.success("Chat history cleared ✅")

# ---------------- Display Chat ----------------
for sender, message in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"<div class='chat-bubble-user'>🙋‍♂️ You: {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-bot'>🤖 Bot: {message}</div>", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Settings")
theme_choice = st.sidebar.radio("Choose Theme", ["🌙 Dark", "☀️ Light"], index=0 if st.session_state.theme == "🌙 Dark" else 1)

if theme_choice != st.session_state.theme:
    st.session_state.theme = theme_choice
    st.rerun()

if st.sidebar.button("🎉 Celebrate Positivity"):
    st.balloons()
    st.success("🌟 Stay Positive, You’re Doing Amazing!")
