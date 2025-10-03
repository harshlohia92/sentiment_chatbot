import streamlit as st
from chatbot_helper import predict_with_keywords, generate_response, bert_model

st.set_page_config(page_title="Motivational Chatbot 💬", page_icon="💬", layout="centered")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #141e30, #243b55);
    color: white;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stTextInput>div>div>input {
    background-color: #1c1c1c;
    color: white;
    border-radius: 12px;
    border: 2px solid #4cafef;
    padding: 10px;
}
.stButton>button {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    color: white;
    border-radius: 12px;
    padding: 12px 26px;
    font-weight: bold;
    transition: 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #2575fc, #6a11cb);
    box-shadow: 0px 0px 12px #6a11cb;
}
.chat-bubble-user {
    background: linear-gradient(135deg, #42a5f5, #1e88e5);
    color: white;
    padding: 14px 20px;
    border-radius: 25px;
    margin: 6px 0;
    text-align: right;
    max-width: 70%;
    word-wrap: break-word;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
}
.chat-bubble-bot {
    background: rgba(44,44,44,0.95);
    color: #f1f1f1;
    padding: 14px 20px;
    border-radius: 25px;
    margin: 6px 0;
    text-align: left;
    max-width: 70%;
    word-wrap: break-word;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
}
.delete-btn {
    background: transparent;
    color: red;
    border: none;
    font-size: 14px;
    cursor: pointer;
    margin-left: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#FFD700;'>💬 Motivational Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#ffffff;'>Get positive and motivational replies instantly!</p>",
            unsafe_allow_html=True)


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Choose Theme", ["🌙 Dark", "☀️ Light"])

if st.sidebar.button("Celebrate! 🎉"):
    st.balloons()
    st.success("🌟 Stay Positive, You’re Doing Amazing!")

if st.sidebar.button("Clear Chat 🗑️"):
    st.session_state.chat_history = []


user_input = st.text_input("Type your message here:")

if st.button("Send") and user_input.strip() != "":

    sentiment = predict_with_keywords(bert_model, user_input)
    reply = generate_response(sentiment, user_input)


    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", reply))


for idx in range(len(st.session_state.chat_history)):
    sender, message = st.session_state.chat_history[idx]
    col1, col2 = st.columns([0.9, 0.1])

    if sender == "user":
        with col1:
            st.markdown(f"<div class='chat-bubble-user'>You: {message}</div>", unsafe_allow_html=True)
    else:
        with col1:
            st.markdown(f"<div class='chat-bubble-bot'>🤖 Bot: {message}</div>", unsafe_allow_html=True)


    with col2:
        if st.button("X", key=f"del_{idx}"):
            st.session_state.chat_history.pop(idx)
            break
