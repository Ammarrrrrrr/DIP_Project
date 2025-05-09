import streamlit as st
import requests
import json

st.set_page_config(page_title="Chat with LLaMA 3.2", layout="centered")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm LLaMA 3.2. How can I help you today?"}]

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box for user message
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": "llama3.2-vision",
                        "messages": st.session_state.messages,
                        "stream": False
                    }
                )
                if response.status_code == 200:
                    result = response.json()
                    content = result["message"]["content"]
                    st.markdown(content)
                    st.session_state.messages.append({"role": "assistant", "content": content})
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
