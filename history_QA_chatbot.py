import os 
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# load envionment 
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    st.error("Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")   

# create history bot function
chat = model.start_chat(history=[])

def get_gemini_response(question):
    try:
        if question:
            # Return the streaming response generator
            return chat.send_message(question, stream=True)
        else:
            return "Please enter a question."
    except Exception as e:
        return f"Error: {str(e)}"

# streamlit UI
st.set_page_config(page_title="AI History Q&A Chatbot")
st.header("Gemini History Q&A Chatbot")

# initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

# text input for question
input = st.text_input("Input :", key="input")
submit = st.button("Ask a question:")

if submit and input:
    st.session_state['chat_history'].append(("You", input))
    streamed_response = ""
    response = get_gemini_response(input)
    st.subheader("Gemini's Answer:")
    response_placeholder = st.empty()
    try:
        for chunk in response:
            streamed_response += chunk.text
            response_placeholder.write(streamed_response)
    except Exception as e:
        streamed_response = f"Error: {str(e)}"
        response_placeholder.write(streamed_response)
    st.session_state['chat_history'].append(("Gemini", streamed_response))
st.subheader("Chat History:")
for role, text in st.session_state['chat_history']:
    st.write(f"**{role}:** {text}")
