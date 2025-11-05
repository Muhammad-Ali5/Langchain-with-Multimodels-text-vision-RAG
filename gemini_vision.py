import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Use a vision-capable model
model = genai.GenerativeModel("gemini-2.5-flash")

def get_gemini_response(prompt, image):
    try:
        if image and prompt:
            response = model.generate_content([prompt, image])
        elif image:
            response = model.generate_content([image])
        else:
            response = model.generate_content([prompt])
        return response.text
    except Exception as e:
        return f"Error: {e}"

st.title("Gemini Vision Chatbot")

prompt = st.text_input("Enter your question (optional):")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

image = None
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

if st.button("Analyze Image"):
    if not prompt and not image:
        st.warning("Please enter a prompt or upload an image.")
    else:
        st.subheader("Gemini Response:")
        st.write(get_gemini_response(prompt, image))
