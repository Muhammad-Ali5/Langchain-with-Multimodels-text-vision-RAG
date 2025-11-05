import os
import time
import streamlit as st
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Use Gemini Flash model (higher free quota)
def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content([input, image[0], prompt])
    return response.text

def input_image_setup(uploaded_file):
    # check if file has been uploaded
    if uploaded_file is not None:
        # Read file bytes first
        image_bytes = uploaded_file.getvalue()

        image_parts = [{"mime_type": uploaded_file.type, "data": image_bytes}]
        return image_parts
    else:
        st.warning("Please upload an image file.")
        return None

# streamlit app
st.set_page_config(page_title="Calorie Counter Chatbot", layout="centered")

st.header("Gemini Health Application")
input = st.text_input("Input Prompt", key ="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = " "
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

submit = st.button("Submit") #tell me about the total calories 
input_prompt = """
    You are a helpful nutrition and health assistant.
    Your task is to analyze the uploaded food image and/or the user’s text input.
    1. Identify the food items in the image (or described in text).
    2. Estimate the portion size and calculate approximate calories.
    3. Provide a short breakdown of nutrients (carbs, protein, fat).
    4. If the food is unhealthy, suggest healthier alternatives.
    5. Keep answers simple, clear, and user-friendly.

    Format your answer like this:
    🍽 Food: [food items]
    🔥 Calories: [estimated total calories]
    ⚡ Nutrients: [carbs %, protein %, fat %]
    💡 Tip: [health advice or alternative]
"""
if submit:
    image_data = input_image_setup(uploaded_file)
    response = get_gemini_response(input, image_data, input_prompt)
    st.subheader("Response:")
    st.write(response)
st.markdown("---")
