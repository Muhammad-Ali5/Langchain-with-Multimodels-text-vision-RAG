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
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Streamlit UI setup
st.set_page_config(page_title="Invoice Extractor with Gemini Vision")
st.header("Invoice Data Extractor (Google Gemini Vision)")

uploaded_file = st.file_uploader(
    "Upload an invoice image (JPG, PNG, etc.)", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read file bytes first
    image_bytes = uploaded_file.getvalue()

    # Show image with PIL
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Invoice", use_column_width=True)

    # User input
    st.subheader("Ask any question about this invoice:")
    user_question = st.text_input("Your question about the invoice:")

    if st.button("Ask about this invoice") and user_question:
        chat_prompt = (
            f"You are an invoice analysis assistant. "
            f"Answer the following question about the uploaded invoice image: {user_question}"
        )
        try:
            chat_response = model.generate_content([
                chat_prompt,
                {"mime_type": uploaded_file.type, "data": image_bytes}
            ])
            st.markdown(f"**Answer:** {chat_response.text}")

        except Exception as e:
            error_msg = str(e)
            st.error(f"Error in chatbot: {error_msg}")

            # Retry if quota exceeded (429 error)
            if "429" in error_msg:
                st.warning("Quota exceeded. Retrying in 40 seconds...")
                time.sleep(40)
                try:
                    chat_response = model.generate_content([
                        chat_prompt,
                        {"mime_type": uploaded_file.type, "data": image_bytes}
                    ])
                    st.markdown(f"**Answer (after retry):** {chat_response.text}")
                except Exception as e2:
                    st.error(f"Retry failed: {str(e2)}")

else:
    st.info("Please upload an invoice image and then ask your question about it.")
