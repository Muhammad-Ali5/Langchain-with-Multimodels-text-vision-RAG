import os 
import streamlit as st 
from dotenv import load_dotenv
import time

from langchain_groq import ChatGroq
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio


# Fix event loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load environment variables
load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

os.environ['GOOGLE_API_KEY'] = "AIzaSyBRe8NDmbb-..........................s"

st.title("ChatGroq + LangChain Demo")
url = st.text_input("Enter the Website link to load content : ")

if url and "vectors" not in st.session_state:
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.loader = WebBaseLoader(url)
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

if "vectors" in st.session_state:
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.3,
        api_key=groq_api_key
    )

    prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    <context>
    {context}
    Question: {input}
    """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    user_prompt = st.text_input(" Enter the prompt here:")

    if user_prompt: 
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        st.write("Answer : ", response["answer"])
        st.write("Response Time :", round(time.process_time() - start, 2), "s")

        with st.expander(" Relevant Document Chunks:"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)

                st.write("-----------------------------------")
