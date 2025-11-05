import os 
import streamlit as st 
from PyPDF2 import PdfReader
from dotenv import load_dotenv 

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or ""

def get_pdf_text(pdf_docs): 
    text = ""
    for pdf in pdf_docs: 
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text): 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks): 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.And make sure that you provide all the details,
    if the answer is not in the provided context, then just say "I don't know".don't provide the wrong answer.
    Context : \n{context}?\n
    Question : \n{question}\n

    
    Answer : 
    """

    model = ChatGoogleGenerativeAI(
        model = "gemini-1.5-flash",
        temperature=0.6,
    )

    prompt = PromptTemplate(
        template = prompt_template, 
        input_variables = ["context", "question"]
    )

    chain = load_qa_chain(
        llm = model, 
        chain_type = "stuff", 
        prompt = prompt
    )
    return chain

def user_input_handler(user_question): 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents" : docs,
        "question" : user_question}
    )

    print(response)
    st.write("Reply: ", response.get("output_text", "Sorry, no reply found."))

def main():
    st.set_page_config(page_title = "🥱Chat With Multiple PDF😉", layout = "centered")
    st.header("chat with Multiple PDF using Gemini")

    user_question = st.text_input("Ask a Question from PDF files")

    if user_question: 
        user_input_handler(user_question)

    with st.sidebar: 
        st.title("Menu : ")
        pdf_docs = st.file_uploader("Upload your PDF files and Click on the Submit And Process Button", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."): 
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()