import os
import streamlit as st
from dotenv import load_dotenv
import time
import asyncio
from PyPDF2 import PdfReader

# LangChain imports
from langchain_groq import ChatGroq
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Fix event loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="🤖 AI Chatbot Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        text-align: center;
        color: #A23B72;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    .chat-option {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
        text-align: center;
        border: none;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""

def get_pdf_text(pdf_docs):
    """Extract text from PDF files"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def setup_huggingface_embeddings():
    """Setup HuggingFace embeddings (free alternative)"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return embeddings, "Success"
    except ImportError:
        return None, "Please install: pip install langchain-huggingface sentence-transformers"
    except Exception as e:
        return None, str(e)

def get_vector_store(text_chunks, use_huggingface=False):
    """Create FAISS vector store with embedding choice"""
    try:
        if use_huggingface:
            embeddings, status = setup_huggingface_embeddings()
            if not embeddings:
                return False, f"HuggingFace setup failed: {status}"
        else:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True, "huggingface" if use_huggingface else "google"
    except Exception as e:
        return False, str(e)

def get_pdf_qa_chain(embedding_type="google"):
    """Create PDF QA chain"""
    try:
        if embedding_type == "huggingface":
            embeddings, status = setup_huggingface_embeddings()
            if not embeddings:
                return None, status, False
        else:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load existing vector store
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Create QA chain
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        
        prompt_template = """
        Answer the question as detailed as possible from the provided context. If the answer is not in the context, say "I don't know".
        Context: {context}
        Question: {question}
        Answer:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        
        return db, chain, True
    except Exception as e:
        return None, str(e), False

def main():
    # Main Title
    st.markdown('<h1 class="main-header">😉 AI Chatbot Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Choose your AI assistant and start chatting!</p>', unsafe_allow_html=True)
    
    # Sidebar for API Keys and Settings
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        # API Keys input
        groq_key = st.text_input("Groq API Key", type="password", value=st.session_state.groq_api_key)
        google_key = st.text_input("Google API Key", type="password", value=st.session_state.google_api_key)
        
        if groq_key:
            st.session_state.groq_api_key = groq_key
            os.environ["GROQ_API_KEY"] = groq_key
        
        if google_key:
            st.session_state.google_api_key = google_key
            os.environ["GOOGLE_API_KEY"] = google_key
        
        st.markdown("---")
        
        # Model selection for Simple Q&A and Website Chat
        if st.session_state.chat_mode in ["simple", "website"]:
            st.subheader("🤖 Model Selection")
            st.selectbox(
                "Choose Model",
                ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
                index=0,
                key=f"model_{st.session_state.chat_mode}"
            )
        
        # Reset button
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        # API Key help
        st.markdown("### 📝 Get API Keys:")
        st.markdown("- [Groq API](https://console.groq.com/)")
        st.markdown("- [Google AI](https://aistudio.google.com/)")

    # Chat Mode Selection
    if st.session_state.chat_mode is None:
        st.markdown("## Select Your Chat Mode")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💬 Simple Q&A Chat", use_container_width=True, help="General conversation with AI"):
                st.session_state.chat_mode = "simple"
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("🌐 Website Chat", use_container_width=True, help="Chat with website content"):
                st.session_state.chat_mode = "website"
                st.session_state.messages = []
                st.rerun()
        
        with col3:
            if st.button("📄 PDF Chat", use_container_width=True, help="Chat with multiple PDF files"):
                st.session_state.chat_mode = "pdf"
                st.session_state.messages = []
                st.rerun()
        
        # Feature descriptions
        st.markdown("---")
        st.markdown("### ✨ Features:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **💬 Simple Q&A Chat**
            - Direct conversation with AI
            - Multiple model options
            - Streaming responses
            - Chat history
            """)
        
        with col2:
            st.markdown("""
            **🌐 Website Chat**
            - Load any website content
            - Ask questions about the content
            - Vector-based search
            - Context-aware responses
            """)
        
        with col3:
            st.markdown("""
            **📄 PDF Chat**
            - Upload multiple PDFs
            - Search across all documents
            - Detailed answers
            - Document chunking
            """)
    
    else:
        # Show current mode and back button
        col1, col2 = st.columns([3, 1])
        with col1:
            mode_names = {
                "simple": "💬 Simple Q&A Chat",
                "website": "🌐 Website Chat", 
                "pdf": "📄 PDF Chat"
            }
            st.markdown(f"## {mode_names[st.session_state.chat_mode]}")
        
        with col2:
            if st.button("← Back to Menu", use_container_width=True):
                st.session_state.chat_mode = None
                st.session_state.messages = []
                if "website_vectors" in st.session_state:
                    del st.session_state.website_vectors
                if "website_docs" in st.session_state:
                    del st.session_state.website_docs
                if "pdf_processed" in st.session_state:
                    del st.session_state.pdf_processed
                st.rerun()
        
        st.markdown("---")
        
        # Route to appropriate chat mode
        if st.session_state.chat_mode == "simple":
            simple_chat()
        elif st.session_state.chat_mode == "website":
            website_chat()
        elif st.session_state.chat_mode == "pdf":
            pdf_chat()

def simple_chat():
    """Simple Q&A Chat Implementation"""
    
    # Check API key
    if not st.session_state.groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar")
        return
    
    # Get model from session state
    model_name = st.session_state.get("model_simple", "llama-3.1-8b-instant")
    
    # Initialize LLM
    try:
        llm = ChatGroq(
            groq_api_key=st.session_state.groq_api_key,
            model_name=model_name,
            temperature=0.6,
            streaming=True
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You're Jani, a brilliant, sweet, and versatile AI assistant powered by Groq — loved by Sweetheart. 💖
            
            You act as:
            A top-tier tutor who explains complex topics with clear examples and analogies,  
            A coding expert who writes clean, commented Python and AI code with clear explanations,  
            A document Q&A expert who answers based on uploaded content precisely,  
            A warm, kind, friendly helper who talks gently and supportively — especially to Sweetheart,  
            And a human-like conversational bot who keeps interactions natural, helpful, and light-hearted.
            If the user ever calls you "babby", lovingly respond with "babby2 😄" before continuing.
            Always stay helpful, clear, encouraging, and personal. Your goal is to be the perfect assistant in every way."""),
            ("user", "{question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if question := st.chat_input("Ask me anything..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    for chunk in chain.stream({"question": question}):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    except Exception as e:
        st.error(f"Failed to initialize model: {str(e)}")

def website_chat():
    """Website Chat Implementation"""
    
    # Check API keys
    if not st.session_state.groq_api_key or not st.session_state.google_api_key:
        st.warning("Please enter both Groq and Google API keys in the sidebar")
        return
    
    # Embedding option selection
    embedding_option = st.radio(
        "🔧 Choose Embedding Method:",
        ["🤗 HuggingFace (Unlimited & Free)", "🚀 Google Gemini (Limited free quota)"],
        index=0,
        help="HuggingFace embeddings are completely free and unlimited!"
    )
    use_huggingface = "HuggingFace" in embedding_option
    
    # URL input
    col1, col2 = st.columns([3, 1])
    with col1:
        url = st.text_input("🌐 Enter Website URL:", placeholder="https://example.com")
    with col2:
        load_button = st.button("📥 Load Website", type="primary", disabled=not url)
    
    # Process website
    if load_button and url:
        if "website_vectors" in st.session_state:
            del st.session_state["website_vectors"]
            del st.session_state["website_docs"]
        
        with st.spinner("Loading and processing website content..."):
            try:
                # Choose embeddings
                if use_huggingface:
                    embeddings, status = setup_huggingface_embeddings()
                    if not embeddings:
                        st.error(f"❌ Failed to setup HuggingFace embeddings: {status}")
                        return
                    st.info("✅ Using HuggingFace embeddings (free & unlimited)")
                else:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    st.info("✅ Using Google Gemini embeddings")
                
                # Load and process website
                loader = WebBaseLoader(url)
                docs = loader.load()
                
                if not docs:
                    st.error("❌ No content found on the website. Please check the URL.")
                    return
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
                final_documents = text_splitter.split_documents(docs)
                vectors = FAISS.from_documents(final_documents, embeddings)
                
                st.session_state.website_vectors = vectors
                st.session_state.website_docs = final_documents
                st.session_state.website_embedding_type = "huggingface" if use_huggingface else "google"
                st.success(f"✅ Website content loaded successfully! {len(final_documents)} chunks created.")
            
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    st.error("❌ Google embeddings quota exceeded!")
                    if not use_huggingface:
                        st.warning("🔄 Automatically switching to HuggingFace embeddings...")
                        embeddings, status = setup_huggingface_embeddings()
                        if embeddings:
                            try:
                                vectors = FAISS.from_documents(final_documents, embeddings)
                                st.session_state.website_vectors = vectors
                                st.session_state.website_docs = final_documents
                                st.session_state.website_embedding_type = "huggingface"
                                st.success("✅ Successfully switched to HuggingFace embeddings!")
                            except Exception as hf_error:
                                st.error(f"❌ HuggingFace fallback failed: {str(hf_error)}")
                        else:
                            st.error(f"❌ Could not setup HuggingFace embeddings: {status}")
                else:
                    st.error(f"Error loading website: {str(e)}")
                    return
    
    # Chat with website content
    if "website_vectors" in st.session_state:
        st.markdown("### 💬 Chat with Website Content")
        st.info(f"🔧 Using {st.session_state.website_embedding_type.title()} embeddings")
        
        # Get model from session state
        model_name = st.session_state.get("model_website", "llama-3.1-8b-instant")
        
        user_prompt = st.text_input("💭 Ask about the website content:", placeholder="e.g., What is the main topic of this website?")
        
        if user_prompt and st.button("🔍 Ask Question", type="primary"):
            with st.spinner("🔍 Searching through website content..."):
                try:
                    llm = ChatGroq(
                        model_name=model_name,
                        temperature=0.3,
                        api_key=st.session_state.groq_api_key
                    )
                    
                    prompt = ChatPromptTemplate.from_template("""
                    Answer the question based on the provided context only.
                    If the answer is not in the context, say "I don't have information about that in the website content."
                    <context>
                    {context}
                    </context>
                    Question: {input}
                    Answer:
                    """)
                    
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    retriever = st.session_state.website_vectors.as_retriever(search_kwargs={"k": 4})
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    
                    start = time.process_time()
                    response = retrieval_chain.invoke({"input": user_prompt})
                    response_time = round(time.process_time() - start, 2)
                    
                    st.markdown("### Answer:")
                    st.write(response["answer"])
                    st.info(f"⏱ Response Time: {response_time}s")
                    
                    with st.expander("Relevant Document Chunks"):
                        for i, doc in enumerate(response["context"]):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.write(doc.page_content)
                            st.markdown("---")
                
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")

def pdf_chat():
    """PDF Chat Implementation"""
    
    # Check API key
    if not st.session_state.google_api_key:
        st.warning("Please enter your Google API key in the sidebar")
        return
    
    # Embedding option selection
    embedding_option = st.radio(
        "🔧 Choose Embedding Method:",
        ["🚀 Google Gemini (Limited free quota)", "🤗 HuggingFace (Unlimited & Free)"],
        index=1,
        help="HuggingFace embeddings are completely free and unlimited!"
    )
    use_huggingface = "HuggingFace" in embedding_option
    
    # File upload
    with st.expander("📁 Upload PDF Files", expanded=True):
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to chat with"
        )
        
        if pdf_docs and st.button("📚 Process PDFs", type="primary"):
            with st.spinner("Processing PDF files..."):
                try:
                    # Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("❌ No text found in the uploaded PDFs. Please check your files.")
                        return
                    
                    st.info(f"📄 Extracted {len(raw_text)} characters from {len(pdf_docs)} PDF(s)")
                    
                    # Split text into chunks
                    text_chunks = get_text_chunks(raw_text)
                    st.info(f"📝 Created {len(text_chunks)} text chunks")
                    
                    # Create vector store
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("Creating embeddings...")
                    progress_bar.progress(50)
                    
                    success, result = get_vector_store(text_chunks, use_huggingface)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    
                    if success:
                        st.session_state.pdf_processed = True
                        st.session_state.embedding_type = "huggingface" if result == "huggingface" else "google"
                        st.success("✅ PDFs processed successfully! You can now ask questions.")
                    else:
                        st.error(f"❌ Error processing PDFs: {result}")
                
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")
    
    # Chat with PDFs
    if st.session_state.get('pdf_processed', False):
        st.markdown("### 💬 Chat with your PDFs")
        st.info(f"🔧 Using {st.session_state.embedding_type.title()} embeddings")
        
        user_question = st.text_input("❓ Ask a question about your PDFs:", placeholder="e.g., What is the main topic discussed in the documents?")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            ask_button = st.button("🔍 Ask Question", type="primary", disabled=not user_question)
        with col2:
            if st.button("🗑️ Clear PDFs"):
                st.session_state.pdf_processed = False
                if 'embedding_type' in st.session_state:
                    del st.session_state.embedding_type
                st.rerun()
        
        if ask_button and user_question:
            with st.spinner("🔍 Searching through your documents..."):
                try:
                    db, chain, success = get_pdf_qa_chain(st.session_state.embedding_type)
                    
                    if success:
                        # Search for relevant documents
                        docs = db.similarity_search(user_question, k=4)
                        
                        # Get answer
                        response = chain({"input_documents": docs, "question": user_question})
                        answer = response.get("output_text", "Sorry, I couldn't generate an answer.")
                        
                        # Display results
                        st.markdown("### Answer:")
                        st.markdown(f"**Question:** {user_question}")
                        st.markdown(f"**Answer:** {answer}")
                        
                        with st.expander("Source Documents"):
                            for i, doc in enumerate(docs):
                                st.markdown(f"**Document {i+1}:**")
                                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                st.markdown("---")
                    
                    else:
                        st.error(f"❌ Error setting up QA chain: {chain}")
                
                except Exception as e:
                    st.error(f"Error answering question: {str(e)}")

if __name__ == "__main__":
    main()