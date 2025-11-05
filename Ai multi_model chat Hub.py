import os
import time
import streamlit as st
import google.generativeai as genai
from PIL import Image
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import requests
import asyncio

# LangChain imports
from langchain_groq import ChatGroq
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Fix event loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🤖 Unified AI Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🤖 Unified AI Assistant - All-in-One Chatbot")
st.markdown("*Your comprehensive AI companion for chat, vision, documents, and more!*")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings & Mode Selection")
    
    # API Keys
    st.subheader("🔑 API Keys")
    google_api_key = st.text_input("Google API Key", type="password", 
                                  value=os.getenv("GOOGLE_API_KEY", ""))
    groq_api_key = st.text_input("Groq API Key", type="password", 
                                value=os.getenv("GROQ_API_KEY", ""))
    
    # Mode selection
    st.subheader("🎯 Choose Mode")
    mode = st.selectbox(
        "Select Assistant Mode",
        [
            "💬 General Chat",
            "🍎 Calorie Counter",
            "👁️ Vision Analysis",
            "📄 PDF Chat",
            "🌐 Website Chat",
            "📋 Invoice Extractor",
            "🎨 Image Generator"
        ]
    )
    
    # Model selection for chat modes
    if mode == "💬 General Chat":
        st.subheader("🤖 Model Selection")
        model_name = st.selectbox(
            "Choose Model",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],  # Updated to remove deprecated gemma2-9b-it
            index=0
        )
    elif mode == "🌐 Website Chat":  # Added model selector for Website Chat
        st.subheader("🤖 Model Selection for Website Chat")
        website_model = st.selectbox(
            "Choose Model",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
            index=0
        )
    
    # Clear conversation
    if st.button("🗑️ Clear Conversation"):
        for key in list(st.session_state.keys()):
            if key.startswith('messages') or key.startswith('chat_history') or key.startswith('gemini_chat') or key in ['vectors', 'docs', 'pdf_processed']:
                del st.session_state[key]
        st.rerun()

# Initialize session state
if f"messages_{mode}" not in st.session_state:
    st.session_state[f"messages_{mode}"] = []

# Configure APIs
if google_api_key:
    genai.configure(api_key=google_api_key)
    os.environ["GOOGLE_API_KEY"] = google_api_key

# Helper Functions
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
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
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
        Use the following context to answer the question. If you don't know the answer, say "I don't know".
        
        Context: {context}
        Question: {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        
        return db, chain, True
        
    except Exception as e:
        return None, str(e), False

@st.cache_resource
def get_chat_chain(_api_key, _model_name):
    """Initialize chat chain"""
    llm = ChatGroq(
        groq_api_key=_api_key,
        model_name=_model_name,
        temperature=0.6,
        streaming=True
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You're Jani, a brilliant, sweet, and versatile AI assistant powered by Groq – loved by Sweetheart. 💖

        You act as:
        A top-tier tutor who explains complex topics with clear examples and analogies,  
        A coding expert who writes clean, commented Python and AI code with clear explanations,  
        A document Q&A expert who answers based on uploaded content precisely,  
        A warm, kind, friendly helper who talks gently and supportively – especially to Sweetheart,  
        And a human-like conversational bot who keeps interactions natural, helpful, and light-hearted.
        If the user ever calls you "babby", lovingly respond with "babby2 😄" before continuing.
        Always stay helpful, clear, encouraging, and personal. Your goal is to be the perfect assistant in every way."""),
        ("user", "{question}")
    ])
    
    return prompt | llm | StrOutputParser()

# Mode-specific implementations
if mode == "💬 General Chat":
    st.subheader("💬 General Chat Mode")
    
    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar")
    else:
        try:
            chain = get_chat_chain(groq_api_key, model_name)
            
            # Display chat history
            for message in st.session_state[f"messages_{mode}"]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Chat input
            if question := st.chat_input("Ask me anything..."):
                st.session_state[f"messages_{mode}"].append({"role": "user", "content": question})
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
                        st.session_state[f"messages_{mode}"].append({"role": "assistant", "content": full_response})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"Failed to initialize model: {str(e)}")

elif mode == "🍎 Calorie Counter":
    st.subheader("🍎 Calorie Counter Mode")
    
    if not google_api_key:
        st.warning("Please enter your Google API key in the sidebar")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            food_description = st.text_area("Describe your food (optional):", height=100)
            uploaded_file = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])
        
        with col2:
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Food Image", use_column_width=True)
        
        if st.button("🔍 Analyze Food"):
            if uploaded_file or food_description:
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash-latest")
                    
                    input_prompt = """
                    You are a helpful nutrition and health assistant.
                    Your task is to analyze the uploaded food image and/or the user's text input.
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
                    
                    if uploaded_file:
                        image_bytes = uploaded_file.getvalue()
                        image_parts = [{"mime_type": uploaded_file.type, "data": image_bytes}]
                        response = model.generate_content([food_description, image_parts[0], input_prompt])
                    else:
                        response = model.generate_content([food_description, input_prompt])
                    
                    st.markdown("### 📊 Analysis Results:")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Error analyzing food: {e}")
            else:
                st.warning("Please upload an image or describe your food!")

elif mode == "👁️ Vision Analysis":
    st.subheader("👁️ Vision Analysis Mode")
    
    if not google_api_key:
        st.warning("Please enter your Google API key in the sidebar")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            prompt = st.text_area("What would you like to know about the image?", height=100)
            uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        
        with col2:
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Analyze Image"):
            if uploaded_file or prompt:
                try:
                    model = genai.GenerativeModel("gemini-2.0-flash-exp")
                    
                    if uploaded_file and prompt:
                        response = model.generate_content([prompt, image])
                    elif uploaded_file:
                        response = model.generate_content([image])
                    else:
                        response = model.generate_content([prompt])
                    
                    st.markdown("### 🔍 Analysis Results:")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please upload an image or enter a prompt!")

elif mode == "📄 PDF Chat":
    st.subheader("📄 PDF Chat Mode")
    
    if not google_api_key:
        st.warning("Please enter your Google API key in the sidebar")
    else:
        # Embedding option selection
        embedding_option = st.radio(
            "🔧 Choose Embedding Method:",
            ["🚀 Google Gemini (Limited free quota)", "🤗 HuggingFace (Unlimited & Free)"],
            index=1,
            help="HuggingFace embeddings are completely free and unlimited!"
        )
        
        use_huggingface = "HuggingFace" in embedding_option
        
        # File upload section
        with st.expander("📁 Upload PDF Files", expanded=True):
            pdf_docs = st.file_uploader(
                "Upload your PDF files", 
                type="pdf", 
                accept_multiple_files=True,
                help="You can upload multiple PDF files at once"
            )
            
            if st.button("📚 Process PDFs", type="primary") and pdf_docs:
                with st.spinner("🔄 Processing PDFs... This may take a moment."):
                    try:
                        # Extract text
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.error("❌ No text found in the uploaded PDFs. Please check your files.")
                        else:
                            st.info(f"📄 Extracted {len(raw_text)} characters from {len(pdf_docs)} PDF(s)")
                            
                            # Create chunks with progress
                            text_chunks = get_text_chunks(raw_text)
                            st.info(f"📝 Created {len(text_chunks)} text chunks")
                            
                            # Show processing progress
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("Creating embeddings...")
                            progress_bar.progress(50)
                            
                            # Create vector store
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
                        st.error(f"❌ Error processing PDFs: {str(e)}")
        
        # Chat with PDFs section
        if st.session_state.get('pdf_processed', False):
            st.markdown("### 💬 Chat with your PDFs")
            
            embedding_type = st.session_state.get('embedding_type', 'google')
            st.info(f"🔧 Using {embedding_type.title()} embeddings")
            
            user_question = st.text_input(
                "❓ Ask a question about your PDFs:",
                placeholder="e.g., What is the main topic discussed in the documents?",
                key="pdf_question"
            )
            
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
                        db, chain, success = get_pdf_qa_chain(embedding_type)
                        
                        if success:
                            # Search for relevant documents
                            docs = db.similarity_search(user_question, k=4)
                            
                            # Get answer
                            response = chain({"input_documents": docs, "question": user_question})
                            answer = response.get("output_text", "Sorry, I couldn't generate an answer.")
                            
                            # Display results
                            st.markdown("### 📖 Answer:")
                            st.markdown(f"**Question:** {user_question}")
                            st.markdown(f"**Answer:** {answer}")
                            
                        else:
                            st.error(f"❌ Error setting up QA chain: {chain}")
                            
                    except Exception as e:
                        st.error(f"❌ Error answering question: {str(e)}")
        else:
            st.info("📁 Please upload and process PDF files first to start chatting!")

elif mode == "🌐 Website Chat":
    st.subheader("🌐 Website Chat Mode")
    
    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar for the chat model")
    else:
        # Embedding option selection
        embedding_option = st.radio(
            "🔧 Choose Embedding Method:",
            ["🤗 HuggingFace (Unlimited & Free)", "🚀 Google Gemini (Limited free quota)"],
            index=0,
            help="HuggingFace embeddings are completely free and unlimited!"
        )
        
        use_huggingface = "HuggingFace" in embedding_option
        
        col1, col2 = st.columns([3, 1])
        with col1:
            url = st.text_input("🔗 Enter website URL:", placeholder="https://example.com")
        with col2:
            load_button = st.button("📥 Load Website", type="primary", disabled=not url)
        
        # Load website content
        if load_button and url:
            if "vectors" in st.session_state:
                del st.session_state["vectors"]
                
            with st.spinner("🔄 Loading website content... This may take a moment."):
                try:
                    # Load website content
                    st.session_state.loader = WebBaseLoader(url)
                    st.session_state.docs = st.session_state.loader.load()
                    
                    if not st.session_state.docs:
                        st.error("❌ No content found on the website. Please check the URL.")
                    else:
                        st.info(f"📄 Loaded content from: {url}")
                        
                        # Split documents
                        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, chunk_overlap=300
                        )
                        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                            st.session_state.docs
                        )
                        st.info(f"📝 Created {len(st.session_state.final_documents)} text chunks")
                        
                        # Create embeddings and vector store
                        try:
                            if use_huggingface:
                                embeddings, status = setup_huggingface_embeddings()
                                if embeddings:
                                    st.info("✅ Using HuggingFace embeddings (free & unlimited)")
                                    st.session_state.website_embedding_type = "huggingface"
                                else:
                                    st.error(f"❌ Failed to setup HuggingFace embeddings: {status}")
                                    raise Exception(f"HuggingFace setup failed: {status}")
                            else:
                                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                                st.info("✅ Using Google Gemini embeddings")
                                st.session_state.website_embedding_type = "google"
                            
                            st.session_state.embeddings = embeddings
                                
                            # Create vector store
                            st.session_state.vectors = FAISS.from_documents(
                                st.session_state.final_documents, st.session_state.embeddings
                            )
                            st.session_state.website_loaded = True
                            
                            st.success("✅ Website content loaded successfully! You can now ask questions.")
                                
                        except Exception as embed_error:
                            error_msg = str(embed_error)
                            if "429" in error_msg or "quota" in error_msg.lower():
                                st.error("❌ Google embeddings quota exceeded!")
                                st.warning("🔄 Automatically switching to HuggingFace embeddings...")
                                
                                embeddings, status = setup_huggingface_embeddings()
                                if embeddings:
                                    try:
                                        st.session_state.embeddings = embeddings
                                        st.session_state.vectors = FAISS.from_documents(
                                            st.session_state.final_documents, st.session_state.embeddings
                                        )
                                        st.session_state.website_loaded = True
                                        st.session_state.website_embedding_type = "huggingface"
                                        st.success("✅ Successfully switched to HuggingFace embeddings!")
                                    except Exception as hf_error:
                                        st.error(f"❌ HuggingFace fallback failed: {str(hf_error)}")
                                else:
                                    st.error(f"❌ Could not setup HuggingFace embeddings: {status}")
                            else:
                                st.error(f"❌ Error creating embeddings: {error_msg}")
                                
                except Exception as e:
                    st.error(f"❌ Error loading website: {str(e)}")
        
        # Chat with website
        if st.session_state.get('website_loaded', False):
            st.markdown("### 💬 Chat with Website Content")
            
            embedding_type = st.session_state.get('website_embedding_type', 'google')
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"🔧 Using {embedding_type.title()} embeddings")
            with col2:
                if st.button("🗑️ Clear Website"):
                    for key in ['vectors', 'website_loaded', 'website_embedding_type', 'docs', 'final_documents', 'embeddings']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            
            user_prompt = st.text_input(
                "💭 Ask about the website content:",
                placeholder="e.g., What is the main topic of this website?",
                key="website_question"
            )
            
            if st.button("🔍 Ask Question", type="primary", disabled=not user_prompt) and user_prompt:
                with st.spinner("🔍 Searching through website content..."):
                    try:
                        # Initialize Groq LLM
                        llm = ChatGroq(
                            model_name=website_model,  # Use selected model from sidebar
                            temperature=0.3, 
                            api_key=groq_api_key
                        )
                        
                        # Create prompt template
                        prompt = ChatPromptTemplate.from_template("""
                        You are a helpful AI assistant analyzing website content.
                        Answer the question based on the provided context only.
                        If the answer is not in the context, say "I don't have information about that in the website content."
                        
                        <context>
                        {context}
                        </context>
                        
                        Question: {input}
                        
                        Answer:
                        """)
                        
                        # Create retrieval chain
                        document_chain = create_stuff_documents_chain(llm, prompt)
                        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 4})
                        retrieval_chain = create_retrieval_chain(retriever, document_chain)
                        
                        # Get response
                        start = time.process_time()
                        response = retrieval_chain.invoke({"input": user_prompt})
                        response_time = round(time.process_time() - start, 2)
                        
                        # Display results
                        st.markdown("### 📖 Answer:")
                        st.markdown(f"**Question:** {user_prompt}")
                        st.markdown(f"**Answer:** {response['answer']}")
                        st.info(f"⏱️ Response time: {response_time}s")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing question: {str(e)}")
        else:
            st.info("🌐 Please enter a website URL and click 'Load Website' first!")

elif mode == "📋 Invoice Extractor":
    st.subheader("📋 Invoice Extractor Mode")
    
    if not google_api_key:
        st.warning("Please enter your Google API key in the sidebar")
    else:
        uploaded_file = st.file_uploader(
            "📄 Upload invoice image", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Invoice", use_column_width=True)
            
            user_question = st.text_input("❓ Ask about this invoice:")
            
            if st.button("🔍 Analyze Invoice") and user_question:
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash-latest")
                    image_bytes = uploaded_file.getvalue()
                    
                    chat_prompt = (
                        f"You are an invoice analysis assistant. "
                        f"Answer the following question about the uploaded invoice image: {user_question}"
                    )
                    
                    response = model.generate_content([
                        chat_prompt,
                        {"mime_type": uploaded_file.type, "data": image_bytes}
                    ])
                    
                    st.markdown("### 📊 Analysis Result:")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Error analyzing invoice: {e}")

elif mode == "🎨 Image Generator":
    st.subheader("🎨 Image Generator Mode")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_input("🎨 Describe the image you want to generate:")
        
        col_a, col_b = st.columns(2)
        with col_a:
            width = st.number_input("Width", min_value=256, max_value=1024, value=1024, step=64)
            height = st.number_input("Height", min_value=256, max_value=1024, value=1024, step=64)
        with col_b:
            seed = st.number_input("Seed (for reproducibility)", min_value=1, value=42)
            model = st.selectbox("Model", ["flux", "turbo", "flux-realism"])
    
    with col2:
        if st.button("🎨 Generate Image") and prompt:
            with st.spinner("Generating image..."):
                try:
                    image_url = f"https://pollinations.ai/p/{prompt}?width={width}&height={height}&seed={seed}&model={model}"
                    
                    st.markdown("### 🖼️ Generated Image:")
                    st.image(image_url, caption=f"Generated: {prompt}")
                    
                    # Download option
                    response = requests.get(image_url)
                    st.download_button(
                        label="📥 Download Image",
                        data=response.content,
                        file_name=f"generated_{seed}.jpg",
                        mime="image/jpeg"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating image: {e}")

# Footer
st.markdown("---")
# st.markdown("*🤖 Powered by Google Gemini, Groq, LangChain & Streamlit | Made with ❤️*")