import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# --- Custom Styling ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Constants ---
PDF_STORAGE_PATH = 'document_store/pdfs/'
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore(OllamaEmbeddings(model="deepseek-r1:1.5b"))

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = {}

# --- LLM ---
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# --- Utility Functions ---
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    return PDFPlumberLoader(file_path).load()

def chunk_documents(raw_documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return splitter.split_documents(raw_documents)

def index_documents(doc_name, chunks):
    st.session_state.vector_store.add_documents(chunks)
    st.session_state.doc_chunks[doc_name] = chunks

def find_related_documents(query):
    return st.session_state.vector_store.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | LANGUAGE_MODEL
    return chain.invoke({"user_query": user_query, "document_context": context_text})

# --- Sidebar ---
st.sidebar.header("Document Settings")

uploaded_pdfs = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True,
    help="Add one or more PDF files"
)

if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared.")

st.sidebar.markdown("*Indexed Documents:*")
if st.session_state.uploaded_docs:
    for doc in st.session_state.uploaded_docs:
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        with col1:
            st.markdown(f"- {doc}")
        with col2:
            if st.button("❌", key=f"delete_{doc}"):
                # Remove from memory
                st.session_state.uploaded_docs.remove(doc)
                st.session_state.doc_chunks.pop(doc, None)

                # Delete file
                try:
                    os.remove(os.path.join(PDF_STORAGE_PATH, doc))
                except:
                    st.sidebar.warning(f"Failed to delete {doc} from disk.")

                # Rebuild vector store
                embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
                new_store = InMemoryVectorStore(embeddings)
                for chunks in st.session_state.doc_chunks.values():
                    new_store.add_documents(chunks)
                st.session_state.vector_store = new_store

                
else:
    st.sidebar.markdown("No documents uploaded yet.")

# --- Upload Handling ---
if uploaded_pdfs:
    for pdf in uploaded_pdfs:
        if pdf.name not in st.session_state.uploaded_docs:
            file_path = save_uploaded_file(pdf)
            raw_docs = load_pdf_documents(file_path)
            chunks = chunk_documents(raw_docs)
            index_documents(pdf.name, chunks)
            st.session_state.uploaded_docs.append(pdf.name)
    st.sidebar.success("All new PDFs processed!")

# --- Main App ---
st.title(" MENTO DOC")
st.markdown("---")

# Display chat history
for role, message in st.session_state.chat_history:
    with st.chat_message(role, avatar="🤖" if role == "assistant" else None):
        st.write(message)

# Chat input
user_input = st.chat_input("Enter your question about the documents...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("Analyzing documents..."):
        related_docs = find_related_documents(user_input)
        response = generate_answer(user_input, related_docs)

    st.session_state.chat_history.append(("assistant", response))
    with st.chat_message("assistant", avatar="🤖"):
        st.write(response)