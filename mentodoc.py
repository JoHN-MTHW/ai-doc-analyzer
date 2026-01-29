import os
import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# -----------------------------
# 🎨 Custom Styling
# -----------------------------
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #FFFFFF; }
.stChatInput input {
    background-color: #1E1E1E !important;
    color: #FFFFFF !important;
    border: 1px solid #3A3A3A !important;
}
.stChatMessage {
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
.stChatMessage:nth-child(odd) {
    background-color: #1E1E1E;
    border: 1px solid #3A3A3A;
}
.stChatMessage:nth-child(even) {
    background-color: #2A2A2A;
    border: 1px solid #404040;
}
.stFileUploader {
    background-color: #1E1E1E;
    border: 1px solid #3A3A3A;
    border-radius: 5px;
    padding: 15px;
}
h1, h2, h3 { color: #00FFAA; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 📌 Constants
# -----------------------------
PDF_STORAGE_PATH = "document_store/pdfs/"
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

AVAILABLE_MODELS = [
    "deepseek-r1:1.5b",
    "gemma3:4b"
]

PROMPT_TEMPLATE = """
You are an expert research assistant.
Answer strictly using the provided context.
If unsure, say you don't know.
Limit your answer to 3 concise sentences.

Query: {user_query}
Context: {document_context}
Answer:
"""

# -----------------------------
# 🧠 Session State Init
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = {}

# -----------------------------
# ⚙️ Sidebar
# -----------------------------
st.sidebar.header("⚙ Document Settings")

selected_model = st.sidebar.selectbox(
    "Choose Model",
    AVAILABLE_MODELS
)

st.sidebar.caption(f"🧠 Active model: `{selected_model}`")

if "last_model" not in st.session_state:
    st.session_state.last_model = selected_model

# 🔄 Handle model switch safely
if st.session_state.last_model != selected_model:
    st.session_state.chat_history.clear()
    st.session_state.doc_chunks.clear()
    st.session_state.uploaded_docs.clear()

    st.session_state.vector_store = InMemoryVectorStore(
        OllamaEmbeddings(model=selected_model)
    )

    st.session_state.last_model = selected_model
    st.sidebar.info("Model changed → Vector store rebuilt")

# Initialize vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore(
        OllamaEmbeddings(model=selected_model)
    )

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history.clear()
    st.sidebar.success("Chat cleared")

# -----------------------------
# 🤖 LLM
# -----------------------------
LLM = OllamaLLM(model=selected_model)

# -----------------------------
# 🛠 Utility Functions
# -----------------------------
def save_uploaded_file(uploaded_file):
    path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def load_pdf(path):
    return PDFPlumberLoader(path).load()

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return splitter.split_documents(docs)

def index_documents(doc_name, chunks):
    st.session_state.vector_store.add_documents(chunks)
    st.session_state.doc_chunks[doc_name] = chunks

def retrieve_docs(query):
    return st.session_state.vector_store.similarity_search(query, k=3)

def generate_answer(query, docs):
    context = "\n\n".join(d.page_content for d in docs)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | LLM
    return chain.invoke({
        "user_query": query,
        "document_context": context
    })

# -----------------------------
# 📄 Sidebar: Upload PDFs
# -----------------------------
uploaded_pdfs = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_pdfs:
    for pdf in uploaded_pdfs:
        if pdf.name not in st.session_state.uploaded_docs:
            path = save_uploaded_file(pdf)
            raw_docs = load_pdf(path)
            chunks = chunk_documents(raw_docs)
            index_documents(pdf.name, chunks)
            st.session_state.uploaded_docs.append(pdf.name)
    st.sidebar.success("PDFs indexed successfully")

st.sidebar.markdown("### 📚 Indexed Documents")
if not st.session_state.uploaded_docs:
    st.sidebar.caption("No documents uploaded")

for doc in st.session_state.uploaded_docs:
    col1, col2 = st.sidebar.columns([0.85, 0.15])
    col1.markdown(f"- {doc}")
    if col2.button("❌", key=f"del_{doc}"):
        st.session_state.uploaded_docs.remove(doc)
        st.session_state.doc_chunks.pop(doc, None)

        try:
            os.remove(os.path.join(PDF_STORAGE_PATH, doc))
        except:
            pass

        # Rebuild vector store safely
        embeddings = OllamaEmbeddings(model=selected_model)
        new_store = InMemoryVectorStore(embeddings)
        for chunks in st.session_state.doc_chunks.values():
            new_store.add_documents(chunks)

        st.session_state.vector_store = new_store
        st.rerun()

# -----------------------------
# 🧠 Main UI
# -----------------------------
st.title("📄 MENTO DOC")
st.markdown("---")

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

user_input = st.chat_input("Ask a question about the documents...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("🔍 Analyzing documents..."):
        related_docs = retrieve_docs(user_input)
        response = generate_answer(user_input, related_docs)

    st.session_state.chat_history.append(("assistant", response))
    with st.chat_message("assistant"):
        st.write(response)

    with st.expander("📄 Retrieved Sources"):
        for doc in related_docs:
            st.write(doc.metadata.get("source", "Unknown"))
