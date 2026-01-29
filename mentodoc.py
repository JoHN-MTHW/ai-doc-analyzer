import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# --------------------------------------------------
# UI STYLING
# --------------------------------------------------
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
.stChatMessage[data-testid="stChatMessage"] {
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
.stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
    background-color: #1E1E1E !important;
    border: 1px solid #3A3A3A !important;
}
.stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
    background-color: #2A2A2A !important;
    border: 1px solid #404040 !important;
}
h1, h2, h3 {
    color: #00FFAA !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
PDF_STORAGE_PATH = "document_store/pdfs/"
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

PROMPT_TEMPLATE = """
You are an expert research assistant.
Answer using ONLY the provided context.
If the answer is not found, say you don't know.
Keep responses concise (max 3 sentences).

Question: {user_query}
Context: {document_context}
Answer:
"""

# --------------------------------------------------
# SIDEBAR – MODEL SETTINGS
# --------------------------------------------------
st.sidebar.header("🧠 Model Settings")

LLM_MODEL = st.sidebar.selectbox(
    "Choose LLM",
    ["deepseek-r1:1.5b", "gemma3:4b"],
    help="Select the local Ollama model for answering"
)

EMBED_MODEL = "nomic-embed-text"

st.sidebar.markdown(f"""
**Embeddings Model:** `{EMBED_MODEL}`  
*(Used for document indexing)*
""")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = {}

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore(
        OllamaEmbeddings(model=EMBED_MODEL)
    )

# --------------------------------------------------
# LLM INITIALIZATION
# --------------------------------------------------
LLM = OllamaLLM(model=LLM_MODEL)

# --------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------
def save_uploaded_file(uploaded_file):
    path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def load_pdf(file_path):
    return PDFPlumberLoader(file_path).load()

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

def rebuild_vector_store():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    store = InMemoryVectorStore(embeddings)
    for chunks in st.session_state.doc_chunks.values():
        store.add_documents(chunks)
    st.session_state.vector_store = store

def generate_answer(query, docs):
    context = "\n\n".join([d.page_content for d in docs])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | LLM
    return chain.invoke({
        "user_query": query,
        "document_context": context
    })

# --------------------------------------------------
# SIDEBAR – DOCUMENT MANAGEMENT
# --------------------------------------------------
st.sidebar.header("📄 Documents")

uploaded_pdfs = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat cleared")

st.sidebar.markdown("**Indexed Documents:**")

for doc in list(st.session_state.uploaded_docs):
    col1, col2 = st.sidebar.columns([0.8, 0.2])
    with col1:
        st.write(doc)
    with col2:
        if st.button("❌", key=f"del_{doc}"):
            st.session_state.uploaded_docs.remove(doc)
            st.session_state.doc_chunks.pop(doc, None)
            try:
                os.remove(os.path.join(PDF_STORAGE_PATH, doc))
            except:
                pass
            rebuild_vector_store()

# --------------------------------------------------
# HANDLE PDF UPLOAD
# --------------------------------------------------
if uploaded_pdfs:
    for pdf in uploaded_pdfs:
        if pdf.name not in st.session_state.uploaded_docs:
            path = save_uploaded_file(pdf)
            raw_docs = load_pdf(path)
            chunks = chunk_documents(raw_docs)

            st.session_state.doc_chunks[pdf.name] = chunks
            st.session_state.uploaded_docs.append(pdf.name)
            st.session_state.vector_store.add_documents(chunks)

    st.sidebar.success("Documents indexed successfully!")

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
st.title("📘 MENTO DOC")
st.caption(f"LLM: {LLM_MODEL} | Embeddings: {EMBED_MODEL}")
st.markdown("---")

# Display chat
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

# Chat input
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Analyzing documents..."):
        docs = st.session_state.vector_store.similarity_search(user_input)
        response = generate_answer(user_input, docs)

    st.session_state.chat_history.append(("assistant", response))
    with st.chat_message("assistant", avatar="🤖"):
        st.write(response)
