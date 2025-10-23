# MENTO DOC 📝

[![Streamlit](https://img.shields.io/badge/Streamlit-App-blue?logo=streamlit)](https://streamlit.io/)  
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)]()   

MENTO DOC is an AI-powered document assistant built with **Streamlit** and **LangChain**. It allows users to upload PDFs, automatically index their content, and ask questions about the documents using a chatbot interface powered by **Ollama** embeddings.

---

## ⚠️ Important

MENTO DOC requires a **local Ollama model** installed on your machine.  
You must have the Ollama desktop app installed and the required model (`deepseek-r1:1.5b`) downloaded locally before running the app.

---

## Features

- Upload multiple PDF documents
- Automatic chunking and indexing of PDF content
- Contextual question-answering using AI
- Chat interface with styled messages
- Clear chat history and remove uploaded documents
- Custom dark-themed UI

---

## Installation

1. **Install Ollama**  
   Download and install the [Ollama app](https://ollama.com/).  
   Download the required model (`deepseek-r1:1.5b`) through Ollama.

2. **Clone the repository:**

```bash
git clone https://github.com/yourusername/mento-doc.git
cd mento-doc
````

3. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

4. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## Usage

1. **Run the Streamlit app:**

```bash
streamlit run app.py
```

2. **Upload PDFs:**
   Use the sidebar to upload one or multiple PDF files.

3. **Ask Questions:**
   Type your query in the chat box. The AI will provide concise, factual answers based on the uploaded documents.

4. **Manage Documents and Chat:**

* Clear chat history via the sidebar button
* Delete indexed documents with the ❌ button next to the document name

---

## File Structure

```
mento-doc/
│
├── app.py               # Main Streamlit app
├── requirements.txt     # Python dependencies
├── document_store/      # Folder to store uploaded PDFs
│   └── pdfs/
└── README.md
```

---

## Technologies Used

* **[Streamlit](https://streamlit.io/)** – Web app framework for Python
* **[LangChain](https://www.langchain.com/)** – LLM orchestration framework
* **[Ollama Embeddings](https://ollama.com/)** – Embeddings and LLM integration (requires local model)
* **PDFPlumber** – PDF parsing
* **InMemoryVectorStore** – Document similarity search

---

## Customization

* Adjust `chunk_size` and `chunk_overlap` in `RecursiveCharacterTextSplitter` to change document chunking behavior.
* Change the LLM model in `OllamaLLM(model="deepseek-r1:1.5b")` to use a different Ollama model (must be downloaded locally).

---


## Contribution

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## Contact

Created by John Mathew – [jhnmathew125@gmail.com](mailto:jhnmathew125@gmail.com)

