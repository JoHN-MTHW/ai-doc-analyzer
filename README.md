# 🧠 MENTO BOT

> An AI-Powered **Local LLM** Python Coding Assistant built with
> **Streamlit**, **LangChain**, and **Ollama**

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)\
[![LangChain](https://img.shields.io/badge/Powered%20by-LangChain-2E7D32?logo=python&logoColor=white)](https://python.langchain.com/)\
[![Ollama](https://img.shields.io/badge/LLM%20Runtime-Ollama-0A66C2?logo=ollama&logoColor=white)](https://ollama.ai/)\
![Local Only](https://img.shields.io/badge/Local%20LLMs-Yes-success)\
[![Python](https://img.shields.io/badge/Made%20with-Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)

------------------------------------------------------------------------

## 💡 Overview

**MENTO BOT** is a local-first AI coding assistant designed to help
developers **debug, review, optimize, and reason about Python code**.

🕰️ **Project Background**\
This project was originally built **\~1 year ago**, at a time when
**local LLM tooling and generative AI workflows were still emerging**.\
Recently, the project was **refined and optimized**, with support added
for **multiple local models** and improved state handling.

⚠️ **Important:**\
This application requires **Ollama running locally**. It **cannot be
deployed to cloud platforms** like Streamlit Cloud due to its local LLM
dependency.

------------------------------------------------------------------------

## 🚀 Key Features

### 🧠 Multi-Mode AI Assistant

-   **Default** -- General Python coding help\
-   **Bug Fixer** -- Identify and fix issues\
-   **Code Reviewer** -- Suggest improvements and best practices\
-   **Optimizer** -- Improve performance and efficiency

### 🔁 Multi-Model Support (via Ollama)

Easily switch between different local LLMs at runtime: -
`deepseek-r1:1.5b` - `gemma3:4b`

> Any Ollama-compatible model can be added with minimal changes.

### ⚙️ Developer Utilities

-   💬 Persistent chat history
-   💾 Downloadable chat logs
-   🧮 Safe Python code execution sandbox
-   🎨 Custom dark UI theme
-   🔒 Fully local inference (no APIs, no data leakage)

------------------------------------------------------------------------

## 🖥️ Tech Stack

  Component          Technology
  ------------------ -----------------
  Frontend           Streamlit
  AI Orchestration   LangChain
  LLM Runtime        Ollama (local)
  Models             DeepSeek, Gemma
  Language           Python 3.10+

------------------------------------------------------------------------

## ⚠️ Local Setup Instructions

### Prerequisites

-   Python 3.10+
-   Ollama installed locally

### Steps

``` bash
ollama pull deepseek-r1:1.5b
ollama pull gemma3:4b
ollama serve
git clone https://github.com/<your-username>/mento-bot.git
cd mento-bot
pip install -r requirements.txt
streamlit run app.py
```

------------------------------------------------------------------------

## 🧩 Project Structure

    mento-bot/
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── .gitignore

------------------------------------------------------------------------

## 🤝 Contributing

Contributions are welcome via issues or pull requests.

------------------------------------------------------------------------

## ❤️ Acknowledgements

-   Streamlit
-   LangChain
-   Ollama


## Contact

Created by John Mathew – [jhnmathew125@gmail.com](mailto:jhnmathew125@gmail.com)
