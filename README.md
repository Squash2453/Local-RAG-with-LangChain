# Local RAG with LangChain

A beginner-friendly Retrieval-Augmented Generation (RAG) system built using **LangChain** and **Ollama**. This project allows users to load any text document, automatically split it into chunks, generate embeddings, store them in a FAISS vector database, retrieve the most relevant information, and answer questions using a local Large Language Model (LLM). It demonstrates how LangChain simplifies the implementation of a complete RAG pipeline while running entirely on your local machine.

---

## Features

* Load any text document (`.txt`) using a graphical file picker.
* Automatic document loading using LangChain's `TextLoader`.
* Smart document chunking using `RecursiveCharacterTextSplitter`.
* Generate embeddings locally using `nomic-embed-text` through Ollama.
* Store embeddings in a FAISS vector store.
* Retrieve relevant document chunks using LangChain's retriever.
* Generate responses using the `gemma4:e4b` language model.
* Support follow-up questions using conversation history.
* 100% local execution with no cloud APIs.

---

## Project Structure

```text
Simple-Local-RAG-With-LangChain/
│
├── main.py             # Entry point of the application
├── config.py           # Model names and application configuration
├── file_loader.py      # File selection and document chunking
├── embedding.py        # Embeddings and FAISS vector store
├── generator.py        # RAG chain and response generation
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Technologies Used

* Python 3
* LangChain
* Ollama
* FAISS
* Gemma 4 (`gemma4:e4b`)
* nomic-embed-text
* Tkinter

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Squash2453/Local-RAG-with-LangChain.git
cd Local-RAG-with-LangChain
```

### 2. (Optional) Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Download and install Ollama from the official website.

### 5. Pull the required models

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text
```

### 6. Start the Ollama server

```bash
ollama serve
```

### 7. Run the application

```bash
python main.py
```

---

## Usage

1. Run the application.

```bash
python main.py
```

2. Select a text (`.txt`) document.

3. The application will:

* Load the document.
* Split it into chunks.
* Generate embeddings.
* Store embeddings in a FAISS vector store.
* Build the LangChain RAG pipeline.

4. Ask questions about the document.

5. Type `exit`, `quit`, or `q` to close the application.

---

## How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline using LangChain:

```text
                Document (.txt)
                       │
                       ▼
                 TextLoader
                       │
                       ▼
     RecursiveCharacterTextSplitter
                       │
                       ▼
           Ollama Embeddings
                       │
                       ▼
             FAISS Vector Store
                       │
                       ▼
                User Question
                       │
                       ▼
                  Retriever
                       │
                       ▼
              PromptTemplate
                       │
                       ▼
              Gemma 4 (Ollama)
                       │
                       ▼
                Generated Answer
```

### Workflow

1. The selected document is loaded using LangChain's `TextLoader`.
2. The document is split into overlapping chunks using `RecursiveCharacterTextSplitter`.
3. Each chunk is converted into an embedding using `OllamaEmbeddings`.
4. All embeddings are stored inside a FAISS vector database.
5. When a question is asked, the retriever finds the most relevant document chunks.
6. The retrieved context is inserted into a prompt template.
7. The prompt is sent to the local `gemma4:e4b` language model through Ollama.
8. The generated answer is returned to the user.

---

## Comparison with the Manual RAG Implementation

This project implements the same RAG pipeline as the manual version but replaces custom implementations with LangChain components.

| Manual RAG                | LangChain Version              |
| ------------------------- | ------------------------------ |
| Manual file loading       | TextLoader                     |
| Manual chunking           | RecursiveCharacterTextSplitter |
| Manual embeddings         | OllamaEmbeddings               |
| Manual cosine similarity  | FAISS Retriever                |
| Manual prompt handling    | PromptTemplate                 |
| Manual retrieval pipeline | LangChain RAG Chain            |

This demonstrates how LangChain abstracts much of the repetitive RAG implementation while preserving the same overall workflow.

---

## Future Improvements

* Support PDF and DOCX documents.
* Persist the FAISS vector database to disk.
* Stream LLM responses.
* Add a Streamlit or Flask web interface.
* Support multiple documents in a single knowledge base.
* Use LangChain Memory instead of manual conversation history.
* Add source citations for retrieved document chunks.

---

## Acknowledgements

This project was built as a learning exercise to understand how LangChain simplifies the implementation of Retrieval-Augmented Generation (RAG) while using local Large Language Models through Ollama.
