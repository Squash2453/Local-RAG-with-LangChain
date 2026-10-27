# RAG System with LangChain, FAISS, Ollama & Gemini

A document-based Retrieval-Augmented Generation (RAG) application built using **Python, LangChain, FAISS, and Ollama**, with support for Google Gemini API and local Gemma models.

The application allows users to upload multiple documents, retrieve relevant information using semantic search, and generate answers grounded in the retrieved document content. It uses local Ollama embeddings and FAISS for retrieval, while allowing users to choose between a cloud-based Gemini model and a locally running Ollama model for answer generation.

## Features

- **Multiple document support:** Load `.txt`, `.csv`, and `.pdf` files using a graphical file picker.
- **Document processing:** Load documents and split them into smaller chunks using LangChain.
- **Local embeddings:** Generate embeddings using `nomic-embed-text` through Ollama.
- **Vector search:** Store and retrieve document embeddings using FAISS.
- **Flexible LLM selection:** Switch between Google Gemini API and a local Ollama model.
- **Conversation history:** Include recent questions and answers to support follow-up questions.
- **Source metadata:** Include retrieved document filenames and, where available, PDF page numbers in the context sent to the model.
- **Configurable retrieval:** Retrieve relevant document chunks to provide context for answer generation.

## Project Architecture

```text
Documents (.txt, .csv, .pdf)
             |
             v
      Document Loading
             |
             v
      Text Chunking
             |
             v
    Ollama Embeddings
   (nomic-embed-text)
             |
             v
       FAISS Index
             |
             v
       User Question
             |
             v
    FAISS Retriever
             |
             v
   Retrieved Document Chunks
             |
             v
  Context + Conversation History
             |
             v
       LLM Selection
          /     \
         v       v
     Gemini    Ollama
         \       /
          v     v
       Generated Answer
```

## Technologies Used

- Python
- LangChain
- FAISS
- Ollama
- Google Gemini API
- Gemma (`gemma4:e4b`)
- `nomic-embed-text`
- PyPDFLoader
- TextLoader
- RecursiveCharacterTextSplitter
- Tkinter

## Prerequisites

Before running the project, make sure you have:

- Python 3.10 or a compatible version supported by your dependencies.
- [Ollama](https://ollama.com/) installed and running.
- The required Ollama embedding model downloaded.
- A Google Gemini API key if you want to use Gemini for answer generation.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Squash2453/Local-RAG-with-LangChain.git
cd Local-RAG-with-LangChain
```

### 2. Create a virtual environment (recommended)

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install and configure Ollama

Download Ollama from [ollama.com](https://ollama.com/).

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

If you plan to use the local Gemma model for answer generation, also run:

```bash
ollama pull gemma4:e4b
```

Ensure Ollama is running before starting the application.

## Configuration

The application supports two answer-generation options, configured through `config.py`.

### Option 1: Google Gemini API

Set the model provider in `config.py`:

```python
MODEL_PROVIDER = "gemini"
```

Configure your Gemini model and read the API key from an environment variable:

```python
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-3.8-flash"
```

Set the API key in PowerShell for the current terminal session:

```powershell
$env:GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"
```

Replace the placeholder with your own API key. Never commit your real API key to GitHub.

**Note:** Gemini is used for answer generation, but document embeddings are still generated locally through Ollama.

### Option 2: Local Ollama model

Set the model provider in `config.py`:

```python
MODEL_PROVIDER = "ollama"
```

Configure the local model and Ollama URL:

```python
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e4b"
```

In this mode, Ollama generates the answers locally, while `nomic-embed-text` continues to generate document embeddings.

## Running the Application

Start the application from the project directory:

```bash
python main.py
```

The application will allow you to select documents and ask questions based on their contents.

## How It Works

1. **Document loading:** The user selects one or more supported files using the graphical file picker.
2. **Chunking:** LangChain splits the loaded documents into smaller, overlapping text chunks.
3. **Embedding generation:** Ollama converts the chunks into numerical vector representations using `nomic-embed-text`.
4. **Vector storage:** FAISS stores the embeddings and associated document information for similarity search.
5. **Retrieval:** When the user asks a question, the retriever searches for relevant document chunks.
6. **Context construction:** Retrieved text and available source metadata are added to the context supplied to the LLM.
7. **Conversation history:** Recent question-and-answer pairs are included in the prompt to help the model interpret follow-up questions.
8. **Answer generation:** The selected Gemini or Ollama model generates a response using the supplied context and conversation history.

## Conversation History

The current implementation maintains recent conversation history in a Python list and includes up to the last five question-and-answer pairs in subsequent prompts.

This supports basic conversational follow-ups, but the history is not automatically persisted across application restarts. Additionally, document retrieval currently uses the latest question directly, so ambiguous follow-up questions may not always retrieve the most relevant chunks.

## Local vs. Cloud-Based Answer Generation

| Component | Gemini mode | Ollama mode |
|---|---|---|
| Embedding generation | Local Ollama | Local Ollama |
| Vector storage and retrieval | Local FAISS | Local FAISS |
| Answer generation | Google Gemini API | Local Gemma model |
| Internet required for answer generation | Yes | Not for local model inference |
| API key required | Yes | No Gemini API key required |

Both modes use the same underlying document retrieval pipeline.

## Future Improvements

- Add history-aware query rewriting for improved follow-up retrieval.
- Persist the FAISS index between application runs.
- Add persistent conversation history using a suitable LangGraph checkpointer.
- Improve source attribution and display citations alongside generated answers.
- Add a web interface using Streamlit or Flask.
- Add automated tests and error handling for document loading and model calls.

## Acknowledgements

This project was developed as a learning exercise to understand Retrieval-Augmented Generation, semantic search, vector stores, embeddings, conversation history, and the integration of local and cloud-based language models using LangChain.
