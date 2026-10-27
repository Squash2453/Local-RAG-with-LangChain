import os
import tkinter as tk
from tkinter import filedialog
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = (".txt", ".csv", ".pdf")


def select_files():
    """Opens a file dialog so the user can pick one or more documents."""
    print("Opening file-selection window...")
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        title="Select Documents for RAG",
        filetypes=[
            ("Supported Documents", "*.txt *.csv *.pdf"),
            ("Text Files", "*.txt"),
            ("CSV Files", "*.csv"),
            ("PDF Files", "*.pdf"),
            ("All Files", "*.*"),
        ],
    )
    root.destroy()

    if not file_paths:
        print("No files were selected.")
        return []
    return list(file_paths)


def load_document(path):
    """Load one file with the loader that matches its extension."""
    if path.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path, encoding="utf-8")
    return loader.load()


def load_and_chunk(filepaths):
    """Load every selected file, then split all of them into chunks."""
    all_documents = []

    for path in filepaths:
        name = os.path.basename(path)

        if not path.lower().endswith(SUPPORTED_EXTENSIONS):
            print(f"  Skipping unsupported file: {name}")
            continue

        try:
            docs = load_document(path)
        except Exception as e:
            print(f"  Could not read {name}: {e}")
            continue

        # Drop pages with no text. Scanned PDFs often have none.
        text_docs = [d for d in docs if d.page_content.strip()]
        if not text_docs:
            print(f"  No extractable text in {name} (scanned PDF?)")
            continue

        all_documents.extend(text_docs)
        print(f"  Loaded {name} ({len(text_docs)} section(s))")

    if not all_documents:
        return []

    # Split once, after combining everything, so chunking works the same way for every file.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(all_documents)