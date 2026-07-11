import tkinter as tk
from tkinter import filedialog
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def select_file():
    """Opens a graphical file dialog for the user to select the document."""
    print("Opening file-selection window...")
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select a Text Document for RAG",
        filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("All Files", "*.*")]
    )

    if not file_path:
        print("No file was selected. Exiting.")
        return None
    return file_path


def load_and_chunk(filepath):
    """Load the file and split into chunks using LangChain."""


    loader = TextLoader(filepath, encoding="utf-8")
    documents = loader.load()


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    return chunks