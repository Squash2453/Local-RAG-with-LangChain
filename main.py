import os

from config import CHAT_MODEL
from file_loader import select_file,load_and_chunk
from embedding import build_vectorstore
from generator import build_chain,ask_with_memory


def main():
    print("=" * 60)
    print("   SIMPLE RAG WITH LANGCHAIN (100% Local Backend)")
    print("=" * 60)

    filepath = select_file()
    if not filepath:
        return

    print(f"\n[1] Loading and chunking: '{os.path.basename(filepath)}'...")
    chunks = load_and_chunk(filepath)
    if not chunks:
        print("File was empty or could not be read.")
        return
    print(f"    Created {len(chunks)} chunks.")

    print(f"\n[2] Building vector store...")
    vectorstore = build_vectorstore(chunks)

    print(f"\n[3] Building RAG chain with '{CHAT_MODEL}'...")
    chain = build_chain(vectorstore)

    print("\n" + "=" * 60)
    print("✅ RAG System Ready! Ask questions. Type 'exit' to quit.")
    print("=" * 60 + "\n")

    history = []

    while True:
        question = input("\nYour Question: ").strip()

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        print("  🔍 Retrieving relevant chunks...")
        print(f"  🤖 Asking '{CHAT_MODEL}'...")

        answer = ask_with_memory(chain, question, history)

        history.append({"question": question, "answer": answer})

        print(f"\n💬 ANSWER:\n{answer}")
        print("-" * 60)


if __name__ == "__main__":
    main()
