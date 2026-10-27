from config import MODEL_PROVIDER
from file_loader import select_files, load_and_chunk
from embedding import build_vectorstore
from generator import build_chain, ask_with_memory


def main():
    print("=" * 60)
    print("   SIMPLE RAG WITH LANGCHAIN (100% Local Backend)")
    print("=" * 60)

    file_paths = select_files()
    if not file_paths:
        return

    print(f"\n[1] Loading and chunking {len(file_paths)} file(s)...")
    chunks = load_and_chunk(file_paths)
    if not chunks:
        print("No readable text was found in the selected files.")
        return
    print(f"    Created {len(chunks)} chunks.")

    print(f"\n[2] Building vector store...")
    vectorstore = build_vectorstore(chunks)

    print(f"\n[3] Building RAG chain with '{MODEL_PROVIDER}'...")
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
        print(f"  🤖 Asking '{MODEL_PROVIDER}'...")

        answer = ask_with_memory(chain, question, history)

        history.append({"question": question, "answer": answer})

        print(f"\n💬 ANSWER:\n{answer}")
        print("-" * 60)


if __name__ == "__main__":
    main()