
from langchain_ollama import OllamaLLM

from langchain_core.prompts import PromptTemplate
from config import OLLAMA_URL,CHAT_MODEL

def build_chain(vectorstore):
    """Build the RAG components using the current LangChain package layout."""

    llm = OllamaLLM(
        model=CHAT_MODEL,
        base_url=OLLAMA_URL
    )

    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template="""
You are a helpful assistant. You have access to a document and your own knowledge.

DOCUMENT CONTEXT:
{context}

QUESTION:
{input}

Instructions:
- First, understand what topic the DOCUMENT CONTEXT is about.
- If the question is related to that topic:
    - If the answer is found in the DOCUMENT CONTEXT above, answer it
    - If the answer is NOT in the document, answer from your own knowledge by mentioning it is not exactly defined in the document
- If the question is completely off-topic and unrelated to the document's subject:
    - Do NOT answer it
    - Politely remind the user what you are here for
    - Example: "I'm a college information assistant for ABC College. I can only help with questions related to the college."
- Use the CONVERSATION HISTORY to understand follow-up questions like "it", "they", "that" etc.
ANSWER:
"""
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return {
        "retriever": retriever,
        "prompt": prompt,
        "llm": llm,
    }


def ask_with_memory(chain, question, history):
    """Add conversation history to the question and invoke the chain."""

    history_text = ""
    for h in history[-5:]:
        history_text += f"User: {h['question']}\nAssistant: {h['answer']}\n\n"

    if history_text:
        full_question = f"""Conversation so far:
{history_text}
New question: {question}"""
    else:
        full_question = question

    retrieved_docs = chain["retriever"].invoke(full_question)
    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt_value = chain["prompt"].invoke(
        {
            "context": context,
            "input": full_question,
        }
    )

    return chain["llm"].invoke(prompt_value)
