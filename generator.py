import os
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from config import MODEL_PROVIDER,GOOGLE_API_KEY,GEMINI_MODEL,OLLAMA_URL,OLLAMA_MODEL

def build_chain(vectorstore):
    """Build the RAG components using the current LangChain package layout."""
    
    if MODEL_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError(
                "Set the GOOGLE_API_KEY environment variable"
            )

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            api_key=GOOGLE_API_KEY,
            temperature = 0
        )
    
    elif MODEL_PROVIDER == "ollama":
        llm = OllamaLLM(
            model = OLLAMA_MODEL,
            base_url = OLLAMA_URL
        )
        
    else:
        raise ValueError(
            "Model provider should be 'gemini' or 'ollama'"
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
- The DOCUMENT CONTEXT may contain text from several documents on different subjects. Treat all of it as in scope.
- Before deciding a question is off-topic, check the whole DOCUMENT CONTEXT, not just the first part.
- If the question relates to any subject in the DOCUMENT CONTEXT:
    - If the answer is in the DOCUMENT CONTEXT, answer it from there.
    - If the answer is NOT in the DOCUMENT CONTEXT, say the documents do not define it exactly, then answer from your own knowledge.
- Only refuse if the question does not relate to any subject in the DOCUMENT CONTEXT. When you refuse, politely remind the user what you cover, listing the subjects of the documents.
- Use the CONVERSATION HISTORY to understand follow-up questions like "it", "they", "that".
- When it helps, say which document an answer came from.

ANSWER:
"""
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
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

    retrieved_docs = chain["retriever"].invoke(question)
    
    content_parts = []
    for doc in retrieved_docs:
        source = os.path.basename(
            doc.metadata.get("source","Unknown Source")
        )
        
        page = doc.metadata.get("page")
        
        if isinstance(page, int):
            source = f"{source} page {page + 1}"
            
        content_parts.append(
            f"[Source {source}]\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(content_parts)

    prompt_value = chain["prompt"].invoke(
        {
            "context": context,
            "input": full_question,
        }
    )

    # return chain["llm"].invoke(prompt_value)
    response = chain["llm"].invoke(prompt_value)
    if isinstance(response, str):
        answer = response
    else:
        answer = response.content

    if isinstance(answer, list):
        answer = "".join(
            block.get("text", "")
            for block in answer
            if isinstance(block, dict) and block.get("type") == "text"
        )

    return answer
