import os
import hashlib
from langsmith import Client, traceable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import (
    OPENAI_API_KEY, 
    OPENAI_BASE_URL, 
    OPENAI_MODEL_NAME, 
    EMBEDDING_MODEL_NAME,
    LANGCHAIN_TRACING_V2,
    LANGCHAIN_API_KEY,
    LANGCHAIN_PROJECT
)
from qa_pairs import QA_PAIRS

# Ensure LangSmith tracing is enabled
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# --- 1. Define Prompt Templates ---

# V1: Concise and direct
SYSTEM_V1 = (
    "You are a concise assistant. Answer the question using ONLY the provided context. "
    "Keep your answer short (1-2 sentences). "
    "If the context doesn't have the answer, say 'Insufficient information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human", "{question}"),
])

# V2: Structured and detailed
SYSTEM_V2 = (
    "You are a technical expert. Provide a structured, detailed answer based on the context.\n\n"
    "Instructions:\n"
    "1. Use only the provided context.\n"
    "2. Be precise and technical.\n"
    "3. Use 3-4 sentences.\n"
    "4. If unknown, state that the context does not provide enough data.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human", "{question}"),
])

# Unique names for Prompt Hub
PROMPT_V1_NAME = "aetheria-rag-v1"
PROMPT_V2_NAME = "aetheria-rag-v2"

# --- 2. Hub Operations ---

def push_prompts_to_hub(client):
    """Pushes both prompt versions to LangSmith Prompt Hub."""
    print("📤 Pushing prompts to Hub...")
    try:
        url1 = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1: Concise RAG prompt")
        print(f"✅ Pushed V1: {url1}")
        url2 = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2: Structured RAG prompt")
        print(f"✅ Pushed V2: {url2}")
    except Exception as e:
        print(f"⚠️ Push failed (likely already exists): {e}")

def pull_prompts_from_hub(client):
    """Pulls prompts from Hub with local fallback."""
    print("📥 Pulling prompts from Hub...")
    prompts = {}
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"✅ Pulled {PROMPT_V1_NAME}")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print("ℹ️ Using local fallback for V1")
        
    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"✅ Pulled {PROMPT_V2_NAME}")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print("ℹ️ Using local fallback for V2")
        
    return prompts

# --- 3. A/B Routing Logic ---

def get_prompt_version(request_id: str) -> str:
    """Deterministic routing using MD5 hash."""
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME

# --- 4. RAG Implementation ---

def build_vector_store():
    with open("data/knowledge_base.txt", "r") as f:
        content = f.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(content)
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=EMBEDDING_MODEL_NAME
    )
    return FAISS.from_texts(chunks, embeddings)

@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version_tag: str) -> str:
    """Run RAG query with a specific prompt version."""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

def main():
    print("🚀 Starting Task 2: Prompt Hub & A/B Routing...")
    
    client = Client(api_key=LANGCHAIN_API_KEY)
    
    # Hub operations
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)
    
    # Setup RAG components
    vector_store = build_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=OPENAI_MODEL_NAME,
        temperature=0
    )
    
    # Run 50 queries with A/B routing
    v1_count = 0
    v2_count = 0
    
    print(f"📊 Running {len(QA_PAIRS)} queries with A/B routing...")
    for i, qa in enumerate(QA_PAIRS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]
        
        if version_tag == "v1": v1_count += 1
        else: v2_count += 1
        
        question = qa["question"]
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:50]}...")
        
        try:
            ask_ab(retriever, llm, prompt, question, version_tag)
        except Exception as e:
            print(f"❌ Error: {e}")
            
    print("\n✅ Task 2 Complete.")
    print(f"📈 Routing Summary: V1: {v1_count} | V2: {v2_count}")
    print("Check LangSmith for traces with 'ab-test' tag.")

if __name__ == "__main__":
    main()
