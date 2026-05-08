import os
from langsmith import traceable
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

# 1. Load and split the dataset
def load_and_split_data(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(content)
    return chunks

# 2. Create Vector Store
def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=EMBEDDING_MODEL_NAME
    )
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# 3. Build RAG Chain
def build_rag_chain(retriever):
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=OPENAI_MODEL_NAME,
        temperature=0
    )
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# 4. Decorated Query Function
@traceable(name="rag-query")
def ask(chain, question: str) -> str:
    return chain.invoke(question)

def main():
    print("🚀 Starting RAG Pipeline Task 1...")
    
    # Load chunks
    chunks = load_and_split_data("data/knowledge_base.txt")
    print(f"✅ Loaded {len(chunks)} chunks.")
    
    # Create vector store
    vector_store = create_vector_store(chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("✅ FAISS Vector Store created.")
    
    # Build chain
    chain = build_rag_chain(retriever)
    print("✅ RAG Chain built.")
    
    # Run all 50 questions
    print(f"📊 Running {len(QA_PAIRS)} questions...")
    for i, qa in enumerate(QA_PAIRS):
        question = qa["question"]
        print(f"[{i+1}/50] Querying: {question}")
        try:
            answer = ask(chain, question)
            # print(f"Response: {answer[:100]}...")
        except Exception as e:
            print(f"❌ Error on question {i+1}: {e}")
            
    print("✅ Task 1 Complete. Check LangSmith for traces.")

if __name__ == "__main__":
    main()
