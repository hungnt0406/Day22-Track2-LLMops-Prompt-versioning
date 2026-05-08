import os
import json
import numpy as np
import warnings
import importlib
from typing import List

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.dataset_schema import SingleTurnSample
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

from config import (
    OPENAI_API_KEY, 
    OPENAI_BASE_URL, 
    OPENAI_MODEL_NAME, 
    EMBEDDING_MODEL_NAME,
    LANGCHAIN_API_KEY
)
from qa_pairs import QA_PAIRS

# Dynamic import for numbered files
step2 = importlib.import_module("02_prompt_hub_ab_routing")
pull_prompts_from_hub = step2.pull_prompts_from_hub
build_vector_store = step2.build_vector_store
PROMPT_V1_NAME = step2.PROMPT_V1_NAME
PROMPT_V2_NAME = step2.PROMPT_V2_NAME

# --- 1. Evaluation Setup ---

def run_evaluation_for_version(version_name: str, prompt, retriever, llm) -> List[SingleTurnSample]:
    """Runs all 50 questions through a specific prompt version and captures samples for RAGAS."""
    print(f"🧪 Generating responses for {version_name}...")
    samples = []
    
    for i, qa in enumerate(QA_PAIRS):
        question = qa["question"]
        reference = qa["answer"]
        
        # Retrieve context
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        
        # Generate answer
        # Using join with a variable to avoid tool expansion issues
        newline_sep = "\n\n"
        full_context = newline_sep.join(contexts)
        
        chain = prompt | llm
        response = chain.invoke({"context": full_context, "question": question})
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Create RAGAS sample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference
        )
        samples.append(sample)
        
        if (i + 1) % 10 == 0:
            print(f"   Done {i+1}/50...")
            
    return samples

def main():
    print("🚀 Starting Task 3: RAGAS Evaluation...")
    
    # Initialize components
    client = Client(api_key=LANGCHAIN_API_KEY)
    prompts = pull_prompts_from_hub(client)
    vector_store = build_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=OPENAI_MODEL_NAME,
        temperature=0
    )
    
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=EMBEDDING_MODEL_NAME
    )

    # Wrap for RAGAS
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    # Run data collection for both versions
    samples_v1 = run_evaluation_for_version("V1", prompts[PROMPT_V1_NAME], retriever, llm)
    samples_v2 = run_evaluation_for_version("V2", prompts[PROMPT_V2_NAME], retriever, llm)

    # Perform Evaluation
    print("📊 Evaluating V1 with RAGAS...")
    dataset_v1 = EvaluationDataset(samples=samples_v1)
    results_v1 = evaluate(dataset_v1, metrics=metrics, llm=ragas_llm, embeddings=ragas_emb)
    
    print("📊 Evaluating V2 with RAGAS...")
    dataset_v2 = EvaluationDataset(samples=samples_v2)
    results_v2 = evaluate(dataset_v2, metrics=metrics, llm=ragas_llm, embeddings=ragas_emb)

    # Calculate means
    scores = {
        "V1": {m.name: float(np.mean(results_v1[m.name])) for m in metrics},
        "V2": {m.name: float(np.mean(results_v2[m.name])) for m in metrics}
    }

    # Print Comparison Table
    print("\n" + "="*50)
    print(f"{'Metric':<20} | {'V1 (Concise)':<12} | {'V2 (Expert)':<12}")
    print("-" * 50)
    for m in metrics:
        name = m.name
        print(f"{name:<20} | {scores['V1'][name]:.4f}      | {scores['V2'][name]:.4f}")
    print("="*50)

    # Verification
    if scores['V1']['faithfulness'] >= 0.8 or scores['V2']['faithfulness'] >= 0.8:
        print("✅ Target met: Faithfulness >= 0.8")
    else:
        print("⚠️ Warning: Faithfulness below 0.8. Consider refining chunks or prompts.")

    # Save report
    os.makedirs("data", exist_ok=True)
    with open("data/ragas_report.json", "w") as f:
        json.dump(scores, f, indent=4)
    print("\n💾 Report saved to data/ragas_report.json")

if __name__ == "__main__":
    main()
