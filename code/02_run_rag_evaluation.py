
# %%
import sys
# Compatibility: newer langchain-community moved ChatVertexAI to langchain-google-vertexai
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except ModuleNotFoundError:
    import langchain_google_vertexai
    sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
import numpy as np
import pandas as pd
import json
import pickle
from tqdm.auto import tqdm

embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")

# Load testset — cap at 100 QA pairs
df = pd.read_csv("outputs/testset/01_esg_testset.csv", index_col=False).head(100)
print(f"Loaded {len(df)} QA pairs for evaluation")

with open("outputs/testset/01_esg_knowledge_graph.pkl", "rb") as f:
    kg = pickle.load(f)

doc_contents = [node.properties["page_content"] for node in kg.nodes]

# Load pre-computed embeddings (run 02a_compute_embeddings.py first)
EMBEDDINGS_PATH = "outputs/testset/doc_embeddings.npy"
print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
doc_embeddings = np.load(EMBEDDINGS_PATH).tolist()
print(f"Loaded {len(doc_embeddings)} embeddings")

import os

ALL_MODELS = [
    "qwen3:4b-instruct",
    "gemma3:4b",
    "gemma4:e2b",
    "gemma4:e4b",
    "ministral-3:8b",
    "nemotron-3-nano:4b",
    "glm-4.7-flash:q4_K_M",
]

filter_env = os.environ.get("EVAL_MODELS", "")
models = [m.strip() for m in filter_env.split(",") if m.strip()] if filter_env else ALL_MODELS
print(f"Models to evaluate: {models}")


def get_most_relevant_doc(query):
    query_embedding = embeddings.embed_query(query)
    similarities = [
        np.dot(query_embedding, doc_emb)
        / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
        for doc_emb in doc_embeddings
    ]
    return [doc_contents[np.argmax(similarities)]]


def generate_answer(llm, query, relevant_doc):
    prompt = f"question: {query}\n\nDocuments: {relevant_doc}"
    messages = [
        ("system", "You are a helpful assistant that answers questions based on given documents only."),
        ("human", prompt),
    ]
    return llm.invoke(messages).content


for ragLLM in models:
    print(f"\nEvaluating model: {ragLLM}")
    llm = ChatOllama(model=ragLLM)
    ragas_dataset = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        relevant_docs = get_most_relevant_doc(row["user_input"])
        response = generate_answer(llm, row["user_input"], relevant_docs)
        ragas_dataset.append({
            "user_input": row["user_input"],
            "retrieved_contexts": relevant_docs,
            "response": response,
            "reference": row["reference"],
        })

    with open(f"outputs/rag_responses/rag_responses_{ragLLM}.json", "w") as f:
        json.dump(ragas_dataset, f)
    print(f"Saved responses for {ragLLM}")
