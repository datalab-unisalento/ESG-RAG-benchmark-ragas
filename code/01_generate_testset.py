# %%
import sys
# Compatibility: newer langchain-community moved ChatVertexAI to langchain-google-vertexai
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except ModuleNotFoundError:
    import langchain_google_vertexai
    sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai

from langchain_community.document_loaders import DirectoryLoader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType
import pickle
import pandas as pd
import json
import os
import random

random.seed(42)

# Use full dataset for experiments
path = "data/documents"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()

# Create knowledge graph with overlapping chunks
kg = KnowledgeGraph()

chunk_size = 1000
overlap = 200
step = chunk_size - overlap

for doc in docs:
    content = doc.page_content
    for i in range(0, len(content), step):
        chunk = content[i:i + chunk_size]
        if len(chunk.strip()) > 50:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": chunk,
                        "document_metadata": doc.metadata,
                        "chunk_id": f"{doc.metadata.get('source', 'doc')}_{i // step}"
                    }
                )
            )

# Remove duplicates
unique_nodes = {}
for node in kg.nodes:
    node_id = node.id
    if node_id not in unique_nodes:
        unique_nodes[node_id] = node

kg.nodes = list(unique_nodes.values())
print(f"Created knowledge graph with {len(kg.nodes)} nodes")

# Create output subdirectory
output_dir = "outputs/testset"
os.makedirs(output_dir, exist_ok=True)

# Save full knowledge graph (used by RAG evaluation for retrieval)
with open(f"{output_dir}/01_esg_knowledge_graph.pkl", "wb") as f:
    pickle.dump(kg, f)

# %%
# Sample 300 nodes randomly for QA generation
TARGET_QA = 300
sampled_nodes = random.sample(kg.nodes, min(TARGET_QA, len(kg.nodes)))
print(f"Sampled {len(sampled_nodes)} nodes for QA generation")

# ESG personas: rotate across sampled nodes
PERSONAS = [
    {
        "name": "Environmental KPI Extractor",
        "system": (
            "You are an environmental analyst specializing in ESG reporting. "
            "You focus on carbon emissions, energy consumption, water usage, waste management, "
            "biodiversity, and climate-related targets."
        ),
    },
    {
        "name": "Social KPI Extractor",
        "system": (
            "You are a social responsibility analyst specializing in ESG reporting. "
            "You focus on employee diversity, labor practices, community investment, "
            "human rights, health and safety, and supply chain social standards."
        ),
    },
    {
        "name": "Governance KPI Extractor",
        "system": (
            "You are a corporate governance analyst specializing in ESG reporting. "
            "You focus on board composition, executive compensation, anti-corruption policies, "
            "risk management frameworks, shareholder rights, and data privacy."
        ),
    },
]

llm = ChatOllama(model="qwen3:4b-instruct", num_ctx=12000)

test_data = []

for i, node in enumerate(sampled_nodes):
    persona = PERSONAS[i % len(PERSONAS)]
    content = node.properties["page_content"]

    question_prompt = (
        f"Based on this ESG document excerpt, generate one specific question that can be "
        f"answered from the text. The question must reflect your area of expertise. "
        f"Respond with only the question, no extra text.\n\n"
        f"Document excerpt:\n{content}"
    )

    answer_prompt_template = (
        "Answer the following question using only the provided document excerpt. "
        "Be concise and factual.\n\n"
        "Question: {question}\n\n"
        "Document excerpt:\n{content}\n\n"
        "Answer:"
    )

    try:
        q_response = llm.invoke([
            ("system", persona["system"]),
            ("human", question_prompt),
        ])
        question = q_response.content.strip()

        a_response = llm.invoke([
            ("system", persona["system"]),
            ("human", answer_prompt_template.format(question=question, content=content)),
        ])
        reference = a_response.content.strip()

        if len(question) > 10 and len(reference) > 10:
            test_data.append({
                "user_input": question,
                "reference": reference,
                "document_source": node.properties.get("document_metadata", {}).get("source", "unknown"),
                "persona": persona["name"],
            })
            if (i + 1) % 50 == 0:
                print(f"  Generated {len(test_data)} QA pairs so far ({i + 1}/{len(sampled_nodes)} nodes processed)")

    except Exception as e:
        print(f"Error on node {i}: {e}")
        continue

print(f"Final testset: {len(test_data)} QA pairs from {len(sampled_nodes)} sampled nodes")

if test_data:
    pd.DataFrame(test_data).to_csv(f"{output_dir}/01_esg_testset.csv", index=False)
    with open(f"{output_dir}/01_esg_testset.json", "w") as f:
        json.dump(test_data, f, indent=2)
else:
    print("No QA pairs generated.")
    pd.DataFrame(columns=["user_input", "reference", "document_source", "persona"]).to_csv(
        f"{output_dir}/01_esg_testset.csv", index=False
    )
