
# %%
import sys
# Compatibility: newer langchain-community moved ChatVertexAI to langchain-google-vertexai
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except ModuleNotFoundError:
    import langchain_google_vertexai
    sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai

from langchain_ollama import OllamaEmbeddings
import numpy as np
import pickle
from tqdm.auto import tqdm

EMBEDDINGS_PATH = "outputs/testset/doc_embeddings.npy"
BATCH_SIZE = 50

embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")

with open("outputs/testset/01_esg_knowledge_graph.pkl", "rb") as f:
    kg = pickle.load(f)

doc_contents = [node.properties["page_content"] for node in kg.nodes]
print(f"Total chunks to embed: {len(doc_contents)}")

doc_embeddings = []
for i in tqdm(range(0, len(doc_contents), BATCH_SIZE), desc="Embedding batches"):
    batch = doc_contents[i:i + BATCH_SIZE]
    doc_embeddings.extend(embeddings.embed_documents(batch))

np.save(EMBEDDINGS_PATH, np.array(doc_embeddings))
print(f"Saved {len(doc_embeddings)} embeddings to {EMBEDDINGS_PATH}")
