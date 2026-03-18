
import chromadb
from chromadb.utils import embedding_functions
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
_client = chromadb.PersistentClient(path=os.path.join(BASE, "data", "vectorstore"))
_collection = _client.get_or_create_collection(
    name="spotify_metadata",
    embedding_function=_ef
)

def retrieve_relevant_tools(query, n_results=3):
    results = _collection.query(query_texts=[query], n_results=n_results)
    return results["ids"][0], results["documents"][0]
