from typing import List, Any, Tuple
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions


DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.abspath("./Airlines_QA_Bot/data/embeddings/chroma"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "hr_policies")


class ChromaRetriever:
    def __init__(self, client: chromadb.ClientAPI, collection: chromadb.Collection, top_k: int = 5):
        self.client = client
        self.collection = collection
        self.top_k = top_k

    def query(self, q: str, k: int | None = None) -> List[str]:
        k = k or self.top_k
        res = self.collection.query(query_texts=[q], n_results=k)
        docs = (res.get("documents") or [[]])[0]
        return docs


def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=False,
    )


def _get_embedding_fn(model: str | None = None):
    model_name = model or DEFAULT_EMBEDDING_MODEL
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)


def _ensure_collection(client: chromadb.ClientAPI, name: str, embedding_fn) -> chromadb.Collection:
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name=name, embedding_function=embedding_fn)


def build_retriever(docs: List[str], top_k: int = 5) -> Any:
    splitter = _get_splitter()
    chunks: List[str] = []
    for doc in docs:
        if not doc:
            continue
        pieces = splitter.split_text(doc)
        chunks.extend([p for p in pieces if p and p.strip()])

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn = _get_embedding_fn()
    collection = _ensure_collection(client, COLLECTION_NAME, emb_fn)

    # Upsert chunks with deterministic IDs
    if chunks:
        ids = [f"chunk-{i}" for i in range(len(chunks))]
        # Clean previous to avoid duplication in MVP flow
        try:
            collection.delete(where={})
        except Exception:
            pass
        collection.add(ids=ids, documents=chunks)

    return ChromaRetriever(client, collection, top_k=top_k)


def retrieve_chunks(retriever: Any, query: str, k: int = 5) -> List[str]:
    return retriever.query(query, k=k)
