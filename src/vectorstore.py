import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Any
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_DIR, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP

class ChromaDBStore:
    def __init__(self, collection_name: str = "textbook_collection"):
        self.persist_directory = DB_DIR
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        print(f"[INFO] Initializing ChromaDB at {self.persist_directory}")
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Initialize SentenceTransformer embedding function via Chroma utils
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )

    def build_from_documents(self, documents: List[Any]):
        if not documents:
            print("[WARNING] No documents provided to build vector store.")
            return

        print(f"[INFO] Chunking {len(documents)} document pages...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split into {len(chunks)} chunks.")

        # Extract texts, metadatas, and generate unique IDs
        texts = [chunk.page_content for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            # clean metadata, chroma requires string, int, float or bool
            meta = {}
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            metadatas.append(meta)
            
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        print(f"[INFO] Adding {len(texts)} chunks to ChromaDB...")
        # Batch add iteratively to avoid potential limits
        batch_size = 5000
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            self.collection.add(
                documents=texts[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
        print(f"[SUCCESS] Vector store built successfully.")

    def query(self, query_text: str, top_k: int = 5, score_threshold: float = 0.0):
        print(f"[INFO] Querying ChromaDB for: '{query_text}'")
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            ids = results['ids'][0]
            
            for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                # Convert distance to similarity score
                similarity_score = 1.0 - distance
                
                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                    })
            
            print(f"[INFO] Retrieved {len(retrieved_docs)} matching chunks (Threshold: >={score_threshold}).")
            
        return retrieved_docs

if __name__ == "__main__":
    from src.data_loader import load_documents
    docs = load_documents()
    store = ChromaDBStore()
    if docs:
        store.build_from_documents(docs)
    
    print("\n--- Testing Retrieval ---")
    res = store.query("What is Machine Learning?", top_k=2)
    for r in res:
        print(f"Score: {r['similarity_score']:.4f}")
        print(f"Source: {r['metadata'].get('source')} - Page {r['metadata'].get('page')}")
        print(f"Preview: {r['content'][:100]}...\n")