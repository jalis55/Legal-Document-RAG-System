# src/retrieval/retriever.py
import pickle
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class LegalRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_dir: str = "data/vector_db"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔧 Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        self.index = None
        self.chunks: List[Dict] = []
        
        self.index_path = self.index_dir / "faiss_index.bin"
        self.chunks_path = self.index_dir / "chunks.pkl"
        
        self._load_index()

    def _load_index(self):
        if self.index_path.exists() and self.chunks_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                print(f"✅ Loaded existing FAISS index with {len(self.chunks)} chunks")
            except Exception as e:
                print(f"⚠️  Failed to load existing index: {e}")
                print("   Creating new empty index...")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.chunks = []
        else:
            print("📦 No existing index found. Creating new empty index...")
            self.index = faiss.IndexFlatL2(self.dimension)

    def add_chunks(self, chunks: List[Dict]):
        """Add chunks to FAISS index"""
        if not chunks:
            return

        print(f"   Encoding {len(chunks)} chunks...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

        # Save index atomically
        temp_index_path = self.index_path.with_suffix('.tmp')
        temp_chunks_path = self.chunks_path.with_suffix('.tmp')
        
        faiss.write_index(self.index, str(temp_index_path))
        with open(temp_chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Atomic rename
        temp_index_path.rename(self.index_path)
        temp_chunks_path.rename(self.chunks_path)

        print(f"✅ Added {len(chunks)} chunks → Total: {len(self.chunks)}")

    def search(self, query: str, top_k: int = 6) -> List[Dict]:
        """Retrieve top relevant chunks"""
        if not self.chunks or self.index.ntotal == 0:
            print("⚠️  Vector database is empty. Please run ingestion first.")
            return []

        query_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        
        # Ensure we don't request more than available
        actual_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_emb, actual_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                # Convert distance to similarity score
                chunk["relevance_score"] = float(1 / (1 + distances[0][i]))
                results.append(chunk)
        
        return results

    def get_stats(self):
        count = len(self.chunks)
        unique_docs = len(self.get_indexed_doc_ids())
        print(f"📊 Vector DB Stats: {count} total chunks across {unique_docs} documents")
        return count

    def get_indexed_doc_ids(self) -> List[str]:
        """Return list of unique doc_ids currently in the index"""
        return list(set(chunk["doc_id"] for chunk in self.chunks))
    
    def clear_index(self):
        """Reset the index (useful for testing)"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        if self.index_path.exists():
            self.index_path.unlink()
        if self.chunks_path.exists():
            self.chunks_path.unlink()
        print("🗑️  Index cleared")