"""Vector Store for historical conversation retrieval with FAISS and Native NumPy acceleration."""
import json
import joblib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from src.utils.config import setup_logger, resolve_path

logger = setup_logger("vector_store")

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

class FAISSVectorStore:
    """
    Production vector store using FAISS (or optimized NumPy fallback).
    Uses L2-normalized dense embeddings for exact inner-product (cosine) search.
    """
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True, stop_words='english')
        self.reducer: Optional[TruncatedSVD] = None
        self.index = None
        self.dense_matrix: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []
        self.is_fitted = False

    def build_index(self, conversations: List[Dict[str, Any]]):
        """Build vector index from historical conversations."""
        logger.info(f"Building vector index from {len(conversations):,} historical conversations...")
        texts = [c["customer_text"] for c in conversations]
        
        tfidf_mat = self.vectorizer.fit_transform(texts)
        n_samples, n_features = tfidf_mat.shape
        
        # Dynamically size SVD components based on data dimensions
        n_comp = min(self.embedding_dim, n_features - 1 if n_features > 1 else 1, n_samples - 1 if n_samples > 1 else 1)
        if n_comp >= 2:
            self.reducer = TruncatedSVD(n_components=n_comp, random_state=42)
            dense_vecs = self.reducer.fit_transform(tfidf_mat)
        else:
            dense_vecs = tfidf_mat.toarray()
            
        norm_vecs = normalize(dense_vecs, norm='l2', axis=1).astype(np.float32)
        self.dense_matrix = norm_vecs
        
        if HAS_FAISS:
            try:
                self.index = faiss.IndexFlatIP(norm_vecs.shape[1])
                self.index.add(norm_vecs)
            except Exception as e:
                logger.warning(f"FAISS initialization failed ({e}). Using NumPy dot-product engine.")
                self.index = None
                
        self.metadata = conversations
        self.is_fitted = True
        logger.info(f"Vector index successfully built with {len(conversations):,} vectors.")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string into normalized dense vector."""
        tfidf_q = self.vectorizer.transform([query])
        if self.reducer is not None:
            dense_q = self.reducer.transform(tfidf_q)
        else:
            dense_q = tfidf_q.toarray()
        norm_q = normalize(dense_q, norm='l2', axis=1).astype(np.float32)
        return norm_q

    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_intent: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for top-k most similar historical customer queries."""
        if not self.is_fitted or self.dense_matrix is None:
            raise RuntimeError("Vector Store is not initialized or fitted.")
            
        q_vec = self.embed_query(query)
        
        # Calculate cosine similarities via inner product
        if self.index is not None and HAS_FAISS:
            k_candidate = min(top_k * 10 if filter_intent else top_k, self.index.ntotal)
            scores, indices = self.index.search(q_vec, k_candidate)
            score_pairs = list(zip(scores[0], indices[0]))
        else:
            # Native NumPy dot product (extremely fast for 10k-50k vectors)
            sims = np.dot(self.dense_matrix, q_vec.T).squeeze()
            if sims.ndim == 0:
                sims = np.array([sims])
            k_candidate = min(top_k * 10 if filter_intent else top_k, len(self.dense_matrix))
            top_indices = np.argsort(sims)[::-1][:k_candidate]
            score_pairs = [(sims[idx], idx) for idx in top_indices]
            
        results = []
        for score, idx in score_pairs:
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            
            if filter_intent and item.get("intent") and item.get("intent") != filter_intent:
                continue
                
            results.append({
                "similarity_score": round(float(score), 4),
                "conversation_id": item.get("conversation_id", f"idx_{idx}"),
                "customer_text": item.get("customer_text", ""),
                "brand_response_text": item.get("brand_response_text", ""),
                "intent": item.get("intent", "UNKNOWN"),
                "is_multi_turn": item.get("is_multi_turn", False)
            })
            
            if len(results) >= top_k:
                break
                
        return results

    def save(self, index_file: str, metadata_file: str, embedder_file: str = "data/indices/embedder.joblib"):
        """Save index matrix and metadata to disk."""
        resolved_idx = resolve_path(index_file)
        resolved_meta = resolve_path(metadata_file)
        resolved_embed = resolve_path(embedder_file)
        
        resolved_idx.parent.mkdir(parents=True, exist_ok=True)
        resolved_meta.parent.mkdir(parents=True, exist_ok=True)
        resolved_embed.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(str(resolved_idx.with_suffix('.npy')), self.dense_matrix)
        
        if HAS_FAISS and self.index is not None:
            try:
                faiss.write_index(self.index, str(resolved_idx))
            except Exception:
                pass
                
        with open(resolved_meta, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
        joblib.dump({"vectorizer": self.vectorizer, "reducer": self.reducer, "dim": self.embedding_dim}, str(resolved_embed))
        logger.info(f"Saved vector index to {resolved_idx} and metadata to {resolved_meta}")

    def load(self, index_file: str, metadata_file: str, embedder_file: str = "data/indices/embedder.joblib"):
        """Load index matrix and metadata from disk."""
        resolved_idx = resolve_path(index_file)
        resolved_meta = resolve_path(metadata_file)
        resolved_embed = resolve_path(embedder_file)
        
        npy_file = resolved_idx.with_suffix('.npy')
        if npy_file.exists():
            self.dense_matrix = np.load(str(npy_file))
        
        if HAS_FAISS and resolved_idx.exists():
            try:
                self.index = faiss.read_index(str(resolved_idx))
            except Exception:
                self.index = None
                
        with open(resolved_meta, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        embed_data = joblib.load(str(resolved_embed))
        self.vectorizer = embed_data["vectorizer"]
        self.reducer = embed_data["reducer"]
        self.embedding_dim = embed_data["dim"]
        self.is_fitted = True
        logger.info(f"Loaded vector store with {len(self.metadata)} items.")
