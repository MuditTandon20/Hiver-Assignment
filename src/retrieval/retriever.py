"""Historical Case Retriever module."""
from typing import List, Dict, Any, Optional
from src.retrieval.vector_store import FAISSVectorStore
from src.utils.config import setup_logger

logger = setup_logger("retriever")

class HistoricalCaseRetriever:
    """Retrieves relevant historical brand support interactions."""
    def __init__(self, vector_store: FAISSVectorStore, top_k: int = 3, similarity_threshold: float = 0.45):
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        predicted_intent: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve top historical cases matching query and optional intent."""
        k = top_k or self.top_k
        results = self.vector_store.search(
            query=query,
            top_k=k,
            filter_intent=predicted_intent
        )
        # If intent filtering yields too few results, fallback to unfiltered search
        if len(results) < k:
            unfiltered = self.vector_store.search(query=query, top_k=k, filter_intent=None)
            for item in unfiltered:
                if item["conversation_id"] not in [r["conversation_id"] for r in results]:
                    results.append(item)
                if len(results) >= k:
                    break
        return results
