"""Unit tests for FAISS vector store and retriever."""
import pytest
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import HistoricalCaseRetriever

def test_faiss_vector_store_indexing():
    sample_conversations = [
        {
            "conversation_id": "c1",
            "customer_text": "Where is my package order? Delivery is late and delayed.",
            "brand_response_text": "Please check your Orders tab for tracking updates. ^AMZ",
            "intent": "DELIVERY_STATUS"
        },
        {
            "conversation_id": "c2",
            "customer_text": "Item arrived damaged, broken and cracked in pieces.",
            "brand_response_text": "Sorry for the damage! You can request a replacement in Your Orders. ^AMZ",
            "intent": "ITEM_ISSUE"
        },
        {
            "conversation_id": "c3",
            "customer_text": "How do I return this shirt and get my refund money back?",
            "brand_response_text": "Start your return by printing a label on Amazon.com. ^AMZ",
            "intent": "RETURN_REFUND"
        }
    ]
    
    vs = FAISSVectorStore(embedding_dim=16)
    vs.build_index(sample_conversations)
    
    retriever = HistoricalCaseRetriever(vector_store=vs, top_k=2)
    results = retriever.retrieve(
        query="My package delivery is late and delayed.",
        predicted_intent="DELIVERY_STATUS"
    )
    
    assert len(results) > 0
    assert "similarity_score" in results[0]
    assert "brand_response_text" in results[0]
    assert results[0]["intent"] == "DELIVERY_STATUS"
    assert "Orders tab" in results[0]["brand_response_text"]
