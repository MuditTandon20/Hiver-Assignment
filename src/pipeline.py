"""Unified End-to-End AI Customer Support Agent Pipeline."""
from typing import Dict, Any, List, Optional
from src.intents.classifier import ProductionIntentClassifier
from src.retrieval.retriever import HistoricalCaseRetriever
from src.retrieval.vector_store import FAISSVectorStore
from src.escalation.policy import EscalationPolicy
from src.generation.generator import create_response_generator
from src.utils.config import load_config, setup_logger, resolve_path
from src.utils.schemas import AgentResponse

logger = setup_logger("agent_pipeline")

class SupportAgentPipeline:
    """End-to-End Production Support Agent."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        
        # 1. Intent Classifier
        model_dir = self.config.get("intents", {}).get("model_dir", "data/models")
        model_file = str(resolve_path(f"{model_dir}/tfidf_classifier.joblib"))
        self.classifier = ProductionIntentClassifier(
            model_path=model_file,
            confidence_threshold=self.config.get("intents", {}).get("confidence_threshold", 0.60)
        )
        
        # 2. Vector Store & Retriever
        retrieval_cfg = self.config.get("retrieval", {})
        self.vector_store = FAISSVectorStore(embedding_dim=retrieval_cfg.get("embedding_dim", 256))
        
        index_file = retrieval_cfg.get("index_file", "data/indices/faiss_index.bin")
        meta_file = retrieval_cfg.get("metadata_file", "data/indices/metadata.json")
        try:
            self.vector_store.load(index_file, meta_file)
        except Exception as e:
            logger.warning(f"Could not load pre-built index: {e}. Index must be built prior to retrieval.")
            
        self.retriever = HistoricalCaseRetriever(
            vector_store=self.vector_store,
            top_k=retrieval_cfg.get("top_k", 3),
            similarity_threshold=retrieval_cfg.get("similarity_threshold", 0.45)
        )
        
        # 3. Escalation Policy
        escalation_cfg = self.config.get("escalation", {})
        self.escalation_policy = EscalationPolicy(
            confidence_threshold=escalation_cfg.get("confidence_threshold", 0.60),
            min_similarity_threshold=escalation_cfg.get("min_retrieval_similarity", 0.35),
            sensitive_keywords=escalation_cfg.get("sensitive_keywords")
        )
        
        # 4. Grounded Response Generator
        self.generator = create_response_generator(self.config)

    def process_message(
        self,
        customer_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process incoming customer support message through the end-to-end pipeline.
        Returns complete structured response adhering to AgentResponse schema.
        """
        history = conversation_history or []
        msg_clean = customer_message.strip() if customer_message else ""
        
        # Step 1: Intent Classification
        intent_res = self.classifier.predict(msg_clean)
        predicted_intent = intent_res["intent"]
        confidence = intent_res["confidence"]
        
        # Step 2: Similar Case Retrieval
        retrieved_examples = []
        if self.vector_store.is_fitted and msg_clean:
            retrieved_examples = self.retriever.retrieve(
                query=msg_clean,
                predicted_intent=predicted_intent,
                top_k=self.retriever.top_k
            )
            
        # Step 3: Escalation Decision
        escalation_res = self.escalation_policy.evaluate(
            customer_message=msg_clean,
            predicted_intent=predicted_intent,
            intent_confidence=confidence,
            retrieved_examples=retrieved_examples,
            conversation_history=history
        )
        
        # Step 4: Grounded Response Drafting
        generated_response = self.generator.generate(
            customer_message=msg_clean,
            conversation_history=history,
            predicted_intent=predicted_intent,
            retrieved_examples=retrieved_examples
        )
        
        raw_result = {
            "intent": predicted_intent,
            "intent_confidence": confidence,
            "intent_reason": intent_res.get("reason", ""),
            "decision": escalation_res["decision"],
            "escalation_reason_code": escalation_res.get("reason_code", "NONE"),
            "escalation_reason": escalation_res["reason"],
            "response": generated_response,
            "retrieved_examples": retrieved_examples,
            "is_multi_turn": len(history) > 0
        }
        
        # Contract validation with Pydantic
        validated = AgentResponse(**raw_result)
        return validated.model_dump()
