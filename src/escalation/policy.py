"""Multi-Factor Escalation & Human Handoff Engine."""
import re
from typing import Dict, Any, List, Optional
from src.intents.taxonomy import get_intent_metadata
from src.utils.schemas import EscalationDecision

def contains_keyword(text: str, keyword: str) -> bool:
    """Check if keyword exists in text as a whole word or phrase with word boundaries."""
    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
    return bool(re.search(pattern, text.lower()))

def contains_any_keyword(text: str, keywords: List[str]) -> List[str]:
    """Return all keywords that match in text using word boundaries."""
    return [k for k in keywords if contains_keyword(text, k)]

class EscalationPolicy:
    """
    Deterministic & rule-calibrated escalation engine.
    Ensures safe, explainable decisions on whether a query can be auto-handled or requires human intervention.
    """
    def __init__(
        self,
        confidence_threshold: float = 0.60,
        min_similarity_threshold: float = 0.35,
        sensitive_keywords: Optional[List[str]] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.min_similarity_threshold = min_similarity_threshold
        self.sensitive_keywords = sensitive_keywords or [
            "lawsuit", "lawyer", "legal action", "sue", "attorney", "court",
            "fraud", "police", "hacked", "stolen card", "unauthorized charge",
            "harassment", "death threat", "fbi", "bbb", "better business bureau"
        ]

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieved_examples: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate escalation criteria in hierarchical priority order.
        Returns decision dict with decision, reason_code, and detailed explanation.
        """
        text_lower = customer_message.lower().strip()
        
        # 1. Critical Legal / Safety / Fraud Triggers (Word-boundary matching)
        matched_sensitive = contains_any_keyword(text_lower, self.sensitive_keywords)
        if matched_sensitive:
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason_code": "SENSITIVE_KEYWORD_TRIGGER",
                "reason": f"Sensitive safety/legal keywords detected ({', '.join(matched_sensitive)}). Immediate human review required.",
                "escalated": True
            }
            
        # 2. High-Risk Account Security
        if predicted_intent == "ACCOUNT_ACCESS":
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason_code": "ACCOUNT_SECURITY_RISK",
                "reason": "Account access and credential compromise inquiries require authenticated human agent verification.",
                "escalated": True
            }
            
        # 3. High-Risk Financial Dispute / Billing Irregularity
        dispute_keywords = ["unauthorized", "charged twice", "double charge", "stolen", "dispute"]
        if predicted_intent == "BILLING_PAYMENT" and bool(contains_any_keyword(text_lower, dispute_keywords)):
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason_code": "FINANCIAL_DISPUTE",
                "reason": "Unrecognized charge / payment dispute requires private ledger access and financial agent intervention.",
                "escalated": True
            }
            
        # 4. Severe Complaint / Hostility / Escalation Demand
        complaint_triggers = ["manager", "supervisor", "unacceptable", "terrible", "worst service"]
        if predicted_intent == "FEEDBACK_COMPLAINT" or bool(contains_any_keyword(text_lower, complaint_triggers)):
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason_code": "SEVERE_COMPLAINT",
                "reason": "High dissatisfaction or customer requested human supervisor intervention.",
                "escalated": True
            }

        # 5. Low Intent Classification Confidence
        if intent_confidence < self.confidence_threshold:
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason_code": "LOW_INTENT_CONFIDENCE",
                "reason": f"Intent classification confidence ({intent_confidence:.2f}) is below safe threshold ({self.confidence_threshold:.2f}).",
                "escalated": True
            }

        # 6. Insufficient Historical Retrieval Grounding
        max_sim = max([ex.get("similarity_score", 0.0) for ex in retrieved_examples]) if retrieved_examples else 0.0
        if max_sim < self.min_similarity_threshold:
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason_code": "INSUFFICIENT_RETRIEVAL_GROUNDING",
                "reason": f"No high-confidence historical precedent found (max similarity {max_sim:.2f} < {self.min_similarity_threshold:.2f}).",
                "escalated": True
            }

        # 7. Default Safe Auto-Handling
        return {
            "decision": "AUTO_HANDLE",
            "reason_code": "SAFE_FOR_AUTOMATION",
            "reason": f"Query matches intent '{predicted_intent}' with high confidence ({intent_confidence:.2f}) and verified historical grounding ({max_sim:.2f}).",
            "escalated": False
        }
