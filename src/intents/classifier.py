"""Production Intent Classification engine with confidence calibration and evidence extraction."""
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from src.intents.tfidf_classifier import TfidfIntentClassifier
from src.intents.taxonomy import INTENT_TAXONOMY, ALL_INTENTS
from src.utils.config import setup_logger, resolve_path

logger = setup_logger("intent_classifier")

def _contains_phrase(text: str, phrases: List[str]) -> bool:
    """Check if any phrase matches in text with word boundary awareness."""
    for phrase in phrases:
        pattern = r'\b' + re.escape(phrase.lower()) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

class ProductionIntentClassifier:
    """
    Production-grade Intent Classifier.
    Combines high-precision deterministic rules with statistical ML and confidence gating.
    """
    def __init__(self, model_path: Optional[str] = "data/models/tfidf_classifier.joblib", confidence_threshold: float = 0.60):
        self.confidence_threshold = confidence_threshold
        self.ml_classifier = TfidfIntentClassifier()
        self.model_path = model_path
        if model_path:
            full_path = resolve_path(model_path)
            if full_path.exists():
                self.ml_classifier.load(str(full_path))
            
    def predict(self, text: str) -> Dict[str, Any]:
        """Classify incoming customer message."""
        if not text or not text.strip():
            return {
                "intent": "OUT_OF_SCOPE",
                "confidence": 0.0,
                "evidence": ["empty_input"],
                "reason": "Empty or whitespace-only customer input.",
                "all_probabilities": {},
                "model": "InputValidation"
            }
            
        text_clean = text.strip()
        
        # 1. Deterministic Rule Matching for high-stakes/critical intents
        security_terms = ["hacked", "stolen account", "unauthorized login", "otp not receiving", "locked out"]
        if _contains_phrase(text_clean, security_terms):
            return {
                "intent": "ACCOUNT_ACCESS",
                "confidence": 0.95,
                "evidence": ["deterministic_security_rule"],
                "reason": "Explicit account security and authentication keywords detected.",
                "model": "Production_Hybrid_RuleEngine"
            }
            
        billing_terms = ["charged twice", "double charge", "unknown charge", "fraudulent charge"]
        if _contains_phrase(text_clean, billing_terms):
            return {
                "intent": "BILLING_PAYMENT",
                "confidence": 0.92,
                "evidence": ["deterministic_billing_rule"],
                "reason": "Explicit monetary dispute and payment keywords detected.",
                "model": "Production_Hybrid_RuleEngine"
            }
            
        # 2. Statistical ML Pipeline
        if self.ml_classifier.pipeline is not None:
            ml_res = self.ml_classifier.predict(text_clean)
            intent = ml_res["intent"]
            conf = ml_res["confidence"]
            evidence = ml_res["evidence"]
            
            # If confidence is below threshold, flag for potential out-of-scope or escalation
            if conf < self.confidence_threshold:
                reason = f"Low classification confidence ({conf:.2f} < threshold {self.confidence_threshold:.2f})."
            else:
                reason = f"Classified as {intent} with {conf*100:.1f}% confidence based on terms: {', '.join(evidence) if evidence else 'context'}."
                
            return {
                "intent": intent,
                "confidence": conf,
                "evidence": evidence,
                "reason": reason,
                "all_probabilities": ml_res.get("all_probabilities", {}),
                "model": "Production_Hybrid_Classifier"
            }
            
        # Fallback if model not trained yet
        return {
            "intent": "DELIVERY_STATUS",
            "confidence": 0.50,
            "evidence": ["fallback_default"],
            "reason": "Default fallback classifier.",
            "model": "FallbackClassifier"
        }
