"""Baseline 1: Trivial Majority Class Classifier & Canned Response Generator."""
from typing import Dict, Any, List

class TrivialMajorityClassifier:
    """Always predicts the empirical majority class (DELIVERY_STATUS)."""
    def __init__(self, majority_intent: str = "DELIVERY_STATUS"):
        self.majority_intent = majority_intent

    def predict(self, text: str) -> Dict[str, Any]:
        return {
            "intent": self.majority_intent,
            "confidence": 0.50, # Arbitrary baseline confidence
            "evidence": ["majority_class_fallback"],
            "model": "Baseline_1_TrivialMajority"
        }

    def generate_response(self, text: str) -> str:
        return "Thank you for contacting Amazon. Please visit Your Orders on Amazon.com to check the status of your delivery. ^AMZ"
