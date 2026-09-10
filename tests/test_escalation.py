"""Unit tests for multi-factor escalation policy."""
import pytest
from src.escalation.policy import EscalationPolicy

def test_sensitive_keyword_escalation():
    policy = EscalationPolicy(confidence_threshold=0.60)
    decision = policy.evaluate(
        customer_message="I will sue Amazon and take legal action with my attorney!",
        predicted_intent="FEEDBACK_COMPLAINT",
        intent_confidence=0.90,
        retrieved_examples=[{"similarity_score": 0.85}]
    )
    assert decision["decision"] == "ESCALATE_TO_HUMAN"
    assert decision["reason_code"] == "SENSITIVE_KEYWORD_TRIGGER"

def test_account_security_escalation():
    policy = EscalationPolicy()
    decision = policy.evaluate(
        customer_message="Someone changed my account password and stole my account.",
        predicted_intent="ACCOUNT_ACCESS",
        intent_confidence=0.92,
        retrieved_examples=[{"similarity_score": 0.80}]
    )
    assert decision["decision"] == "ESCALATE_TO_HUMAN"
    assert decision["reason_code"] == "ACCOUNT_SECURITY_RISK"

def test_low_confidence_escalation():
    policy = EscalationPolicy(confidence_threshold=0.70)
    decision = policy.evaluate(
        customer_message="weird ambiguous query here",
        predicted_intent="DELIVERY_STATUS",
        intent_confidence=0.45,
        retrieved_examples=[{"similarity_score": 0.50}]
    )
    assert decision["decision"] == "ESCALATE_TO_HUMAN"
    assert decision["reason_code"] == "LOW_INTENT_CONFIDENCE"

def test_safe_auto_handle():
    policy = EscalationPolicy(confidence_threshold=0.60, min_similarity_threshold=0.40)
    decision = policy.evaluate(
        customer_message="Where can I track my package delivery?",
        predicted_intent="DELIVERY_STATUS",
        intent_confidence=0.88,
        retrieved_examples=[{"similarity_score": 0.75}]
    )
    assert decision["decision"] == "AUTO_HANDLE"
    assert decision["reason_code"] == "SAFE_FOR_AUTOMATION"
