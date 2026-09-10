"""Unit tests for Intent taxonomy and classifiers."""
import pytest
from src.intents.taxonomy import INTENT_TAXONOMY, ALL_INTENTS, heuristic_intent_match
from src.intents.trivial_classifier import TrivialMajorityClassifier
from src.intents.tfidf_classifier import TfidfIntentClassifier

def test_intent_taxonomy_structure():
    assert len(ALL_INTENTS) == 11
    for intent in ALL_INTENTS:
        meta = INTENT_TAXONOMY[intent]
        assert "name" in meta
        assert "description" in meta
        assert "keywords" in meta
        assert "inclusion_criteria" in meta
        assert "exclusion_criteria" in meta
        assert "default_escalation" in meta

def test_trivial_classifier():
    clf = TrivialMajorityClassifier()
    res = clf.predict("Where is my package?")
    assert res["intent"] == "DELIVERY_STATUS"
    assert res["model"] == "Baseline_1_TrivialMajority"
    resp = clf.generate_response("Where is my package?")
    assert "^AMZ" in resp

def test_tfidf_classifier_lifecycle(tmp_path):
    texts = [
        "Where is my package? Tracking is stuck.",
        "My box was smashed and headphones broken.",
        "How do I return this item for a refund?",
        "Someone hacked my account and I cannot log in.",
        "Charged twice for my prime membership order."
    ]
    labels = [
        "DELIVERY_STATUS",
        "ITEM_ISSUE",
        "RETURN_REFUND",
        "ACCOUNT_ACCESS",
        "BILLING_PAYMENT"
    ]
    
    clf = TfidfIntentClassifier()
    clf.train(texts, labels)
    
    res = clf.predict("My delivery is delayed and package missing.")
    assert "intent" in res
    assert "confidence" in res
    assert res["confidence"] >= 0.0 and res["confidence"] <= 1.0
    assert "all_probabilities" in res
    
    # Test serialization
    save_file = str(tmp_path / "test_model.joblib")
    clf.save(save_file)
    
    clf_loaded = TfidfIntentClassifier(model_path=save_file)
    res_loaded = clf_loaded.predict("My delivery is delayed and package missing.")
    assert res_loaded["intent"] == res["intent"]
