"""Evaluation metrics computation for Classification, Retrieval, and Escalation."""
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.intents.taxonomy import ALL_INTENTS

def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Compute comprehensive multi-class classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-intent metrics
    labels = sorted(list(set(y_true + y_pred)))
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    
    per_intent = {}
    for lbl, p, r, f, s in zip(labels, p_per_class, r_per_class, f1_per_class, support):
        per_intent[lbl] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s)
        }
        
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(float(f1_macro), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_intent_metrics": per_intent,
        "confusion_matrix": cm,
        "labels": labels
    }

def compute_retrieval_metrics(
    queries: List[Dict[str, Any]],
    retrieved_results: List[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Compute Recall@k and Mean Reciprocal Rank (MRR) for intent-grounded retrieval."""
    recall_at_1 = []
    recall_at_3 = []
    reciprocal_ranks = []
    
    for q, retrieved in zip(queries, retrieved_results):
        target_intent = q.get("ground_truth_intent") or q.get("intent")
        if not target_intent or not retrieved:
            continue
            
        matches = [1 if r.get("intent") == target_intent else 0 for r in retrieved]
        
        recall_at_1.append(matches[0] if len(matches) > 0 else 0)
        recall_at_3.append(1 if any(matches[:3]) else 0)
        
        # Calculate RR
        rr = 0.0
        for rank, match in enumerate(matches, 1):
            if match == 1:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
        
    return {
        "recall@1": round(float(np.mean(recall_at_1)) if recall_at_1 else 0.0, 4),
        "recall@3": round(float(np.mean(recall_at_3)) if recall_at_3 else 0.0, 4),
        "mrr": round(float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0, 4),
        "evaluated_samples": len(queries)
    }

def compute_escalation_metrics(y_true_escalated: List[bool], y_pred_escalated: List[bool]) -> Dict[str, Any]:
    """Compute precision, recall, and F1 for human escalation decisions."""
    acc = float(accuracy_score(y_true_escalated, y_pred_escalated))
    p, r, f1, _ = precision_recall_fscore_support(y_true_escalated, y_pred_escalated, average='binary', zero_division=0)
    
    return {
        "escalation_accuracy": round(acc, 4),
        "escalation_precision": round(float(p), 4),
        "escalation_recall": round(float(r), 4),
        "escalation_f1": round(float(f1), 4),
        "total_evaluated": len(y_true_escalated)
    }
