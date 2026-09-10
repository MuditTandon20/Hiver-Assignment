"""Step 11 & 15 Script: Automated Evaluation Harness comparing Baseline 1, Baseline 2, and Final System."""
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.intents.trivial_classifier import TrivialMajorityClassifier
from src.intents.tfidf_classifier import TfidfIntentClassifier
from src.pipeline import SupportAgentPipeline
from src.evaluation.metrics import compute_classification_metrics, compute_retrieval_metrics, compute_escalation_metrics
from src.utils.config import load_config, setup_logger, resolve_path

logger = setup_logger("run_evaluation")

def evaluate_models():
    config = load_config()
    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})
    golden_path = resolve_path(data_cfg.get("golden_eval_set_path", "data/processed/golden_eval_set.json"))
    results_dir = resolve_path(eval_cfg.get("results_dir", "evaluation/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading Golden Evaluation Set from {golden_path}...")
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_set: List[Dict[str, Any]] = json.load(f)
        
    logger.info(f"Evaluating {len(golden_set)} golden examples across 3 systems...")
    
    y_true_intent = [item["ground_truth_intent"] for item in golden_set]
    y_true_escalated = [item["ground_truth_escalation"] == "ESCALATE_TO_HUMAN" for item in golden_set]
    
    # -------------------------------------------------------------
    # 1. BASELINE 1: Trivial Majority Classifier
    # -------------------------------------------------------------
    logger.info("Evaluating Baseline 1 (Trivial Majority)...")
    b1 = TrivialMajorityClassifier()
    start_time = time.time()
    b1_preds = [b1.predict(item["customer_text"])["intent"] for item in golden_set]
    b1_latency_ms = ((time.time() - start_time) / len(golden_set)) * 1000.0
    b1_metrics = compute_classification_metrics(y_true_intent, b1_preds)
    
    # -------------------------------------------------------------
    # 2. BASELINE 2: TF-IDF + Logistic Regression
    # -------------------------------------------------------------
    logger.info("Evaluating Baseline 2 (TF-IDF + Logistic Regression)...")
    model_dir = config.get("intents", {}).get("model_dir", "data/models")
    model_path = str(resolve_path(f"{model_dir}/tfidf_classifier.joblib"))
    b2 = TfidfIntentClassifier(model_path=model_path)
    start_time = time.time()
    b2_preds = [b2.predict(item["customer_text"])["intent"] for item in golden_set]
    b2_latency_ms = ((time.time() - start_time) / len(golden_set)) * 1000.0
    b2_metrics = compute_classification_metrics(y_true_intent, b2_preds)
    
    # -------------------------------------------------------------
    # 3. FINAL PRODUCTION SYSTEM: Hybrid Pipeline + FAISS + Escalation
    # -------------------------------------------------------------
    logger.info("Evaluating Final Production Support Agent...")
    pipeline = SupportAgentPipeline(config)
    
    pipeline_preds = []
    pipeline_escalated = []
    retrieved_all = []
    full_results = []
    
    start_time = time.time()
    for item in golden_set:
        res = pipeline.process_message(
            customer_message=item["customer_text"],
            conversation_history=item.get("conversation_history", [])
        )
        pipeline_preds.append(res["intent"])
        pipeline_escalated.append(res["decision"] == "ESCALATE_TO_HUMAN")
        retrieved_all.append(res.get("retrieved_examples", []))
        
        full_results.append({
            "example_id": item.get("example_id"),
            "customer_text": item["customer_text"],
            "difficulty_tier": item.get("difficulty_tier", "EASY"),
            "ground_truth_intent": item["ground_truth_intent"],
            "predicted_intent": res["intent"],
            "intent_confidence": res["intent_confidence"],
            "ground_truth_escalation": item["ground_truth_escalation"],
            "predicted_decision": res["decision"],
            "escalation_reason": res["escalation_reason"],
            "generated_response": res["response"],
            "retrieved_examples": res["retrieved_examples"]
        })
    pipeline_latency_ms = ((time.time() - start_time) / len(golden_set)) * 1000.0
    
    final_intent_metrics = compute_classification_metrics(y_true_intent, pipeline_preds)
    final_retrieval_metrics = compute_retrieval_metrics(golden_set, retrieved_all)
    final_escalation_metrics = compute_escalation_metrics(y_true_escalated, pipeline_escalated)
    
    # -------------------------------------------------------------
    # Compile Benchmark Results Summary
    # -------------------------------------------------------------
    benchmark_summary = {
        "experiment_comparison": [
            {
                "experiment": "Baseline 1 (Trivial Majority)",
                "intent_accuracy": b1_metrics["accuracy"],
                "macro_f1": b1_metrics["macro_f1"],
                "weighted_f1": b1_metrics["weighted_f1"],
                "retrieval_mrr": "N/A",
                "escalation_f1": "N/A",
                "avg_latency_ms": round(b1_latency_ms, 3),
                "cost_per_1k": "$0.00"
            },
            {
                "experiment": "Baseline 2 (TF-IDF + LogReg)",
                "intent_accuracy": b2_metrics["accuracy"],
                "macro_f1": b2_metrics["macro_f1"],
                "weighted_f1": b2_metrics["weighted_f1"],
                "retrieval_mrr": "N/A",
                "escalation_f1": "N/A",
                "avg_latency_ms": round(b2_latency_ms, 3),
                "cost_per_1k": "$0.00"
            },
            {
                "experiment": "Final Production System",
                "intent_accuracy": final_intent_metrics["accuracy"],
                "macro_f1": final_intent_metrics["macro_f1"],
                "weighted_f1": final_intent_metrics["weighted_f1"],
                "retrieval_mrr": final_retrieval_metrics["mrr"],
                "escalation_f1": final_escalation_metrics["escalation_f1"],
                "avg_latency_ms": round(pipeline_latency_ms, 3),
                "cost_per_1k": "$0.00 (Local / Mock)"
            }
        ],
        "detailed_metrics": {
            "baseline_1": b1_metrics,
            "baseline_2": b2_metrics,
            "final_intent_metrics": final_intent_metrics,
            "final_retrieval_metrics": final_retrieval_metrics,
            "final_escalation_metrics": final_escalation_metrics
        },
        "individual_predictions": full_results
    }
    
    # Save benchmark JSON
    benchmark_file = results_dir / "benchmark_results.json"
    with open(benchmark_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Saved benchmark results to {benchmark_file}")
    
    # Print formatted comparison table
    print("\n" + "="*80)
    print("                      BENCHMARK EXPERIMENT RESULTS TABLE")
    print("="*80)
    print(f"{'System / Model':<30} | {'Accuracy':<9} | {'Macro F1':<9} | {'MRR':<7} | {'Esc F1':<7} | {'Latency':<8}")
    print("-"*80)
    for exp in benchmark_summary["experiment_comparison"]:
        mrr_str = f"{exp['retrieval_mrr']:.4f}" if isinstance(exp['retrieval_mrr'], float) else exp['retrieval_mrr']
        esc_str = f"{exp['escalation_f1']:.4f}" if isinstance(exp['escalation_f1'], float) else exp['escalation_f1']
        print(f"{exp['experiment']:<30} | {exp['intent_accuracy']:<9.4f} | {exp['macro_f1']:<9.4f} | {mrr_str:<7} | {esc_str:<7} | {exp['avg_latency_ms']:<6.2f}ms")
    print("="*80 + "\n")
    
    return benchmark_summary

if __name__ == "__main__":
    evaluate_models()
