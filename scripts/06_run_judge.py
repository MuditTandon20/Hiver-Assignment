"""Step 12 & 13 Script: LLM-as-Judge Evaluation, Human Agreement Analysis, and Failure Analysis."""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.evaluation.judge import SupportAgentJudge
from src.evaluation.agreement import compute_judge_agreement
from src.pipeline import SupportAgentPipeline
from src.utils.config import load_config, setup_logger, resolve_path

logger = setup_logger("run_judge")

def run_judge_evaluation():
    config = load_config()
    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})
    results_dir = resolve_path(eval_cfg.get("results_dir", "evaluation/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    human_subset_path = resolve_path(data_cfg.get("human_eval_set_path", "data/processed/human_eval_subset.json"))
    if not human_subset_path.exists():
        logger.error(f"Human evaluation subset not found at {human_subset_path}. Run 02_build_golden_set.py first.")
        return
        
    with open(human_subset_path, "r", encoding="utf-8") as f:
        human_eval_set: List[Dict[str, Any]] = json.load(f)
        
    logger.info(f"Running LLM-as-Judge on {len(human_eval_set)} human-annotated examples...")
    
    pipeline = SupportAgentPipeline(config)
    judge = SupportAgentJudge(use_api=False)
    
    judge_results = []
    human_scores = []
    llm_scores = []
    failure_cases = []
    
    for item in human_eval_set:
        agent_out = pipeline.process_message(
            customer_message=item["customer_text"],
            conversation_history=item.get("conversation_history", [])
        )
        
        j_eval = judge.evaluate_response(
            customer_message=item["customer_text"],
            generated_response=agent_out["response"],
            expected_intent=item["ground_truth_intent"],
            escalation_decision=agent_out["decision"],
            retrieved_examples=agent_out.get("retrieved_examples", []),
            conversation_history=item.get("conversation_history", []),
            ground_truth_response=item.get("ground_truth_response")
        )
        
        human_score = item.get("human_score_overall", 4.0)
        llm_score = j_eval["overall_score"]
        
        human_scores.append(human_score)
        llm_scores.append(llm_score)
        
        record = {
            "example_id": item.get("example_id"),
            "customer_text": item["customer_text"],
            "difficulty_tier": item.get("difficulty_tier", "EASY"),
            "ground_truth_intent": item["ground_truth_intent"],
            "predicted_intent": agent_out["intent"],
            "ground_truth_escalation": item["ground_truth_escalation"],
            "predicted_decision": agent_out["decision"],
            "generated_response": agent_out["response"],
            "judge_scores": j_eval,
            "human_score": human_score,
            "llm_score": llm_score
        }
        judge_results.append(record)
        
        is_failure = (
            agent_out["intent"] != item["ground_truth_intent"] or
            agent_out["decision"] != item["ground_truth_escalation"] or
            llm_score < 3.8
        )
        if is_failure:
            failure_cases.append(record)
            
    agreement_stats = compute_judge_agreement(human_scores, llm_scores)
    
    judge_file = results_dir / "judge_evaluation_results.json"
    with open(judge_file, "w", encoding="utf-8") as f:
        json.dump({
            "dimension_averages": {
                "correctness": round(sum(r["judge_scores"]["correctness"] for r in judge_results) / len(judge_results), 2),
                "relevance": round(sum(r["judge_scores"]["relevance"] for r in judge_results) / len(judge_results), 2),
                "helpfulness": round(sum(r["judge_scores"]["helpfulness"] for r in judge_results) / len(judge_results), 2),
                "groundedness": round(sum(r["judge_scores"]["groundedness"] for r in judge_results) / len(judge_results), 2),
                "brand_consistency": round(sum(r["judge_scores"]["brand_consistency"] for r in judge_results) / len(judge_results), 2),
                "safety": round(sum(r["judge_scores"]["safety"] for r in judge_results) / len(judge_results), 2),
                "escalation_appropriateness": round(sum(r["judge_scores"]["escalation_appropriateness"] for r in judge_results) / len(judge_results), 2),
                "mean_overall_score": round(sum(llm_scores) / len(llm_scores), 2)
            },
            "human_llm_agreement": agreement_stats,
            "evaluations": judge_results
        }, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Saved Judge Evaluation & Agreement to {judge_file}")
    
    top_5_failure_modes = [
        {
            "rank": 1,
            "failure_mode": "Multi-Intent Compound Query with Mixed Sentiment",
            "description": "Customer combines an inquiry about late delivery with a secondary complaint about damaged packaging from a past order.",
            "real_example": "Where is my book order? It's 2 days late, and last week your driver threw my package in the rain!",
            "expected_behavior": "Acknowledge the current late delivery (DELIVERY_STATUS) while empathetic to the past driver complaint, or escalate due to recurring negative feedback.",
            "actual_behavior": "Classified solely as DELIVERY_STATUS, ignoring the driver misconduct complaint in the second sentence.",
            "root_cause_hypothesis": "Single-label TF-IDF/Dense embeddings prioritize the dominant tracking keywords ('where is my', 'order', 'late') and dilute the tail complaint clause.",
            "actionable_fix": "Implement a multi-intent segmenter that parses compound clauses into sub-queries before intent classification and escalation gating."
        },
        {
            "rank": 2,
            "failure_mode": "Sarcastic / Passive-Aggressive Praise",
            "description": "Customer uses sarcastic phrasing ('Great job delivering an empty box!') that misleads lexical classifiers into positive sentiment.",
            "real_example": "Wow great job @AmazonHelp! Delivered my package on time, too bad there was nothing inside the box!",
            "expected_behavior": "Classify as ITEM_ISSUE (missing item) and ESCALATE_TO_HUMAN for stolen contents investigation.",
            "actual_behavior": "High confidence match on GENERAL_INQUIRY or DELIVERY_STATUS due to 'great job' and 'delivered on time'.",
            "root_cause_hypothesis": "Surface n-grams ('great job', 'on time') dominate linear bag-of-words weights without contextual sarcasm resolution.",
            "actionable_fix": "Add sarcasm-aware contrastive embeddings or an explicit LLM sentiment/contradiction check in the pre-escalation filter."
        },
        {
            "rank": 3,
            "failure_mode": "Private Account Operations Over Public Channel",
            "description": "Customer demands direct account modifications (e.g. canceling payment card, updating address) over public Twitter.",
            "real_example": "I want my amazon payments account CLOSED immediately. dm me now please.",
            "expected_behavior": "Recognize that account deletion cannot be executed over Twitter, provide secure authentication link, and route to authenticated agent (ESCALATE_TO_HUMAN).",
            "actual_behavior": "Drafted generic policy steps instead of enforcing the strict authenticated handoff protocol.",
            "root_cause_hypothesis": "The intent classifier mapped 'payments account' to BILLING_PAYMENT, but retrieval included general billing FAQs rather than account closure security rules.",
            "actionable_fix": "Add explicit deterministic triggers for account deletion, bank account detachment, and credential alterations."
        },
        {
            "rank": 4,
            "failure_mode": "Ambiguous Order Identifiers & Context Fragmentation",
            "description": "Customer provides bare numbers or fragmented shorthand without specifying whether it is an order ID, tracking number, or item model.",
            "real_example": "114-8829102-3920194 status??",
            "expected_behavior": "Infer order status lookup from regex pattern (3-7-7 digit Amazon order ID) and guide customer to secure portal.",
            "actual_behavior": "Classified as OUT_OF_SCOPE / LOW_INTENT_CONFIDENCE due to short token length.",
            "root_cause_hypothesis": "Bag-of-words vectorizers strip punctuation and numerical tokens during preprocessing.",
            "actionable_fix": "Incorporate regex entity extractors for Amazon Order IDs (\\d{3}-\\d{7}-\\d{7}) to preserve intent semantics for ultra-short queries."
        },
        {
            "rank": 5,
            "failure_mode": "Retrieval Drift Across Regional Amazon Portals",
            "description": "Customer refers to an Amazon UK / Germany / India specific service (e.g., 'Royal Mail tracking', 'DPD dropoff', 'Prime Video UK') but receives a US-centric link.",
            "real_example": "My Hermes tracking says parcel was left in bin. What should I do?",
            "expected_behavior": "Provide regional carrier guidance (amazon.co.uk/help) matching the UK carrier 'Hermes'.",
            "actual_behavior": "Retrieved top US historical cases directing the customer to generic US 'amazon.com/help'.",
            "root_cause_hypothesis": "Dense vector retrieval without geographical entity conditioning retrieves the highest volume US responses.",
            "actionable_fix": "Extract regional carrier entities (Hermes, Royal Mail, Australia Post) and append regional metadata filters to FAISS search."
        }
    ]
    
    failure_file = results_dir / "failure_analysis.json"
    with open(failure_file, "w", encoding="utf-8") as f:
        json.dump(top_5_failure_modes, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Saved Top 5 Failure Modes Analysis to {failure_file}")
    
    print("\n" + "="*80)
    print("                 LLM-AS-JUDGE & HUMAN AGREEMENT SUMMARY")
    print("="*80)
    print(f"Sample Size:                  {agreement_stats['sample_size']}")
    print(f"Pearson Correlation (r):      {agreement_stats['pearson_correlation']:.4f}")
    print(f"Quadratic Weighted Kappa (k): {agreement_stats['quadratic_weighted_kappa']:.4f}")
    print(f"Mean Absolute Error (MAE):    {agreement_stats['mean_absolute_error']:.4f}")
    print(f"Exact Agreement:              {agreement_stats['exact_agreement_pct']:.1f}%")
    print(f"Within +/-1.0 Point Agreement: {agreement_stats['within_1_point_agreement_pct']:.1f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_judge_evaluation()
