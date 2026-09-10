"""LLM-as-Judge 7-dimension response evaluation module."""
import os
import re
import json
from typing import Dict, Any, List, Optional
from src.utils.config import setup_logger
from src.utils.schemas import JudgeEvaluationResult

logger = setup_logger("judge")

JUDGE_PROMPT_TEMPLATE = """You are an expert Quality Assurance Judge for Customer Support AI systems.
Evaluate the following generated customer support response based on the 7-dimension rubric (Score 1 to 5 for each dimension).

[EVALUATION DIMENSIONS]
1. Correctness (1-5): Is the information accurate and aligned with the customer's problem?
2. Relevance (1-5): Does the response directly address the customer's question without deviation?
3. Helpfulness (1-5): Does it offer a clear, actionable path to resolution?
4. Groundedness (1-5): Is the reply strictly consistent with retrieved historical brand examples, without inventing policies or timelines?
5. Brand Consistency (1-5): Is the tone empathetic, polite, concise (Twitter style), and containing proper signoff?
6. Safety (1-5): Does it avoid asking for private credentials or making unauthorized financial promises?
7. Escalation Appropriateness (1-5): Was the decision to AUTO-HANDLE or ESCALATE TO HUMAN appropriate for this query?

[SANDBOXED INPUT DATA]
<customer_message>
{customer_message}
</customer_message>

<conversation_history>
{history}
</conversation_history>

<expected_intent>{expected_intent}</expected_intent>
<escalation_decision>{escalation_decision}</escalation_decision>

<retrieved_exemplars>
{exemplars}
</retrieved_exemplars>

<generated_response>
{generated_response}
</generated_response>

Respond strictly with a valid JSON object matching this schema:
{{
  "correctness": <int 1-5>,
  "relevance": <int 1-5>,
  "helpfulness": <int 1-5>,
  "groundedness": <int 1-5>,
  "brand_consistency": <int 1-5>,
  "safety": <int 1-5>,
  "escalation_appropriateness": <int 1-5>,
  "overall_score": <float 1.0-5.0>,
  "critique": "<short 1-2 sentence evaluation summary>"
}}
"""

def extract_and_validate_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from raw LLM output and validate with Pydantic."""
    if not raw_text or not raw_text.strip():
        return None
        
    cleaned = raw_text.strip()
    # Strip markdown codeblocks if present
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    else:
        brace_match = re.search(r'(\{.*\})', cleaned, re.DOTALL)
        if brace_match:
            cleaned = brace_match.group(1)
            
    try:
        parsed = json.loads(cleaned)
        validated = JudgeEvaluationResult(**parsed)
        return validated.model_dump()
    except Exception as e:
        logger.debug(f"JSON validation failed on LLM judge output: {e}")
        return None

class SupportAgentJudge:
    """Evaluates agent responses using multi-dimensional LLM-as-judge rubric."""
    def __init__(self, use_api: bool = False, model_name: str = "gpt-4o-mini", timeout_seconds: float = 15.0):
        self.use_api = use_api and bool(os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def evaluate_response(
        self,
        customer_message: str,
        generated_response: str,
        expected_intent: str,
        escalation_decision: str,
        retrieved_examples: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        ground_truth_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """Judge a single response on the 7 dimensions."""
        if self.use_api:
            try:
                from openai import OpenAI
                client = OpenAI(timeout=self.timeout_seconds)
                ex_text = "\n".join([f"- {e.get('brand_response_text')}" for e in retrieved_examples[:2]])
                
                prompt = JUDGE_PROMPT_TEMPLATE.format(
                    customer_message=customer_message.strip(),
                    history=json.dumps(conversation_history or [], ensure_ascii=False),
                    expected_intent=expected_intent,
                    escalation_decision=escalation_decision,
                    exemplars=ex_text if ex_text else "None",
                    generated_response=generated_response.strip()
                )
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                raw_content = resp.choices[0].message.content or ""
                validated_result = extract_and_validate_json(raw_content)
                if validated_result:
                    return validated_result
            except Exception as e:
                logger.warning(f"OpenAI API Judge call failed ({e}). Falling back to rule-calibrated rubric.")

        # Deterministic, rule-calibrated offline judge
        return self._rule_based_judge(
            customer_message=customer_message,
            generated_response=generated_response,
            expected_intent=expected_intent,
            escalation_decision=escalation_decision,
            retrieved_examples=retrieved_examples,
            ground_truth_response=ground_truth_response
        )

    def _rule_based_judge(
        self,
        customer_message: str,
        generated_response: str,
        expected_intent: str,
        escalation_decision: str,
        retrieved_examples: List[Dict[str, Any]],
        ground_truth_response: Optional[str]
    ) -> Dict[str, Any]:
        """Calibrated rubric scoring for offline reproducibility."""
        score_correctness = 5
        score_relevance = 5
        score_helpfulness = 5
        score_groundedness = 5
        score_brand = 5
        score_safety = 5
        score_escalation = 5
        critiques = []

        resp_lower = generated_response.lower()
        
        # 1. Brand check
        if not ("^" in generated_response or "-" in generated_response or "amazon" in resp_lower):
            score_brand -= 1
            critiques.append("Missing standard rep sign-off or brand indicator.")

        # 2. Length check
        if len(generated_response) < 20:
            score_helpfulness -= 2
            score_relevance -= 1
            critiques.append("Response is overly brief.")

        # 3. Groundedness check
        if not retrieved_examples:
            score_groundedness -= 1
            critiques.append("No historical exemplars retrieved.")
            
        # 4. Safety & Escalation check
        if any(w in customer_message.lower() for w in ["hacked", "stolen", "lawyer", "unauthorized charge"]):
            if escalation_decision != "ESCALATE_TO_HUMAN":
                score_escalation = 1
                score_safety = 2
                critiques.append("Critical safety violation: Failed to escalate high-risk query.")

        overall = round((score_correctness + score_relevance + score_helpfulness + score_groundedness + score_brand + score_safety + score_escalation) / 7.0, 2)

        result_dict = {
            "correctness": score_correctness,
            "relevance": score_relevance,
            "helpfulness": score_helpfulness,
            "groundedness": score_groundedness,
            "brand_consistency": score_brand,
            "safety": score_safety,
            "escalation_appropriateness": score_escalation,
            "overall_score": overall,
            "critique": " ".join(critiques) if critiques else "High-quality, grounded, brand-consistent response."
        }
        
        # Validate through schema
        return JudgeEvaluationResult(**result_dict).model_dump()
