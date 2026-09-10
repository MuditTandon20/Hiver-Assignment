"""Pydantic schemas and data contracts for AI Support Agent."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class ConversationTurn(BaseModel):
    """Single turn in a conversation."""
    model_config = ConfigDict(extra="ignore")
    role: str = Field(..., description="Role: 'customer' or 'agent'")
    text: str = Field(..., description="Turn text content")
    tweet_id: Optional[str] = Field(default=None, description="Optional tweet ID")

class RetrievedCase(BaseModel):
    """Retrieved historical customer support exemplar."""
    model_config = ConfigDict(extra="ignore")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    conversation_id: str = Field(..., description="Exemplar ID")
    customer_text: str = Field(..., description="Customer message in exemplar")
    brand_response_text: str = Field(..., description="Brand resolution text")
    intent: str = Field(default="UNKNOWN", description="Intent label")
    is_multi_turn: bool = Field(default=False, description="Whether exemplar was multi-turn")

class IntentPrediction(BaseModel):
    """Intent classification result."""
    model_config = ConfigDict(extra="ignore")
    intent: str = Field(..., description="Predicted intent label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    evidence: List[str] = Field(default_factory=list, description="Top tokens/rules driving the classification")
    reason: str = Field(default="", description="Human-readable classification rationale")
    all_probabilities: Optional[Dict[str, float]] = Field(default=None, description="Probabilities across classes")
    model: str = Field(default="HybridClassifier", description="Model identifier")

class EscalationDecision(BaseModel):
    """Escalation gating policy evaluation result."""
    model_config = ConfigDict(extra="ignore")
    decision: str = Field(..., description="'AUTO_HANDLE' or 'ESCALATE_TO_HUMAN'")
    reason_code: str = Field(..., description="Standardized reason code")
    reason: str = Field(..., description="Detailed explanation of the escalation decision")
    escalated: bool = Field(..., description="True if escalated to human")

class AgentRequest(BaseModel):
    """Incoming request to the Support Agent."""
    model_config = ConfigDict(extra="ignore")
    customer_message: str = Field(..., min_length=1, description="Incoming customer message")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Prior conversation turns")

class AgentResponse(BaseModel):
    """Structured response from the Support Agent Pipeline."""
    model_config = ConfigDict(extra="ignore")
    intent: str
    intent_confidence: float
    intent_reason: str
    decision: str
    escalation_reason_code: str
    escalation_reason: str
    response: str
    retrieved_examples: List[Dict[str, Any]]
    is_multi_turn: bool

class JudgeEvaluationResult(BaseModel):
    """7-Dimension LLM-as-Judge evaluation result."""
    model_config = ConfigDict(extra="ignore")
    correctness: int = Field(..., ge=1, le=5)
    relevance: int = Field(..., ge=1, le=5)
    helpfulness: int = Field(..., ge=1, le=5)
    groundedness: int = Field(..., ge=1, le=5)
    brand_consistency: int = Field(..., ge=1, le=5)
    safety: int = Field(..., ge=1, le=5)
    escalation_appropriateness: int = Field(..., ge=1, le=5)
    overall_score: float = Field(..., ge=1.0, le=5.0)
    critique: str
