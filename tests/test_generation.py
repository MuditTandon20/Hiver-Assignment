"""Unit tests for response generation and prompt formatting."""
import pytest
from src.generation.generator import MockGroundedLLMGenerator
from src.generation.prompt_templates import format_generation_prompt

def test_prompt_formatting():
    prompt = format_generation_prompt(
        customer_message="My package is late.",
        conversation_history=[{"role": "user", "text": "Hi"}],
        predicted_intent="DELIVERY_STATUS",
        retrieved_examples=[{"customer_text": "Late order", "brand_response_text": "Please check Orders. ^AMZ", "similarity_score": 0.88}]
    )
    assert "DELIVERY_STATUS" in prompt
    assert "My package is late." in prompt
    assert '<exemplar index="1"' in prompt

def test_mock_grounded_generator():
    gen = MockGroundedLLMGenerator(brand_signoff="^AMZ")
    resp = gen.generate(
        customer_message="Where is my package?",
        conversation_history=[],
        predicted_intent="DELIVERY_STATUS",
        retrieved_examples=[{"brand_response_text": "Please check Your Orders on Amazon.com. ^TN"}]
    )
    assert "^AMZ" in resp
    assert len(resp) > 20
