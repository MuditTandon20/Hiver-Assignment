"""Grounded Response Generation module with multi-provider and offline fallback support."""
import os
import re
from typing import Dict, Any, List, Optional
from src.generation.prompt_templates import AMAZON_SUPPORT_SYSTEM_PROMPT, format_generation_prompt
from src.utils.config import setup_logger

logger = setup_logger("generator")

class MockGroundedLLMGenerator:
    """
    High-fidelity offline grounded response generator.
    Synthesizes brand-aligned responses from top retrieved historical exemplars
    and domain-specific Amazon resolution guidelines.
    Ensures 100% offline reproducibility without API keys.
    """
    def __init__(self, brand_signoff: str = "^AMZ"):
        self.brand_signoff = brand_signoff

    def generate(
        self,
        customer_message: str,
        conversation_history: List[Dict[str, str]],
        predicted_intent: str,
        retrieved_examples: List[Dict[str, Any]]
    ) -> str:
        # If we have high similarity retrieved examples, adapt the best historical response
        if retrieved_examples and len(retrieved_examples) > 0:
            best_ex = retrieved_examples[0]
            best_resp = best_ex.get("brand_response_text", "")
            
            # Clean up rep tags and replace with standard signoff
            clean_resp = re.sub(r'(\^[A-Z]{2,3}|-[A-Z]{2,3})$', '', best_resp).strip()
            
            # If the response is solid and grounded, adapt it
            if len(clean_resp) > 15:
                return f"{clean_resp} {self.brand_signoff}"
                
        # Template-grounded fallback based on predicted intent
        intent_responses = {
            "DELIVERY_STATUS": "I'm sorry for the delay with your delivery! Please check your tracking status in Your Orders, or reach out to us at amazon.com/help so we can investigate.",
            "ITEM_ISSUE": "I'm so sorry your package arrived in this condition! Please reach out to us directly at amazon.com/contact-us so we can arrange a replacement or refund.",
            "RETURN_REFUND": "You can easily start a return and track your refund status by visiting Your Orders on Amazon.com. Let us know if you need further assistance!",
            "ACCOUNT_ACCESS": "We take account security very seriously. Please never share personal details on Twitter. Visit amazon.com/help to securely verify your account credentials.",
            "BILLING_PAYMENT": "We'd like to look into this charge with you. For account security, please reach out via our secure support page: amazon.com/help.",
            "PRIME_DIGITAL": "Sorry to hear you're experiencing issues with your digital service! Please try restarting the app or device. For more help, visit amazon.com/videohelp.",
            "PRODUCT_STOCK": "Thanks for your interest! Restock timelines vary by supplier. Please check the product page and select 'Notify Me' for stock updates.",
            "TECH_APP_ISSUE": "Sorry for the inconvenience! Please try clearing your app cache or restarting your device. You can also report this at amazon.com/help.",
            "FEEDBACK_COMPLAINT": "I'm truly sorry to hear about your experience. We want to make this right. Please contact our leadership team via secure chat at amazon.com/help.",
            "GENERAL_INQUIRY": "Thanks for reaching out to Amazon! Please let us know how we can assist you today, or visit amazon.com/help for general information.",
            "OUT_OF_SCOPE": "Thanks for contacting Amazon Support. If you have an inquiry regarding an Amazon order or service, please let us know how we can help."
        }
        
        reply = intent_responses.get(predicted_intent, intent_responses["DELIVERY_STATUS"])
        return f"{reply} {self.brand_signoff}"

class OpenAILLMGenerator:
    """OpenAI API-backed Grounded Generator with timeout, retry, and offline fallback."""
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None, brand_signoff: str = "^AMZ", timeout_seconds: float = 15.0):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), timeout=timeout_seconds)
        self.model_name = model_name
        self.brand_signoff = brand_signoff
        self.fallback = MockGroundedLLMGenerator(brand_signoff=brand_signoff)

    def generate(
        self,
        customer_message: str,
        conversation_history: List[Dict[str, str]],
        predicted_intent: str,
        retrieved_examples: List[Dict[str, Any]]
    ) -> str:
        prompt = format_generation_prompt(
            customer_message=customer_message,
            conversation_history=conversation_history,
            predicted_intent=predicted_intent,
            retrieved_examples=retrieved_examples
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": AMAZON_SUPPORT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=150
            )
            content = response.choices[0].message.content
            if content and content.strip():
                reply = content.strip()
                if self.brand_signoff not in reply:
                    reply = f"{reply} {self.brand_signoff}"
                return reply
        except Exception as e:
            logger.warning(f"OpenAI Generation failed ({e}). Gracefully falling back to grounded template generator.")
            
        return self.fallback.generate(
            customer_message=customer_message,
            conversation_history=conversation_history,
            predicted_intent=predicted_intent,
            retrieved_examples=retrieved_examples
        )

def create_response_generator(config: Dict[str, Any]):
    """Factory function for creating appropriate response generator."""
    provider = config.get("generation", {}).get("provider", "mock_rule_llm")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if provider == "openai" and openai_key:
        logger.info("Using OpenAI LLM Response Generator.")
        return OpenAILLMGenerator(
            model_name=config.get("generation", {}).get("model_name", "gpt-4o-mini"),
            api_key=openai_key
        )
    else:
        logger.info("Using Offline High-Fidelity Grounded Response Generator.")
        return MockGroundedLLMGenerator(
            brand_signoff=config.get("generation", {}).get("brand_signoff", "^AMZ")
        )
