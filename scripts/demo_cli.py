"""Interactive CLI demo for testing the AI Customer Support Agent in real-time."""
import sys
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.pipeline import SupportAgentPipeline
from src.utils.config import load_config

def main():
    print("="*75)
    print("  AmazonHelp AI Customer Support Agent — Interactive CLI Demo")
    print("="*75)
    print("Initializing pipeline components...")
    
    config = load_config()
    pipeline = SupportAgentPipeline(config)
    
    print("\nReady! Enter customer messages to test the pipeline (type 'exit' or 'quit' to quit).\n")
    
    while True:
        try:
            user_input = input("\n[Customer Tweet] > ").strip()
            if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting demo. Goodbye!")
                break
                
            res = pipeline.process_message(user_input)
            
            print("\n" + "-"*60)
            print(f"🤖 PREDICTED INTENT:    {res['intent']} (Confidence: {res['intent_confidence']*100:.1f}%)")
            print(f"⚡ INTENT REASON:       {res['intent_reason']}")
            print(f"🚨 ESCALATION DECISION: {res['decision']} [{res['escalation_reason_code']}]")
            print(f"📋 ESCALATION REASON:   {res['escalation_reason']}")
            print(f"💬 DRAFTED RESPONSE:    \n   \"{res['response']}\"")
            
            if res.get("retrieved_examples"):
                print(f"\n🔍 RETRIEVED HISTORICAL EXEMPLARS ({len(res['retrieved_examples'])}):")
                for i, ex in enumerate(res["retrieved_examples"], 1):
                    print(f"   [{i}] (Sim: {ex['similarity_score']:.2f}) Cust: {ex['customer_text'][:60]}... | Rep: {ex['brand_response_text'][:60]}...")
            print("-"*60)
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"Error processing message: {e}")

if __name__ == "__main__":
    main()
