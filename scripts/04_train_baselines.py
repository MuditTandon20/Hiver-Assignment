"""Step 6 Script: Train Baseline 1 (Majority Class) and Baseline 2 (TF-IDF + Logistic Regression)."""
import sys
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.intents.tfidf_classifier import TfidfIntentClassifier
from src.intents.taxonomy import heuristic_intent_match
from src.utils.config import load_config, setup_logger, resolve_path

logger = setup_logger("train_baselines")

def main():
    config = load_config()
    data_cfg = config.get("data", {})
    intents_cfg = config.get("intents", {})
    
    train_convs_path = data_cfg.get("train_conversations_path", "data/processed/train_conversations.json")
    resolved_train_path = resolve_path(train_convs_path)
    model_dir = resolve_path(intents_cfg.get("model_dir", "data/models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading training data from {resolved_train_path}...")
    with open(resolved_train_path, "r", encoding="utf-8") as f:
        train_convs = json.load(f)
        
    texts = [c["customer_text"] for c in train_convs]
    labels = [c.get("intent") or heuristic_intent_match(c["customer_text"]) for c in train_convs]
    
    logger.info(f"Training Baseline 2 TF-IDF Intent Classifier on {len(texts):,} samples...")
    clf = TfidfIntentClassifier()
    clf.train(texts, labels)
    
    model_path = model_dir / "tfidf_classifier.joblib"
    clf.save(str(model_path))
    logger.info(f"Baseline 2 model successfully trained and saved to: {model_path}")

if __name__ == "__main__":
    main()
