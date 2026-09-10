"""Baseline 2: TF-IDF + Calibrated Logistic Regression Intent Classifier."""
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.intents.taxonomy import ALL_INTENTS
from src.utils.config import setup_logger

logger = setup_logger("tfidf_classifier")

class TfidfIntentClassifier:
    """TF-IDF N-gram feature extractor + Logistic Regression Classifier."""
    def __init__(self, model_path: Optional[str] = None):
        self.pipeline: Optional[Pipeline] = None
        self.classes_: List[str] = ALL_INTENTS
        if model_path and Path(model_path).exists():
            self.load(model_path)
            
    def train(self, texts: List[str], labels: List[str]):
        """Train TF-IDF + Logistic Regression pipeline with balanced class weights."""
        logger.info(f"Training TF-IDF Logistic Regression on {len(texts):,} samples...")
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,
                stop_words='english'
            )),
            ('clf', LogisticRegression(
                C=2.0,
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs'
            ))
        ])
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.classes_)
        logger.info("TF-IDF Classifier training complete.")

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict intent label, confidence, and top driving keywords."""
        if not self.pipeline:
            raise RuntimeError("Classifier has not been trained or loaded.")
            
        probs = self.pipeline.predict_proba([text])[0]
        top_idx = int(np.argmax(probs))
        predicted_intent = self.classes_[top_idx]
        confidence = float(probs[top_idx])
        
        # Extract top matching ngrams as evidence
        vectorizer: TfidfVectorizer = self.pipeline.named_steps['tfidf']
        clf: LogisticRegression = self.pipeline.named_steps['clf']
        
        feature_names = np.array(vectorizer.get_feature_names_out())
        x_vec = vectorizer.transform([text])
        nonzero_indices = x_vec.nonzero()[1]
        
        evidence = []
        if len(nonzero_indices) > 0:
            coefs = clf.coef_[top_idx][nonzero_indices]
            top_words_idx = nonzero_indices[np.argsort(coefs)[::-1][:3]]
            evidence = [feature_names[idx] for idx in top_words_idx if clf.coef_[top_idx][idx] > 0]
            
        return {
            "intent": predicted_intent,
            "confidence": round(confidence, 4),
            "all_probabilities": {c: round(float(p), 4) for c, p in zip(self.classes_, probs)},
            "evidence": evidence,
            "model": "Baseline_2_TfidfLogReg"
        }

    def save(self, model_path: str):
        """Serialize pipeline to disk."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "classes": self.classes_}, path)
        logger.info(f"Saved TF-IDF model to {model_path}")

    def load(self, model_path: str):
        """Load serialized pipeline from disk."""
        data = joblib.load(model_path)
        self.pipeline = data["pipeline"]
        self.classes_ = data["classes"]
        logger.info(f"Loaded TF-IDF model from {model_path}")
