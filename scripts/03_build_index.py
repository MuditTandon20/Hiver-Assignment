"""Step 8 Script: Build FAISS Vector Index on Historical Training Conversations."""
import sys
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.retrieval.vector_store import FAISSVectorStore
from src.intents.taxonomy import heuristic_intent_match
from src.utils.config import load_config, setup_logger, resolve_path

logger = setup_logger("build_index")

def main():
    config = load_config()
    data_cfg = config.get("data", {})
    retrieval_cfg = config.get("retrieval", {})
    
    train_convs_path = data_cfg.get("train_conversations_path", "data/processed/train_conversations.json")
    index_file = retrieval_cfg.get("index_file", "data/indices/faiss_index.bin")
    metadata_file = retrieval_cfg.get("metadata_file", "data/indices/metadata.json")
    
    resolved_train_path = resolve_path(train_convs_path)
    logger.info(f"Loading training conversations from {resolved_train_path}...")
    with open(resolved_train_path, "r", encoding="utf-8") as f:
        train_convs = json.load(f)
        
    logger.info(f"Assigning intent annotations to {len(train_convs):,} training conversations...")
    for item in train_convs:
        if "intent" not in item:
            item["intent"] = heuristic_intent_match(item["customer_text"])
            
    vector_store = FAISSVectorStore(embedding_dim=retrieval_cfg.get("embedding_dim", 256))
    vector_store.build_index(train_convs)
    vector_store.save(index_file=index_file, metadata_file=metadata_file)
    logger.info(f"FAISS index successfully built and saved to {index_file} with {len(train_convs):,} items.")

if __name__ == "__main__":
    main()
