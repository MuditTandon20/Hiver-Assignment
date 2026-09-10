"""Step 1 & 3 Script: Download, clean, and build conversation threads for AmazonHelp."""
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.data.downloader import download_data
from src.data.conversation_builder import build_conversations, save_conversations
from src.utils.config import load_config, setup_logger

logger = setup_logger("prepare_data")

def main():
    config = load_config()
    data_cfg = config.get("data", {})
    brand_name = config.get("project", {}).get("selected_brand", "AmazonHelp")
    
    logger.info("=== STEP 1: DOWNLOADING DATASET ===")
    raw_csv = download_data(
        url=data_cfg.get("raw_data_url"),
        output_path=data_cfg.get("raw_data_path"),
        max_lines=data_cfg.get("sample_size_raw", 250000)
    )
    
    logger.info(f"=== STEP 2 & 3: CONSTRUCTING CONVERSATION THREADS FOR {brand_name} ===")
    train_convs, test_convs = build_conversations(
        raw_csv_path=str(raw_csv),
        brand_name=brand_name,
        max_conversations=15000,
        train_ratio=0.80,
        seed=config.get("project", {}).get("seed", 42)
    )
    
    # Save conversation datasets
    save_conversations(train_convs, data_cfg.get("train_conversations_path", "data/processed/train_conversations.json"))
    save_conversations(test_convs, data_cfg.get("test_conversations_path", "data/processed/test_conversations.json"))
    
    logger.info(f"Data preparation complete! Train pool: {len(train_convs):,}, Test pool: {len(test_convs):,}")

if __name__ == "__main__":
    main()
