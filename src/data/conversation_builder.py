"""Conversation thread reconstruction and splitting module."""
import csv
import io
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from src.data.cleaner import clean_tweet_text, is_english
from src.utils.config import setup_logger, resolve_path

logger = setup_logger("conversation_builder")

def build_conversations(
    raw_csv_path: str,
    brand_name: str = "AmazonHelp",
    max_conversations: Optional[int] = 20000,
    train_ratio: float = 0.80,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Construct multi-turn conversation threads for a specific brand from raw Twitter support CSV.
    Returns (train_conversations, test_conversations) ensuring zero thread leakage.
    """
    random.seed(seed)
    resolved_csv = resolve_path(raw_csv_path)
    logger.info(f"Loading raw tweets from {resolved_csv} for brand: {brand_name}...")
    
    tweets_by_id: Dict[str, Dict[str, Any]] = {}
    brand_response_ids: List[str] = []
    
    with open(resolved_csv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweet_id = row.get("tweet_id", "").strip()
            if not tweet_id:
                continue
            tweets_by_id[tweet_id] = row
            # If this is a brand reply
            if row.get("inbound", "").lower() == "false" and row.get("author_id", "") == brand_name:
                brand_response_ids.append(tweet_id)
                
    logger.info(f"Loaded {len(tweets_by_id):,} total tweets. Found {len(brand_response_ids):,} {brand_name} responses.")
    
    conversations: List[Dict[str, Any]] = []
    processed_customer_tweets = set()
    
    for resp_id in brand_response_ids:
        resp_row = tweets_by_id[resp_id]
        parent_id = resp_row.get("in_response_to_tweet_id", "").strip()
        
        if not parent_id or parent_id not in tweets_by_id:
            continue
            
        cust_row = tweets_by_id[parent_id]
        if cust_row.get("inbound", "").lower() != "true":
            continue
            
        # Avoid duplicate evaluations on identical customer tweet IDs
        if parent_id in processed_customer_tweets:
            continue
            
        cust_raw_text = cust_row.get("text", "")
        brand_raw_text = resp_row.get("text", "")
        
        # Filter for English language and non-empty content
        if not is_english(cust_raw_text) or not is_english(brand_raw_text):
            continue
            
        cust_clean = clean_tweet_text(cust_raw_text)
        brand_clean = clean_tweet_text(brand_raw_text, remove_brand_mentions=False)
        
        if len(cust_clean) < 5 or len(brand_clean) < 5:
            continue
            
        # Trace prior turns recursively if any (up to 3 turns) with cycle detection
        history: List[Dict[str, str]] = []
        curr_parent = cust_row.get("in_response_to_tweet_id", "").strip()
        depth = 0
        seen_turn_ids = {cust_row["tweet_id"], resp_row["tweet_id"]}
        while curr_parent and curr_parent in tweets_by_id and depth < 3 and curr_parent not in seen_turn_ids:
            seen_turn_ids.add(curr_parent)
            prev_tweet = tweets_by_id[curr_parent]
            role = "customer" if prev_tweet.get("inbound", "").lower() == "true" else "agent"
            prev_clean = clean_tweet_text(prev_tweet.get("text", ""), remove_brand_mentions=False)
            if prev_clean:
                history.insert(0, {"role": role, "text": prev_clean, "tweet_id": curr_parent})
            curr_parent = prev_tweet.get("in_response_to_tweet_id", "").strip()
            depth += 1
            
        conv = {
            "conversation_id": f"conv_{cust_row['tweet_id']}",
            "customer_tweet_id": cust_row["tweet_id"],
            "customer_author_id": cust_row.get("author_id", "unknown"),
            "customer_text": cust_clean,
            "raw_customer_text": cust_raw_text,
            "brand_response_id": resp_row["tweet_id"],
            "brand_response_text": brand_clean,
            "raw_brand_text": brand_raw_text,
            "conversation_history": history,
            "is_multi_turn": len(history) > 0,
            "created_at": cust_row.get("created_at", "")
        }
        
        conversations.append(conv)
        processed_customer_tweets.add(parent_id)
        
        if max_conversations and len(conversations) >= max_conversations:
            break
            
    logger.info(f"Constructed {len(conversations):,} clean conversations for {brand_name}.")
    
    # Shuffle and split
    random.shuffle(conversations)
    split_idx = int(len(conversations) * train_ratio)
    train_convs = conversations[:split_idx]
    test_convs = conversations[split_idx:]
    
    logger.info(f"Split dataset into: {len(train_convs):,} Train / Retrieval items, {len(test_convs):,} Test pool items.")
    return train_convs, test_convs

def save_conversations(conversations: List[Dict[str, Any]], filepath: str):
    """Save conversations list to JSON file."""
    path = resolve_path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(conversations):,} conversations to {path}")
