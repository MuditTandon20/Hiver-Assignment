"""Unit tests for data cleaning and conversation construction."""
import pytest
from src.data.cleaner import clean_tweet_text, is_english, extract_agent_signoff

def test_clean_tweet_text():
    raw_tweet = "@AmazonHelp @115821 My order hasn't arrived yet! &amp; tracking is broken."
    cleaned = clean_tweet_text(raw_tweet)
    assert "@AmazonHelp" not in cleaned
    assert "@115821" not in cleaned
    assert "&" in cleaned
    assert "My order hasn't arrived yet!" in cleaned

def test_is_english():
    assert is_english("Where is my package? It is late.") == True
    assert is_english("amazonのfireTVstickが見れない") == False
    assert is_english("") == False

def test_extract_agent_signoff():
    assert extract_agent_signoff("We would love to help you! ^TN") == "^TN"
    assert extract_agent_signoff("Please reach out to us at amazon.com/help -KC") == "-KC"
    assert extract_agent_signoff("Hello customer") is None
