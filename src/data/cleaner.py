"""Text cleaning, language filtering, and normalization module."""
import re
import html
from typing import Optional

# Non-English detection heuristic: check ratio of ASCII/Latin alphabetic characters
def is_english(text: str, min_ascii_ratio: float = 0.75) -> bool:
    """Check if the text is predominantly English / Latin characters."""
    if not text or len(text.strip()) == 0:
        return False
    # Filter out common Japanese/Chinese/Cyrillic scripts
    latin_chars = len(re.findall(r'[a-zA-Z0-9\s\.,!\?\'\"\-_:;/@#\$%\^&\*\(\)]', text))
    total_chars = len(text)
    return (latin_chars / total_chars) >= min_ascii_ratio

def clean_tweet_text(text: str, remove_brand_mentions: bool = True) -> str:
    """
    Clean and normalize tweet text.
    - Decodes HTML entities (&amp;, &gt;, &lt;)
    - Normalizes excessive whitespace
    - Removes leading brand/user handles if requested
    - Preserves links/URLs in clean format
    """
    if not isinstance(text, str):
        return ""
    
    # Decode HTML entities
    cleaned = html.unescape(text)
    
    # Replace newlines and multiple spaces with a single space
    cleaned = re.sub(r'[\r\n\t]+', ' ', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    
    if remove_brand_mentions:
        # Remove @handles at the beginning or middle while preserving message body
        cleaned = re.sub(r'@\w+', '', cleaned)
    
    # Clean redundant spaces
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned

def extract_agent_signoff(text: str) -> Optional[str]:
    """Extract support rep sign-off tags like ^TN, ^AG, ^CH, -KC, ET."""
    match = re.search(r'(\^[A-Z]{2,3}|-[A-Z]{2,3}|\b[A-Z]{2}$)', text.strip())
    if match:
        return match.group(1)
    return None
