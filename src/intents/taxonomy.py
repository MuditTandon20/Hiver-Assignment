"""Intent taxonomy definitions, rules, and schemas for AmazonHelp support."""
from typing import Dict, List, Any

INTENT_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "DELIVERY_STATUS": {
        "name": "Order Tracking & Delivery Status",
        "description": "Customer asking where their package is, tracking updates, late delivery, or carrier delays.",
        "keywords": ["track", "tracking", "delivery", "delivered", "where is my package", "order status", "delayed", "late", "courier", "dispatch", "shipped", "hasn't arrived", "still waiting", "estimated delivery"],
        "inclusion_criteria": "Inquiries regarding transit progress, estimated delivery date, tracking status, or package not yet received.",
        "exclusion_criteria": "Package already received with damaged or missing contents.",
        "default_escalation": "AUTO_HANDLE",
        "sample_phrases": [
            "Where is my order? It was supposed to arrive yesterday.",
            "Tracking says delivered but I haven't received anything.",
            "My package has been stuck in transit for 4 days."
        ]
    },
    "ITEM_ISSUE": {
        "name": "Damaged, Defective or Missing Items",
        "description": "Customer received an open, damaged, defective, wrong, or incomplete order.",
        "keywords": ["broken", "damaged", "defective", "missing item", "empty box", "wrong item", "tampered", "opened package", "cracked", "not working", "faulty"],
        "inclusion_criteria": "Issues regarding physical state of received goods or missing contents.",
        "exclusion_criteria": "Delays where nothing has arrived yet.",
        "default_escalation": "ESCALATE_TO_HUMAN",
        "sample_phrases": [
            "My box was torn open and the headphones are missing!",
            "I ordered a blue shirt but received a red jacket instead.",
            "The screen on the monitor I received is cracked."
        ]
    },
    "RETURN_REFUND": {
        "name": "Returns, Replacements & Refund Status",
        "description": "Customer asking how to return an item, return label issues, replacement requests, or refund timeline.",
        "keywords": ["refund", "return", "replacement", "send back", "return label", "drop off", "exchange", "money back", "reimburse", "refund status"],
        "inclusion_criteria": "Inquiries regarding the return process, replacement eligibility, or money refunded.",
        "exclusion_criteria": "General billing disputes without return context.",
        "default_escalation": "AUTO_HANDLE",
        "sample_phrases": [
            "How do I return this item and get a refund?",
            "I dropped off my return 5 days ago, when will I get my refund?",
            "Can I get a replacement for an item that didn't fit?"
        ]
    },
    "ACCOUNT_ACCESS": {
        "name": "Account Security, Login & Authentication",
        "description": "Customer locked out of account, 2FA/OTP issues, password resets, unauthorized access alerts.",
        "keywords": ["locked out", "login", "password", "2fa", "otp", "verification code", "account suspended", "sign in", "hack", "unauthorized login", "reset password"],
        "inclusion_criteria": "Account authentication, credential issues, 2-factor verification, and profile lockouts.",
        "exclusion_criteria": "Subscription settings within an active accessible account.",
        "default_escalation": "ESCALATE_TO_HUMAN",
        "sample_phrases": [
            "I am locked out of my account and not getting the OTP on my phone.",
            "Someone hacked my account and changed the email address.",
            "Need help resetting my password, link is expired."
        ]
    },
    "BILLING_PAYMENT": {
        "name": "Charges, Invoices & Payment Methods",
        "description": "Unexpected charges, double charges, payment declines, gift card redemption, invoice requests.",
        "keywords": ["charged twice", "double charge", "unknown charge", "payment failed", "credit card", "gift card", "invoice", "receipt", "deducted", "overcharged", "bank statement"],
        "inclusion_criteria": "Questions regarding monetary transactions, card charges, promo codes, and invoices.",
        "exclusion_criteria": "Prime video subscription charge (categorize under PRIME_DIGITAL if specific to Prime).",
        "default_escalation": "ESCALATE_TO_HUMAN",
        "sample_phrases": [
            "I see an unknown charge of $49.99 from Amazon on my bank statement.",
            "My payment method keeps getting declined at checkout.",
            "How do I download the VAT invoice for my last purchase?"
        ]
    },
    "PRIME_DIGITAL": {
        "name": "Prime Membership & Digital Services",
        "description": "Prime benefits, Prime Video streaming errors, Kindle ebooks, Amazon Music, Audible, digital content.",
        "keywords": ["prime", "prime video", "kindle", "amazon music", "audible", "fire tv", "fire stick", "membership fee", "stream", "ebook", "cancel prime"],
        "inclusion_criteria": "Digital media, streaming troubleshooting, e-readers, and Prime membership features.",
        "exclusion_criteria": "Physical hardware defects out of warranty.",
        "default_escalation": "AUTO_HANDLE",
        "sample_phrases": [
            "Prime Video keeps buffering and showing error code 5004.",
            "How do I cancel my Amazon Prime membership auto-renewal?",
            "My Kindle book is not syncing across devices."
        ]
    },
    "PRODUCT_STOCK": {
        "name": "Product Inquiries & Stock Availability",
        "description": "Pre-orders, restock dates, product compatibility, seller authenticity, specifications.",
        "keywords": ["in stock", "restock", "preorder", "pre-order", "release date", "warranty", "compatible", "specification", "is this genuine", "seller"],
        "inclusion_criteria": "Pre-purchase questions about items, inventory availability, and specs.",
        "exclusion_criteria": "Issues with an already delivered item.",
        "default_escalation": "AUTO_HANDLE",
        "sample_phrases": [
            "When will the 256GB version of this phone be back in stock?",
            "Does this charger work with iPhone 13?",
            "Is the pre-order guaranteed to arrive on release day?"
        ]
    },
    "TECH_APP_ISSUE": {
        "name": "Website & App Technical Glitches",
        "description": "Amazon app crashing, website checkout buttons not working, search glitch, browser errors.",
        "keywords": ["app crashing", "website down", "error on page", "can't checkout", "cart not updating", "glitch", "bug", "server error", "broken button"],
        "inclusion_criteria": "Technical bugs on Amazon website or mobile applications.",
        "exclusion_criteria": "Streaming video player bugs (use PRIME_DIGITAL).",
        "default_escalation": "AUTO_HANDLE",
        "sample_phrases": [
            "The Amazon iOS app crashes every time I open the search tab.",
            "Getting a 500 internal server error when clicking place order.",
            "My cart shows 0 items even though I added 3 items."
        ]
    },
    "FEEDBACK_COMPLAINT": {
        "name": "Complaints & Dissatisfaction",
        "description": "Venting frustration about poor service, driver misconduct, repeated unresolved issues, legal threats.",
        "keywords": ["worst service", "terrible", "useless support", "horrible", "unacceptable", "scam", "never buying again", "lawyer", "complaint", "rude rep", "disgusted"],
        "inclusion_criteria": "Strong expressions of dissatisfaction, escalation demands, or staff complaints.",
        "exclusion_criteria": "Calm, objective inquiry about delivery or returns.",
        "default_escalation": "ESCALATE_TO_HUMAN",
        "sample_phrases": [
            "Your customer service is completely useless! 4 reps gave me false promises.",
            "Delivery driver threw my package over the fence and broke my flower pot!",
            "I've been waiting 2 weeks with zero resolution, unacceptable!"
        ]
    },
    "GENERAL_INQUIRY": {
        "name": "General Inquiries & Gratitude",
        "description": "Greetings, thank you notes, compliments, general policies, operating hours.",
        "keywords": ["thank you", "thanks", "appreciate", "hello", "hi", "good morning", "customer service hours", "how does amazon work"],
        "inclusion_criteria": "Positive feedback, casual polite banter, or broad non-order inquiries.",
        "exclusion_criteria": "Complaints or specific problem statements.",
        "default_escalation": "AUTO_HANDLE",
        "sample_phrases": [
            "Thank you so much for the quick help earlier today!",
            "Hi, what are the customer support phone hours?",
            "Just wanted to say the delivery guy was super polite today!"
        ]
    },
    "OUT_OF_SCOPE": {
        "name": "Out of Scope / Spam",
        "description": "Unintelligible gibberish, spam links, unrelated third-party products, completely irrelevant text.",
        "keywords": ["crypto", "bitcoin", "follow back", "random spam", "asdfgh"],
        "inclusion_criteria": "Irrelevant spam, advertisements, or non-actionable gibberish.",
        "exclusion_criteria": "Any intelligible customer support request.",
        "default_escalation": "ESCALATE_TO_HUMAN",
        "sample_phrases": [
            "Check out my new music video on YouTube!",
            "Invest in bitcoin today dm for details",
            "asdfjkl; 12345"
        ]
    }
}

ALL_INTENTS = list(INTENT_TAXONOMY.keys())

def get_intent_metadata(intent_name: str) -> Dict[str, Any]:
    """Retrieve metadata for a specific intent."""
    return INTENT_TAXONOMY.get(intent_name, INTENT_TAXONOMY["OUT_OF_SCOPE"])

import re

def _contains_any(text: str, keywords: List[str]) -> bool:
    """Helper to check if any keyword matches as a word boundary match in text."""
    for kw in keywords:
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def heuristic_intent_match(text: str) -> str:
    """Keyword-based intent rule matching for bootstrapping/labeling validation."""
    if not text or not text.strip():
        return "OUT_OF_SCOPE"
    text_clean = text.strip()
    
    # Priority order checks with word-boundary awareness
    if _contains_any(text_clean, ["hacked", "locked out", "otp", "password reset", "unauthorized login", "sign in"]):
        return "ACCOUNT_ACCESS"
    if _contains_any(text_clean, ["charged twice", "unknown charge", "double charge", "overcharged", "bank statement", "declined"]):
        return "BILLING_PAYMENT"
    if _contains_any(text_clean, ["prime video", "fire tv", "firestick", "kindle", "audible", "stream"]):
        return "PRIME_DIGITAL"
    if _contains_any(text_clean, ["broken", "damaged", "missing item", "empty box", "tampered", "wrong item", "cracked", "defective"]):
        return "ITEM_ISSUE"
    if _contains_any(text_clean, ["refund", "return label", "send back", "return item", "exchange", "reimburse"]):
        return "RETURN_REFUND"
    if _contains_any(text_clean, ["track", "tracking", "delivery", "delivered", "where is my", "shipped", "delayed", "courier", "package", "dispatch"]):
        return "DELIVERY_STATUS"
    if _contains_any(text_clean, ["app crash", "website down", "500 error", "bug", "glitch", "can't checkout"]):
        return "TECH_APP_ISSUE"
    if _contains_any(text_clean, ["in stock", "restock", "preorder", "compatible", "release date", "specification"]):
        return "PRODUCT_STOCK"
    if _contains_any(text_clean, ["terrible", "worst", "unacceptable", "horrible", "scam", "lawyer", "rude", "useless", "lawsuit"]):
        return "FEEDBACK_COMPLAINT"
    if _contains_any(text_clean, ["thank", "thanks", "hello", "hi there", "great service", "appreciate"]):
        return "GENERAL_INQUIRY"
        
    return "DELIVERY_STATUS" # Default most frequent class
