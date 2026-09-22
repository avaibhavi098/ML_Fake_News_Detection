"""
Text Preprocessing Pipeline for Fake News Detection
Handles normalization, cleaning, noise removal, and publisher bias mitigation.
"""

import re
import html
import unicodedata

# Regex to detect Reuters / standard news agency datelines that cause publisher bias
# e.g., "WASHINGTON (Reuters) -", "LONDON (Reuters) -", "(Reuters) -"
REUTERS_DATELINE_REGEX = re.compile(
    r"^\s*([A-Z\s,]+)?\((?:reuters|ap|associated press|afp)\)\s*[-—–:]\s*",
    re.IGNORECASE
)

# Standalone news agency mentions that skew classification weights
AGENCY_MENTIONS_REGEX = re.compile(
    r"\b(?:reuters|associated press|thomson reuters)\b",
    re.IGNORECASE
)

# URL detection regex
URL_REGEX = re.compile(
    r"https?://\S+|www\.\S+|ftp://\S+",
    re.IGNORECASE
)

# HTML tags regex
HTML_TAG_REGEX = re.compile(r"<[^>]+>")

# Email detection regex
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Excessive whitespace / non-standard characters
WHITESPACE_REGEX = re.compile(r"\s+")


def clean_text(text: str, remove_agency_bias: bool = True) -> str:
    """
    Cleans and normalizes input text for TF-IDF vectorization and inference.
    
    Steps:
    1. Unescape HTML entities (&amp;, &lt;, etc.)
    2. Strip HTML tags
    3. Remove URLs and hyperlinks
    4. Remove email addresses
    5. Strip agency datelines (e.g. 'WASHINGTON (Reuters) -') to prevent source bias
    6. Normalize unicode characters (NFKD)
    7. Remove non-printable / control characters
    8. Condense excessive whitespace
    
    Args:
        text: Raw news text string
        remove_agency_bias: Whether to strip publisher-specific signatures
        
    Returns:
        Cleaned, normalized text string
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Unescape HTML entities
    cleaned = html.unescape(text)

    # 2. Strip HTML tags
    cleaned = HTML_TAG_REGEX.sub(" ", cleaned)

    # 3. Strip URLs
    cleaned = URL_REGEX.sub(" ", cleaned)

    # 4. Strip Emails
    cleaned = EMAIL_REGEX.sub(" ", cleaned)

    # 5. Mitigate publisher bias (ISOT dataset artifacts like "(Reuters) -")
    if remove_agency_bias:
        cleaned = REUTERS_DATELINE_REGEX.sub("", cleaned)
        cleaned = AGENCY_MENTIONS_REGEX.sub("", cleaned)

    # 6. Unicode normalization
    cleaned = unicodedata.normalize("NFKD", cleaned)

    # 7. Normalize whitespace
    cleaned = WHITESPACE_REGEX.sub(" ", cleaned).strip()

    return cleaned
