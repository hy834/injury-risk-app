import re

INJECTION_PATTERNS = [
    r"ignore\s+(?:previous|all\s+prior|the\s+above)\s+(?:instructions|rules|prompts)",
    r"disregard\s+(?:the\s+protocol|your\s+instructions|the\s+rules)",
    r"(?:you\s+are\s+now|pretend\s+you\s+are|act\s+as\s+(?:a|an|the|if|my|your)\b)",
    r"(?:print|reveal|show\s+me)\s+(?:your\s+system\s+prompt|the\s+api\s+key|the\s+instructions)",
    r"api\s*key",
    r"\boverride\b",
    r"new\s+instructions",
    r"no\s+need\s+for\s+medical",
    r"</?(?:staff_notes|system)>",
    r"forget\s+(?:your\s+rules|everything)",
    r"developer\s+mode",
    r"\bjailbreak\b",
    r"(?:say|state|declare|mark)\s+.*?(?:cleared|fit\s+to\s+play)",
    r"(?:clear|bench)\s+(?:this|the)\s+player",
    r"approve\s+selection",
    r"bench\s+the\s+player"
]

import unicodedata

# Look-alike characters used to dodge keyword filters ("ign0re", "1gnore", "@ct as").
_LEET = str.maketrans({"0": "o", "1": "i", "!": "i", "|": "i", "3": "e", "4": "a", "@": "a",
                       "5": "s", "$": "s", "7": "t", "+": "t", "8": "b", "9": "g"})
_INVISIBLE = re.compile(r"[\u200b-\u200f\u2060\ufeff\u00ad]")


def _variants(text: str) -> list[str]:
    """The note as written, plus versions with common disguises undone:
    accents and full-width letters, invisible characters, look-alike digits and symbols,
    letters split by dots or dashes ("i.g.n.o.r.e") and letters split by spaces ("i g n o r e")."""
    base = unicodedata.normalize("NFKC", text)
    base = "".join(c for c in unicodedata.normalize("NFKD", base) if not unicodedata.combining(c))
    base = _INVISIBLE.sub("", base).lower()
    leet = base.translate(_LEET)
    joined = re.sub(r"(?<=[a-z])[.\-_*~'`]+(?=[a-z])", "", leet)
    spaced = re.sub(r"\b(?:[a-z] ){2,}[a-z]\b", lambda m: m.group(0).replace(" ", ""), joined)
    squeezed = re.sub(r"([a-z])\1{2,}", r"\1", spaced)  # "ignoooore" -> "ignore"
    return [text, base, leet, joined, spaced, squeezed]


def check_prompt_injection(text: str) -> tuple[bool, str]:
    """Checks staff notes for prompt-injection attempts, including disguised spellings.

    Returns:
        (is_injected, sanitized_or_original_text)
    """
    if not text or not text.strip():
        return False, text

    for variant in _variants(text):
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, variant, re.IGNORECASE):
                sanitized = "[notes removed: possible prompt injection, sent to medical team for manual reading]"
                return True, sanitized

    return False, text

