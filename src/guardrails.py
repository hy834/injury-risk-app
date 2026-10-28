import re

INJECTION_PATTERNS = [
    r"ignore\s+(?:previous|all\s+prior|the\s+above)\s+(?:instructions|rules|prompts)",
    r"disregard\s+(?:the\s+protocol|your\s+instructions|the\s+rules)",
    r"(?:you\s+are\s+now|pretend\s+you\s+are|act\s+as)",
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

def check_prompt_injection(text: str) -> tuple[bool, str]:
    """Checks staff notes for prompt-injection attempts.
    
    Returns:
        (is_injected, sanitized_or_original_text)
    """
    if not text or not text.strip():
        return False, text

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            sanitized = "[notes removed: possible prompt injection, sent to medical team for manual reading]"
            return True, sanitized

    return False, text

