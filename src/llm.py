"""LLM briefing layer (Groq, free tier).

The LLM proposes, Python enforces:
- JSON mode + Pydantic validation
- every cited clause must be one the app retrieved
- no clearing / benching wording
- the final line is fixed
- one retry with the validation error, then a deterministic template fallback
- 429 rate limits are retried with back-off before falling back
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

MAIN_MODEL = "llama-3.3-70b-versatile"
BACKUP_MODEL = "llama-3.1-8b-instant"
# Preference order. Groq retires models over time, so the app asks Groq which of these
# the key can use and falls back down this list automatically.
PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
]


MODEL_FALLBACKS: list[str] = []  # set by the app from available_models()


def available_models(api_key: str | None) -> tuple[list[str], str | None]:
    """Ask Groq which chat models this key can use. Returns (models in preference order, error or None)."""
    if not api_key:
        return list(PREFERRED_MODELS), None
    try:
        from groq import Groq
        ids = {m.id for m in Groq(api_key=api_key, max_retries=0, timeout=15).models.list().data}
    except Exception as e:  # network or key problem: keep the default list, report the reason
        return list(PREFERRED_MODELS), f"{type(e).__name__}: {str(e)[:160]}"
    ordered = [m for m in PREFERRED_MODELS if m in ids]
    extra = sorted(m for m in ids if m not in ordered and not any(x in m for x in ("whisper", "guard", "tts", "compound", "prompt")))
    return (ordered + extra) or list(PREFERRED_MODELS), None
FINAL_LINE = "Final decision: medical team."
NOT_COVERED = "Protocol does not cover this; refer to the medical team."

# Wording a briefing may never use (the system never clears or benches a player [P8]).
FORBIDDEN_OUTPUT = [
    r"\b(?:is|are|be|been|now|has been)\s+cleared\b",
    r"\bcleared\s+(?:to|for)\s+(?:play|train|full|selection|match|the match)",
    r"\bfit\s+to\s+(?:play|train)\b",
    r"\bfully\s+fit\b",
    r"\bsafe\s+to\s+(?:play|train|return)\b",
    r"\b(?:should|must|will|can)\s+be\s+benched\b",
    r"\bbench\s+(?:him|her|them|the player|this player)\b",
    r"\bdrop\s+(?:him|her|them|the player)\s+from\b",
    r"\bready\s+for\s+(?:full\s+)?selection\b",
]


class Action(BaseModel):
    clause_id: str = Field(pattern=r"^P\d{1,2}$")
    action: str = Field(min_length=3, max_length=400)


class Briefing(BaseModel):
    risk_summary: str = Field(min_length=3, max_length=500)
    actions: list[Action] = Field(default_factory=list, max_length=6)
    edge_case_note: str = ""
    covered: bool
    final_line: str


# ---------------------------------------------------------------- helpers
def read_text(path: str) -> str:
    """Read a rule file fresh from disk on every call (never cached)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def text_hash(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:16]


def forbidden_hits(text: str) -> list[str]:
    low = (text or "").lower()
    return [p for p in FORBIDDEN_OUTPUT if re.search(p, low)]


def all_text(b: Briefing) -> str:
    return " ".join([b.risk_summary, b.edge_case_note, b.final_line] + [a.action for a in b.actions])


def validate_briefing(raw: str, allowed_ids: list[str], required_ids: list[str] | None = None,
                      expect_not_covered: bool = False) -> tuple[Briefing | None, list[str]]:
    """Parse and check one LLM reply. Returns (briefing or None, list of problems)."""
    try:
        data = json.loads(raw)
    except Exception:
        return None, ["Reply was not valid JSON."]
    try:
        b = Briefing.model_validate(data)
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(x) for x in first.get("loc", []))
        return None, [f"JSON does not match the schema ({loc}: {first.get('msg')})."]

    problems: list[str] = []
    allowed = set(allowed_ids)
    if b.final_line.strip() != FINAL_LINE:
        problems.append(f'final_line must be exactly "{FINAL_LINE}"')
    cited = {a.clause_id for a in b.actions} | set(re.findall(r"\[(P\d{1,2})\]", all_text(b)))
    unknown = sorted(cited - allowed, key=lambda c: int(c[1:]))
    if unknown:
        problems.append(f"cites clauses that were not provided: {', '.join(unknown)}")
    hits = forbidden_hits(all_text(b))
    if hits:
        problems.append("uses clearing or benching wording, which only the medical team may decide")
    if expect_not_covered is False and required_ids:
        missing = [c for c in required_ids if c not in {a.clause_id for a in b.actions}]
        if missing:
            problems.append(f"must include an action for every triggered clause; missing: {', '.join(missing)}")
    if not b.covered and NOT_COVERED.lower()[:30] not in all_text(b).lower():
        problems.append(f'when covered is false the reply must say "{NOT_COVERED}"')
    return (b if not problems else None), problems


# ---------------------------------------------------------------- Groq call
class LLMUnavailable(Exception):
    """Raised when the API cannot be used (no key, quota, outage). The caller falls back."""


def _retry_after_seconds(err) -> float | None:
    try:
        v = err.response.headers.get("retry-after")
        return float(v) if v is not None else None
    except Exception:
        return None


def call_groq(api_key: str, model: str, system: str, user: str, json_mode: bool = True,
              status: Callable[[str], None] | None = None, max_attempts: int = 4,
              fallbacks: list[str] | None = None) -> tuple[str, str]:
    """Call Groq with back-off on 429. Tries the backup model if the main one stays unavailable.
    Returns (reply_text, model_used). Raises LLMUnavailable if nothing works."""
    from groq import Groq, RateLimitError, APIConnectionError, APITimeoutError, APIStatusError, NotFoundError

    client = Groq(api_key=api_key, max_retries=0, timeout=30)
    models = [model] + [m for m in (fallbacks or MODEL_FALLBACKS or [BACKUP_MODEL]) if m != model][:3]
    last_err = "unknown error"
    for m in models:
        delay = 2.0
        for attempt in range(max_attempts):
            try:
                kwargs = dict(model=m, temperature=0, max_tokens=1200,
                              messages=[{"role": "system", "content": system},
                                        {"role": "user", "content": user}])
                if "gpt-oss" in m:
                    kwargs["reasoning_effort"] = "low"      # keep hidden reasoning short
                elif "qwen3" in m:
                    kwargs["reasoning_format"] = "hidden"   # never return thinking text
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or "", m
            except RateLimitError as e:
                wait = _retry_after_seconds(e) or delay
                if wait > 60:            # daily quota exhausted: stop, don't hang the app
                    last_err = "daily rate limit reached"
                    break
                if status:
                    status(f"Rate limit reached, retrying in {wait:.0f}s...")
                time.sleep(wait)
                delay = min(delay * 2, 16)
                last_err = "rate limit"
            except NotFoundError as e:
                last_err = f"model {m} not available ({str(e)[:120]})"
                break
            except (APIConnectionError, APITimeoutError):
                last_err = "could not reach Groq"
                if attempt >= 1:
                    break
                time.sleep(delay)
            except APIStatusError as e:
                code = getattr(e, "status_code", "")
                if code == 401:
                    raise LLMUnavailable("Groq rejected the API key (check .streamlit/secrets.toml)")
                last_err = f"Groq error {code} on {m}: {str(e)[:140]}"
                break
    raise LLMUnavailable(last_err)


# ---------------------------------------------------------------- prompts
def build_user_message(player_summary: str, risk_label: str, risk_score: float, threshold: float,
                       rules: list[str], edge_note: str | None, clauses: dict[str, str], notes: str) -> str:
    passages = "\n".join(clauses.values()) if clauses else "(no protocol passages)"
    lines = [
        player_summary,
        f"Risk model: {risk_label} (score {risk_score:.2f}, cut-off {threshold:.2f}; model output for review).",
        f"Rules triggered by this week's data: {', '.join(rules) if rules else 'none'}.",
        f"Clause IDs you may cite: {', '.join(clauses.keys()) if clauses else 'none'}.",
    ]
    if edge_note:
        lines.append(f"Edge case: {edge_note}")
    if notes and notes.strip():
        lines.append(f"<staff_notes>{notes.strip()}</staff_notes>")
    lines += ["", "Protocol passages:", passages, "",
              "Write the briefing for coaches and the medical team. Include one action for every triggered clause. Reply with the JSON object only."]
    return "\n".join(lines)


def build_question_message(question: str, clauses: dict[str, str]) -> str:
    passages = "\n".join(clauses.values()) if clauses else "(no protocol passages)"
    return "\n".join([
        "A staff member asks a question about a player (coded ID D000, football).",
        f"<staff_notes>{question}</staff_notes>",
        f"Clause IDs you may cite: {', '.join(clauses.keys()) if clauses else 'none'}.",
        "", "Protocol passages:", passages, "",
        "Answer only from the passages, using the same JSON format. If the passages do not answer the question, set covered to false.",
    ])


GENERAL_SYSTEM = "You are a sports medicine assistant. Write a short injury-risk briefing for this player for the coach, under 120 words."


# ---------------------------------------------------------------- main entry
def get_briefing(api_key: str | None, model: str, live: bool, rules_path: str,
                 user_message: str, allowed_ids: list[str], required_ids: list[str],
                 fallback_text: Callable[[], str], status: Callable[[str], None] | None = None) -> dict:
    """Return a dict: source ('LLM' / 'Template (offline)' / 'Template (fallback)'),
    briefing (Briefing or None), text (template text when used), problems, model, attempts."""
    if not live or not api_key:
        return {"source": "Template (offline)", "briefing": None, "text": fallback_text(),
                "problems": [], "model": None, "attempts": 0}
    system = read_text(rules_path)
    msg = user_message
    problems: list[str] = []
    first_problems: list[str] = []
    model_used = model
    for attempt in (1, 2):
        try:
            raw, model_used = call_groq(api_key, model, system, msg, json_mode=True, status=status)
        except LLMUnavailable as e:
            return {"source": "Template (fallback)", "briefing": None, "text": fallback_text(),
                    "problems": [f"LLM unavailable: {e}"], "first_problems": first_problems,
                    "model": None, "attempts": attempt, "unavailable": True}
        b, problems = validate_briefing(raw, allowed_ids, required_ids)
        if attempt == 1:
            first_problems = list(problems)
        if b is not None:
            return {"source": "LLM", "briefing": b, "text": None, "problems": [],
                    "first_problems": first_problems, "model": model_used,
                    "attempts": attempt, "retried": attempt == 2}
        msg = (user_message + "\n\nYour previous reply was rejected for these reasons: "
               + "; ".join(problems) + ". Fix them and reply again with the JSON object only.")
    return {"source": "Template (fallback)", "briefing": None, "text": fallback_text(),
            "problems": problems, "first_problems": first_problems, "model": model_used, "attempts": 2}


def _all_strings(obj) -> str:
    """Every string value in a JSON reply, so a reply in a slightly different shape can still be judged."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return " ".join(_all_strings(v) for v in obj.values())
    if isinstance(obj, list):
        return " ".join(_all_strings(v) for v in obj)
    return ""


def ask_question(api_key: str, model: str, rules_path: str, question: str, clauses: dict[str, str],
                 status: Callable[[str], None] | None = None) -> dict:
    """Out-of-scope test: the LLM must say the protocol does not cover it (one retry if unreadable)."""
    system = read_text(rules_path)
    base = build_question_message(question, clauses) + (
        '\nIf the passages do not answer it, reply exactly: {"risk_summary": "' + NOT_COVERED
        + '", "actions": [], "edge_case_note": "", "covered": false, "final_line": "' + FINAL_LINE + '"}')
    msg = base
    raw, m, data, parsed = "", model, None, None
    for attempt in (1, 2):
        raw, m = call_groq(api_key, model, system, msg, status=status)
        try:
            data = json.loads(raw)
        except Exception:
            data = None
        try:
            parsed = Briefing.model_validate(data) if data is not None else None
        except ValidationError:
            parsed = None
        if parsed is not None or (data is not None and NOT_COVERED.lower()[:30] in _all_strings(data).lower()):
            break
        msg = base + "\n\nYour previous reply could not be read. Reply with the JSON object only, using exactly the keys shown."
    text = all_text(parsed) if parsed is not None else (_all_strings(data) if data is not None else (raw or ""))
    covered_flag = parsed.covered if parsed is not None else (data.get("covered") if isinstance(data, dict) else None)
    said_not_covered = NOT_COVERED.lower()[:30] in text.lower() and covered_flag is not True
    if parsed is not None:
        answer = parsed.risk_summary
    else:
        answer = ("(reply in a different format) " + (text or raw or "(empty reply)"))[:220]
    return {"passed": said_not_covered and not forbidden_hits(text), "answer": answer,
            "reply": parsed, "raw_ok": parsed is not None, "model": m}


def general_knowledge_briefing(api_key: str, model: str, player_summary: str, risk_label: str,
                               risk_score: float, status: Callable[[str], None] | None = None) -> str:
    """RAG comparison: the same player with NO protocol passages and no rules file."""
    user = f"{player_summary}\nRisk model: {risk_label} (score {risk_score:.2f})."
    raw, _ = call_groq(api_key, model, GENERAL_SYSTEM, user, json_mode=False, status=status)
    return raw


def render_briefing_text(b: Briefing) -> str:
    """Plain-text version for CSV export and the audit log."""
    parts = [b.risk_summary] + [f"[{a.clause_id}] {a.action}" for a in b.actions]
    if b.edge_case_note:
        parts.append(f"Edge case: {b.edge_case_note}")
    parts.append(b.final_line)
    return "\n".join(parts)
