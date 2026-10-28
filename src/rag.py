import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.rules import load_protocol_clauses

SPORT_CLAUSE_MAP = {
    "football": "P10",
    "cricket": "P11",
    "basketball": "P12",
    "kabaddi": "P13"
}

def retrieve_clauses(
    sport: str,
    acwr: float,
    sleep_hours: float,
    soreness: float,
    prior_injury: int,
    triggered_rule_ids: list[str],
    protocol_path: str = "club_protocol.md"
) -> dict[str, str]:
    """Performs TF-IDF retrieval on club_protocol.md and adds deterministic mandatory clauses.
    
    Returns:
        Ordered dictionary mapping clause ID (e.g. 'P1') to the clause text.
    """
    # 1. Load protocol fresh from disk (never cached)
    all_clauses = load_protocol_clauses(protocol_path)
    if not all_clauses:
        return {}

    clause_ids = list(all_clauses.keys())
    clause_texts = [all_clauses[cid] for cid in clause_ids]

    # 2. Build Query
    query = f"{sport.lower()} workload ratio {acwr} sleep {sleep_hours} soreness {soreness} prior injury {prior_injury}"

    # 3. TF-IDF retrieval
    retrieved_set: dict[str, str] = {}
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(clause_texts)
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # Top 3 clauses with score >= 0.05
        scored_pairs = sorted(zip(clause_ids, scores), key=lambda x: x[1], reverse=True)
        top_pairs = [pair for pair in scored_pairs if pair[1] >= 0.05][:3]
        for cid, _ in top_pairs:
            retrieved_set[cid] = all_clauses[cid]
    except Exception:
        # Fallback if vocabulary is empty or single word issue
        pass

    # 4. Always add mandatory clauses:
    # - Sport-specific clause
    sport_key = sport.lower()
    if sport_key in SPORT_CLAUSE_MAP:
        sp_cid = SPORT_CLAUSE_MAP[sport_key]
        if sp_cid in all_clauses:
            retrieved_set[sp_cid] = all_clauses[sp_cid]

    # - P8: Clearance rule
    if "P8" in all_clauses:
        retrieved_set["P8"] = all_clauses["P8"]

    # - Every clause triggered by the rule check
    for r_cid in triggered_rule_ids:
        if r_cid in all_clauses:
            retrieved_set[r_cid] = all_clauses[r_cid]

    # Sort clauses numerically by ID (e.g. P1, P2, ... P14)
    def parse_id(cid: str) -> int:
        digits = re.findall(r"\d+", cid)
        return int(digits[0]) if digits else 999

    sorted_keys = sorted(retrieved_set.keys(), key=parse_id)
    return {k: retrieved_set[k] for k in sorted_keys}



def retrieve_for_question(question: str, k: int = 3, protocol_path: str = "club_protocol.md") -> dict[str, str]:
    """Top-k clauses for a free-text question (no minimum score, so the LLM also sees
    loosely related passages and must decide whether they really answer the question)."""
    all_clauses = load_protocol_clauses(protocol_path)
    ids = list(all_clauses.keys())
    if not ids:
        return {}
    vec = TfidfVectorizer(stop_words="english")
    m = vec.fit_transform([all_clauses[c] for c in ids])
    scores = cosine_similarity(vec.transform([question]), m).flatten()
    top = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)[:k]
    return {cid: all_clauses[cid] for cid, _ in top}
