import os
import streamlit as st

def load_protocol_clauses(path: str = "club_protocol.md") -> dict[str, str]:
    """Reads club_protocol.md fresh from disk on every invocation (no caching).
    Returns a dictionary mapping clause ID (e.g. 'P1') to full text."""
    if not os.path.exists(path):
        st.error(f"Missing protocol file: `{path}`.")
        st.stop()
    
    clauses = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("[P"):
                # Extract clause ID like P1, P2, etc.
                end_idx = line_str.find("]")
                if end_idx != -1:
                    cid = line_str[1:end_idx]
                    clauses[cid] = line_str
    return clauses

def check_player_rules(row: dict | object, is_high_risk: bool) -> tuple[list[str], str | None]:
    """Applies deterministic protocol rules and detects edge cases.
    Returns:
        (triggered_clause_ids, edge_case_note)
    """
    sport = str(row["sport"]).lower()
    acwr = float(row["acwr"])
    sleep_hours = float(row["sleep_hours"])
    soreness = float(row["soreness"])
    prior_injury = int(row["prior_injury"])

    rules = []

    # General rules
    if acwr > 1.5:
        rules.append("P1")
    elif 1.3 <= acwr <= 1.5:
        rules.append("P2")

    if sleep_hours < 6.0:
        rules.append("P3")

    if soreness >= 7.0:
        rules.append("P4")

    if prior_injury == 1:
        rules.append("P6")

    # Sport-specific rules
    if sport == "football" and acwr >= 1.2:
        rules.append("P10")

    if sport == "cricket" and acwr > 1.3:
        rules.append("P11")

    if sport == "basketball" and soreness >= 6.0:
        rules.append("P12")

    if sport == "kabaddi" and prior_injury == 1:
        rules.append("P13")

    # Edge cases
    edge_case = None

    # Edge Case 1: Conflicting rules (Precedence P14)
    # football with 1.2 <= acwr < 1.3 (P10 applies, P2 does not)
    # basketball with 6.0 <= soreness < 7.0 (P12 applies, P4 does not)
    if sport == "football" and 1.2 <= acwr < 1.3:
        edge_case = "Conflicting rules: Football threshold [P10] (acwr ≥ 1.2) triggers caution zone before general rule [P2] (1.3–1.5). By precedence [P14], sport-specific rule applies."
        if "P14" not in rules:
            rules.append("P14")
    elif sport == "basketball" and 6.0 <= soreness < 7.0:
        edge_case = "Conflicting rules: Basketball soreness threshold [P12] (soreness ≥ 6) triggers assessment before general rule [P4] (soreness ≥ 7). By precedence [P14], sport-specific rule applies."
        if "P14" not in rules:
            rules.append("P14")
    # Edge Case 2: High score, no rule (Manual review P7)
    # Model says High but none of P1, P2, P3, P4, P10, P11, P12 fired
    elif is_high_risk and not any(r in rules for r in ["P1", "P2", "P3", "P4", "P10", "P11", "P12"]):
        edge_case = "High score, no rule: Predictive model classifies player as High risk, but no primary workload, sleep, or soreness rules fired. Flagged under [P7] for manual medical review."
        if "P7" not in rules:
            rules.append("P7")

    return rules, edge_case

