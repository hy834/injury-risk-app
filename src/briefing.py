from src.rules import load_protocol_clauses

def generate_template_briefing(
    risk_label: str,
    risk_score: float,
    threshold: float,
    triggered_clauses: list[str],
    edge_case_note: str | None,
    protocol_path: str = "club_protocol.md"
) -> str:
    """Generates deterministic template briefing according to Section 5 requirements.
    
    Format:
    - one line with risk label and score
    - one bullet per triggered clause quoting its text with its ID
    - the edge-case note if any
    - final line: 'Final decision: medical team.'
    - If no rule fired: 'No protocol rule is triggered by this week's data. [P8] Only the medical team may clear or bench a player.'
    """
    all_clauses = load_protocol_clauses(protocol_path)
    
    lines = [f"Risk assessment: {risk_label} (predicted probability: {risk_score:.3f}, cut-off: {threshold:.2f})."]

    if triggered_clauses:
        for cid in triggered_clauses:
            clause_text = all_clauses.get(cid, f"[{cid}] Protocol clause {cid}")
            lines.append(f"• {clause_text}")
    else:
        lines.append("No protocol rule is triggered by this week's data. [P8] Only the medical team may clear or bench a player.")

    if edge_case_note:
        lines.append(f"Edge case: {edge_case_note}")

    lines.append("Final decision: medical team.")

    return "\n".join(lines)
