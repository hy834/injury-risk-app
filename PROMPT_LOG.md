# Prompt Log (submit as an appendix)

Record every important prompt you give the coding assistant (Antigravity) and the key prompts inside the app. For each one, note what went wrong and what you changed.

## A. Prompts to the coding assistant

| # | Date | Who | Prompt (short) | What happened | What went wrong | What we changed |
|---|---|---|---|---|---|---|
| 1 | 28 Oct | Team (Antigravity, Gemini) | "Read BUILD_PROMPT.md. Do Phase 1 only." | Built data loading, both models, baselines, Model Performance and Squad tabs | Agent waited on a terminal approval ("Allow probe Streamlit server?") and the `streamlit run` command never "finishes" | Approved the probe; told the agent the app was already running |
| 2 | 28 Oct | Team (Antigravity) | Phase 2: RAG, rule check, template briefing, Player Assessment, sign-off, Audit Log, built as a clinical patient chart | Patient chart, vital-sign tiles, gauge, sign-off form | Vital-sign "normal ranges" were backwards or invented (e.g. sleep "< 6 h" shown as normal, HRV 40–54 ms made up); "Immutable record" claim was untrue; demo set made the squad look 62% High risk | Prompted fixes: ranges taken only from club_protocol.md; HRV grey (no fixed threshold, P5); wording fixed; squad view defaults to the 500-player test set |
| 3 | 28–29 Oct | Team (Antigravity) | Redesign as clinical software (font, header, sidebar menu, cards, plain-English labels) | Looked like a clinical tool instead of a dashboard | Our CSS forced the font onto icon spans, so icon names showed as text ("_arrow_right"); raw HTML cards left empty white bars; "Cleared Correctly" label in the confusion matrix; wrong claim that sport helps cricket; sign-off defaulted to "Cleared for full training" | Restored the icon font, used st.container cards, renamed to "Correctly rated Low", corrected the cricket text, safe default "Refer for further assessment" |
| 4 | 29 Oct | Team | Phase 3 prompt | — | Antigravity "Individual quota reached" on Gemini and Claude models | Moved the project to another laptop and continued with Claude (chat) |
| 5 | 29 Oct | Claude (chat) | Finish Phase 3 and 4 from BUILD_PROMPT.md | Groq LLM briefing (JSON mode, Pydantic validation, one retry, template fallback, 429 back-off, backup model, session cache); AI safety testing page; About & risks page; cost chart, table and button fixes | Tested with a fake Groq reply because the build machine could not reach Groq; found the injection filter caught 20/20 known attempts but 0/10 new ones | Kept the 10 new attempts as an honest held-out test and added a full-pipeline test showing the output check still blocks unsafe briefings |
| 6 | 29 Oct | Claude (chat) + team test on Mac | First live run with the Groq key | Every briefing fell back to the template | Groq had retired the Llama 3.3 / 3.1 models ("model not available") | App now asks Groq which models the key can use and picks the best (openai/gpt-oss-120b); falls back down a preference list |
| 7 | 29 Oct | Claude (chat) | Live briefing for D010 | Correct clauses (P10, P1, P3) and actions | LLM invented an "edge case" (claimed P10 overrides P1) that the rule check had not found: a small hallucination | Rules file now says edge_case_note must be empty unless the app reports one; the app removes any invented edge case and counts it in the safety tests |

## B. Prompts inside the app (the LLM)

| Version | Date | Change to briefing_rules.md | Why | Effect on test results |
|---|---|---|---|---|
| v1 | 28 Oct | First version | | |
| v2 | 29 Oct | Actions must cover every triggered clause; up to 6 actions; write actions in own words; do not cite P8 unless no rule fired | Matches what the Python output check enforces, so fewer retries | Re-run the briefing test with the live key and record here |

## C. Test results over time

| Date | Injection attempts caught | Normal notes wrongly flagged | Briefings blocked by output check | Citation failures | Out-of-scope answered correctly |
|---|---|---|---|---|---|
| | /20 | /10 | /63 | /63 | /5 |

## D. Failure examples (for the presentation)

| Player / test ID | What the LLM did wrong | How the guardrail handled it |
|---|---|---|
| | | |
