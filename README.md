# Player Health Hub (Project 4: Player Injury Risk System)

Educational prototype. Synthetic data only. Not medical advice.

## Status
All four build phases are done: predictive model with baselines, RAG over the club protocol, Groq LLM briefing with validation and fallback, guardrails, medical sign-off and audit log, AI safety testing page, About & risks page.

## Run it on your computer
1. Put your Groq key in `.streamlit/secrets.toml` (copy `secrets.toml.example`):
   `GROQ_API_KEY = "gsk_..."`
2. In a terminal, inside this folder:
   - Mac: `python3 -m pip install -r requirements.txt` then `python3 -m streamlit run app.py`
   - Windows: `python -m pip install -r requirements.txt` then `python -m streamlit run app.py`
3. Open http://localhost:8501. Without a key the app still works in Offline mode (template briefings).

## Put it online (Streamlit Community Cloud)
1. Publish this folder to a **public** GitHub repo (GitHub Desktop → Add local repository → Publish, untick "Keep this code private"). Check that `.streamlit/secrets.toml` is NOT in the repo.
2. share.streamlit.io → Create app → "Yup, I have an app" → repo, branch `main`, main file `app.py`.
3. Advanced settings → Python 3.12 → Secrets: `GROQ_API_KEY = "gsk_..."` → Save → Deploy.
4. Open the link 10 minutes before the demo (free apps sleep).

## Demo script (7 minutes)
1. **Squad list**: this week's squad, availability by sport.
2. **Player chart** quick buttons: 🟢 Low risk (D000), 🔴 High risk (D010), ⚖️ Rule conflict (D060), 🔍 High score, no rule (D061), 🚨 Injection test (D006). Show "Source: LLM", the cited clauses and the sign-off being blocked without a name.
3. **Sign-offs**: the decision appears in the audit log.
4. **Model evaluation**: model vs baselines, cost of errors, fairness.
5. **AI safety testing**: run the tests (best done once before presenting; results stay for the session).

---

# Original build and deploy guide


Educational prototype. Synthetic data only. Not medical advice.

## What is in this folder

| File | What it is |
|---|---|
| `BUILD_PROMPT.md` | The full specification for the coding agent (Antigravity). It builds the app from this. |
| `data/` | Training, test and demo datasets + data dictionary (same files as the progress report). |
| `club_protocol.md` | The fictional club protocol, clauses P1 to P14, used by the RAG. |
| `briefing_rules.md` | The LLM's rules and output format (its system prompt). |
| `tests/injection_tests.csv` | 20 injection attempts, 10 normal notes, 5 out-of-scope questions for testing. |
| `business_case.md` | Buy / rent / build comparison (placeholder figures to check). |
| `PROMPT_LOG.md` | Template for the prompt log you must submit. |
| `requirements.txt` | Python packages Streamlit Cloud installs. |
| `.streamlit/` | Theme, plus `secrets.toml.example` showing where the API key goes. |
| `.gitignore` | Stops your API key from being uploaded to GitHub. |

## Localhost or Streamlit?

Use **Streamlit Community Cloud**. "Localhost" means the app runs only on your laptop, and nobody else can open it. The brief needs a public link. Streamlit Community Cloud is free, runs straight from a GitHub repo, keeps the API key secret, and gives a link like `https://your-app.streamlit.app`. You still run it on localhost while building, to test.

## Step 0: Accounts (one-time, 15 minutes)
1. **Google account** for Antigravity.
2. **GitHub account** (github.com) to store the code.
3. **Groq API key** (free): console.groq.com → sign in → API Keys → Create API Key. Copy it (starts with `gsk_`). Never paste it into slides, chat or GitHub.
4. **Streamlit Community Cloud**: share.streamlit.io → sign in with GitHub.
5. **Python 3.12** from python.org (Windows: tick "Add Python to PATH" during install).

## Step 1: Set up Antigravity
1. Download and install Google Antigravity, sign in with Google.
2. When asked for a development mode, choose **Agent-assisted development** (recommended: you stay in control).
3. Pick the strongest model offered (for example Gemini 3 Pro or a Claude model).
4. Unzip this kit. In Antigravity: **File → Open Folder** → choose `injury-risk-app`.
5. Add your key: copy `.streamlit/secrets.toml.example` to a new file `.streamlit/secrets.toml` and paste your Groq key in it. This file stays on your laptop only.

## Step 2: Vibe code it, one phase at a time
Open the **Agent Manager**, click **New Task**, choose **Planning** mode (it shows a plan before changing anything), and paste the first prompt. Read the plan, approve it, and let it work. When it finishes, it will start the app on `http://localhost:8501`; open that in your browser and check it.

**Phase 1 prompt**
> Read BUILD_PROMPT.md fully. Then do Phase 1 only: project structure, data loading and checks, the model and baselines, and the Model Performance and Squad Overview tabs. Set up a virtual environment, install requirements.txt, and run the app with `streamlit run app.py`. Confirm the test-1 numbers match the table in section 4 exactly. Then stop and tell me how to test it.

**Phase 2 prompt**
> Phase 1 works. Now do Phase 2 from BUILD_PROMPT.md: RAG retrieval, the rule check, the template briefing, the Player Assessment tab in Offline mode, the medical sign-off and the Audit Log. Stop and tell me how to test it.

**Phase 3 prompt**
> Phase 2 works. Now do Phase 3 from BUILD_PROMPT.md: the Groq LLM in JSON mode with Pydantic validation, retry and fallback, all guardrails in section 6, and the LLM & Guardrail Testing tab. The key is in .streamlit/secrets.toml under GROQ_API_KEY. Stop and tell me how to test it.

**Phase 4 prompt**
> Phase 3 works. Now do Phase 4: the About & Risks tab, styling, README, and a final check of every item in section 10 of BUILD_PROMPT.md. List anything that does not pass.

**When something breaks**, paste the exact error and say what you clicked:
> When I click "Assess player" on D060 I get this error: [paste]. Fix it without changing anything else, and tell me what was wrong.

**Fill in `PROMPT_LOG.md` as you go**: what you asked, what went wrong, what you changed. It is a required deliverable.

### What to check after each phase
- Phase 1: the test-1 table shows F1 0.53 for "with sport" and 0.48 for "no sport". The squad table shows 63 players.
- Phase 2: D000 shows Low, D010 shows High, D060 and D062 show a "conflicting rules" banner, D061 shows "high score, no rule". Recording a decision without a staff name is blocked.
- Phase 3: with the key, briefings say "Source: LLM" and cite clause IDs; the injection example shows "Injection blocked"; turning the key off still works (template).
- Phase 4: every item in section 10 passes.

## Step 3: Put the code on GitHub
Easiest without coding: **GitHub Desktop** (desktop.github.com).
1. On github.com click **New repository**, name it `injury-risk-app`, set it to **Public**, create it.
2. In GitHub Desktop: **File → Add local repository** → choose the `injury-risk-app` folder (create a repository there if asked) → **Publish repository** (untick "Keep this code private").
3. **Check on github.com that `.streamlit/secrets.toml` is NOT in the repo.** If it is, delete it from GitHub and create a new Groq key (the old one is exposed).

Or ask the agent: "Initialise git, commit everything except secrets, and push to https://github.com/<you>/injury-risk-app." You will have to sign in to GitHub when asked.

## Step 4: Deploy on Streamlit Community Cloud (the public link)
1. Go to share.streamlit.io → **Create app** → **Yup, I have an app**.
2. Repository: `<you>/injury-risk-app`, Branch: `main`, Main file path: `app.py`.
3. App URL: pick a name, e.g. `riverside-injury-risk` → link becomes `https://riverside-injury-risk.streamlit.app`.
4. **Advanced settings** → Python version **3.12** → **Secrets**: paste
   `GROQ_API_KEY = "gsk_your_key"` → Save.
5. **Deploy**. It takes a few minutes. If it fails, open the logs (bottom right), copy the error and give it to the agent.
6. Every time you push a change to GitHub, the app updates itself.
7. Free apps go to sleep when unused. **Open the link 10 minutes before the demo** and click "Yes, get this app back up" if you see it.
8. Make a QR code of the link (any free QR generator) for the Demo slide.

## How the app covers the presentation template

| Slide | What to show | Where it comes from |
|---|---|---|
| 1 Title | Project title, team D1, section | — |
| 2 Team | Name, role, what each built | Roles in the progress report |
| 3 Problem | Injuries cost availability and wages; objective in one line | About & Risks tab |
| 4 Dataset | 2,000 rows train / 500 test / 63 demo; 10 columns; target `high_risk`; synthetic, season weeks 1–52 | `data/`, `data_dictionary.csv` |
| 5 Model | Baselines vs models table, why logistic regression, cut-off 0.35, confusion matrix, error costs | Model Performance tab |
| 6 GenAI | Feature 1: protocol RAG briefing; Feature 2: guardrails (injection, output check); Feature 3: human sign-off + audit log. Tech stack: Groq Llama 3.3 70B, Streamlit, scikit-learn, Streamlit Cloud | Player Assessment, LLM & Guardrail Testing tabs |
| 7 Demo | Public link + QR; low-risk, high-risk, edge case, injection attempt | Player Assessment quick buttons |
| 8 Insights | e.g. sport matters (F1 0.48 → 0.53); RAG beats the LLM alone (club-specific rules, citations); one cut-off may not suit every sport (cricket); a missed injury costs more than a false alarm; why a club would pay, B2B revenue, compliance; buy/rent/build recommendation; limitations | Model Performance, LLM & Guardrail Testing, About & Risks |

**Final deliverables also asked for:** precision/recall and the cost of each error, how the LLM was tested and how often it failed (with examples), where the human checkpoint is and why, risks (bias, privacy, misuse), the 3-year buy/rent/build cost, and the prompt log. All of these are in the app's tabs or in `business_case.md` / `PROMPT_LOG.md`.

**Peer challenge round:** other teams will try to break your app. The usual attacks are injection notes, extreme or missing values, conflicting rules, and asking the LLM about things the protocol doesn't cover. The LLM & Guardrail Testing tab is your evidence that you tested these.
