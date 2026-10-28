# Business Case

## Why a club would pay
- **Injuries are expensive.** An injured player is still paid but cannot play, often for weeks. Results suffer, and so do prize money, sponsorship and ticket income.
- **Squad availability wins seasons.** The more of the squad fit each week, the more choice the coach has. The app shows availability as the share of players at Low risk.
- **Consistency.** Today staff check load, sleep and soreness by hand and not always the same way. The system checks every player every week against the club's own protocol.
- **Accountability.** Every flag and decision is recorded with the staff member and clause used, which protects the club and the medical team.

## B2B revenue potential (if we sold this to clubs)
**Placeholder assumptions for the team to check.**

| Tier | Customer | Price per year (assumption) |
|---|---|---|
| Academy | Academies, college and state teams | ₹2–3 lakh |
| Pro club | One professional squad | ₹10–12 lakh |
| Multi-sport | A club or franchise group with several squads | ₹20–25 lakh |

Example: by year 3, 10 pro clubs + 30 academies = 10 × ₹12 lakh + 30 × ₹3 lakh = **about ₹2.1 crore a year**. Customers: top-tier leagues in cricket, football, kabaddi and basketball, plus academies. Each club adds its own protocol, so the product scales without rebuilding the model.

## Compliance
- **India's Digital Personal Data Protection Act, 2023:** players must consent; data is used only for injury prevention; players can ask for their data to be deleted.
- Coded player IDs only; real data stays on club-controlled hosting, not a public prototype.
- The medical team makes every decision, so the system is decision support, not a diagnosis.

# Buy, Rent or Build (3-year view)

**All figures are planning assumptions for a single multi-sport club, in Indian rupees. They are placeholders for the team to check and replace with real quotes before the final presentation.**

| Option | What it means | Year 1 | Years 2–3 (per year) | 3-year total (approx.) | Main trade-off |
|---|---|---|---|---|---|
| Buy | One-time licence of an existing athlete-monitoring product, customised to our protocol | ₹20–30 lakh (licence + set-up) | ₹4–6 lakh (support, about 20% of licence) | ₹28–42 lakh | Fast to start, but the club protocol and briefings must fit the vendor's product |
| Rent | Monthly or yearly software subscription (SaaS) | ₹10–15 lakh (subscription + integration) | ₹10–12 lakh | ₹30–39 lakh | Lowest risk and no maintenance, but the club's health data sits with a vendor |
| Build | Our own app: model + protocol RAG + briefing, as in this prototype | ₹25–35 lakh (2 developers + part-time data scientist for about 6 months) | ₹8–10 lakh (maintenance) + ₹1–2 lakh (hosting and LLM) | ₹43–59 lakh | Full control of data and protocol, but highest cost and needs in-house skills |

**Running costs we know are small:** the LLM writes about 100 short briefings a week; on a paid API this is a few hundred rupees a month. The model trains in seconds.

**How to judge the value (fill in with the club's numbers):**
Value per year = injuries prevented × average weeks out per injury × weekly cost of a player who cannot play (wages plus the effect on results).
Example: if the system helps prevent 3 injuries a year, each costing 6 weeks out at ₹2 lakh a week, that is ₹36 lakh a year, which would cover any option within about a year.

**Our current leaning (team to confirm):** **Rent** a monitoring platform to collect the data, and **Build** only the thin protocol-briefing layer on top, because the club protocol is specific to us and the health data should stay under the club's control.
