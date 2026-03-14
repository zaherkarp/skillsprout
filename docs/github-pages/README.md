# SkillSprout Lite (GitHub Pages Frontend)

This folder contains a fully static frontend that can be published directly to **GitHub Pages** without running FastAPI, PostgreSQL, or Redis.

## Why this exists

- Enables zero-infrastructure demos.
- Gives a visual walkthrough of SkillSprout bucket logic.
- Supports persona QA for people transitioning from fashion into new fields.

## Design summary

- **Static-only architecture:** plain HTML + CSS + vanilla JavaScript (no external CDN dependencies).
- **Built-in mock dataset:** sample roles, skill baselines, and transition targets are embedded in `app.js`.
- **Persona QA flow:** `Nia (fashion -> new field)` tests three fields (`Project Coordination`, `Operations`, `Data Analysis`) in one run.
- **Deterministic recommendation rules:**
  - Ready Now: `match >= 75` and `gap <= 25`
  - Trainable: `match >= 50` OR `gap in [26, 55]`
  - Long Reskill: all others

## Local test run

```bash
cd docs/github-pages
python -m http.server 4173
# open http://localhost:4173
```

## Publish to GitHub Pages

1. Push this repository to GitHub.
2. In repository settings, enable **Pages**.
3. Set source to your branch and `/docs/github-pages` folder.
4. Save and wait for deployment.

The site will then be hosted on `https://<org-or-user>.github.io/<repo>/`.

## Career Transition Report Generator

`report.html` is an in-browser career transition report generator powered by the Claude API.

- **Structured data** (skills profile, BLS labor market table, recommendation cards) renders immediately without any API call.
- **Narrative sections** (executive summary, labor market context, next steps) require a network connection and an Anthropic API key to call the Claude API at runtime.
- If the Claude API is unavailable or no API key is provided, the report still renders all structured data sections with a warning banner.
