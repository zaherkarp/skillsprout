# SkillSprout - GitHub Pages Frontend

A fully static frontend for [SkillSprout](../../README.md) that runs on GitHub Pages without any backend infrastructure.

## What's Included

| Page | Purpose |
|------|---------|
| `index.html` | Landing page with interactive skill-matching demo |
| `report.html` | Career transition report generator (Claude AI-powered narratives) |
| `app.js` | Scoring engine + embedded mock dataset |
| `styles.css` | Responsive design system |

## Features

- **Interactive skill matcher** - adjust skill sliders and see recommendation buckets update in real time
- **Career report generator** - generates structured reports with O*NET skill profiles, BLS labor market projections, and training paths
- **Claude AI narratives** - optional AI-generated executive summaries and next-steps (requires Anthropic API key)
- **Three demo personas** - Maria (nurse), Aisha (bootcamp developer), James (retail supervisor)
- **Fully client-side** - no API keys, no server, no database required for the core demo
- **Print-ready reports** - reports render cleanly for PDF export via browser print

## Scoring Rules

- **Ready Now**: match >= 75% and gap <= 25%
- **Trainable**: match >= 50% or gap in [26%, 55%]
- **Long Reskill**: everything else

## Local Development

```bash
cd docs/github-pages
python -m http.server 4173
# open http://localhost:4173
```

## Deploy to GitHub Pages

1. Push the repository to GitHub
2. Go to Settings > Pages
3. Set source to your branch and `/docs/github-pages` folder
4. Save - the site deploys to `https://<user>.github.io/skillsprout/`

## LinkedIn Sharing

The pages include Open Graph meta tags for rich previews when shared on LinkedIn and other platforms. The OG image (`og-image.svg`) shows project stats and branding.
