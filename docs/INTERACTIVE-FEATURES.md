# SkillSprout Interactive Features Guide

How the demo at [zaherkarp.github.io/skillsprout](https://zaherkarp.github.io/skillsprout) works, and what makes it interactive.

---

## The User Journey

```
                                    +---------------------------+
                                    |     zaherkarp.github.io   |
                                    |       /skillsprout        |
                                    +------------+--------------+
                                                 |
                                                 v
                        +------------------------------------------------+
                        |              STEP PROGRESS BAR                  |
                        |   (1)-----(2)-----(3)-----(4)                  |
                        |  Choose   Rate    View    Take                 |
                        |  Persona  Skills  Matches Action               |
                        +------------------------------------------------+
                                                 |
                    +----------------------------+----------------------------+
                    |                            |                            |
                    v                            v                            v
          +-----------------+         +------------------+         +------------------+
          |  Persona Picker |         |  Skill Sliders   |         |  Career Matches  |
          |                 |         |                   |         |                  |
          | +-------------+ |         |  Communication    |         | +- Ready Now --+ |
          | | Nia     [*] | |         |  ====O========    |         | | Project Coord | |
          | +-------------+ |         |        ^          |         | | 87% [ring]    | |
          | | Marcus  [ ] | |   gap --+--------+          |         | +---------------+ |
          | +-------------+ |  marker |  Programming      |         | +- Trainable --+ |
          | | Priya   [ ] | |         |  ==O============  |         | | Data Analyst  | |
          | +-------------+ |         |     ^             |         | | 62% [ring]    | |
          +-----------------+         +------------------+         | +---------------+ |
                    |                         |                    | +- Long Reskill + |
                    |    sets baseline        |    live-updates    | | Software Dev   | |
                    +--------+----------------+--------+-----------+ | 41% [ring]    | |
                             |                         |           +------------------+
                             v                         v
                    +-----------------+      +------------------+
                    |   Radar Chart   |      |   Quick Wins     |
                    |                 |      |   Advisor        |
                    |    Comm         |      |                  |
                    |   / | \        |      | 1. Programming   |
                    |  Ops  Cust     |      |    +1 unlocks 2  |
                    |  |     |       |      | 2. Data Analysis |
                    |  Prog--Data    |      |    +8% total     |
                    |   \ | /        |      | 3. Operations    |
                    |   Crit         |      |    +4% total     |
                    +-----------------+      +------------------+
```

---

## Feature Breakdown

### 1. Sticky Step Progress Bar

A 4-step progress indicator sticks to the top of the viewport as users scroll, providing constant orientation.

```
  [ (1) Choose Persona ]---[ (2) Rate Skills ]---[ (3) View Matches ]---[ (4) Take Action ]
        completed               active                 upcoming              upcoming
        (filled)             (highlighted)             (dimmed)              (dimmed)
```

Steps auto-advance as the user interacts:
- **Step 1**: Activates when a persona or role is selected
- **Step 2**: Activates when any skill slider is adjusted
- **Step 3**: Activates when results render (automatically after step 2)
- **Step 4**: Activates when the user bookmarks a career to their shortlist

---

### 2. Interactive Radar Chart

An SVG spider/radar chart renders your 6 skill dimensions as a filled polygon. When you click "Compare on Chart" on any career card, the target career's requirements overlay as a second polygon.

```
                  Communication
                      /\
                     /  \
                    / YOU \
           Ops ---/  . . . \--- Customer
                  \ TARGET /     Service
                   \  . . /
                    \    /
           Crit ----\  /---- Data
            Think    \/     Analysis
                Programming
```

- **Teal fill** = your current skill profile
- **Dashed orange overlay** = target career requirements
- Skill gap is visible as the space between the two shapes

---

### 3. Expandable Result Cards with Progress Rings

Each career match is a clickable card that expands to show a detailed gap analysis.

```
  +------------------------------------------------------+
  |  [87% ring]  Project Coordinator                 [*]  |
  |              Match: 87%  Gap: 11%  Project Coord      |
  |              Gaps: Critical Thinking                   |
  |                                                        |
  |  v  (click to expand)                                  |
  |------------------------------------------------------  |
  |  Skill Breakdown:                                      |
  |  Communication   [=============|====] 4 -> 4          |
  |  Customer Svc    [=============|====] 4 -> 3          |
  |  Critical Think  [=========|........] 3 -> 3  GAP!    |
  |  Programming     [===|..............]  1 -> 0         |
  |  Data Analysis   [======|...........] 2 -> 2          |
  |  Operations      [=============|====] 3 -> 3          |
  |                                                        |
  |  Training Tip: PMP or CAPM certification is the       |
  |  standard entry point. Coursera and LinkedIn Learning  |
  |  have affordable prep courses.                         |
  |                                                        |
  |  [Compare on Chart]                                    |
  +------------------------------------------------------+
```

- **Progress ring**: Circular SVG showing match % with color-coded stroke
- **Bookmark star**: Click to save to personal shortlist
- **Skill bars**: Green for met, red for gaps, with exact level numbers
- **Training tip**: Actionable next step for each career path

---

### 4. Skill Gap Target Markers

When a career card is selected for comparison, the skill sliders gain visual markers showing where the target career needs you to be.

```
  Programming
  ====O========================
       ^           ^
       |           |
    You: 1      Target: 4 (red arrow = gap)

  Communication
  =============================O
                               ^
                               |
                          You: 4, Target: 4 (green check)
```

---

### 5. Quick Wins Advisor

Answers the question: "If I could only improve ONE skill, which should it be?"

```
  +--------------------------------------------------+
  |  Quick Wins                                       |
  |                                                    |
  |  #1  Programming  (Basic -> Intermediate)          |
  |      Unlocks 2 tier upgrades, +12% total match    |
  |                                                    |
  |  #2  Data Analysis  (Intermediate -> Advanced)     |
  |      Unlocks 1 tier upgrade, +8% total match       |
  |                                                    |
  |  #3  Critical Thinking  (Advanced -> Expert)       |
  |      +4% total match improvement                   |
  +--------------------------------------------------+
```

For each skill below max (4), the advisor simulates a +1 boost and measures:
- How many careers change tier (e.g., Trainable -> Ready Now)
- Total match % gain across all careers

---

### 6. Career Bookmarks & Shortlist

Users can star any career to build a personal shortlist. Data persists in `localStorage`.

```
  +--------------------------------------------------+
  |  My Shortlist                          [Saved]    |
  |                                                    |
  |  [87% ring]  Project Coordinator               X  |
  |              Project Coordination - Ready Now      |
  |                                                    |
  |  [62% ring]  Operations Specialist             X  |
  |              Operations - Trainable                |
  |                                                    |
  |  [Clear Shortlist]                                 |
  +--------------------------------------------------+
```

---

### 7. Confetti Celebration

When a slider adjustment causes a career to newly enter the **Ready Now** tier, a burst of confetti particles animates from the results section. This provides satisfying feedback for skill improvements.

```
         *  .  *
       .    *    .
      *  READY!   *     <-- confetti burst
       .    *    .
         *  .  *
  +-------------------+
  | Ready Now         |
  | + Project Coord!  |  <-- newly promoted
  +-------------------+
```

---

### 8. Shareable URL Profiles

Skill profiles are encoded into the URL hash so users can share their exact configuration:

```
https://zaherkarp.github.io/skillsprout/github-pages/#communication=4&customer_service=4&critical_thinking=3&programming=2&data_analysis=2&operations=3&persona=Nia+%28fashion+-%3E+new+field%29
```

The "Share Profile" button copies this URL to the clipboard. Anyone opening the link sees the exact same skill configuration and results.

---

## Data Flow

```
  +-----------+     +-----------+     +------------+     +-------------+
  |  Persona  | --> |  Skill    | --> |  Scoring   | --> |  Render     |
  |  Selector |     |  Sliders  |     |  Engine    |     |  Pipeline   |
  +-----------+     +-----------+     +------------+     +-------------+
       |                 |                  |                   |
       | sets baseline   | user adjusts    | scoreTarget()     | updates:
       | profile         | values          | per career        | - bucket cards
       |                 |                  |                   | - radar chart
       v                 v                  v                   | - quick wins
  catalog.personas  readSkills()     { bucket, match,          | - QA table
                                       gap, gapSkills }        | - shortlist
                                                               | - step progress
                                                               | - URL hash
                                                               v
                                                          DOM + localStorage
```

### Scoring Formula

For each target career:

```
  match_earned  = SUM( min(user_skill, required_skill) )  for each skill
  match_possible = SUM( required_skill )                   for each skill
  match_pct     = match_earned / match_possible * 100

  gap_total     = SUM( max(0, required - actual) )         for each skill
  gap_possible  = SUM( required_skill )                    for each skill
  gap_pct       = gap_total / gap_possible * 100
```

### Bucket Assignment

```
  match >= 75% AND gap <= 25%  -->  READY NOW      (green)
  match >= 50% OR gap 26-55%   -->  TRAINABLE      (amber)
  everything else              -->  LONG RESKILL   (red)
```

---

## Tech Stack (Demo Site)

| Layer | Technology | Notes |
|-------|-----------|-------|
| Markup | HTML5 | Semantic, accessible |
| Styling | CSS3 (custom properties) | Teal + amber palette, responsive |
| Logic | Vanilla JS | Zero dependencies, ~700 lines |
| Charts | Inline SVG | Radar chart + progress rings |
| Persistence | localStorage | Shortlist bookmarks |
| State Sharing | URL hash params | Shareable skill profiles |
| Hosting | GitHub Pages | Auto-deployed via Actions |
| Fonts | Inter (Google Fonts) | Clean, modern typography |

No frameworks. No build step. No npm. Just HTML, CSS, and JS.

---

## Local Development

```bash
cd docs/github-pages
python -m http.server 4173
# open http://localhost:4173
```

---

*Built with care for people exploring their next career move.*
