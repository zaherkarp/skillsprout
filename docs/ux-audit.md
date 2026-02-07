# UX & Accessibility Audit: SkillSprout

**Audit Date:** 2026-02-07
**Scope:** Full user-facing surface -- Jinja2 templates (`base.html`, `index.html`, `flow.html`, `docs.html`), inline JavaScript, `style.css`, API schemas, and scoring engine output.
**Methodology:** Manual code-level review against WCAG 2.1 AA, Nielsen heuristics, and real-world persona modeling.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [User Journey Map](#2-user-journey-map)
3. [Accessibility Gaps](#3-accessibility-gaps)
4. [Input Barrier Analysis](#4-input-barrier-analysis)
5. [Output Clarity](#5-output-clarity)
6. [Consolidated Findings Matrix](#6-consolidated-findings-matrix)
7. [Recommendations Roadmap](#7-recommendations-roadmap)

---

## 1. Executive Summary

SkillSprout is an API-first career transition tool that maps user skills against O\*NET occupation data and produces recommendations in three buckets: READY\_NOW, TRAINABLE, and LONG\_RESKILL. The current frontend is a thin Jinja2 layer over the API, originally intended as a developer demonstration rather than a consumer-facing product.

**The core finding of this audit is that the application is functionally inaccessible to its intended audience.** The people who most need career transition guidance -- displaced workers, career changers, non-traditional workers -- are the least equipped to navigate the current interface. The system requires knowledge of O\*NET taxonomy, comfort with technical vocabulary, and tolerance for an unstyled, non-responsive UI. There are six P0 issues that individually block core usage, twelve P1 issues that degrade the experience for users who do get through, and several P2 improvement opportunities.

**Priority distribution:**
- **P0 (Blocks core usage):** 6 findings
- **P1 (Degrades experience):** 12 findings
- **P2 (Improvement opportunity):** 8 findings

---

## 2. User Journey Map

### 2.1 Current Flow: Step-by-Step

```
VISIT /                           STEP 0: Landing Page
  |                               Time: 0s
  v
Click "Create Profile & Start"   STEP 1: Profile creation (POST /api/v1/user/profile)
  |                               The system sends an empty JSON body {}.
  |                               User sees alert(): "Profile created! User ID: 7"
  |                               Time: ~2s
  v
Redirect to /flow/{user_id}      STEP 2: Flow page loads
  |                               User sees a search input for current occupation.
  |                               Banner: "This is a simplified UI flow. For full
  |                               functionality, use the API Documentation."
  |                               Time: ~3s
  v
Type occupation name, click       STEP 3: Search occupations
"Search"                          (GET /api/v1/occupations/search?q=...)
  |                               Results appear as clickable list items.
  |                               In demo mode, only 3 occupations exist.
  |                               Time: ~8s
  v
Click an occupation result        STEP 4: Set current occupation + fetch skills
  |                               (POST .../current-occupation, GET .../skills)
  |                               User sees alert(): "Selected: Software Developers"
  |                               Skills section appears with up to 15 range sliders.
  |                               Time: ~15s
  v
Adjust 15 sliders (0-4),         STEP 5: Rate skills
click "Submit Ratings"            (POST .../skills/ratings)
  |                               Labels show O*NET skill names (e.g., "Systems
  |                               Evaluation", "Complex Problem Solving").
  |                               All sliders default to "2" (Intermediate).
  |                               User sees alert(): "Skills rated successfully!"
  |                               Time: ~60-180s (highly variable)
  v
Click "Get My Recommendations"   STEP 6: Generate recommendations
  |                               (POST .../recommendations)
  |                               Results render inline: buckets with occupation
  |                               cards showing match_score, gap_severity, top
  |                               gaps, and a training_suggestion string.
  |                               Time: ~3s
  v
Read results                     STEP 7: Interpret output
                                  No next action available. No links to training
                                  resources. No save/export. No way to adjust
                                  ratings and re-run.
                                  Time: variable (user likely leaves)
```

### 2.2 Time-to-Value Analysis

| Metric | Current State |
|---|---|
| **Time to first meaningful interaction** | ~15s (after profile creation + occupation search + selection) |
| **Time to first recommendation** | ~90-200s (depends on slider adjustment care) |
| **Time to actionable outcome** | Never reached -- no links to training programs, job boards, or next steps |
| **Minimum clicks to recommendation** | 5 (Create Profile -> Search -> Select Occupation -> Submit Ratings -> Get Recommendations) |
| **Cognitive load peak** | Step 5: rating 15 unfamiliar skills on an unlabeled numeric scale |

### 2.3 Where Non-Professional Users Get Stuck

**Stuck Point 1: "What is my occupation?" (Step 3)**
A retail cashier searching for "cashier" in demo mode gets zero results because the mock data only contains 3 tech occupations. Even in live mode, O\*NET titles often differ from colloquial job names (e.g., "Cashiers" is the O\*NET title, but a user might search "checkout clerk" or "register operator"). There is no fuzzy matching, no suggested occupations, and no fallback for zero results beyond "Try a different search term."

**Stuck Point 2: "What do these skills mean?" (Step 5)**
Skills are presented by their O\*NET names: "Systems Evaluation", "Complex Problem Solving", "Active Listening". No definitions, examples, or behavioral anchors are provided. The scale labels (0=None, 1=Basic, 2=Intermediate, 3=Advanced, 4=Expert) are shown once in the section header but not on individual sliders. A warehouse worker being asked to rate their "Systems Analysis" proficiency has no frame of reference.

**Stuck Point 3: "What does this number mean?" (Step 7)**
The output shows "Match: 67%, Gap: 34%" with no explanation of what these percentages represent, what thresholds determine bucket placement, or why a 67% match is "Trainable" rather than "Ready Now."

**Stuck Point 4: "What do I do now?" (Step 7)**
The training suggestions are generic templates: "Fill 3 skill gap(s) through a bootcamp, certificate program, or self-directed learning with portfolio projects (typically 3-18 months)." No specific programs, no links, no cost estimates, no local resources. The user journey dead-ends.

### 2.4 Personas and Failure Modes

| Persona | Failure Point | Severity |
|---|---|---|
| **Maria, 52, displaced factory worker** | Cannot find "assembly line worker" in demo data. Would not understand "Systems Evaluation" or "Complex Problem Solving" as O\*NET defines them. | Total failure |
| **James, 28, Uber driver wanting to transition** | "Gig worker" or "rideshare driver" does not map cleanly to any O\*NET code. Skills like "customer service under pressure" or "navigation/routing" have no element\_id equivalent. | Total failure |
| **Priya, 35, church volunteer coordinator** | Volunteer experience is invisible to the system. Coordination, event planning, budgeting, and mentoring skills cannot be entered because they must map to O\*NET element\_ids. | Total failure |
| **Alex, 24, bootcamp graduate** | Could navigate the technical UI, but would not know how to rate O\*NET skills vs. bootcamp-acquired skills. The 0-4 scale does not distinguish between "learned in bootcamp" and "used professionally for 5 years." | Partial failure |

---

## 3. Accessibility Gaps

### 3.1 Progressive Enhancement

| Issue | Status | Priority |
|---|---|---|
| **JavaScript dependency** | The entire flow page (`flow.html`) is non-functional without JavaScript. Profile creation, occupation search, skill rating, and recommendation retrieval all require JS. There are no `<noscript>` fallbacks, no server-side form submissions, no progressive enhancement strategy. | **P1** |
| **`style.css` is effectively empty** | The external stylesheet (`static/css/style.css`) contains only a comment: `/* Additional styles can be added here */`. All styling is inline in `base.html`'s `<style>` block or applied via inline `style` attributes on individual elements. This prevents user stylesheets from having meaningful specificity and makes the CSS non-cacheable across pages. | **P2** |
| **No service worker or offline support** | For users on unreliable connections (rural areas, mobile data), there is no caching strategy. Every page load requires a full round-trip. | **P2** |

### 3.2 Keyboard Navigation

| Issue | Details | Priority |
|---|---|---|
| **Occupation results are `<li>` elements with `onclick` handlers** | In `flow.html` (lines 63-73), search results are rendered as `<li>` elements with `li.onclick = () => selectOccupation(...)`. These are not focusable, have no `tabindex`, no `role="button"`, and no keyboard event handlers. A keyboard-only user cannot select an occupation. | **P0** |
| **No visible focus indicators** | The base CSS does not define any `:focus` or `:focus-visible` styles. The browser defaults are suppressed by the `* { margin: 0; padding: 0; }` reset. There is a `cursor: pointer` on `.btn` but no focus ring. | **P1** |
| **Range sliders lack keyboard context** | The `<input type="range">` elements are natively keyboard-accessible (arrow keys), but there are no visible tick marks, no ARIA value descriptions, and the current value display (`<span>`) updates only via `oninput`, which works with keyboard but is not announced. | **P1** |
| **No skip-to-content link** | The page has a header and footer but no skip navigation mechanism. | **P2** |

### 3.3 Screen Reader Compatibility

| Issue | Details | Priority |
|---|---|---|
| **`alert()` for system feedback** | Profile creation, occupation selection, skill submission, and errors all use JavaScript `alert()`. Screen readers will announce these, but they are modal, non-customizable, and interruptive. There are no ARIA live regions for status updates. | **P1** |
| **Dynamic content not announced** | When the skills section or recommendations section becomes visible (via `style.display = 'block'`), screen readers receive no notification. No `aria-live` regions, no `role="status"`, no focus management after content changes. | **P1** |
| **Occupation search results have no semantic role** | The results list uses `<ul>` with `listStyle: 'none'`, which some screen readers interpret as a non-list. The `<li>` items have no `role="option"` or `aria-label`. The `<small>` tag wrapping the O\*NET code provides no context about what the code represents. | **P1** |
| **Slider labels do not include skill descriptions** | The `<label>` elements contain the skill name and importance value but no `for` attribute linking to the corresponding `<input>`. The `id` attributes on sliders use the O\*NET element\_id (e.g., `skill_2.B.1.a`) which contains dots, making them technically invalid HTML IDs per the HTML4 spec (valid in HTML5 but unusual). | **P1** |
| **Recommendation output is semantically flat** | Bucket headings use `<h4>` inside dynamically created `<div>` elements. Individual occupation cards are unstyled `<div>` elements with no landmark roles, no `aria-label`, and data presented as raw `<strong>` and `<small>` tags with no structure. A screen reader user would hear a wall of undifferentiated text. | **P1** |
| **Language attribute is correct** | `<html lang="en">` is set. This is one of the few accessibility requirements met. | Pass |
| **Viewport meta tag is present** | `<meta name="viewport" content="width=device-width, initial-scale=1.0">` is set. | Pass |

### 3.4 Color Contrast

| Element | Foreground | Background | Ratio | WCAG AA (4.5:1) | Priority |
|---|---|---|---|---|---|
| Body text (`#333` on `#f5f5f5`) | `#333333` | `#f5f5f5` | ~9.7:1 | Pass | -- |
| Header text (white on `#2c3e50`) | `#ffffff` | `#2c3e50` | ~8.6:1 | Pass | -- |
| Demo badge (`#fff` on `#f39c12`) | `#ffffff` | `#f39c12` | ~2.1:1 | **Fail** | **P1** |
| Alert-info text (`#0c5460` on `#d1ecf1`) | `#0c5460` | `#d1ecf1` | ~6.4:1 | Pass | -- |
| Alert-warning text (`#856404` on `#fff3cd`) | `#856404` | `#fff3cd` | ~4.6:1 | Borderline pass | **P2** |
| Footer text (`#7f8c8d` on `#f5f5f5`) | `#7f8c8d` | `#f5f5f5` | ~2.9:1 | **Fail** | **P1** |
| Occupation result items (default on `#ecf0f1`) | `#333333` | `#ecf0f1` | ~8.5:1 | Pass | -- |
| `.btn` text (white on `#3498db`) | `#ffffff` | `#3498db` | ~3.3:1 | **Fail** | **P1** |
| `.btn-success` text (white on `#27ae60`) | `#ffffff` | `#27ae60` | ~3.1:1 | **Fail** | **P1** |
| `.btn-secondary` text (white on `#95a5a6`) | `#ffffff` | `#95a5a6` | ~2.3:1 | **Fail** | **P1** |

Five of the UI's interactive/informational elements fail WCAG AA contrast requirements. The primary action button (`.btn`), success button (`.btn-success`), and secondary button (`.btn-secondary`) all fail, meaning every clickable call-to-action in the application has insufficient contrast.

### 3.5 Mobile Usability

| Issue | Details | Priority |
|---|---|---|
| **No responsive design** | The `.container` has `max-width: 1200px` but no media queries exist anywhere in the codebase. On mobile screens, the layout does not adapt. The search input has a hardcoded `width: 300px` which may overflow on small screens. Range sliders have `width: 200px` which is difficult to manipulate on touch devices. | **P0** |
| **No touch target sizing** | Button padding is `10px 20px` which may produce tap targets smaller than the recommended 44x44px minimum for touch interfaces. Occupation result `<li>` items have `padding: 10px` which is also potentially undersized. | **P1** |
| **Inline styles prevent responsive overrides** | Many layout-critical properties are set via inline `style` attributes (e.g., `style="width: 300px"`, `style="margin-left: 20px"`), which have the highest CSS specificity and cannot be overridden by media queries in external stylesheets without `!important`. | **P1** |

### 3.6 Low Bandwidth Considerations

| Issue | Details | Priority |
|---|---|---|
| **No loading indicators** | API calls (occupation search, skill fetch, recommendation generation) have no visual feedback during loading. The user clicks a button and nothing visibly happens until the `alert()` fires or the DOM updates. On slow connections, this can take 5-30 seconds with zero feedback. | **P1** |
| **No error recovery** | All API errors result in `alert('Error: ' + error.message)` with no retry option, no graceful degradation, and no state preservation. A timeout during skill submission loses all entered ratings. | **P1** |
| **All CSS is inline in `<head>`** | The `<style>` block in `base.html` is ~128 lines and is re-downloaded on every page load. This is a minor concern but prevents caching. | **P2** |

---

## 4. Input Barrier Analysis

### 4.1 How Skills Are Specified

The skill rating system requires users to interact with O\*NET's internal taxonomy at every step:

**Occupation selection** requires matching the user's self-description of their job to O\*NET's standardized occupation titles. The search endpoint (`/api/v1/occupations/search`) performs keyword matching against O\*NET titles. A user who describes themselves as a "front desk person" would need to know to search for "Receptionists and Information Clerks" (O\*NET code 43-4171.00).

**Skill identification** is done entirely via O\*NET `element_id` values. When the user selects an occupation, the system fetches skills for that O\*NET code and presents them by their canonical O\*NET names. The API schema (`SkillRating` in `schemas.py`, line 71-74) requires:

```python
class SkillRating(BaseModel):
    element_id: str
    rating_0_4: int = Field(..., ge=0, le=4)
```

The `element_id` field is an opaque O\*NET identifier like `2.B.1.a` (Reading Comprehension) or `2.B.8.d` (Systems Analysis). While the UI auto-populates these from the fetched occupation skills, the API itself demands them -- meaning any programmatic or alternative interface integration must also speak O\*NET's taxonomy natively.

**Skill rating** uses a 0-4 integer scale mapped internally to a capability scalar:

```python
RATING_TO_CAPABILITY = {
    0: 0.0,   # None
    1: 0.25,  # Basic
    2: 0.5,   # Intermediate
    3: 0.75,  # Advanced
    4: 1.0,   # Expert
}
```

This mapping (from `scoring.py`, lines 16-22) is never shown to the user. The labels "None/Basic/Intermediate/Advanced/Expert" appear only once in the section header text on the flow page. Individual sliders display only the numeric value (0-4). The user has no behavioral anchors to calibrate their self-assessment.

### 4.2 Vocabulary Barrier

The O\*NET skill taxonomy uses standardized workforce science terminology that differs significantly from how people describe their abilities:

| O\*NET Term | What a User Might Call It | Gap |
|---|---|---|
| Systems Evaluation | "Figuring out if something is working" | High -- academic vs. colloquial |
| Complex Problem Solving | "Fixing things when they go wrong" | Medium -- O\*NET definition has specific criteria |
| Active Listening | "Paying attention" | Medium -- most people think they do this |
| Critical Thinking | "Using common sense" or "analyzing things" | Medium -- term is overloaded |
| Reading Comprehension | "Reading" | Low -- but the importance rating confuses |
| Programming | "Coding" or "writing scripts" | Low -- reasonably well-understood |
| Systems Analysis | "Understanding how parts fit together" | High -- very abstract |
| Equipment Maintenance | "Fixing equipment" | Low |
| Troubleshooting | "Debugging" or "figuring out what's broken" | Low for tech; high for other domains |

**Key observation:** The vocabulary barrier is not uniform. Technical workers may recognize most terms; service workers, tradespeople, and non-traditional workers face significantly higher barriers. The system implicitly assumes the user has workforce development literacy.

### 4.3 Non-Traditional Worker Analysis

The system fundamentally fails three categories of non-traditional workers:

**Informal experience holders (volunteers, community organizers, caregivers)**

A church volunteer coordinator who has managed event logistics for 200-person gatherings, coordinated 30+ volunteers, managed budgets, and handled conflict resolution possesses skills that map to "Coordination", "Management of Personnel Resources", "Negotiation", and "Service Orientation" in O\*NET terms. However:

- They would not describe their current role as any O\*NET occupation
- They would not recognize their experience in O\*NET's skill names
- The system provides no way to enter skills outside the context of a recognized O\*NET occupation
- There is no "I don't currently have a formal job" path

**Gig economy workers (rideshare, delivery, freelance)**

A rideshare driver possesses skills in customer service, navigation, time management, vehicle operation, conflict de-escalation, and financial management (self-employment). The nearest O\*NET occupation might be "Taxi Drivers and Chauffeurs" (53-3054.00), but:

- The mock data does not include this occupation
- Even if found, the O\*NET skills for this occupation would not capture the entrepreneurial, customer-facing, and platform-management skills that distinguish gig workers
- Multi-gig workers (driving + delivery + freelance tasks) have no way to represent a composite skill profile

**Career gap returners**

A parent returning to the workforce after a 10-year gap has skills that have both decayed (technical proficiency) and grown (project management, communication, crisis management). The 0-4 scale provides no way to indicate "I was a 4 ten years ago but I'm probably a 2 now" or "I developed this skill informally."

### 4.4 What Gets Lost

The input model produces systematic biases:

1. **Skills not in O\*NET's taxonomy are invisible.** Digital literacy, AI prompt engineering, social media management, remote collaboration, and many emerging skills have no element\_id.

2. **Transferable soft skills are underweighted.** The system rates skills by O\*NET importance weights, which are occupation-specific. A user's strong "Active Listening" (element\_id `2.B.2.a`) is weighted differently for every target occupation, but the user rates it once and that rating is applied uniformly.

3. **Self-assessment bias is unmitigated.** Research consistently shows that less-skilled individuals overestimate their abilities (Dunning-Kruger effect) while highly skilled individuals underestimate. The system has no calibration for self-assessment bias. All sliders default to "2" (Intermediate), which anchors users toward the middle of the scale.

4. **Context is erased.** A "3" in Programming means very different things for a web developer (might mean "competent in one language") vs. a data scientist (might mean "can write analysis scripts"). The system treats all 3s identically.

---

## 5. Output Clarity

### 5.1 Do Users Understand WHY Each Bucket Assignment?

**No.** The bucket assignment logic (from `scoring.py`, lines 223-253) uses configurable thresholds:

```python
# From config.py defaults:
ready_now_match_threshold: float = 75.0
ready_now_gap_threshold: float = 25.0
trainable_match_min: float = 50.0
trainable_match_max: float = 74.0
trainable_gap_min: float = 26.0
trainable_gap_max: float = 55.0
```

The bucket assignment rules are:
- **READY\_NOW:** match\_score >= 75 AND gap\_severity <= 25
- **TRAINABLE:** match\_score between 50-74 OR gap\_severity between 26-55
- **LONG\_RESKILL:** everything else

These thresholds are never shown to the user. The output includes a plain-language `explanation` field (e.g., "Good foundation for Web Developers (67% match), but 3 key skill gap(s) to address (34% severity). With focused training, this role is achievable."), but this explanation does not state the threshold rules. A user seeing a 74% match in TRAINABLE and 75% in READY\_NOW has no way to understand the 1-point difference that changed the bucket.

**Critical issue:** The TRAINABLE bucket uses OR logic between match\_score and gap\_severity ranges. This means a user with a 90% match but 30% gap severity is TRAINABLE (not READY\_NOW) because gap\_severity > 25. The explanation says "Strong match" but the bucket says "Trainable." This is confusing and feels contradictory.

### 5.2 Do Users Understand WHAT Specific Gaps Exist?

**Partially.** The output includes `top_gaps` with up to 5 skill gaps per recommended occupation. Each gap contains:

```python
class SkillGapInfo(BaseModel):
    element_id: str           # e.g., "2.B.8.d"
    skill_name: str           # e.g., "Systems Analysis"
    required_importance: float # e.g., 78.0
    required_level: float     # e.g., 5.38
    user_capability: float    # e.g., 0.25
    gap_weight: float         # e.g., 0.12
```

The UI displays only `skill_name` from the gaps: `<small><strong>Top gaps:</strong> Systems Analysis, Programming, Critical Thinking</small>`. The numeric fields (`required_importance`, `required_level`, `user_capability`, `gap_weight`) are available in the API response but not rendered in the UI.

**Problems:**
- Users see skill names but not how large each gap is
- `user_capability` is a 0.0-1.0 scalar, not the original 0-4 rating, making it opaque even if displayed
- `required_importance` is on a 0-100 scale, `required_level` is on a 0-7 scale -- two different scales for the same skill, neither explained
- The gap list is truncated to the top 3 in the UI display (line 204: `.slice(0, 3)`) even though 5 are available in the data

### 5.3 Do Users Understand WHAT To Do About Gaps?

**No.** The `training_suggestion` field provides templated text based on bucket and job zone:

```
"Fill 3 skill gap(s) through a bootcamp, certificate program, or self-directed
 learning with portfolio projects (typically 3-18 months)."
```

This text is generated by `_generate_training_suggestion()` in `scoring.py` (lines 255-299). There are exactly 7 template variations across 3 buckets and 3 job zone ranges. The suggestions are:

- Generic: "bootcamp, certificate program, or self-directed learning"
- Non-specific to the actual skill gaps identified
- Not linked to any training providers, course catalogs, or resources
- Not personalized to the user's location, budget, or time availability
- Not differentiated by the specific skills that need development

The `app/features/training_paths/__init__.py` module exists as an empty stub, confirming that specific training path generation was planned but never implemented.

### 5.4 Do Recommendations Feel Trustworthy?

**No, for several reasons:**

1. **No explanation of methodology.** The landing page says "Baseline scoring using match scores and skill gap analysis" but never explains what these terms mean or how they are calculated.

2. **No confidence intervals or uncertainty communication.** A 67% match is presented as a precise number. There is no indication of how sensitive this score is to the user's self-assessment accuracy. If the user had rated one skill differently, the score might change by 10+ points.

3. **The model version is displayed but not explained.** The output includes `Model Version: v1_baseline` which is developer-facing metadata that erodes user trust. It signals "this is unfinished."

4. **Decision guidance is a single sentence.** The `decision_guidance` field (from `endpoints.py`, lines 605-610) has only 3 possible values, selected by whether the READY\_NOW bucket has items. This is not personalized guidance.

5. **No social proof or validation.** There are no success stories, no data about how many users have followed recommendations, no external validation of the scoring methodology.

6. **Visual presentation undermines credibility.** The minimal, unstyled interface with `alert()` popups, raw percentage displays, and a visible "DEMO MODE" badge communicates "prototype" rather than "trusted career tool."

---

## 6. Consolidated Findings Matrix

### P0: Blocks Core Usage

| ID | Finding | Location | Impact |
|---|---|---|---|
| **P0-1** | Users must know O\*NET element IDs to rate skills via the API. The UI auto-populates these, but the underlying dependency means no alternative input modes are possible without mapping to O\*NET taxonomy first. | `schemas.py` line 71-74, `flow.html` line 141-143 | Anyone interacting outside the single UI flow (API consumers, integrators, alternative frontends) must speak O\*NET. Non-traditional workers whose skills do not map to O\*NET are excluded entirely. |
| **P0-2** | No plain-language input mode exists. There is no way to describe skills in natural language, select from behavioral examples, or import skills from a resume/LinkedIn profile. The `skills_translator` feature module is an empty stub. | `app/features/skills_translator/__init__.py` (empty) | The primary audience (career changers) cannot use the tool without workforce science literacy. |
| **P0-3** | No explanation of thresholds used for bucket assignment. Users cannot understand why an occupation is READY\_NOW vs. TRAINABLE vs. LONG\_RESKILL. The 75/25/50-74/26-55 threshold rules are in code only. | `scoring.py` lines 223-253, `config.py` lines 53-58 | Users cannot calibrate trust in the system's recommendations or make informed decisions. |
| **P0-4** | No actionable training recommendations. Training suggestions are generic templates (7 total variants) with no links to specific programs, courses, or resources. The `training_paths` feature module is an empty stub. | `scoring.py` lines 255-299, `app/features/training_paths/__init__.py` (empty) | The user journey dead-ends at recommendations. There is no path from "you need to learn Systems Analysis" to "here is how to learn Systems Analysis." |
| **P0-5** | Minimal CSS/styling makes the application feel unfinished and untrustworthy. The external stylesheet is effectively empty (1 comment line). All styling is inline in `base.html`. Interactive elements like occupation results are unstyled `<li>` items. `alert()` is used for all system messages. | `static/css/style.css`, `base.html` lines 7-128, `flow.html` multiple `alert()` calls | Users evaluating whether to trust career-changing advice from this tool will be deterred by the prototype appearance. Credibility is critical for an application that asks users to make life decisions based on its output. |
| **P0-6** | Occupation search results in `flow.html` are not keyboard-accessible. Results are rendered as `<li>` elements with `onclick` handlers and no `tabindex`, `role`, or keyboard event support. Keyboard-only and screen reader users cannot select an occupation, completely blocking the flow. | `flow.html` lines 63-73 | WCAG 2.1 SC 2.1.1 (Keyboard) failure. Any user who cannot use a mouse -- motor impairments, screen reader users, power keyboard users -- is completely blocked at step 3 of the flow. |

### P1: Degrades Experience

| ID | Finding | Location | Impact |
|---|---|---|---|
| **P1-1** | Five interactive elements fail WCAG AA color contrast (`.btn` at 3.3:1, `.btn-success` at 3.1:1, `.btn-secondary` at 2.3:1, `.demo-badge` at 2.1:1, footer text at 2.9:1). | `base.html` inline styles | Users with low vision or in bright environments cannot reliably read buttons or footer content. Every call-to-action in the application is affected. |
| **P1-2** | No visible focus indicators defined. The CSS reset (`* { margin: 0; padding: 0 }`) does not explicitly remove outlines, but no custom `:focus` styles are provided, leaving users dependent on inconsistent browser defaults. | `base.html` lines 8-11 | Keyboard users cannot track their position in the interface. |
| **P1-3** | All system feedback uses JavaScript `alert()`. Profile creation, occupation selection, skill submission, and all errors surface as modal alert dialogs. | `index.html` lines 68-88, `flow.html` lines 81-103, 140-159 | Disruptive, non-customizable, and prevents users from referencing context while reading the message. Cannot be styled, logged, or dismissed programmatically. |
| **P1-4** | No loading indicators for API calls. Occupation search, skill fetch, and recommendation generation provide no visual feedback during network requests. | `flow.html` all `async function` blocks | Users on slow connections click a button and see nothing happen for seconds, leading to repeat clicks and confusion. |
| **P1-5** | Dynamic content changes are not announced to assistive technology. Skills section and recommendations section appear via `style.display = 'block'` with no ARIA live regions, focus management, or screen reader announcements. | `flow.html` lines 107-138, 179-218 | Screen reader users do not know when new content appears. |
| **P1-6** | No error recovery or state preservation. API failures show an alert and do nothing else. Skill ratings entered into sliders are lost on page refresh or navigation. | `flow.html` all `catch` blocks | Users who invest time in rating 15 skills lose all progress on any error. |
| **P1-7** | All sliders default to 2 (Intermediate), creating anchoring bias. Users must actively move each slider, and the default suggests "Intermediate" is normal, discouraging honest self-assessment of skills rated 0 or 1. | `flow.html` line 122 | Systematically inflated skill ratings lead to overly optimistic match scores, placing users in higher buckets than warranted. |
| **P1-8** | No responsive design. The container is fixed at `max-width: 1200px`, the search input is `width: 300px`, and range sliders are `width: 200px`. No media queries exist. | `base.html` line 22, `flow.html` lines 17, 124 | Mobile users (a likely majority of the target audience) see a desktop layout on small screens. Input elements may overflow or be difficult to interact with on touch devices. |
| **P1-9** | Touch targets may be undersized. Button padding (`10px 20px`) and list item padding (`10px`) may produce tap targets below the 44x44px WCAG recommendation. | `base.html` line 58, `flow.html` line 66 | Mobile users may struggle to tap buttons and list items accurately. |
| **P1-10** | Demo mode shows only 3 tech occupations. Users testing with non-tech occupations get zero results with no explanation of why and no guidance toward working alternatives. | `onet_client.py` lines 312-371 | Most users evaluating the product in demo mode will hit a dead end at occupation search. |
| **P1-11** | The flow page banner says "For full functionality, use the API Documentation" and links to `/api/v1/docs`. This is unhelpful for non-technical users and undermines confidence in the UI they are actively using. | `flow.html` lines 9-11 | Users feel like they are using a second-class interface. |
| **P1-12** | The `<html>` document uses no ARIA landmarks beyond semantic HTML. There are no `role="main"`, `role="navigation"`, or `aria-label` attributes on sections. The `<header>` and `<footer>` are semantic but contain no navigation links (header) or have insufficient contrast (footer). | `base.html` lines 132-152 | Screen reader users cannot efficiently navigate between page sections. |

### P2: Improvement Opportunities

| ID | Finding | Location | Impact |
|---|---|---|---|
| **P2-1** | No ability to re-run recommendations after adjusting skill ratings without reloading the entire flow. | `flow.html` | Users cannot iterate on their self-assessment. |
| **P2-2** | No save/export/share functionality for recommendation results. | `flow.html` | Users cannot save results for later review or share with advisors/counselors. |
| **P2-3** | No progress indicator across the multi-step flow. Users see individual cards but no visual indication of where they are in the process or how many steps remain. | `flow.html` | Increased cognitive load and drop-off risk. |
| **P2-4** | The user profile is created with an empty JSON body and no user-facing data (name, email, preferences). | `endpoints.py` line 275, `schemas.py` line 43-45 | No personalization, no session persistence, no way to return to previous results. |
| **P2-5** | No feedback mechanism visible in the UI. The API supports feedback (`POST /api/v1/feedback`) but the recommendation results page has no thumbs up/down, no "this is wrong," no save/hide buttons. | `flow.html` `displayRecommendations()` | The calibration model (v2) depends on feedback data that the UI never collects. |
| **P2-6** | CSS is not externalized or organized. Inline styles and the `<style>` block in `base.html` cannot be cached, versioned, or theme-switched. | `base.html`, `static/css/style.css` | Maintenance burden and inability to support theming, dark mode, or user style preferences. |
| **P2-7** | The skill importance value is shown as a raw number (e.g., "Importance: 75") with no context about the scale (0-100) or what constitutes high vs. low importance. | `flow.html` line 120 | Users cannot prioritize which skills to invest in rating accurately. |
| **P2-8** | No internationalization (i18n) support. All text is hardcoded in English in templates and Python code. | All templates, `scoring.py` | Excludes non-English-speaking users, who are often overrepresented in the career-transition population. |

---

## 7. Recommendations Roadmap

### Phase 1: Unblock Core Usage (Address P0s)

**Goal:** Make the application minimally usable by its target audience.

1. **Implement a plain-language skills input mode.** Allow users to describe what they do in their own words ("I manage schedules for a team of 12 people") and map those descriptions to O\*NET skills via NLP or a guided questionnaire. Populate the `skills_translator` module.

2. **Add behavioral anchors to skill ratings.** For each skill, provide 2-3 example behaviors at each proficiency level. For "Active Listening" at level 3: "You can summarize back what someone said and identify their underlying concern." This replaces the abstract 0-4 scale with recognizable descriptions.

3. **Surface bucket assignment logic to users.** Add a tooltip or expandable section explaining: "You are in TRAINABLE because your match score (67%) is between 50-74%. To move to READY NOW, you would need a match score above 75% and a gap severity below 25%." Make the math visible and understandable.

4. **Build actionable training paths.** Populate the `training_paths` module with links to specific resources for each O\*NET skill: online courses (Coursera, edX, Khan Academy), certification programs, community college offerings, apprenticeships. Display these per-gap, not per-occupation.

5. **Implement real CSS and visual design.** Move all styles to the external stylesheet. Add a responsive grid system, proper component styling, feedback toast notifications instead of `alert()`, loading spinners, and a progress stepper for the multi-step flow.

6. **Make occupation results keyboard-accessible.** Add `tabindex="0"`, `role="button"`, `aria-label`, and `keydown` event handlers (Enter and Space) to occupation search result elements.

### Phase 2: Fix Accessibility and Experience (Address P1s)

**Goal:** Make the application usable by people with disabilities and on all devices.

1. Fix all color contrast failures (buttons, badge, footer).
2. Add visible focus indicators (`:focus-visible` styles).
3. Replace `alert()` with inline toast/notification components with ARIA live regions.
4. Add loading states (spinners, skeleton screens) for all API calls.
5. Implement ARIA live regions for dynamic content (skills section, recommendations).
6. Add error recovery with state preservation (save slider values to `localStorage`).
7. Change slider defaults to unset/blank rather than "2" to avoid anchoring.
8. Add responsive breakpoints (mobile-first media queries).
9. Increase touch target sizes to meet 44x44px minimum.
10. Expand demo data to include non-tech occupations.
11. Remove or rephrase the "use the API Documentation" banner.
12. Add ARIA landmarks and skip-to-content navigation.

### Phase 3: Enhance and Iterate (Address P2s)

**Goal:** Make the application a compelling, trustworthy career tool.

1. Allow iterative re-rating and re-running of recommendations.
2. Add save/export/share functionality (PDF report, shareable link).
3. Build a visual progress stepper component.
4. Add user profile fields (name, goals, constraints) for personalization.
5. Surface feedback UI (thumbs up/down, save, hide) on recommendation cards.
6. Externalize and organize CSS with a design system.
7. Contextualize importance values with labels ("High importance", "Moderate importance").
8. Plan for i18n/l10n infrastructure.

---

## Appendix A: Files Reviewed

| File | Path | Role |
|---|---|---|
| `base.html` | `/home/user/skillsprout/templates/base.html` | Base template with all CSS and page structure |
| `index.html` | `/home/user/skillsprout/templates/pages/index.html` | Landing page with profile creation |
| `flow.html` | `/home/user/skillsprout/templates/pages/flow.html` | Core user flow: search, rate, recommend |
| `docs.html` | `/home/user/skillsprout/templates/pages/docs.html` | Developer documentation page |
| `style.css` | `/home/user/skillsprout/static/css/style.css` | External stylesheet (effectively empty) |
| `endpoints.py` | `/home/user/skillsprout/app/api/endpoints.py` | API route handlers |
| `schemas.py` | `/home/user/skillsprout/app/schemas/schemas.py` | Pydantic request/response schemas |
| `scoring.py` | `/home/user/skillsprout/app/ml/scoring.py` | Baseline scoring engine and bucket assignment |
| `calibration.py` | `/home/user/skillsprout/app/ml/calibration.py` | Calibration model (v2, not yet in use) |
| `config.py` | `/home/user/skillsprout/app/core/config.py` | Application settings including scoring thresholds |
| `main.py` | `/home/user/skillsprout/app/main.py` | FastAPI application entry point and UI routes |
| `models.py` | `/home/user/skillsprout/app/models/models.py` | SQLAlchemy ORM models |
| `onet_client.py` | `/home/user/skillsprout/app/services/onet_client.py` | O\*NET API client and mock data |
| `seed_demo.py` | `/home/user/skillsprout/scripts/seed_demo.py` | Demo data seeding script |
| `skills_translator/__init__.py` | `/home/user/skillsprout/app/features/skills_translator/__init__.py` | Empty stub -- planned skills translation |
| `training_paths/__init__.py` | `/home/user/skillsprout/app/features/training_paths/__init__.py` | Empty stub -- planned training paths |
| `explainability/__init__.py` | `/home/user/skillsprout/app/features/explainability/__init__.py` | Empty stub -- planned explainability |

## Appendix B: Scoring Threshold Reference

These values from `config.py` determine bucket assignment and are not visible anywhere in the UI:

```
ready_now_match_threshold  = 75.0   (minimum match_score for READY_NOW)
ready_now_gap_threshold    = 25.0   (maximum gap_severity for READY_NOW)
trainable_match_min        = 50.0   (minimum match_score for TRAINABLE)
trainable_match_max        = 74.0   (maximum match_score for TRAINABLE)
trainable_gap_min          = 26.0   (minimum gap_severity for TRAINABLE)
trainable_gap_max          = 55.0   (maximum gap_severity for TRAINABLE)
```

Note the OR logic in TRAINABLE assignment: an occupation qualifies if EITHER the match score falls in 50-74 OR the gap severity falls in 26-55. This means high-match, moderate-gap occupations can be classified as TRAINABLE rather than READY\_NOW, which contradicts the user's intuition.

## Appendix C: Capability Mapping Reference

The internal mapping from user rating to scoring scalar, from `scoring.py`:

| User Rating | Label (shown once in header) | Internal Scalar | Classified as "Gap"? |
|---|---|---|---|
| 0 | None | 0.0 | Yes (capability <= 0.25) |
| 1 | Basic | 0.25 | Yes (capability <= 0.25) |
| 2 | Intermediate | 0.5 | No |
| 3 | Advanced | 0.75 | No |
| 4 | Expert | 1.0 | No |

A skill rated 1 (Basic) is treated as a gap with the same weight as a skill rated 0 (None). A skill rated 2 (Intermediate) is treated as a non-gap with 50% capability. This cliff between 1 and 2 is never communicated to the user and creates a non-linear scoring effect that could surprise users who rate honestly.
