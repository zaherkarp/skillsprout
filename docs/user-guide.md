# SkillSprout User Guide

## What Is SkillSprout?

SkillSprout helps you discover career transition opportunities based on your current skills. It compares what you can do today with what different jobs require, then tells you which jobs you are ready for now, which ones you could reach with some training, and which ones would require a longer reskilling effort.

SkillSprout uses skill data from O*NET, the U.S. Department of Labor's occupational information database. O*NET defines the skills, knowledge, and abilities required for over 1,000 occupations, along with how important each skill is for each job.

---

## How It Works

### Step 1: Tell Us Your Current Job

Search for your current occupation by title. SkillSprout uses O*NET occupation codes to identify your role. If you cannot find your exact job title, pick the closest match.

### Step 2: Rate Your Skills

Once you select your current occupation, SkillSprout shows you the skills associated with that job. Rate each skill on a 0-4 scale:

| Rating | Meaning | Description |
|--------|---------|-------------|
| 0 | None | No experience or knowledge in this area |
| 1 | Basic | Familiar with the concepts, limited practical experience |
| 2 | Intermediate | Can work independently with this skill |
| 3 | Advanced | Highly proficient, can handle complex tasks |
| 4 | Expert | Deep mastery, can teach others |

Be honest. Overrating your skills will lead to recommendations that are too optimistic (you will be told you are "Ready Now" for jobs you are not actually prepared for). Underrating will show you more "Long-term Reskill" results than necessary.

You do not need to rate every skill. Unrated skills are treated as "0 - None." If you have not rated a skill, SkillSprout assumes you have no experience with it. You can always come back and add ratings later.

### Step 3: Get Recommendations

SkillSprout scores every occupation in its database against your skill profile and sorts the results into three buckets.

---

## Understanding Your Recommendations

### The Three Buckets

**Ready Now** -- These are jobs where your current skills are a strong match. Your match score is 75% or higher, and your skill gaps are minor (25% or less severity). You could start applying to these roles today.

**Trainable** -- These are jobs within your reach if you invest in focused training. You either have a moderate skill match (50-74%) or moderate skill gaps (26-55% severity). Depending on the job, training could take 3 months to 2 years.

**Long-term Reskill** -- These are jobs that require significant skill development. Your current skills do not closely align with the requirements, and the gaps are substantial. Reaching these roles typically takes 1-4+ years of dedicated training or education.

### Match Score

Your match score (0-100) represents how well your rated skills align with what the job requires. It is a weighted average: skills that are more important to the job count more heavily.

A match score of 80 means your skills cover 80% of what the job requires, weighted by importance. It does not mean you are missing exactly 20% of the skills -- it means the weighted importance of skills you are strong in covers 80% of the total.

### Gap Severity

Gap severity (0-100) measures how critical your skill gaps are. It focuses on skills where you rated yourself 0 (None) or 1 (Basic). If you have no experience in a skill that is very important to the job, that creates a high gap severity.

Low gap severity (0-25) means your gaps are in skills that are not critical to the job. High gap severity (55+) means you are missing skills that are central to the role.

### Skill Gaps

For each recommended occupation, SkillSprout shows your top skill gaps -- the specific skills you would need to develop. These are sorted by importance: the most critical missing skills appear first.

Each gap shows:
- The skill name
- How important it is to the target job
- Your current capability level

This tells you exactly where to focus your training efforts.

### Training Suggestions

SkillSprout provides training path suggestions based on the job's education level (called "job zone" in O*NET):

| Job Zone | Typical Education | Training Suggestion |
|----------|-------------------|---------------------|
| 1-2 | High school / some training | Certificate programs, apprenticeships, on-the-job training |
| 3 | Associate's degree or equivalent | Bootcamps, certificate programs, self-directed learning with portfolio |
| 4-5 | Bachelor's degree or higher | Degree programs, extended bootcamps, comprehensive self-study |

These are general suggestions. The actual training path depends on your specific situation, your learning style, and the resources available to you.

### AI Exposure Index

Each occupation includes an AI exposure indicator, drawn from the [Anthropic Labor Market Report](https://www.anthropic.com/research/labor-market-impacts) (March 2026). This shows how much of a job's tasks could be sped up or performed by AI:

- **Theoretical exposure**: The share of tasks an AI could theoretically speed up by at least 2x.
- **Observed exposure**: The share of tasks where AI is actually being used in professional settings today.

For example, Computer Programmers have 94% theoretical exposure and 75% observed exposure -- meaning AI is already performing a large share of programming tasks in practice. In contrast, Electricians have near-zero AI exposure.

This data helps you make informed transition decisions: moving toward roles with lower AI exposure may offer more long-term job security, while high-exposure roles may evolve significantly in the coming years.

### Employment Outlook

SkillSprout shows Bureau of Labor Statistics (BLS) 2024-2034 employment projections for each occupation:

- **Strong growth** (>8% projected growth): High demand, expanding field
- **Moderate growth** (3-8%): Stable, steady demand
- **Stable** (-2% to 3%): Flat, neither growing nor shrinking
- **Declining** (<-2%): Shrinking field, fewer openings expected

Use the employment outlook alongside AI exposure to evaluate transition targets. A role with low AI exposure and strong growth (e.g., Electrician at +11.4%) may be a safer long-term bet than one with high exposure and declining outlook (e.g., Data Entry at -32%).

---

## Providing Feedback

SkillSprout improves over time based on your feedback. When you interact with a recommendation, the system learns what kinds of recommendations are useful and adjusts future results.

### Actions You Can Take

**Save** -- Bookmark a recommendation for later. This tells us you found it interesting enough to revisit.

**Hide** -- Remove a recommendation from your results. This tells us the recommendation was not useful to you. Use this for jobs you are certain you are not interested in.

**Apply** -- Mark that you applied to this position. This is a strong signal that the recommendation was relevant.

**Interview** -- Mark that you received an interview. This tells us the recommendation was not just relevant but that your skills were credible to an employer.

**Offer** -- Mark that you received a job offer. This is the strongest possible signal that the recommendation was accurate.

Your feedback directly improves recommendations for you and for other users with similar skill profiles. The more specific your feedback (interview and offer outcomes are much more valuable than clicks), the better the system becomes.

---

## Privacy

### What Data SkillSprout Collects

- Your skill self-assessment ratings (the 0-4 ratings you provide)
- Your current occupation selection
- Your interactions with recommendations (save, hide, apply, interview, offer)
- Timestamps of your actions

### What SkillSprout Does NOT Collect

- Your name, email, or other identifying information (unless you provide it)
- Your resume or work history
- Your employer information
- Your demographic data (unless you voluntarily provide it for bias auditing purposes)

### How Your Data Is Used

- **Personalization:** Your skill ratings are used to score occupations and generate your recommendations.
- **Model training:** Your feedback (apply, interview, offer, hide) is used in aggregate to train the calibration model that improves recommendation quality for all users. No individual feedback record is exposed to other users.
- **Bias auditing:** Aggregate statistics (e.g., distribution of recommendation buckets) are analyzed to detect systematic biases. Individual user data is not examined in bias audits.

### Data Retention

- Your profile, ratings, and feedback are stored in the database as long as your account exists.
- You can request deletion of your data at any time.

---

## Frequently Asked Questions

### Why did I get zero "Ready Now" recommendations?

This usually means one of two things:
1. **Your skill ratings are conservative.** If you rated most skills as 0 or 1, you will have a low match score against most occupations. Try re-evaluating whether some of your skills deserve a higher rating.
2. **Your current occupation is very specialized.** Some career transitions are inherently difficult. If you are in a highly specialized role, the number of "Ready Now" alternatives may be small.

### Why is a job I know I can do listed as "Trainable" or "Long-term Reskill"?

SkillSprout scores based on the skills you have rated. If you have not rated some of your skills, the system assumes you have no experience with them. Go back and rate additional skills to get more accurate results.

Also check whether the job requires skills you did not expect. O*NET's skill requirements reflect the full scope of a job, including soft skills and general abilities that job listings might not emphasize.

### Can I change my skill ratings?

Yes. You can update your skill ratings at any time. Your recommendations will be recalculated based on your updated ratings.

### How often are recommendations updated?

Recommendations are generated fresh each time you request them. If you change your skill ratings or current occupation, your next recommendation request will reflect those changes.

The calibration model (which adjusts rankings based on collective user feedback) is retrained daily. Changes from the calibration model are gradual and affect ranking order, not bucket assignment.

### What is a "job zone"?

Job zone is an O*NET concept that groups occupations by the amount of education, training, and experience they typically require:

| Job Zone | Education Level | Example Occupations |
|----------|----------------|---------------------|
| 1 | Little or no preparation | Cashiers, food preparation workers |
| 2 | Some preparation | Customer service representatives, medical assistants |
| 3 | Medium preparation | Electricians, dental hygienists |
| 4 | Considerable preparation | Software developers, registered nurses |
| 5 | Extensive preparation | Physicians, lawyers, engineers |

Moving to a higher job zone generally requires more training time.

### Why do some occupations not appear in my results?

SkillSprout scores occupations that are in its database cache. Not all 1,000+ O*NET occupations may be cached. If you are looking for a specific occupation and it does not appear, it may need to be added to the system's cache.

Also, your current occupation is excluded from results (you already have that job).

### How accurate are the training time estimates?

The training time estimates (e.g., "3-12 months for a certificate program") are rough guidelines based on the job zone. They do not account for your specific learning pace, prior related experience, or the intensity of the training program. Use them as a starting point for planning, not as a guarantee.

### What does "AI exposure" mean for my job?

AI exposure measures how much of a job's day-to-day tasks could be sped up or performed by artificial intelligence. A high exposure percentage does not mean you will lose your job -- it means AI tools are increasingly capable of assisting with or automating parts of the work. Jobs with high observed exposure (like programming at 75%) are already seeing significant AI adoption, while jobs with 0% observed exposure (like electricians) involve physical, hands-on work that AI cannot currently perform.

Use this information to plan ahead: if your current role has high AI exposure, developing skills for lower-exposure roles can provide career insurance.

### Is SkillSprout a job board?

No. SkillSprout helps you discover which occupations match your skills and identify what you would need to learn for other occupations. It does not list job openings, connect you with employers, or help you apply. Use SkillSprout to figure out what to aim for, then use job boards and networking to find specific opportunities.
