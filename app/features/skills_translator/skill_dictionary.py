"""Curated mapping of plain-language phrases to O*NET skill element IDs.

This dictionary enables the skills translator to convert informal descriptions
of work experience into standardized O*NET skill identifiers. It covers
common language used by people in caregiving, retail, trades, administrative,
military, gig economy, and volunteer contexts who may not describe their
abilities using formal skill terminology.

Each entry maps a lowercase phrase to a dict containing:
    - element_id: The O*NET element ID (e.g., "2.B.1.a")
    - skill_name: The canonical O*NET skill name

Reference: O*NET Content Model — Skills (2.B)
https://www.onetcenter.org/content.html
"""

from typing import Dict, TypedDict


class SkillMapping(TypedDict):
    """Type for a single skill dictionary entry."""

    element_id: str
    skill_name: str


# ---------------------------------------------------------------------------
# O*NET skill ID constants (for maintainability)
# ---------------------------------------------------------------------------

READING_COMPREHENSION = "2.B.1.a"
ACTIVE_LEARNING = "2.B.1.b"
LEARNING_STRATEGIES = "2.B.1.e"
SERVICE_ORIENTATION = "2.B.1.f"
PROGRAMMING = "2.B.1.g"
ACTIVE_LISTENING = "2.B.2.a"
WRITING = "2.B.3.a"
SCIENCE = "2.B.3.b"
SPEAKING = "2.B.4.a"
MANAGEMENT_PERSONNEL = "2.B.4.e"
MANAGEMENT_FINANCIAL = "2.B.4.f"
MONITORING = "2.B.4.g"
EQUIPMENT_MAINTENANCE = "2.B.4.h"
MATHEMATICS = "2.B.5.a"
DESIGN = "2.B.5.c"
OPERATIONS_ANALYSIS = "2.B.6.a"
TIME_MANAGEMENT = "2.B.6.b"
SOCIAL_PERCEPTIVENESS = "2.B.7.a"
COORDINATION = "2.B.7.b"
PERSUASION = "2.B.7.c"
NEGOTIATION = "2.B.7.d"
INSTRUCTING = "2.B.7.e"
CRITICAL_THINKING = "2.B.8.a"
COMPLEX_PROBLEM_SOLVING = "2.B.8.b"
JUDGMENT_DECISION_MAKING = "2.B.8.c"
SYSTEMS_ANALYSIS = "2.B.8.d"
SYSTEMS_EVALUATION = "2.B.8.e"
TROUBLESHOOTING = "2.B.9.a"
EQUIPMENT_SELECTION = "2.B.9.b"

# ---------------------------------------------------------------------------
# Canonical skill names keyed by element ID
# ---------------------------------------------------------------------------

ONET_SKILLS: Dict[str, str] = {
    READING_COMPREHENSION: "Reading Comprehension",
    ACTIVE_LEARNING: "Active Learning",
    LEARNING_STRATEGIES: "Learning Strategies",
    SERVICE_ORIENTATION: "Service Orientation",
    PROGRAMMING: "Programming",
    ACTIVE_LISTENING: "Active Listening",
    WRITING: "Writing",
    SCIENCE: "Science",
    SPEAKING: "Speaking",
    MANAGEMENT_PERSONNEL: "Management of Personnel Resources",
    MANAGEMENT_FINANCIAL: "Management of Financial Resources",
    MONITORING: "Monitoring",
    EQUIPMENT_MAINTENANCE: "Equipment Maintenance",
    MATHEMATICS: "Mathematics",
    DESIGN: "Design",
    OPERATIONS_ANALYSIS: "Operations Analysis",
    TIME_MANAGEMENT: "Time Management",
    SOCIAL_PERCEPTIVENESS: "Social Perceptiveness",
    COORDINATION: "Coordination",
    PERSUASION: "Persuasion",
    NEGOTIATION: "Negotiation",
    INSTRUCTING: "Instructing",
    CRITICAL_THINKING: "Critical Thinking",
    COMPLEX_PROBLEM_SOLVING: "Complex Problem Solving",
    JUDGMENT_DECISION_MAKING: "Judgment and Decision Making",
    SYSTEMS_ANALYSIS: "Systems Analysis",
    SYSTEMS_EVALUATION: "Systems Evaluation",
    TROUBLESHOOTING: "Troubleshooting",
    EQUIPMENT_SELECTION: "Equipment Selection",
}

# ---------------------------------------------------------------------------
# O*NET skill descriptions (used by TF-IDF similarity engine)
# ---------------------------------------------------------------------------

ONET_SKILL_DESCRIPTIONS: Dict[str, str] = {
    READING_COMPREHENSION: (
        "Understanding written sentences and paragraphs in work-related "
        "documents. Reading reports, emails, manuals, and instructions."
    ),
    ACTIVE_LEARNING: (
        "Understanding the implications of new information for both current "
        "and future problem-solving and decision-making. Learning new skills "
        "quickly and adapting to change."
    ),
    LEARNING_STRATEGIES: (
        "Selecting and using training or instructional methods and procedures "
        "appropriate for the situation when learning or teaching new things."
    ),
    SERVICE_ORIENTATION: (
        "Actively looking for ways to help people. Providing excellent "
        "customer service. Understanding and meeting the needs of others."
    ),
    PROGRAMMING: (
        "Writing computer programs for various purposes. Coding software "
        "applications, scripting, and developing automated solutions."
    ),
    ACTIVE_LISTENING: (
        "Giving full attention to what other people are saying, taking time "
        "to understand the points being made, asking questions as appropriate, "
        "and not interrupting at inappropriate times."
    ),
    WRITING: (
        "Communicating effectively in writing as appropriate for the needs "
        "of the audience. Writing reports, memos, emails, and documentation."
    ),
    SCIENCE: (
        "Using scientific rules and methods to solve problems. Applying "
        "scientific knowledge and research methodology."
    ),
    SPEAKING: (
        "Talking to others to convey information effectively. Presenting "
        "information to groups, explaining ideas, and communicating clearly."
    ),
    MANAGEMENT_PERSONNEL: (
        "Motivating, developing, and directing people as they work, "
        "identifying the best people for the job. Managing teams, hiring, "
        "scheduling staff, and overseeing employee performance."
    ),
    MANAGEMENT_FINANCIAL: (
        "Determining how money will be spent to get the work done and "
        "accounting for these expenditures. Budgeting, financial planning, "
        "and tracking expenses."
    ),
    MONITORING: (
        "Monitoring and assessing performance of yourself, other individuals, "
        "or organizations to make improvements or take corrective action."
    ),
    EQUIPMENT_MAINTENANCE: (
        "Performing routine maintenance on equipment and determining when "
        "and what kind of maintenance is needed. Repairing machines and "
        "systems."
    ),
    MATHEMATICS: (
        "Using mathematics to solve problems. Calculating, counting, "
        "measuring, and applying mathematical formulas."
    ),
    DESIGN: (
        "Generating or adapting equipment and technology to serve user needs. "
        "Creating layouts, visual designs, blueprints, and prototypes."
    ),
    OPERATIONS_ANALYSIS: (
        "Analyzing needs and product requirements to create a design. "
        "Evaluating processes and systems for efficiency."
    ),
    TIME_MANAGEMENT: (
        "Managing one's own time and the time of others. Scheduling, "
        "prioritizing tasks, meeting deadlines, and organizing workload."
    ),
    SOCIAL_PERCEPTIVENESS: (
        "Being aware of others' reactions and understanding why they react "
        "as they do. Reading body language, empathy, and emotional awareness."
    ),
    COORDINATION: (
        "Adjusting actions in relation to others' actions. Coordinating "
        "work across teams, organizing group efforts, and logistics."
    ),
    PERSUASION: (
        "Persuading others to change their minds or behavior. Selling, "
        "convincing, influencing, and motivating others."
    ),
    NEGOTIATION: (
        "Bringing others together and trying to reconcile differences. "
        "Negotiating contracts, resolving disputes, and finding compromises."
    ),
    INSTRUCTING: (
        "Teaching others how to do something. Training new employees, "
        "mentoring, tutoring, and coaching."
    ),
    CRITICAL_THINKING: (
        "Using logic and reasoning to identify the strengths and weaknesses "
        "of alternative solutions, conclusions, or approaches to problems."
    ),
    COMPLEX_PROBLEM_SOLVING: (
        "Identifying complex problems and reviewing related information to "
        "develop and evaluate options and implement solutions."
    ),
    JUDGMENT_DECISION_MAKING: (
        "Considering the relative costs and benefits of potential actions "
        "to choose the most appropriate one. Making decisions under pressure."
    ),
    SYSTEMS_ANALYSIS: (
        "Determining how a system should work and how changes in conditions, "
        "operations, and the environment will affect outcomes."
    ),
    SYSTEMS_EVALUATION: (
        "Identifying measures or indicators of system performance and the "
        "actions needed to improve or correct performance relative to goals."
    ),
    TROUBLESHOOTING: (
        "Determining causes of operating errors and deciding what to do "
        "about it. Diagnosing problems and finding fixes."
    ),
    EQUIPMENT_SELECTION: (
        "Determining the kind of tools and equipment needed to do a job. "
        "Selecting the right materials and machinery."
    ),
}


def _m(element_id: str) -> SkillMapping:
    """Shorthand helper to build a SkillMapping from an element ID."""
    return {"element_id": element_id, "skill_name": ONET_SKILLS[element_id]}


# ---------------------------------------------------------------------------
# PHRASE DICTIONARY  (~200 everyday phrases -> O*NET skills)
#
# Organised loosely by domain so maintainers can add entries in context.
# All phrase keys MUST be lowercase.
# ---------------------------------------------------------------------------

PHRASE_TO_SKILL: Dict[str, SkillMapping] = {
    # =================================================================
    # CAREGIVING / STAY-AT-HOME PARENT
    # =================================================================
    "took care of kids": _m(SERVICE_ORIENTATION),
    "raised children": _m(SERVICE_ORIENTATION),
    "managed household": _m(TIME_MANAGEMENT),
    "ran the household": _m(TIME_MANAGEMENT),
    "managed a household budget": _m(MANAGEMENT_FINANCIAL),
    "managed the household budget": _m(MANAGEMENT_FINANCIAL),
    "household budget": _m(MANAGEMENT_FINANCIAL),
    "home budgeting": _m(MANAGEMENT_FINANCIAL),
    "family budgeting": _m(MANAGEMENT_FINANCIAL),
    "planned meals": _m(TIME_MANAGEMENT),
    "meal planning": _m(TIME_MANAGEMENT),
    "organized family schedule": _m(TIME_MANAGEMENT),
    "organized family schedules": _m(TIME_MANAGEMENT),
    "scheduled appointments": _m(TIME_MANAGEMENT),
    "coordinated schedules": _m(COORDINATION),
    "coordinated activities": _m(COORDINATION),
    "coordinated carpooling": _m(COORDINATION),
    "drove kids to school": _m(TIME_MANAGEMENT),
    "carpooling": _m(COORDINATION),
    "potty training": _m(INSTRUCTING),
    "taught kids to read": _m(INSTRUCTING),
    "helped with homework": _m(INSTRUCTING),
    "tutored my children": _m(INSTRUCTING),
    "homeschooled": _m(INSTRUCTING),
    "cared for elderly parent": _m(SERVICE_ORIENTATION),
    "elder care": _m(SERVICE_ORIENTATION),
    "caregiver": _m(SERVICE_ORIENTATION),
    "home health aide": _m(SERVICE_ORIENTATION),
    "patient care": _m(SERVICE_ORIENTATION),
    "managed medications": _m(MONITORING),
    "administered medicine": _m(MONITORING),
    "monitored health": _m(MONITORING),
    "listened to people's problems": _m(ACTIVE_LISTENING),
    "emotional support": _m(SOCIAL_PERCEPTIVENESS),
    "comforted others": _m(SOCIAL_PERCEPTIVENESS),
    "de-escalated tantrums": _m(NEGOTIATION),
    "mediated sibling fights": _m(NEGOTIATION),
    "resolved family conflicts": _m(NEGOTIATION),
    "read bedtime stories": _m(READING_COMPREHENSION),
    "researched schools": _m(CRITICAL_THINKING),
    "chose daycare": _m(JUDGMENT_DECISION_MAKING),

    # =================================================================
    # RETAIL / FOOD SERVICE
    # =================================================================
    "worked the register": _m(MATHEMATICS),
    "cashier": _m(MATHEMATICS),
    "handled cash": _m(MATHEMATICS),
    "counted the drawer": _m(MATHEMATICS),
    "rang up customers": _m(MATHEMATICS),
    "made change": _m(MATHEMATICS),
    "worked in retail": _m(SERVICE_ORIENTATION),
    "retail sales": _m(PERSUASION),
    "sold products": _m(PERSUASION),
    "upsold items": _m(PERSUASION),
    "met sales goals": _m(PERSUASION),
    "exceeded sales targets": _m(PERSUASION),
    "greeted customers": _m(SERVICE_ORIENTATION),
    "customer service": _m(SERVICE_ORIENTATION),
    "helped customers": _m(SERVICE_ORIENTATION),
    "answered customer questions": _m(SERVICE_ORIENTATION),
    "dealt with complaints": _m(NEGOTIATION),
    "handled complaints": _m(NEGOTIATION),
    "resolved customer issues": _m(COMPLEX_PROBLEM_SOLVING),
    "handled returns": _m(NEGOTIATION),
    "stocked shelves": _m(COORDINATION),
    "inventory management": _m(MONITORING),
    "managed inventory": _m(MONITORING),
    "did inventory counts": _m(MATHEMATICS),
    "ordered supplies": _m(OPERATIONS_ANALYSIS),
    "opened the store": _m(TIME_MANAGEMENT),
    "closed the store": _m(TIME_MANAGEMENT),
    "trained new hires": _m(INSTRUCTING),
    "trained employees": _m(INSTRUCTING),
    "shift leader": _m(MANAGEMENT_PERSONNEL),
    "shift manager": _m(MANAGEMENT_PERSONNEL),
    "assistant manager": _m(MANAGEMENT_PERSONNEL),
    "store manager": _m(MANAGEMENT_PERSONNEL),
    "managed a team": _m(MANAGEMENT_PERSONNEL),
    "supervised employees": _m(MANAGEMENT_PERSONNEL),
    "scheduled staff": _m(TIME_MANAGEMENT),
    "created work schedules": _m(TIME_MANAGEMENT),
    "food service": _m(SERVICE_ORIENTATION),
    "waited tables": _m(SERVICE_ORIENTATION),
    "server": _m(SERVICE_ORIENTATION),
    "bartender": _m(SERVICE_ORIENTATION),
    "cooked meals": _m(TIME_MANAGEMENT),
    "line cook": _m(TIME_MANAGEMENT),
    "fast food": _m(TIME_MANAGEMENT),
    "drive through": _m(SERVICE_ORIENTATION),

    # =================================================================
    # TRADES / MANUAL LABOR / CONSTRUCTION
    # =================================================================
    "fixed things around the house": _m(TROUBLESHOOTING),
    "home repairs": _m(TROUBLESHOOTING),
    "plumbing": _m(TROUBLESHOOTING),
    "electrical work": _m(TROUBLESHOOTING),
    "wiring": _m(TROUBLESHOOTING),
    "carpentry": _m(EQUIPMENT_SELECTION),
    "woodworking": _m(EQUIPMENT_SELECTION),
    "welding": _m(EQUIPMENT_SELECTION),
    "painting houses": _m(EQUIPMENT_SELECTION),
    "drywall": _m(EQUIPMENT_SELECTION),
    "roofing": _m(EQUIPMENT_SELECTION),
    "construction": _m(COORDINATION),
    "construction work": _m(COORDINATION),
    "built things": _m(DESIGN),
    "framing": _m(DESIGN),
    "blueprint reading": _m(READING_COMPREHENSION),
    "read blueprints": _m(READING_COMPREHENSION),
    "used power tools": _m(EQUIPMENT_SELECTION),
    "operated heavy equipment": _m(EQUIPMENT_SELECTION),
    "heavy machinery": _m(EQUIPMENT_SELECTION),
    "forklift operator": _m(EQUIPMENT_SELECTION),
    "fixed cars": _m(TROUBLESHOOTING),
    "auto mechanic": _m(TROUBLESHOOTING),
    "changed oil": _m(EQUIPMENT_MAINTENANCE),
    "brake repair": _m(EQUIPMENT_MAINTENANCE),
    "engine repair": _m(EQUIPMENT_MAINTENANCE),
    "vehicle maintenance": _m(EQUIPMENT_MAINTENANCE),
    "hvac": _m(TROUBLESHOOTING),
    "heating and cooling": _m(TROUBLESHOOTING),
    "appliance repair": _m(TROUBLESHOOTING),
    "landscaping": _m(EQUIPMENT_SELECTION),
    "lawn care": _m(EQUIPMENT_MAINTENANCE),
    "measured materials": _m(MATHEMATICS),
    "estimated costs": _m(MANAGEMENT_FINANCIAL),
    "quoted jobs": _m(MANAGEMENT_FINANCIAL),
    "safety inspections": _m(MONITORING),

    # =================================================================
    # ADMINISTRATIVE / OFFICE
    # =================================================================
    "answered phones": _m(ACTIVE_LISTENING),
    "receptionist": _m(SERVICE_ORIENTATION),
    "front desk": _m(SERVICE_ORIENTATION),
    "filed paperwork": _m(READING_COMPREHENSION),
    "data entry": _m(READING_COMPREHENSION),
    "typed documents": _m(WRITING),
    "wrote reports": _m(WRITING),
    "wrote emails": _m(WRITING),
    "drafted correspondence": _m(WRITING),
    "took meeting notes": _m(WRITING),
    "took minutes": _m(WRITING),
    "organized files": _m(TIME_MANAGEMENT),
    "office management": _m(COORDINATION),
    "office manager": _m(COORDINATION),
    "handled scheduling": _m(TIME_MANAGEMENT),
    "booked appointments": _m(TIME_MANAGEMENT),
    "used spreadsheets": _m(MATHEMATICS),
    "excel": _m(MATHEMATICS),
    "made presentations": _m(SPEAKING),
    "gave presentations": _m(SPEAKING),
    "public speaking": _m(SPEAKING),
    "led meetings": _m(SPEAKING),
    "ran meetings": _m(SPEAKING),
    "project management": _m(COORDINATION),
    "managed projects": _m(COORDINATION),
    "tracked deadlines": _m(TIME_MANAGEMENT),
    "managed budgets": _m(MANAGEMENT_FINANCIAL),
    "processed invoices": _m(MANAGEMENT_FINANCIAL),
    "bookkeeping": _m(MANAGEMENT_FINANCIAL),
    "accounting": _m(MANAGEMENT_FINANCIAL),
    "payroll": _m(MANAGEMENT_FINANCIAL),
    "hiring": _m(MANAGEMENT_PERSONNEL),
    "recruited employees": _m(MANAGEMENT_PERSONNEL),
    "onboarded new employees": _m(INSTRUCTING),
    "conducted interviews": _m(JUDGMENT_DECISION_MAKING),
    "performance reviews": _m(MONITORING),
    "evaluated employees": _m(MONITORING),

    # =================================================================
    # MILITARY / VETERAN
    # =================================================================
    "military service": _m(COORDINATION),
    "served in the military": _m(COORDINATION),
    "army": _m(COORDINATION),
    "navy": _m(COORDINATION),
    "marines": _m(COORDINATION),
    "air force": _m(COORDINATION),
    "led a squad": _m(MANAGEMENT_PERSONNEL),
    "led a platoon": _m(MANAGEMENT_PERSONNEL),
    "team leader": _m(MANAGEMENT_PERSONNEL),
    "squad leader": _m(MANAGEMENT_PERSONNEL),
    "platoon sergeant": _m(MANAGEMENT_PERSONNEL),
    "commanded troops": _m(MANAGEMENT_PERSONNEL),
    "mission planning": _m(SYSTEMS_ANALYSIS),
    "mission briefing": _m(SPEAKING),
    "tactical planning": _m(SYSTEMS_ANALYSIS),
    "strategic planning": _m(SYSTEMS_ANALYSIS),
    "logistics": _m(COORDINATION),
    "supply chain": _m(COORDINATION),
    "inventory control": _m(MONITORING),
    "maintained equipment": _m(EQUIPMENT_MAINTENANCE),
    "weapons maintenance": _m(EQUIPMENT_MAINTENANCE),
    "vehicle mechanic": _m(EQUIPMENT_MAINTENANCE),
    "radio communications": _m(SPEAKING),
    "field medic": _m(SERVICE_ORIENTATION),
    "combat medic": _m(SERVICE_ORIENTATION),
    "first aid": _m(SERVICE_ORIENTATION),
    "security operations": _m(MONITORING),
    "risk assessment": _m(JUDGMENT_DECISION_MAKING),
    "intelligence analysis": _m(CRITICAL_THINKING),
    "wrote after-action reports": _m(WRITING),
    "situation reports": _m(WRITING),
    "trained soldiers": _m(INSTRUCTING),
    "drill instructor": _m(INSTRUCTING),
    "followed standard operating procedures": _m(READING_COMPREHENSION),
    "decision making under pressure": _m(JUDGMENT_DECISION_MAKING),
    "made split-second decisions": _m(JUDGMENT_DECISION_MAKING),

    # =================================================================
    # GIG ECONOMY / FREELANCE / SELF-EMPLOYED
    # =================================================================
    "drove for uber": _m(SERVICE_ORIENTATION),
    "drove for lyft": _m(SERVICE_ORIENTATION),
    "rideshare driver": _m(SERVICE_ORIENTATION),
    "delivery driver": _m(TIME_MANAGEMENT),
    "doordash": _m(TIME_MANAGEMENT),
    "grubhub": _m(TIME_MANAGEMENT),
    "instacart": _m(TIME_MANAGEMENT),
    "amazon delivery": _m(TIME_MANAGEMENT),
    "ran my own business": _m(MANAGEMENT_FINANCIAL),
    "self-employed": _m(MANAGEMENT_FINANCIAL),
    "freelance": _m(TIME_MANAGEMENT),
    "freelancer": _m(TIME_MANAGEMENT),
    "gig work": _m(TIME_MANAGEMENT),
    "side hustle": _m(TIME_MANAGEMENT),
    "sold on etsy": _m(PERSUASION),
    "sold online": _m(PERSUASION),
    "ebay seller": _m(PERSUASION),
    "social media marketing": _m(PERSUASION),
    "managed social media": _m(WRITING),
    "content creation": _m(WRITING),
    "created content": _m(WRITING),
    "graphic design": _m(DESIGN),
    "designed logos": _m(DESIGN),
    "built websites": _m(PROGRAMMING),
    "web design": _m(DESIGN),
    "photography": _m(DESIGN),
    "videography": _m(DESIGN),
    "edited videos": _m(DESIGN),
    "tax preparation": _m(MATHEMATICS),
    "did my own taxes": _m(MATHEMATICS),
    "negotiated rates": _m(NEGOTIATION),
    "negotiated contracts": _m(NEGOTIATION),
    "found my own clients": _m(PERSUASION),
    "marketed my services": _m(PERSUASION),

    # =================================================================
    # CHURCH / COMMUNITY VOLUNTEER
    # =================================================================
    "church volunteer": _m(SERVICE_ORIENTATION),
    "volunteered at church": _m(SERVICE_ORIENTATION),
    "sunday school teacher": _m(INSTRUCTING),
    "taught sunday school": _m(INSTRUCTING),
    "youth group leader": _m(INSTRUCTING),
    "organized events": _m(COORDINATION),
    "planned events": _m(COORDINATION),
    "event planning": _m(COORDINATION),
    "fundraising": _m(PERSUASION),
    "raised money": _m(PERSUASION),
    "grant writing": _m(WRITING),
    "pta president": _m(MANAGEMENT_PERSONNEL),
    "pta volunteer": _m(COORDINATION),
    "coached sports": _m(INSTRUCTING),
    "coached little league": _m(INSTRUCTING),
    "mentored youth": _m(INSTRUCTING),
    "mentored others": _m(INSTRUCTING),
    "community outreach": _m(SOCIAL_PERCEPTIVENESS),
    "food bank volunteer": _m(SERVICE_ORIENTATION),
    "soup kitchen": _m(SERVICE_ORIENTATION),
    "habitat for humanity": _m(COORDINATION),
    "disaster relief": _m(COORDINATION),
    "led a committee": _m(MANAGEMENT_PERSONNEL),
    "chaired a committee": _m(MANAGEMENT_PERSONNEL),
    "board member": _m(JUDGMENT_DECISION_MAKING),
    "nonprofit board": _m(JUDGMENT_DECISION_MAKING),
    "organized volunteers": _m(MANAGEMENT_PERSONNEL),
    "recruited volunteers": _m(PERSUASION),
    "counseled people": _m(SOCIAL_PERCEPTIVENESS),
    "peer counseling": _m(ACTIVE_LISTENING),

    # =================================================================
    # TECHNOLOGY / GENERAL
    # =================================================================
    "used computers": _m(ACTIVE_LEARNING),
    "learned new software": _m(ACTIVE_LEARNING),
    "picked up new tools quickly": _m(ACTIVE_LEARNING),
    "quick learner": _m(ACTIVE_LEARNING),
    "fast learner": _m(ACTIVE_LEARNING),
    "self-taught": _m(LEARNING_STRATEGIES),
    "took online courses": _m(LEARNING_STRATEGIES),
    "watched tutorials": _m(LEARNING_STRATEGIES),
    "figured things out on my own": _m(COMPLEX_PROBLEM_SOLVING),
    "problem solver": _m(COMPLEX_PROBLEM_SOLVING),
    "troubleshot issues": _m(TROUBLESHOOTING),
    "fixed computer problems": _m(TROUBLESHOOTING),
    "tech support": _m(TROUBLESHOOTING),
    "helped people with technology": _m(INSTRUCTING),

    # =================================================================
    # GENERAL / CROSS-DOMAIN
    # =================================================================
    "good with people": _m(SOCIAL_PERCEPTIVENESS),
    "people person": _m(SOCIAL_PERCEPTIVENESS),
    "team player": _m(COORDINATION),
    "worked in a team": _m(COORDINATION),
    "worked well with others": _m(COORDINATION),
    "multitasked": _m(TIME_MANAGEMENT),
    "juggled multiple responsibilities": _m(TIME_MANAGEMENT),
    "stayed organized": _m(TIME_MANAGEMENT),
    "detail oriented": _m(MONITORING),
    "attention to detail": _m(MONITORING),
    "analyzed data": _m(CRITICAL_THINKING),
    "solved problems": _m(COMPLEX_PROBLEM_SOLVING),
    "made decisions": _m(JUDGMENT_DECISION_MAKING),
    "kept records": _m(WRITING),
    "documentation": _m(WRITING),
    "read manuals": _m(READING_COMPREHENSION),
    "followed instructions": _m(READING_COMPREHENSION),
    "explained things to others": _m(SPEAKING),
    "taught others": _m(INSTRUCTING),
    "trained people": _m(INSTRUCTING),
    "supervised others": _m(MANAGEMENT_PERSONNEL),
    "managed people": _m(MANAGEMENT_PERSONNEL),
    "handled money": _m(MANAGEMENT_FINANCIAL),
    "managed a budget": _m(MANAGEMENT_FINANCIAL),
    "fixed things": _m(TROUBLESHOOTING),
    "repaired equipment": _m(EQUIPMENT_MAINTENANCE),
    "maintained equipment": _m(EQUIPMENT_MAINTENANCE),
    "selected tools": _m(EQUIPMENT_SELECTION),
    "chose the right equipment": _m(EQUIPMENT_SELECTION),
    "designed systems": _m(SYSTEMS_ANALYSIS),
    "evaluated performance": _m(SYSTEMS_EVALUATION),
    "quality control": _m(SYSTEMS_EVALUATION),
    "quality assurance": _m(SYSTEMS_EVALUATION),
}


def get_all_skill_ids() -> set:
    """Return the set of all O*NET element IDs referenced in the dictionary.

    Returns:
        Set of unique element ID strings.
    """
    return {entry["element_id"] for entry in PHRASE_TO_SKILL.values()}


def get_phrases_for_skill(element_id: str) -> list:
    """Return all phrases that map to a given O*NET element ID.

    Args:
        element_id: O*NET skill element ID (e.g. "2.B.1.a").

    Returns:
        List of phrase strings.
    """
    return [
        phrase
        for phrase, mapping in PHRASE_TO_SKILL.items()
        if mapping["element_id"] == element_id
    ]
