"""Training resource catalog with real-world programs.

Contains a curated catalog of 40+ training resources mapped to O*NET skill
codes. Resources span the full spectrum of accessibility:

    - Free certificates (Google, IBM, Meta via Coursera)
    - Community college programs
    - Government programs (WIOA, TAA, veteran benefits)
    - Bootcamps (paid and free)
    - Self-directed platforms (freeCodeCamp, Khan Academy, MIT OCW)
    - Library and community resources (for no-computer users)

Each resource is tagged with:
    - Cost tier and estimated cost
    - Time commitment (hours/week, total weeks)
    - Delivery format (online, in-person, hybrid, self-paced)
    - Computer/internet requirements
    - O*NET skill codes it addresses
    - Prerequisites
"""
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ==================== Enums ====================

class CostTier(str, Enum):
    """Cost classification for training resources."""
    FREE = "free"
    LOW = "low"            # Under $500
    MODERATE = "moderate"  # $500 - $5,000
    HIGH = "high"          # Over $5,000


class DeliveryFormat(str, Enum):
    """How the training is delivered."""
    ONLINE_SELF_PACED = "online_self_paced"
    ONLINE_COHORT = "online_cohort"
    IN_PERSON = "in_person"
    HYBRID = "hybrid"
    OFFLINE_MATERIALS = "offline_materials"  # Books, printed materials


class ResourceCategory(str, Enum):
    """Category of training resource."""
    FREE_CERTIFICATE = "free_certificate"
    COMMUNITY_COLLEGE = "community_college"
    GOVERNMENT_PROGRAM = "government_program"
    BOOTCAMP = "bootcamp"
    SELF_DIRECTED = "self_directed"
    LIBRARY_COMMUNITY = "library_community"
    APPRENTICESHIP = "apprenticeship"
    EMPLOYER_SPONSORED = "employer_sponsored"


# ==================== O*NET Skill Code Constants ====================

# Common O*NET element IDs for skill mapping
SKILL_CODES = {
    "programming": "2.B.3.e",
    "critical_thinking": "2.A.2.a",
    "reading_comprehension": "2.A.1.a",
    "active_listening": "2.A.1.b",
    "writing": "2.A.1.c",
    "speaking": "2.A.1.d",
    "mathematics": "2.A.1.e",
    "science": "2.A.1.f",
    "active_learning": "2.A.2.b",
    "monitoring": "2.A.2.d",
    "social_perceptiveness": "2.B.1.a",
    "coordination": "2.B.1.b",
    "persuasion": "2.B.1.c",
    "negotiation": "2.B.1.d",
    "instructing": "2.B.1.e",
    "service_orientation": "2.B.1.f",
    "complex_problem_solving": "2.B.2.i",
    "operations_analysis": "2.B.3.a",
    "technology_design": "2.B.3.b",
    "equipment_selection": "2.B.3.c",
    "installation": "2.B.3.d",
    "quality_control": "2.B.3.g",
    "operations_monitoring": "2.B.3.h",
    "troubleshooting": "2.B.3.k",
    "repairing": "2.B.3.l",
    "systems_analysis": "2.B.4.e",
    "systems_evaluation": "2.B.4.f",
    "judgment_decision_making": "2.B.4.g",
    "time_management": "2.B.4.h",
    "management_financial": "2.B.5.a",
    "management_material": "2.B.5.b",
    "management_personnel": "2.B.5.c",
}


# ==================== Resource Model ====================

class TrainingResource(BaseModel):
    """A single training resource in the catalog.

    Attributes:
        id: Unique identifier for the resource.
        name: Human-readable name of the program.
        provider: Organization offering the resource.
        url: Link to the resource (empty for in-person/offline).
        description: Brief description of what the resource covers.
        category: Resource category classification.
        cost_tier: Cost classification.
        estimated_cost_usd: Estimated total cost in USD (0 for free).
        delivery_format: How the training is delivered.
        hours_per_week: Expected weekly time commitment.
        total_weeks: Expected total duration in weeks.
        requires_computer: Whether a personal computer is needed.
        requires_internet: Whether internet access is needed.
        skill_codes: Set of O*NET element IDs this resource addresses.
        skill_names: Human-readable skill names for display.
        prerequisites: List of prerequisite skill codes or descriptions.
        credential_awarded: Type of credential (if any).
        geographic_scope: national, state, or local availability.
        notes: Additional notes or eligibility requirements.
    """
    id: str = Field(..., description="Unique resource identifier")
    name: str = Field(..., description="Resource name")
    provider: str = Field(..., description="Providing organization")
    url: str = Field("", description="Resource URL")
    description: str = Field(..., description="Brief description")
    category: ResourceCategory
    cost_tier: CostTier
    estimated_cost_usd: float = Field(0.0, ge=0.0)
    delivery_format: DeliveryFormat
    hours_per_week: float = Field(..., gt=0.0, le=80.0)
    total_weeks: int = Field(..., gt=0, le=520)
    requires_computer: bool = True
    requires_internet: bool = True
    skill_codes: List[str] = Field(default_factory=list)
    skill_names: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    credential_awarded: Optional[str] = None
    geographic_scope: str = "national"
    notes: str = ""

    class Config:
        use_enum_values = True


# ==================== Catalog Data ====================

def _build_catalog() -> List[TrainingResource]:
    """Build the full training resource catalog.

    Returns:
        List of all TrainingResource entries.
    """
    resources = [
        # ============================================================
        # FREE CERTIFICATES - Google
        # ============================================================
        TrainingResource(
            id="google-it-support",
            name="Google IT Support Professional Certificate",
            provider="Google (via Coursera)",
            url="https://www.coursera.org/professional-certificates/google-it-support",
            description="Entry-level IT support skills including troubleshooting, networking, operating systems, system administration, and security.",
            category=ResourceCategory.FREE_CERTIFICATE,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=10.0,
            total_weeks=24,
            skill_codes=[SKILL_CODES["troubleshooting"], SKILL_CODES["complex_problem_solving"], SKILL_CODES["technology_design"], SKILL_CODES["operations_monitoring"]],
            skill_names=["Troubleshooting", "Complex Problem Solving", "Technology Design", "Operations Monitoring"],
            credential_awarded="Professional Certificate",
            notes="Financial aid available on Coursera. No prerequisites.",
        ),
        TrainingResource(
            id="google-data-analytics",
            name="Google Data Analytics Professional Certificate",
            provider="Google (via Coursera)",
            url="https://www.coursera.org/professional-certificates/google-data-analytics",
            description="Foundations of data analytics including spreadsheets, SQL, R programming, Tableau, and data visualization.",
            category=ResourceCategory.FREE_CERTIFICATE,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=10.0,
            total_weeks=24,
            skill_codes=[SKILL_CODES["mathematics"], SKILL_CODES["critical_thinking"], SKILL_CODES["programming"], SKILL_CODES["systems_analysis"]],
            skill_names=["Mathematics", "Critical Thinking", "Programming", "Systems Analysis"],
            credential_awarded="Professional Certificate",
            notes="Financial aid available. No prior experience required.",
        ),
        TrainingResource(
            id="google-project-management",
            name="Google Project Management Professional Certificate",
            provider="Google (via Coursera)",
            url="https://www.coursera.org/professional-certificates/google-project-management",
            description="Project management fundamentals including Agile, Scrum, risk management, stakeholder communication, and project planning.",
            category=ResourceCategory.FREE_CERTIFICATE,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=10.0,
            total_weeks=24,
            skill_codes=[SKILL_CODES["coordination"], SKILL_CODES["time_management"], SKILL_CODES["management_personnel"], SKILL_CODES["judgment_decision_making"]],
            skill_names=["Coordination", "Time Management", "Management of Personnel Resources", "Judgment and Decision Making"],
            credential_awarded="Professional Certificate",
            notes="Financial aid available. No prior experience required.",
        ),
        TrainingResource(
            id="google-ux-design",
            name="Google UX Design Professional Certificate",
            provider="Google (via Coursera)",
            url="https://www.coursera.org/professional-certificates/google-ux-design",
            description="UX design foundations including wireframing, prototyping, user research, and usability testing.",
            category=ResourceCategory.FREE_CERTIFICATE,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=10.0,
            total_weeks=28,
            skill_codes=[SKILL_CODES["technology_design"], SKILL_CODES["active_listening"], SKILL_CODES["critical_thinking"], SKILL_CODES["social_perceptiveness"]],
            skill_names=["Technology Design", "Active Listening", "Critical Thinking", "Social Perceptiveness"],
            credential_awarded="Professional Certificate",
            notes="Financial aid available. No prior experience required.",
        ),
        TrainingResource(
            id="google-cybersecurity",
            name="Google Cybersecurity Professional Certificate",
            provider="Google (via Coursera)",
            url="https://www.coursera.org/professional-certificates/google-cybersecurity",
            description="Entry-level cybersecurity skills including network security, Linux, Python, SIEM tools, and incident detection.",
            category=ResourceCategory.FREE_CERTIFICATE,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=10.0,
            total_weeks=24,
            skill_codes=[SKILL_CODES["programming"], SKILL_CODES["operations_monitoring"], SKILL_CODES["troubleshooting"], SKILL_CODES["systems_analysis"]],
            skill_names=["Programming", "Operations Monitoring", "Troubleshooting", "Systems Analysis"],
            credential_awarded="Professional Certificate",
            notes="Financial aid available. No prior experience required.",
        ),

        # ============================================================
        # FREE CERTIFICATES - IBM
        # ============================================================
        TrainingResource(
            id="ibm-data-science",
            name="IBM Data Science Professional Certificate",
            provider="IBM (via Coursera)",
            url="https://www.coursera.org/professional-certificates/ibm-data-science",
            description="Data science methodology, Python, SQL, data visualization, machine learning, and applied data science capstone.",
            category=ResourceCategory.FREE_CERTIFICATE,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=10.0,
            total_weeks=40,
            skill_codes=[SKILL_CODES["programming"], SKILL_CODES["mathematics"], SKILL_CODES["critical_thinking"], SKILL_CODES["systems_analysis"]],
            skill_names=["Programming", "Mathematics", "Critical Thinking", "Systems Analysis"],
            credential_awarded="Professional Certificate",
            notes="Financial aid available. Basic computer literacy recommended.",
        ),
        TrainingResource(
            id="ibm-full-stack",
            name="IBM Full Stack Software Developer Professional Certificate",
            provider="IBM (via Coursera)",
            url="https://www.coursera.org/professional-certificates/ibm-full-stack-cloud-developer",
            description="Full stack development with HTML, CSS, JavaScript, React, Node.js, Python, Django, and cloud deployment.",
            category=ResourceCategory.FREE_CERTIFICATE,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=10.0,
            total_weeks=48,
            skill_codes=[SKILL_CODES["programming"], SKILL_CODES["technology_design"], SKILL_CODES["complex_problem_solving"], SKILL_CODES["systems_analysis"]],
            skill_names=["Programming", "Technology Design", "Complex Problem Solving", "Systems Analysis"],
            credential_awarded="Professional Certificate",
            notes="Financial aid available. No prior programming experience required.",
        ),

        # ============================================================
        # FREE CERTIFICATES - Meta
        # ============================================================
        TrainingResource(
            id="meta-front-end",
            name="Meta Front-End Developer Professional Certificate",
            provider="Meta (via Coursera)",
            url="https://www.coursera.org/professional-certificates/meta-front-end-developer",
            description="Front-end development with HTML, CSS, JavaScript, React, version control, and UX/UI design principles.",
            category=ResourceCategory.FREE_CERTIFICATE,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=7.0,
            total_weeks=32,
            skill_codes=[SKILL_CODES["programming"], SKILL_CODES["technology_design"], SKILL_CODES["complex_problem_solving"]],
            skill_names=["Programming", "Technology Design", "Complex Problem Solving"],
            credential_awarded="Professional Certificate",
            notes="Financial aid available. No prior experience required.",
        ),
        TrainingResource(
            id="meta-back-end",
            name="Meta Back-End Developer Professional Certificate",
            provider="Meta (via Coursera)",
            url="https://www.coursera.org/professional-certificates/meta-back-end-developer",
            description="Back-end development with Python, Django, databases, REST APIs, and cloud deployment.",
            category=ResourceCategory.FREE_CERTIFICATE,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=7.0,
            total_weeks=32,
            skill_codes=[SKILL_CODES["programming"], SKILL_CODES["systems_analysis"], SKILL_CODES["complex_problem_solving"], SKILL_CODES["operations_analysis"]],
            skill_names=["Programming", "Systems Analysis", "Complex Problem Solving", "Operations Analysis"],
            credential_awarded="Professional Certificate",
            notes="Financial aid available. Basic Python knowledge helpful but not required.",
        ),

        # ============================================================
        # SELF-DIRECTED / FREE PLATFORMS
        # ============================================================
        TrainingResource(
            id="freecodecamp-web",
            name="freeCodeCamp Responsive Web Design + JavaScript",
            provider="freeCodeCamp",
            url="https://www.freecodecamp.org",
            description="Free, self-paced curriculum covering HTML, CSS, JavaScript, algorithms, data structures, front-end and back-end development.",
            category=ResourceCategory.SELF_DIRECTED,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=10.0,
            total_weeks=40,
            skill_codes=[SKILL_CODES["programming"], SKILL_CODES["technology_design"], SKILL_CODES["complex_problem_solving"], SKILL_CODES["mathematics"]],
            skill_names=["Programming", "Technology Design", "Complex Problem Solving", "Mathematics"],
            credential_awarded="Verified Certification",
            notes="Completely free. Project-based learning with portfolio pieces.",
        ),
        TrainingResource(
            id="khan-academy-math",
            name="Khan Academy Mathematics",
            provider="Khan Academy",
            url="https://www.khanacademy.org/math",
            description="Comprehensive math curriculum from arithmetic through statistics, linear algebra, and calculus.",
            category=ResourceCategory.SELF_DIRECTED,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=5.0,
            total_weeks=52,
            skill_codes=[SKILL_CODES["mathematics"], SKILL_CODES["critical_thinking"], SKILL_CODES["active_learning"]],
            skill_names=["Mathematics", "Critical Thinking", "Active Learning"],
            notes="Completely free. Adaptive learning with mastery-based progression.",
        ),
        TrainingResource(
            id="khan-academy-computing",
            name="Khan Academy Computing",
            provider="Khan Academy",
            url="https://www.khanacademy.org/computing",
            description="Introduction to computer science, programming with JavaScript, SQL, and HTML/CSS.",
            category=ResourceCategory.SELF_DIRECTED,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=5.0,
            total_weeks=24,
            skill_codes=[SKILL_CODES["programming"], SKILL_CODES["critical_thinking"], SKILL_CODES["complex_problem_solving"]],
            skill_names=["Programming", "Critical Thinking", "Complex Problem Solving"],
            notes="Completely free. Interactive coding exercises.",
        ),
        TrainingResource(
            id="mit-ocw-intro-cs",
            name="MIT OpenCourseWare - Introduction to Computer Science",
            provider="MIT OpenCourseWare",
            url="https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/",
            description="MIT's introductory computer science course using Python. Covers computational thinking, algorithms, and data structures.",
            category=ResourceCategory.SELF_DIRECTED,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=12.0,
            total_weeks=16,
            skill_codes=[SKILL_CODES["programming"], SKILL_CODES["mathematics"], SKILL_CODES["critical_thinking"], SKILL_CODES["complex_problem_solving"]],
            skill_names=["Programming", "Mathematics", "Critical Thinking", "Complex Problem Solving"],
            notes="Free. University-level rigor. No certificate but high-quality content.",
        ),
        TrainingResource(
            id="osha-safety",
            name="OSHA Outreach Training Program",
            provider="OSHA (Department of Labor)",
            url="https://www.osha.gov/training/outreach",
            description="Workplace safety training for construction and general industry. 10-hour and 30-hour courses available.",
            category=ResourceCategory.SELF_DIRECTED,
            cost_tier=CostTier.LOW,
            estimated_cost_usd=25.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=10.0,
            total_weeks=1,
            skill_codes=[SKILL_CODES["monitoring"], SKILL_CODES["quality_control"], SKILL_CODES["operations_monitoring"]],
            skill_names=["Monitoring", "Quality Control Analysis", "Operations Monitoring"],
            credential_awarded="OSHA Card",
            notes="Required for many construction and industrial jobs.",
        ),

        # ============================================================
        # GOVERNMENT PROGRAMS
        # ============================================================
        TrainingResource(
            id="wioa-adult",
            name="WIOA Adult Training Program",
            provider="U.S. Department of Labor (via American Job Centers)",
            url="https://www.dol.gov/agencies/eta/wioa",
            description="Workforce Innovation and Opportunity Act funding for occupational training, on-the-job training, and supportive services for adults.",
            category=ResourceCategory.GOVERNMENT_PROGRAM,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.IN_PERSON,
            hours_per_week=20.0,
            total_weeks=52,
            requires_computer=False,
            requires_internet=False,
            skill_codes=[SKILL_CODES["active_learning"], SKILL_CODES["critical_thinking"], SKILL_CODES["service_orientation"]],
            skill_names=["Active Learning", "Critical Thinking", "Service Orientation"],
            credential_awarded="Varies by program (industry-recognized credentials)",
            geographic_scope="national",
            notes="Eligibility based on income, employment status. Apply at local American Job Center.",
        ),
        TrainingResource(
            id="wioa-dislocated",
            name="WIOA Dislocated Worker Program",
            provider="U.S. Department of Labor (via American Job Centers)",
            url="https://www.dol.gov/agencies/eta/wioa",
            description="Training and employment services for workers who have been laid off, including career counseling, skills assessment, and occupational training.",
            category=ResourceCategory.GOVERNMENT_PROGRAM,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.IN_PERSON,
            hours_per_week=20.0,
            total_weeks=52,
            requires_computer=False,
            requires_internet=False,
            skill_codes=[SKILL_CODES["active_learning"], SKILL_CODES["critical_thinking"], SKILL_CODES["complex_problem_solving"]],
            skill_names=["Active Learning", "Critical Thinking", "Complex Problem Solving"],
            credential_awarded="Varies by program",
            geographic_scope="national",
            notes="For recently laid-off workers. Apply at local American Job Center. May include stipends.",
        ),
        TrainingResource(
            id="taa-program",
            name="Trade Adjustment Assistance (TAA)",
            provider="U.S. Department of Labor",
            url="https://www.dol.gov/agencies/eta/tradeact",
            description="Training, job search allowances, relocation allowances, and income support for workers displaced by foreign trade.",
            category=ResourceCategory.GOVERNMENT_PROGRAM,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.IN_PERSON,
            hours_per_week=20.0,
            total_weeks=104,
            requires_computer=False,
            requires_internet=False,
            skill_codes=[SKILL_CODES["active_learning"], SKILL_CODES["critical_thinking"]],
            skill_names=["Active Learning", "Critical Thinking"],
            credential_awarded="Varies (up to associate degree)",
            geographic_scope="national",
            notes="Must be affected by foreign trade. Provides income support during training.",
        ),
        TrainingResource(
            id="gi-bill",
            name="GI Bill Education Benefits",
            provider="U.S. Department of Veterans Affairs",
            url="https://www.va.gov/education/about-gi-bill-benefits/",
            description="Education benefits for veterans covering tuition, fees, books, and housing for degree programs, vocational training, and certifications.",
            category=ResourceCategory.GOVERNMENT_PROGRAM,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.IN_PERSON,
            hours_per_week=20.0,
            total_weeks=156,
            requires_computer=False,
            requires_internet=False,
            skill_codes=[SKILL_CODES["active_learning"], SKILL_CODES["critical_thinking"], SKILL_CODES["complex_problem_solving"]],
            skill_names=["Active Learning", "Critical Thinking", "Complex Problem Solving"],
            credential_awarded="Varies (certificates through doctoral degrees)",
            geographic_scope="national",
            notes="For eligible veterans and service members. Includes housing allowance.",
        ),
        TrainingResource(
            id="vet-tec",
            name="VET TEC (Veteran Employment Through Technology Education Courses)",
            provider="U.S. Department of Veterans Affairs",
            url="https://www.va.gov/education/about-gi-bill-benefits/how-to-use-benefits/vettec-high-tech-program/",
            description="Fast-track technology training for veterans including coding bootcamps, data science, and IT operations.",
            category=ResourceCategory.GOVERNMENT_PROGRAM,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.HYBRID,
            hours_per_week=40.0,
            total_weeks=20,
            skill_codes=[SKILL_CODES["programming"], SKILL_CODES["technology_design"], SKILL_CODES["complex_problem_solving"], SKILL_CODES["systems_analysis"]],
            skill_names=["Programming", "Technology Design", "Complex Problem Solving", "Systems Analysis"],
            credential_awarded="Program completion certificate",
            geographic_scope="national",
            notes="For veterans with remaining GI Bill entitlement. Includes housing allowance.",
        ),
        TrainingResource(
            id="pell-grant",
            name="Federal Pell Grant (Community College)",
            provider="U.S. Department of Education",
            url="https://studentaid.gov/understand-aid/types/grants/pell",
            description="Need-based federal grant covering tuition at community colleges for certificate and associate degree programs.",
            category=ResourceCategory.GOVERNMENT_PROGRAM,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.IN_PERSON,
            hours_per_week=15.0,
            total_weeks=78,
            requires_computer=False,
            requires_internet=False,
            skill_codes=[SKILL_CODES["active_learning"], SKILL_CODES["critical_thinking"], SKILL_CODES["reading_comprehension"], SKILL_CODES["writing"]],
            skill_names=["Active Learning", "Critical Thinking", "Reading Comprehension", "Writing"],
            credential_awarded="Associate degree or certificate",
            geographic_scope="national",
            notes="Income-based eligibility. Apply via FAFSA. Does not need to be repaid.",
        ),

        # ============================================================
        # COMMUNITY COLLEGE PROGRAMS
        # ============================================================
        TrainingResource(
            id="cc-nursing-assistant",
            name="Certified Nursing Assistant (CNA) Program",
            provider="Community Colleges (nationwide)",
            url="",
            description="State-approved CNA training program covering patient care, vital signs, infection control, and clinical practice.",
            category=ResourceCategory.COMMUNITY_COLLEGE,
            cost_tier=CostTier.LOW,
            estimated_cost_usd=400.0,
            delivery_format=DeliveryFormat.IN_PERSON,
            hours_per_week=20.0,
            total_weeks=8,
            requires_computer=False,
            requires_internet=False,
            skill_codes=[SKILL_CODES["service_orientation"], SKILL_CODES["monitoring"], SKILL_CODES["social_perceptiveness"], SKILL_CODES["active_listening"]],
            skill_names=["Service Orientation", "Monitoring", "Social Perceptiveness", "Active Listening"],
            credential_awarded="CNA Certification",
            notes="Often funded by WIOA or employer sponsorship. High demand occupation.",
        ),
        TrainingResource(
            id="cc-welding",
            name="Welding Certificate Program",
            provider="Community Colleges (nationwide)",
            url="",
            description="Hands-on welding training covering MIG, TIG, stick welding, blueprint reading, and safety procedures.",
            category=ResourceCategory.COMMUNITY_COLLEGE,
            cost_tier=CostTier.MODERATE,
            estimated_cost_usd=3000.0,
            delivery_format=DeliveryFormat.IN_PERSON,
            hours_per_week=20.0,
            total_weeks=24,
            requires_computer=False,
            requires_internet=False,
            skill_codes=[SKILL_CODES["equipment_selection"], SKILL_CODES["quality_control"], SKILL_CODES["operations_monitoring"], SKILL_CODES["troubleshooting"]],
            skill_names=["Equipment Selection", "Quality Control Analysis", "Operations Monitoring", "Troubleshooting"],
            credential_awarded="Welding Certificate (AWS certifications available)",
            notes="Often Pell Grant eligible. Strong job placement rates.",
        ),
        TrainingResource(
            id="cc-hvac",
            name="HVAC Technician Certificate",
            provider="Community Colleges (nationwide)",
            url="",
            description="Heating, ventilation, and air conditioning technology including refrigeration, electrical systems, and EPA certification prep.",
            category=ResourceCategory.COMMUNITY_COLLEGE,
            cost_tier=CostTier.MODERATE,
            estimated_cost_usd=4000.0,
            delivery_format=DeliveryFormat.IN_PERSON,
            hours_per_week=20.0,
            total_weeks=36,
            requires_computer=False,
            requires_internet=False,
            skill_codes=[SKILL_CODES["installation"], SKILL_CODES["repairing"], SKILL_CODES["troubleshooting"], SKILL_CODES["operations_monitoring"]],
            skill_names=["Installation", "Repairing", "Troubleshooting", "Operations Monitoring"],
            credential_awarded="HVAC Certificate + EPA 608 Certification",
            notes="Pell Grant eligible at most institutions. Year-round demand.",
        ),
        TrainingResource(
            id="cc-medical-coding",
            name="Medical Coding and Billing Certificate",
            provider="Community Colleges (nationwide)",
            url="",
            description="Medical terminology, ICD-10/CPT coding, health information management, and medical billing procedures.",
            category=ResourceCategory.COMMUNITY_COLLEGE,
            cost_tier=CostTier.LOW,
            estimated_cost_usd=500.0,
            delivery_format=DeliveryFormat.HYBRID,
            hours_per_week=15.0,
            total_weeks=24,
            skill_codes=[SKILL_CODES["reading_comprehension"], SKILL_CODES["critical_thinking"], SKILL_CODES["active_learning"], SKILL_CODES["monitoring"]],
            skill_names=["Reading Comprehension", "Critical Thinking", "Active Learning", "Monitoring"],
            credential_awarded="Certificate + CPC/CCS exam prep",
            notes="Remote work opportunities after certification.",
        ),
        TrainingResource(
            id="cc-it-support",
            name="IT Help Desk / CompTIA A+ Preparation",
            provider="Community Colleges (nationwide)",
            url="",
            description="Computer hardware, software troubleshooting, networking fundamentals, and CompTIA A+ exam preparation.",
            category=ResourceCategory.COMMUNITY_COLLEGE,
            cost_tier=CostTier.LOW,
            estimated_cost_usd=400.0,
            delivery_format=DeliveryFormat.HYBRID,
            hours_per_week=15.0,
            total_weeks=16,
            skill_codes=[SKILL_CODES["troubleshooting"], SKILL_CODES["complex_problem_solving"], SKILL_CODES["technology_design"], SKILL_CODES["operations_monitoring"]],
            skill_names=["Troubleshooting", "Complex Problem Solving", "Technology Design", "Operations Monitoring"],
            credential_awarded="Certificate + CompTIA A+ exam prep",
            notes="Entry point for IT career. Often WIOA-eligible.",
        ),
        TrainingResource(
            id="cc-accounting",
            name="Accounting Certificate Program",
            provider="Community Colleges (nationwide)",
            url="",
            description="Principles of accounting, bookkeeping, payroll, QuickBooks, and financial statement preparation.",
            category=ResourceCategory.COMMUNITY_COLLEGE,
            cost_tier=CostTier.LOW,
            estimated_cost_usd=500.0,
            delivery_format=DeliveryFormat.HYBRID,
            hours_per_week=10.0,
            total_weeks=32,
            skill_codes=[SKILL_CODES["mathematics"], SKILL_CODES["critical_thinking"], SKILL_CODES["management_financial"], SKILL_CODES["monitoring"]],
            skill_names=["Mathematics", "Critical Thinking", "Management of Financial Resources", "Monitoring"],
            credential_awarded="Accounting Certificate",
            notes="Pell Grant eligible. Good pathway to bookkeeper or accounting clerk roles.",
        ),

        # ============================================================
        # BOOTCAMPS
        # ============================================================
        TrainingResource(
            id="bootcamp-general-assembly",
            name="General Assembly Software Engineering Immersive",
            provider="General Assembly",
            url="https://generalassemb.ly/education/software-engineering-immersive",
            description="Full-time immersive bootcamp covering JavaScript, Python, React, databases, and software engineering practices.",
            category=ResourceCategory.BOOTCAMP,
            cost_tier=CostTier.HIGH,
            estimated_cost_usd=15950.0,
            delivery_format=DeliveryFormat.ONLINE_COHORT,
            hours_per_week=40.0,
            total_weeks=12,
            skill_codes=[SKILL_CODES["programming"], SKILL_CODES["technology_design"], SKILL_CODES["complex_problem_solving"], SKILL_CODES["systems_analysis"]],
            skill_names=["Programming", "Technology Design", "Complex Problem Solving", "Systems Analysis"],
            credential_awarded="Certificate of Completion",
            notes="ISA and financing options available. Career support included.",
        ),
        TrainingResource(
            id="bootcamp-flatiron",
            name="Flatiron School Software Engineering Bootcamp",
            provider="Flatiron School",
            url="https://flatironschool.com/courses/coding-bootcamp/",
            description="Intensive coding bootcamp covering Ruby, JavaScript, React, and full-stack web development.",
            category=ResourceCategory.BOOTCAMP,
            cost_tier=CostTier.HIGH,
            estimated_cost_usd=16900.0,
            delivery_format=DeliveryFormat.ONLINE_COHORT,
            hours_per_week=40.0,
            total_weeks=15,
            skill_codes=[SKILL_CODES["programming"], SKILL_CODES["technology_design"], SKILL_CODES["complex_problem_solving"]],
            skill_names=["Programming", "Technology Design", "Complex Problem Solving"],
            credential_awarded="Certificate of Completion",
            notes="Money-back guarantee if no job within 6 months. Financing available.",
        ),
        TrainingResource(
            id="bootcamp-thinkful",
            name="Thinkful Software Engineering Program",
            provider="Thinkful (Chegg)",
            url="https://www.thinkful.com/bootcamp/web-development/",
            description="Mentor-driven software engineering program with 1-on-1 mentorship, portfolio projects, and career coaching.",
            category=ResourceCategory.BOOTCAMP,
            cost_tier=CostTier.HIGH,
            estimated_cost_usd=9500.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=20.0,
            total_weeks=24,
            skill_codes=[SKILL_CODES["programming"], SKILL_CODES["technology_design"], SKILL_CODES["complex_problem_solving"], SKILL_CODES["systems_analysis"]],
            skill_names=["Programming", "Technology Design", "Complex Problem Solving", "Systems Analysis"],
            credential_awarded="Certificate of Completion",
            notes="Flexible schedule. ISA option available. 1-on-1 mentorship.",
        ),

        # ============================================================
        # APPRENTICESHIP PROGRAMS
        # ============================================================
        TrainingResource(
            id="apprenticeship-ibew",
            name="IBEW Electrical Apprenticeship",
            provider="International Brotherhood of Electrical Workers",
            url="https://www.ibew.org",
            description="4-5 year registered apprenticeship combining classroom instruction with paid on-the-job training for electricians.",
            category=ResourceCategory.APPRENTICESHIP,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.HYBRID,
            hours_per_week=40.0,
            total_weeks=208,
            requires_computer=False,
            requires_internet=False,
            skill_codes=[SKILL_CODES["installation"], SKILL_CODES["repairing"], SKILL_CODES["troubleshooting"], SKILL_CODES["mathematics"], SKILL_CODES["operations_monitoring"]],
            skill_names=["Installation", "Repairing", "Troubleshooting", "Mathematics", "Operations Monitoring"],
            credential_awarded="Journeyman Electrician License",
            notes="Earn while you learn. Competitive entry. Union membership included.",
        ),
        TrainingResource(
            id="apprenticeship-ua-plumbing",
            name="UA Plumbing & Pipefitting Apprenticeship",
            provider="United Association of Plumbers and Pipefitters",
            url="https://www.ua.org",
            description="5-year registered apprenticeship for plumbers and pipefitters with classroom and hands-on training.",
            category=ResourceCategory.APPRENTICESHIP,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.HYBRID,
            hours_per_week=40.0,
            total_weeks=260,
            requires_computer=False,
            requires_internet=False,
            skill_codes=[SKILL_CODES["installation"], SKILL_CODES["repairing"], SKILL_CODES["troubleshooting"], SKILL_CODES["equipment_selection"], SKILL_CODES["quality_control"]],
            skill_names=["Installation", "Repairing", "Troubleshooting", "Equipment Selection", "Quality Control Analysis"],
            credential_awarded="Journeyman Plumber License",
            notes="Earn while you learn. Full benefits from day one.",
        ),

        # ============================================================
        # LIBRARY AND COMMUNITY RESOURCES (NO COMPUTER REQUIRED)
        # ============================================================
        TrainingResource(
            id="library-digital-literacy",
            name="Public Library Digital Literacy Program",
            provider="Public Libraries (nationwide)",
            url="https://www.digitallearn.org",
            description="Free computer and digital literacy classes at local libraries including basic computing, internet use, email, and job search skills.",
            category=ResourceCategory.LIBRARY_COMMUNITY,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.IN_PERSON,
            hours_per_week=4.0,
            total_weeks=8,
            requires_computer=False,
            requires_internet=False,
            skill_codes=[SKILL_CODES["active_learning"], SKILL_CODES["reading_comprehension"]],
            skill_names=["Active Learning", "Reading Comprehension"],
            notes="Libraries provide free computer access. No prerequisites.",
        ),
        TrainingResource(
            id="library-linkedin-learning",
            name="LinkedIn Learning via Public Library",
            provider="Public Libraries (with LinkedIn Learning access)",
            url="https://www.linkedin.com/learning/",
            description="Free access to LinkedIn Learning courses through many public library systems. Covers business, technology, and creative skills.",
            category=ResourceCategory.LIBRARY_COMMUNITY,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=5.0,
            total_weeks=52,
            requires_computer=False,
            requires_internet=False,
            skill_codes=[SKILL_CODES["active_learning"], SKILL_CODES["critical_thinking"], SKILL_CODES["time_management"]],
            skill_names=["Active Learning", "Critical Thinking", "Time Management"],
            notes="Free with library card. Library provides computer access. Wide topic range.",
        ),
        TrainingResource(
            id="community-toastmasters",
            name="Toastmasters International",
            provider="Toastmasters International",
            url="https://www.toastmasters.org",
            description="Communication and leadership development through a supportive club environment with structured speaking practice.",
            category=ResourceCategory.LIBRARY_COMMUNITY,
            cost_tier=CostTier.LOW,
            estimated_cost_usd=100.0,
            delivery_format=DeliveryFormat.IN_PERSON,
            hours_per_week=2.0,
            total_weeks=52,
            requires_computer=False,
            requires_internet=False,
            skill_codes=[SKILL_CODES["speaking"], SKILL_CODES["persuasion"], SKILL_CODES["social_perceptiveness"], SKILL_CODES["active_listening"]],
            skill_names=["Speaking", "Persuasion", "Social Perceptiveness", "Active Listening"],
            credential_awarded="Pathways education awards",
            notes="Low annual dues (~$100/year). Clubs meet weekly in most cities.",
        ),
        TrainingResource(
            id="community-score-mentoring",
            name="SCORE Business Mentoring",
            provider="SCORE (SBA partner)",
            url="https://www.score.org",
            description="Free business mentoring and workshops for entrepreneurs covering business planning, marketing, finance, and operations.",
            category=ResourceCategory.LIBRARY_COMMUNITY,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.HYBRID,
            hours_per_week=3.0,
            total_weeks=26,
            requires_computer=False,
            requires_internet=False,
            skill_codes=[SKILL_CODES["management_financial"], SKILL_CODES["negotiation"], SKILL_CODES["judgment_decision_making"], SKILL_CODES["persuasion"]],
            skill_names=["Management of Financial Resources", "Negotiation", "Judgment and Decision Making", "Persuasion"],
            notes="Free mentoring from experienced business professionals.",
        ),

        # ============================================================
        # ADDITIONAL SELF-DIRECTED RESOURCES
        # ============================================================
        TrainingResource(
            id="coursera-financial-markets",
            name="Financial Markets (Yale) on Coursera",
            provider="Yale University (via Coursera)",
            url="https://www.coursera.org/learn/financial-markets-global",
            description="Overview of financial markets including stocks, bonds, insurance, banking, and behavioral finance.",
            category=ResourceCategory.SELF_DIRECTED,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=7.0,
            total_weeks=7,
            skill_codes=[SKILL_CODES["management_financial"], SKILL_CODES["critical_thinking"], SKILL_CODES["judgment_decision_making"], SKILL_CODES["mathematics"]],
            skill_names=["Management of Financial Resources", "Critical Thinking", "Judgment and Decision Making", "Mathematics"],
            notes="Audit for free. Certificate available with Coursera Plus.",
        ),
        TrainingResource(
            id="edx-cs50",
            name="CS50: Introduction to Computer Science (Harvard)",
            provider="Harvard University (via edX)",
            url="https://www.edx.org/learn/computer-science/harvard-university-cs50-s-introduction-to-computer-science",
            description="Harvard's introduction to computer science covering C, Python, SQL, HTML/CSS/JS, algorithms, and data structures.",
            category=ResourceCategory.SELF_DIRECTED,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=12.0,
            total_weeks=12,
            skill_codes=[SKILL_CODES["programming"], SKILL_CODES["complex_problem_solving"], SKILL_CODES["critical_thinking"], SKILL_CODES["mathematics"]],
            skill_names=["Programming", "Complex Problem Solving", "Critical Thinking", "Mathematics"],
            credential_awarded="Verified Certificate (optional, paid)",
            notes="Audit for free. One of the most popular CS courses worldwide.",
        ),
        TrainingResource(
            id="alison-project-management",
            name="Alison Diploma in Project Management",
            provider="Alison",
            url="https://alison.com/course/diploma-in-project-management",
            description="Free online diploma covering project lifecycle, planning, scheduling, cost management, and quality assurance.",
            category=ResourceCategory.SELF_DIRECTED,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.ONLINE_SELF_PACED,
            hours_per_week=5.0,
            total_weeks=8,
            skill_codes=[SKILL_CODES["coordination"], SKILL_CODES["time_management"], SKILL_CODES["management_personnel"], SKILL_CODES["judgment_decision_making"]],
            skill_names=["Coordination", "Time Management", "Management of Personnel Resources", "Judgment and Decision Making"],
            credential_awarded="Alison Diploma",
            notes="Free with ads. Digital certificate available.",
        ),
        TrainingResource(
            id="red-cross-first-aid",
            name="American Red Cross First Aid/CPR/AED",
            provider="American Red Cross",
            url="https://www.redcross.org/take-a-class/first-aid",
            description="Certification in first aid, CPR, and AED use. Required for many healthcare, education, and childcare positions.",
            category=ResourceCategory.SELF_DIRECTED,
            cost_tier=CostTier.LOW,
            estimated_cost_usd=90.0,
            delivery_format=DeliveryFormat.HYBRID,
            hours_per_week=8.0,
            total_weeks=1,
            requires_computer=False,
            requires_internet=False,
            skill_codes=[SKILL_CODES["service_orientation"], SKILL_CODES["monitoring"], SKILL_CODES["critical_thinking"]],
            skill_names=["Service Orientation", "Monitoring", "Critical Thinking"],
            credential_awarded="Red Cross Certification (2-year validity)",
            notes="Blended learning: online + in-person skills session.",
        ),
        TrainingResource(
            id="goodwill-career-services",
            name="Goodwill Career Services",
            provider="Goodwill Industries",
            url="https://www.goodwill.org/jobs-training/",
            description="Free career coaching, resume writing, interview preparation, computer training, and job placement assistance.",
            category=ResourceCategory.LIBRARY_COMMUNITY,
            cost_tier=CostTier.FREE,
            estimated_cost_usd=0.0,
            delivery_format=DeliveryFormat.IN_PERSON,
            hours_per_week=5.0,
            total_weeks=12,
            requires_computer=False,
            requires_internet=False,
            skill_codes=[SKILL_CODES["speaking"], SKILL_CODES["writing"], SKILL_CODES["active_listening"], SKILL_CODES["social_perceptiveness"]],
            skill_names=["Speaking", "Writing", "Active Listening", "Social Perceptiveness"],
            notes="Free. Available at most Goodwill locations. No eligibility requirements.",
        ),
    ]

    return resources


# ==================== Catalog Access ====================

# Module-level catalog singleton
_catalog: Optional[List[TrainingResource]] = None


def get_catalog() -> List[TrainingResource]:
    """Get the full training resource catalog.

    Returns:
        List of all ``TrainingResource`` entries.
    """
    global _catalog
    if _catalog is None:
        _catalog = _build_catalog()
    return _catalog


def get_resource_by_id(resource_id: str) -> Optional[TrainingResource]:
    """Look up a single resource by its ID.

    Args:
        resource_id: Unique resource identifier.

    Returns:
        The matching ``TrainingResource``, or ``None`` if not found.
    """
    for resource in get_catalog():
        if resource.id == resource_id:
            return resource
    return None


def get_resources_by_skill(skill_code: str) -> List[TrainingResource]:
    """Find all resources that address a given O*NET skill code.

    Args:
        skill_code: O*NET element ID (e.g., ``"2.B.3.e"`` for Programming).

    Returns:
        List of matching resources.
    """
    return [r for r in get_catalog() if skill_code in r.skill_codes]


def get_resources_by_skill_name(skill_name: str) -> List[TrainingResource]:
    """Find all resources that address a given skill by name (case-insensitive).

    Args:
        skill_name: Human-readable skill name (e.g., ``"programming"``).

    Returns:
        List of matching resources.
    """
    skill_lower = skill_name.lower()
    return [
        r for r in get_catalog()
        if any(skill_lower in sn.lower() for sn in r.skill_names)
    ]


def get_resources_by_category(
    category: ResourceCategory,
) -> List[TrainingResource]:
    """Find all resources in a given category.

    Args:
        category: Resource category to filter by.

    Returns:
        List of matching resources.
    """
    cat_value = category.value if isinstance(category, ResourceCategory) else category
    return [r for r in get_catalog() if r.category == cat_value]


def get_resources_by_cost_tier(cost_tier: CostTier) -> List[TrainingResource]:
    """Find all resources at a given cost tier.

    Args:
        cost_tier: Cost tier to filter by.

    Returns:
        List of matching resources.
    """
    tier_value = cost_tier.value if isinstance(cost_tier, CostTier) else cost_tier
    return [r for r in get_catalog() if r.cost_tier == tier_value]


def get_no_computer_resources() -> List[TrainingResource]:
    """Find all resources that do not require a personal computer.

    Returns:
        List of resources usable without a personal computer.
    """
    return [r for r in get_catalog() if not r.requires_computer]


def get_catalog_stats() -> Dict[str, Any]:
    """Get summary statistics about the catalog.

    Returns:
        Dictionary with counts by category, cost tier, and format.
    """
    catalog = get_catalog()
    stats: Dict[str, Any] = {
        "total_resources": len(catalog),
        "by_category": {},
        "by_cost_tier": {},
        "by_format": {},
        "no_computer_required": 0,
        "no_internet_required": 0,
    }

    for resource in catalog:
        cat = resource.category
        stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

        tier = resource.cost_tier
        stats["by_cost_tier"][tier] = stats["by_cost_tier"].get(tier, 0) + 1

        fmt = resource.delivery_format
        stats["by_format"][fmt] = stats["by_format"].get(fmt, 0) + 1

        if not resource.requires_computer:
            stats["no_computer_required"] += 1
        if not resource.requires_internet:
            stats["no_internet_required"] += 1

    return stats
