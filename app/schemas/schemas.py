"""Pydantic schemas for API request/response models."""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


# ==================== Occupation Schemas ====================

class OccupationSearchResult(BaseModel):
    """Search result for an occupation."""
    code: str
    title: str


class OccupationSkill(BaseModel):
    """Skill associated with an occupation."""
    element_id: str
    skill_name: str
    importance: Optional[float] = None
    level: Optional[float] = None


class OccupationDetail(BaseModel):
    """Detailed occupation information."""
    code: str
    title: str
    description: Optional[str] = None
    job_zone: Optional[int] = None
    education: Optional[str] = None


class OccupationWithSkills(BaseModel):
    """Occupation with its associated skills."""
    code: str
    title: str
    description: Optional[str] = None
    job_zone: Optional[int] = None
    skills: List[OccupationSkill]


# ==================== User Schemas ====================

class UserProfileCreate(BaseModel):
    """Request to create a user profile."""
    metadata: Optional[Dict[str, Any]] = None


class UserProfileResponse(BaseModel):
    """User profile response."""
    id: int
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class UserCurrentOccupationRequest(BaseModel):
    """Request to set user's current occupation."""
    onet_code: str = Field(..., description="O*NET code for current occupation")


class UserCurrentOccupationResponse(BaseModel):
    """Response for user's current occupation."""
    user_id: int
    onet_code: str
    occupation_title: str
    selected_at: datetime


class SkillRating(BaseModel):
    """A single skill rating."""
    element_id: str
    rating_0_4: int = Field(..., ge=0, le=4, description="Rating from 0 (none) to 4 (expert)")


class UserSkillRatingsRequest(BaseModel):
    """Request to update user skill ratings."""
    ratings: List[SkillRating] = Field(..., min_items=1)


class UserSkillRatingsResponse(BaseModel):
    """Response after updating skill ratings."""
    user_id: int
    updated_count: int
    ratings: List[SkillRating]


# ==================== Recommendation Schemas ====================

class SkillGapInfo(BaseModel):
    """Information about a skill gap."""
    element_id: str
    skill_name: str
    required_importance: float
    required_level: Optional[float] = None
    user_capability: float
    gap_weight: float


class RecommendedOccupation(BaseModel):
    """A single recommended occupation."""
    onet_code: str
    title: str
    rank: int
    bucket: str  # ready_now, trainable, long_reskill
    match_score: float
    gap_severity: float
    top_gaps: List[SkillGapInfo]
    training_suggestion: str
    explanation: str
    is_exploration: bool = False
    metadata: Optional[Dict[str, Any]] = None


class RecommendationBucket(BaseModel):
    """A bucket of recommendations."""
    bucket_name: str
    bucket_label: str
    description: str
    occupations: List[RecommendedOccupation]


class RecommendationRequest(BaseModel):
    """Request for recommendations."""
    limit_per_bucket: int = Field(10, ge=1, le=50, description="Max recommendations per bucket")
    use_calibration: bool = Field(False, description="Use calibrated model if available")
    enable_exploration: bool = Field(False, description="Enable exploration for online learning")


class RecommendationResponse(BaseModel):
    """Response with recommendations."""
    event_id: int
    user_id: int
    current_occupation: OccupationDetail
    model_version: str
    created_at: datetime
    buckets: List[RecommendationBucket]
    decision_guidance: str
    total_recommendations: int


# ==================== Feedback Schemas ====================

class UserFeedbackRequest(BaseModel):
    """User feedback on a recommendation."""
    event_id: int
    target_onet_code: str
    action_type: str = Field(
        ...,
        description="Action type: click, save, hide, apply, interview, offer"
    )
    metadata: Optional[Dict[str, Any]] = None

    @validator("action_type")
    def validate_action_type(cls, v):
        valid_actions = {"click", "save", "hide", "apply", "interview", "offer"}
        if v not in valid_actions:
            raise ValueError(f"action_type must be one of {valid_actions}")
        return v


class UserFeedbackResponse(BaseModel):
    """Response after submitting feedback."""
    feedback_id: int
    event_id: int
    target_onet_code: str
    action_type: str
    action_at: datetime


# ==================== Model Status Schemas ====================

class ModelMetrics(BaseModel):
    """Model performance metrics."""
    accuracy: Optional[float] = None
    roc_auc: Optional[float] = None
    train_samples: Optional[int] = None
    test_samples: Optional[int] = None
    positive_rate: Optional[float] = None


class ModelStatusResponse(BaseModel):
    """Status of the current model."""
    current_model_version: str
    is_calibrated: bool
    last_trained_at: Optional[datetime] = None
    training_samples: Optional[int] = None
    metrics: Optional[ModelMetrics] = None
    total_feedback_events: int
    total_recommendations: int


# ==================== Health Check ====================

class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    demo_mode: bool
    database: str
    redis: str


# ==================== UI Support Schemas ====================

class QuickRateOption(BaseModel):
    """Quick rate option for UI."""
    label: str
    ratings: Dict[str, int]  # element_id -> rating


class SkillRatingUI(BaseModel):
    """Skill rating for UI display."""
    element_id: str
    skill_name: str
    current_rating: int
    importance: Optional[float] = None


class OccupationSearchRequest(BaseModel):
    """Search request."""
    query: str = Field(..., min_length=1)
    limit: int = Field(20, ge=1, le=100)


# ==================== Decision Flow Schemas ====================

class DecisionFlowResponse(BaseModel):
    """Decision flow guidance."""
    question: str
    options: List[Dict[str, str]]
    recommendation: str


# ==================== Error Schemas ====================

class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
