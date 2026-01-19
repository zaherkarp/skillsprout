"""SQLAlchemy ORM models for the application."""
from datetime import datetime
from typing import Optional
import enum

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, ForeignKey,
    Text, JSON, Enum as SQLEnum, UniqueConstraint, Index, Boolean
)
from sqlalchemy.orm import relationship

from app.db.session import Base


class ActionType(str, enum.Enum):
    """User feedback action types."""
    CLICK = "click"
    SAVE = "save"
    HIDE = "hide"
    APPLY = "apply"
    INTERVIEW = "interview"
    OFFER = "offer"


class RecommendationBucket(str, enum.Enum):
    """Recommendation bucket categories."""
    READY_NOW = "ready_now"
    TRAINABLE = "trainable"
    LONG_RESKILL = "long_reskill"


class Occupation(Base):
    """O*NET occupation data."""
    __tablename__ = "occupation"

    onet_code = Column(String(10), primary_key=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    job_zone = Column(Integer, nullable=True)
    education_level = Column(String(100), nullable=True)
    last_fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    raw_json = Column(JSON, nullable=True)

    # Relationships
    occupation_skills = relationship("OccupationSkill", back_populates="occupation", cascade="all, delete-orphan")
    user_current_occupations = relationship("UserCurrentOccupation", back_populates="occupation")
    recommended_occupations = relationship("RecommendedOccupation", back_populates="occupation")


class Skill(Base):
    """Skills from O*NET."""
    __tablename__ = "skill"

    element_id = Column(String(20), primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Relationships
    occupation_skills = relationship("OccupationSkill", back_populates="skill")
    user_skill_ratings = relationship("UserSkillRating", back_populates="skill")


class OccupationSkill(Base):
    """Junction table linking occupations to skills with importance/level."""
    __tablename__ = "occupation_skill"
    __table_args__ = (
        UniqueConstraint('onet_code', 'element_id', name='uq_occupation_skill'),
        Index('ix_occupation_skill_onet_code', 'onet_code'),
        Index('ix_occupation_skill_element_id', 'element_id'),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    onet_code = Column(String(10), ForeignKey("occupation.onet_code", ondelete="CASCADE"), nullable=False)
    element_id = Column(String(20), ForeignKey("skill.element_id", ondelete="CASCADE"), nullable=False)
    importance = Column(Float, nullable=True)  # 0-100 scale
    level = Column(Float, nullable=True)  # 0-7 scale
    last_fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    occupation = relationship("Occupation", back_populates="occupation_skills")
    skill = relationship("Skill", back_populates="occupation_skills")


class UserProfile(Base):
    """User profile information."""
    __tablename__ = "user_profile"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Optional demographic fields can be added here
    metadata_json = Column(JSON, nullable=True)

    # Relationships
    current_occupations = relationship("UserCurrentOccupation", back_populates="user", cascade="all, delete-orphan")
    skill_ratings = relationship("UserSkillRating", back_populates="user", cascade="all, delete-orphan")
    recommendation_events = relationship("RecommendationEvent", back_populates="user", cascade="all, delete-orphan")


class UserCurrentOccupation(Base):
    """User's current occupation selection."""
    __tablename__ = "user_current_occupation"
    __table_args__ = (
        Index('ix_user_current_occupation_user_id', 'user_id'),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user_profile.id", ondelete="CASCADE"), nullable=False)
    onet_code = Column(String(10), ForeignKey("occupation.onet_code", ondelete="CASCADE"), nullable=False)
    selected_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    user = relationship("UserProfile", back_populates="current_occupations")
    occupation = relationship("Occupation", back_populates="user_current_occupations")


class UserSkillRating(Base):
    """User's self-assessed skill proficiency."""
    __tablename__ = "user_skill_rating"
    __table_args__ = (
        UniqueConstraint('user_id', 'element_id', name='uq_user_skill'),
        Index('ix_user_skill_rating_user_id', 'user_id'),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user_profile.id", ondelete="CASCADE"), nullable=False)
    element_id = Column(String(20), ForeignKey("skill.element_id", ondelete="CASCADE"), nullable=False)
    rating_0_4 = Column(Integer, nullable=False)  # 0=none, 1=basic, 2=intermediate, 3=advanced, 4=expert
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("UserProfile", back_populates="skill_ratings")
    skill = relationship("Skill", back_populates="user_skill_ratings")


class RecommendationEvent(Base):
    """A recommendation generation event (one per user request)."""
    __tablename__ = "recommendation_event"
    __table_args__ = (
        Index('ix_recommendation_event_user_id', 'user_id'),
        Index('ix_recommendation_event_created_at', 'created_at'),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user_profile.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    current_onet_code = Column(String(10), nullable=False)
    model_version = Column(String(50), nullable=False)
    params_json = Column(JSON, nullable=True)  # Stores thresholds, epsilon, etc.

    # Relationships
    user = relationship("UserProfile", back_populates="recommendation_events")
    recommended_occupations = relationship("RecommendedOccupation", back_populates="event", cascade="all, delete-orphan")
    feedbacks = relationship("UserFeedback", back_populates="event", cascade="all, delete-orphan")


class RecommendedOccupation(Base):
    """Individual occupation recommendation within an event."""
    __tablename__ = "recommended_occupation"
    __table_args__ = (
        UniqueConstraint('event_id', 'target_onet_code', name='uq_event_occupation'),
        Index('ix_recommended_occupation_event_id', 'event_id'),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("recommendation_event.id", ondelete="CASCADE"), nullable=False)
    target_onet_code = Column(String(10), ForeignKey("occupation.onet_code", ondelete="CASCADE"), nullable=False)
    rank = Column(Integer, nullable=False)
    bucket = Column(SQLEnum(RecommendationBucket), nullable=False, index=True)
    score_json = Column(JSON, nullable=False)  # match_score, gap_severity, top_gaps, etc.
    is_exploration = Column(Boolean, default=False, nullable=False)  # For explore/exploit tracking

    # Relationships
    event = relationship("RecommendationEvent", back_populates="recommended_occupations")
    occupation = relationship("Occupation", back_populates="recommended_occupations")
    feedbacks = relationship("UserFeedback", back_populates="recommended_occupation", cascade="all, delete-orphan")


class UserFeedback(Base):
    """User feedback on recommendations (clicks, saves, outcomes)."""
    __tablename__ = "user_feedback"
    __table_args__ = (
        Index('ix_user_feedback_event_id', 'event_id'),
        Index('ix_user_feedback_action_type', 'action_type'),
        Index('ix_user_feedback_action_at', 'action_at'),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("recommendation_event.id", ondelete="CASCADE"), nullable=False)
    target_onet_code = Column(String(10), nullable=False)
    recommended_occupation_id = Column(Integer, ForeignKey("recommended_occupation.id", ondelete="CASCADE"), nullable=True)
    action_type = Column(SQLEnum(ActionType), nullable=False)
    action_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    metadata_json = Column(JSON, nullable=True)  # Additional context

    # Relationships
    event = relationship("RecommendationEvent", back_populates="feedbacks")
    recommended_occupation = relationship("RecommendedOccupation", back_populates="feedbacks")


class ModelRegistry(Base):
    """Track trained model versions and artifacts."""
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(50), nullable=False, unique=True, index=True)
    trained_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    training_samples = Column(Integer, nullable=False)
    metrics_json = Column(JSON, nullable=True)  # Accuracy, AUC, etc.
    artifact_path = Column(String(500), nullable=True)  # Path to serialized model
    is_active = Column(Boolean, default=False, nullable=False)
    notes = Column(Text, nullable=True)
