"""Integration test for end-to-end recommendation flow."""
import pytest
from datetime import datetime

from app.models.models import (
    Occupation, Skill, OccupationSkill, UserProfile,
    UserCurrentOccupation, UserSkillRating, RecommendationEvent,
)
from app.ml.scoring import get_baseline_scorer


@pytest.mark.asyncio
async def test_end_to_end_recommendation_flow(test_db):
    """Test complete recommendation flow from user creation to recommendations."""

    # ===== Setup: Create test data =====

    # Create skills
    skills = [
        Skill(element_id="2.B.1.a", name="Reading Comprehension"),
        Skill(element_id="2.B.8.a", name="Critical Thinking"),
        Skill(element_id="2.B.1.g", name="Programming"),
        Skill(element_id="2.B.5.c", name="Design"),
    ]
    for skill in skills:
        test_db.add(skill)

    # Create current occupation
    current_occ = Occupation(
        onet_code="15-1252.00",
        title="Software Developers",
        description="Research, design, and develop software",
        job_zone=4,
        education_level="Bachelor's degree",
        last_fetched_at=datetime.utcnow(),
    )
    test_db.add(current_occ)

    # Add skills to current occupation
    current_occ_skills = [
        OccupationSkill(
            onet_code="15-1252.00",
            element_id="2.B.1.a",
            importance=72.0,
            level=5.12,
        ),
        OccupationSkill(
            onet_code="15-1252.00",
            element_id="2.B.8.a",
            importance=81.0,
            level=5.62,
        ),
        OccupationSkill(
            onet_code="15-1252.00",
            element_id="2.B.1.g",
            importance=84.0,
            level=5.75,
        ),
    ]
    for occ_skill in current_occ_skills:
        test_db.add(occ_skill)

    # Create target occupation (Web Developer)
    target_occ = Occupation(
        onet_code="15-1299.08",
        title="Web Developers",
        description="Develop websites and web applications",
        job_zone=3,
        education_level="Associate's degree",
        last_fetched_at=datetime.utcnow(),
    )
    test_db.add(target_occ)

    # Add skills to target occupation
    target_occ_skills = [
        OccupationSkill(
            onet_code="15-1299.08",
            element_id="2.B.1.a",
            importance=69.0,
            level=4.88,
        ),
        OccupationSkill(
            onet_code="15-1299.08",
            element_id="2.B.8.a",
            importance=75.0,
            level=5.25,
        ),
        OccupationSkill(
            onet_code="15-1299.08",
            element_id="2.B.1.g",
            importance=81.0,
            level=5.62,
        ),
        OccupationSkill(
            onet_code="15-1299.08",
            element_id="2.B.5.c",
            importance=72.0,
            level=5.00,
        ),
    ]
    for occ_skill in target_occ_skills:
        test_db.add(occ_skill)

    await test_db.commit()

    # ===== Step 1: Create user profile =====
    user = UserProfile(created_at=datetime.utcnow())
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)

    assert user.id is not None

    # ===== Step 2: Set current occupation =====
    user_current = UserCurrentOccupation(
        user_id=user.id,
        onet_code="15-1252.00",
        selected_at=datetime.utcnow(),
        is_active=True,
    )
    test_db.add(user_current)
    await test_db.commit()

    # ===== Step 3: Rate skills =====
    # User is strong in programming and critical thinking, weak in design
    skill_ratings = [
        UserSkillRating(user_id=user.id, element_id="2.B.1.a", rating_0_4=3),  # Advanced
        UserSkillRating(user_id=user.id, element_id="2.B.8.a", rating_0_4=4),  # Expert
        UserSkillRating(user_id=user.id, element_id="2.B.1.g", rating_0_4=4),  # Expert
        UserSkillRating(user_id=user.id, element_id="2.B.5.c", rating_0_4=1),  # Basic (gap)
    ]
    for rating in skill_ratings:
        test_db.add(rating)
    await test_db.commit()

    # ===== Step 4: Generate recommendations =====

    # Build user ratings dict
    user_ratings_dict = {sr.element_id: sr.rating_0_4 for sr in skill_ratings}

    # Get target occupation skills
    target_skills = [
        {
            "element_id": occ_skill.element_id,
            "skill_name": occ_skill.skill.name,
            "importance": occ_skill.importance,
            "level": occ_skill.level,
        }
        for occ_skill in target_occ_skills
    ]

    # Score the target occupation
    scorer = get_baseline_scorer()
    score = scorer.score_occupation(
        onet_code=target_occ.onet_code,
        occupation_title=target_occ.title,
        occupation_skills=target_skills,
        user_skill_ratings=user_ratings_dict,
        current_job_zone=current_occ.job_zone,
        target_job_zone=target_occ.job_zone,
    )

    # ===== Assertions =====

    # Check match score
    # User is strong in 3/4 skills (Reading: 3, Critical: 4, Programming: 4, Design: 1)
    # Should have high match score
    assert score.match_score > 70.0, f"Expected match_score > 70, got {score.match_score}"

    # Check gap severity
    # Only Design is a gap (rating 1 = 0.25 capability)
    assert score.gap_severity < 30.0, f"Expected gap_severity < 30, got {score.gap_severity}"

    # Check bucket
    # High match + low gaps = READY_NOW
    assert score.bucket == "ready_now", f"Expected ready_now bucket, got {score.bucket}"

    # Check top gaps
    # Should identify Design as the main gap
    assert len(score.top_gaps) == 1
    assert score.top_gaps[0].skill_name == "Design"

    # Check training suggestion
    assert "apply now" in score.training_suggestion.lower() or "ready" in score.training_suggestion.lower()

    # Check explanation
    assert "strong match" in score.explanation.lower() or "ready" in score.explanation.lower()

    # Check metadata
    assert score.metadata["total_skills"] == 4
    assert score.metadata["skills_with_gaps"] == 1
    assert score.metadata["job_zone_diff"] == -1  # Target job_zone (3) < Current (4)

    # ===== Step 5: Save recommendation event =====

    event = RecommendationEvent(
        user_id=user.id,
        created_at=datetime.utcnow(),
        current_onet_code="15-1252.00",
        model_version="v1_baseline",
        params_json={"test": True},
    )
    test_db.add(event)
    await test_db.commit()
    await test_db.refresh(event)

    assert event.id is not None

    print(f"""
    ✅ Integration test passed!

    User {user.id} rated skills:
      - Reading Comprehension: 3 (Advanced)
      - Critical Thinking: 4 (Expert)
      - Programming: 4 (Expert)
      - Design: 1 (Basic - GAP!)

    Recommendation for Web Developer:
      - Match Score: {score.match_score:.1f}%
      - Gap Severity: {score.gap_severity:.1f}%
      - Bucket: {score.bucket}
      - Top Gap: {score.top_gaps[0].skill_name}
      - Suggestion: {score.training_suggestion[:100]}...
    """)


@pytest.mark.asyncio
async def test_trainable_bucket_scenario(test_db):
    """Test scenario that should result in TRAINABLE bucket."""

    # Create skills
    skills = [
        Skill(element_id="skill1", name="Skill A"),
        Skill(element_id="skill2", name="Skill B"),
        Skill(element_id="skill3", name="Skill C"),
    ]
    for skill in skills:
        test_db.add(skill)

    # Create occupation
    occupation = Occupation(
        onet_code="99-9999.00",
        title="Test Occupation",
        job_zone=3,
        last_fetched_at=datetime.utcnow(),
    )
    test_db.add(occupation)

    # Add skills with moderate importance
    occ_skills = [
        OccupationSkill(onet_code="99-9999.00", element_id="skill1", importance=70.0),
        OccupationSkill(onet_code="99-9999.00", element_id="skill2", importance=60.0),
        OccupationSkill(onet_code="99-9999.00", element_id="skill3", importance=50.0),
    ]
    for occ_skill in occ_skills:
        test_db.add(occ_skill)

    await test_db.commit()

    # Create user
    user = UserProfile(created_at=datetime.utcnow())
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)

    # User has moderate skills (2/3 rated intermediate, 1/3 is gap)
    user_ratings_dict = {
        "skill1": 2,  # Intermediate
        "skill2": 2,  # Intermediate
        "skill3": 0,  # None - gap
    }

    # Score occupation
    scorer = get_baseline_scorer()
    score = scorer.score_occupation(
        onet_code="99-9999.00",
        occupation_title="Test Occupation",
        occupation_skills=[
            {"element_id": "skill1", "skill_name": "Skill A", "importance": 70.0, "level": None},
            {"element_id": "skill2", "skill_name": "Skill B", "importance": 60.0, "level": None},
            {"element_id": "skill3", "skill_name": "Skill C", "importance": 50.0, "level": None},
        ],
        user_skill_ratings=user_ratings_dict,
        target_job_zone=3,
    )

    # Should be TRAINABLE (moderate match, some gaps)
    assert score.bucket == "trainable"
    assert 40 < score.match_score < 75
    assert len(score.top_gaps) == 1
    assert "training" in score.training_suggestion.lower()


@pytest.mark.asyncio
async def test_long_reskill_scenario(test_db):
    """Test scenario that should result in LONG_RESKILL bucket."""

    # Create skills
    skills = [
        Skill(element_id="skill1", name="Skill A"),
        Skill(element_id="skill2", name="Skill B"),
        Skill(element_id="skill3", name="Skill C"),
        Skill(element_id="skill4", name="Skill D"),
    ]
    for skill in skills:
        test_db.add(skill)

    # Create occupation
    occupation = Occupation(
        onet_code="99-8888.00",
        title="Advanced Occupation",
        job_zone=5,
        last_fetched_at=datetime.utcnow(),
    )
    test_db.add(occupation)

    # Add skills with high importance
    occ_skills = [
        OccupationSkill(onet_code="99-8888.00", element_id="skill1", importance=90.0),
        OccupationSkill(onet_code="99-8888.00", element_id="skill2", importance=85.0),
        OccupationSkill(onet_code="99-8888.00", element_id="skill3", importance=80.0),
        OccupationSkill(onet_code="99-8888.00", element_id="skill4", importance=75.0),
    ]
    for occ_skill in occ_skills:
        test_db.add(occ_skill)

    await test_db.commit()

    # Create user
    user = UserProfile(created_at=datetime.utcnow())
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)

    # User has low skills in all areas
    user_ratings_dict = {
        "skill1": 1,  # Basic
        "skill2": 1,  # Basic
        "skill3": 0,  # None
        "skill4": 0,  # None
    }

    # Score occupation
    scorer = get_baseline_scorer()
    score = scorer.score_occupation(
        onet_code="99-8888.00",
        occupation_title="Advanced Occupation",
        occupation_skills=[
            {"element_id": "skill1", "skill_name": "Skill A", "importance": 90.0, "level": None},
            {"element_id": "skill2", "skill_name": "Skill B", "importance": 85.0, "level": None},
            {"element_id": "skill3", "skill_name": "Skill C", "importance": 80.0, "level": None},
            {"element_id": "skill4", "skill_name": "Skill D", "importance": 75.0, "level": None},
        ],
        user_skill_ratings=user_ratings_dict,
        current_job_zone=2,
        target_job_zone=5,
    )

    # Should be LONG_RESKILL (low match, high gaps)
    assert score.bucket == "long_reskill"
    assert score.match_score < 50
    assert score.gap_severity > 40
    assert len(score.top_gaps) >= 2
    assert "significant" in score.training_suggestion.lower() or "long" in score.training_suggestion.lower()
