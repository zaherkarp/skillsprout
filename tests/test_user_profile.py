"""Tests for user profile features: profile, saved occupations, progress, engagement."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.user_profile.profile import (
    ProfileConstraints,
    ProfileCreateRequest,
    ProfileUpdateRequest,
    ProfileResponse,
    RiskTolerance,
    RemotePreference,
    SkillSnapshot,
    _pack_profile_metadata,
    _unpack_profile_response,
)
from app.features.user_profile.saved_occupations import (
    SaveOccupationRequest,
    SavedOccupationResponse,
    SavedOccupationUpdate,
    TrainingStatus,
    _get_saved_list,
    _set_saved_list,
    _next_saved_id,
)
from app.features.user_profile.progress_tracker import (
    SkillUpdate,
    SkillsUpdateRequest,
    ProgressMetrics,
    BucketImprovement,
    _is_improvement,
    _estimate_time_to_ready,
    BUCKET_ORDER,
)
from app.features.user_profile.return_engagement import (
    ProgressSummaryResponse,
    NextMilestone,
    _count_bucket_improvements,
    _determine_next_milestone,
    _build_summary_text,
)


# =========================================================================
# Profile schemas and helpers
# =========================================================================

class TestProfileSchemas:
    """Tests for profile Pydantic schemas and helper functions."""

    def test_profile_create_defaults(self):
        """Default profile creation request should have sensible defaults."""
        req = ProfileCreateRequest()
        assert req.display_name is None
        assert req.current_occupation_id is None
        assert req.skills_snapshot == []
        assert req.constraints is None
        assert req.risk_tolerance == RiskTolerance.STANDARD

    def test_profile_create_full(self):
        """Full profile creation request should parse correctly."""
        req = ProfileCreateRequest(
            display_name="Alice",
            current_occupation_id="15-1252.00",
            skills_snapshot=[
                SkillSnapshot(element_id="2.B.1.a", rating=3, skill_name="Programming"),
            ],
            constraints=ProfileConstraints(
                salary_minimum=60000,
                location="Austin, TX",
                remote_preference=RemotePreference.HYBRID,
                industry_interests=["tech", "finance"],
                timeline_months=12,
            ),
            risk_tolerance=RiskTolerance.RELAXED,
        )
        assert req.display_name == "Alice"
        assert req.constraints.salary_minimum == 60000
        assert req.constraints.remote_preference == RemotePreference.HYBRID
        assert len(req.constraints.industry_interests) == 2

    def test_display_name_strip(self):
        """Display name should be stripped of whitespace."""
        req = ProfileCreateRequest(display_name="  Bob  ")
        assert req.display_name == "Bob"

    def test_display_name_empty_becomes_none(self):
        """Display name that is only whitespace becomes None."""
        req = ProfileCreateRequest(display_name="   ")
        assert req.display_name is None

    def test_constraints_validation(self):
        """Constraints should enforce min/max on fields."""
        c = ProfileConstraints(salary_minimum=0, timeline_months=1)
        assert c.salary_minimum == 0
        assert c.timeline_months == 1

        with pytest.raises(Exception):
            ProfileConstraints(salary_minimum=-1)

        with pytest.raises(Exception):
            ProfileConstraints(timeline_months=61)

    def test_risk_tolerance_values(self):
        """RiskTolerance enum should have expected values."""
        assert RiskTolerance.RELAXED.value == "relaxed"
        assert RiskTolerance.STANDARD.value == "standard"
        assert RiskTolerance.STRICT.value == "strict"

    def test_pack_profile_metadata(self):
        """_pack_profile_metadata should produce a serialisable dict."""
        meta = _pack_profile_metadata(
            display_name="Charlie",
            current_occupation_id="15-1252.00",
            skills_snapshot=[SkillSnapshot(element_id="s1", rating=2)],
            constraints=ProfileConstraints(salary_minimum=50000),
            risk_tolerance=RiskTolerance.STRICT,
        )

        assert meta["display_name"] == "Charlie"
        assert meta["risk_tolerance"] == "strict"
        assert len(meta["skills_snapshot"]) == 1
        assert meta["constraints"]["salary_minimum"] == 50000

    def test_pack_profile_metadata_empty(self):
        """Packing with None values should produce valid metadata."""
        meta = _pack_profile_metadata()
        assert meta["display_name"] is None
        assert meta["skills_snapshot"] == []
        assert meta["constraints"] is None

    def test_unpack_profile_response(self):
        """_unpack_profile_response should reconstruct a ProfileResponse."""
        now = datetime.utcnow()
        mock_row = MagicMock()
        mock_row.id = 1
        mock_row.created_at = now
        mock_row.updated_at = now
        mock_row.metadata_json = {
            "display_name": "Dana",
            "current_occupation_id": "11-3021.00",
            "risk_tolerance": "relaxed",
            "skills_snapshot": [
                {"element_id": "s1", "rating": 3, "skill_name": "Python"}
            ],
            "constraints": {
                "salary_minimum": 70000,
                "location": "Remote",
                "remote_preference": "remote_only",
                "industry_interests": ["tech"],
                "timeline_months": 6,
            },
        }

        resp = _unpack_profile_response(mock_row)
        assert resp.user_id == 1
        assert resp.display_name == "Dana"
        assert resp.risk_tolerance == RiskTolerance.RELAXED
        assert resp.constraints.salary_minimum == 70000
        assert resp.constraints.remote_preference == RemotePreference.REMOTE_ONLY
        assert len(resp.skills_snapshot) == 1

    def test_unpack_profile_response_empty_metadata(self):
        """Unpacking with empty metadata should not raise."""
        now = datetime.utcnow()
        mock_row = MagicMock()
        mock_row.id = 2
        mock_row.created_at = now
        mock_row.updated_at = now
        mock_row.metadata_json = {}

        resp = _unpack_profile_response(mock_row)
        assert resp.display_name is None
        assert resp.constraints is None
        assert resp.risk_tolerance == RiskTolerance.STANDARD

    def test_profile_update_partial(self):
        """ProfileUpdateRequest should allow partial updates."""
        req = ProfileUpdateRequest(display_name="New Name")
        assert req.display_name == "New Name"
        assert req.constraints is None  # Not set

        req2 = ProfileUpdateRequest(risk_tolerance=RiskTolerance.STRICT)
        assert req2.display_name is None
        assert req2.risk_tolerance == RiskTolerance.STRICT


# =========================================================================
# Saved Occupations helpers
# =========================================================================

class TestSavedOccupationsHelpers:
    """Tests for saved occupations helper functions."""

    def test_get_saved_list_empty(self):
        """Empty metadata should return an empty list."""
        assert _get_saved_list({}) == []

    def test_get_saved_list(self):
        """Should extract saved_occupations from metadata."""
        items = [{"saved_id": "saved_1", "onet_code": "15-1252.00"}]
        meta = {"saved_occupations": items}
        assert _get_saved_list(meta) == items

    def test_set_saved_list(self):
        """Should update saved_occupations in metadata."""
        meta = {}
        items = [{"saved_id": "saved_1"}]
        result = _set_saved_list(meta, items)
        assert result["saved_occupations"] == items

    def test_next_saved_id_empty(self):
        """First saved ID should be saved_1."""
        assert _next_saved_id([]) == "saved_1"

    def test_next_saved_id_increment(self):
        """Should increment from the max existing ID."""
        items = [
            {"saved_id": "saved_1"},
            {"saved_id": "saved_3"},
        ]
        assert _next_saved_id(items) == "saved_4"

    def test_training_status_values(self):
        """TrainingStatus enum should have expected values."""
        assert TrainingStatus.NOT_STARTED.value == "not_started"
        assert TrainingStatus.IN_PROGRESS.value == "in_progress"
        assert TrainingStatus.COMPLETED.value == "completed"
        assert TrainingStatus.PAUSED.value == "paused"

    def test_save_occupation_request_schema(self):
        """SaveOccupationRequest should validate correctly."""
        req = SaveOccupationRequest(
            user_id=1,
            onet_code="15-1252.00",
            notes="Interested in this role",
        )
        assert req.user_id == 1
        assert req.onet_code == "15-1252.00"

    def test_saved_occupation_update_schema(self):
        """SavedOccupationUpdate should allow partial updates."""
        upd = SavedOccupationUpdate(training_status=TrainingStatus.IN_PROGRESS)
        assert upd.training_status == TrainingStatus.IN_PROGRESS
        assert upd.notes is None


# =========================================================================
# Progress Tracker helpers
# =========================================================================

class TestProgressTrackerHelpers:
    """Tests for progress tracker helper functions."""

    def test_is_improvement_trainable_to_ready(self):
        """Trainable to ready_now is an improvement."""
        assert _is_improvement("trainable", "ready_now") is True

    def test_is_improvement_long_to_trainable(self):
        """Long_reskill to trainable is an improvement."""
        assert _is_improvement("long_reskill", "trainable") is True

    def test_is_not_improvement_same(self):
        """Same bucket is not an improvement."""
        assert _is_improvement("trainable", "trainable") is False

    def test_is_not_improvement_regression(self):
        """Ready_now to trainable is not an improvement."""
        assert _is_improvement("ready_now", "trainable") is False

    def test_estimate_time_ready_now(self):
        """ready_now should return 0 months."""
        assert _estimate_time_to_ready(90, 5, "ready_now") == 0

    def test_estimate_time_trainable(self):
        """Trainable should return months based on gap severity."""
        months = _estimate_time_to_ready(60, 30, "trainable")
        assert months == 6  # 30 / 5 = 6

    def test_estimate_time_trainable_min(self):
        """Trainable with low gap should still return at least 1 month."""
        months = _estimate_time_to_ready(70, 2, "trainable")
        assert months == 1

    def test_estimate_time_trainable_cap(self):
        """Trainable should cap at 24 months."""
        months = _estimate_time_to_ready(50, 200, "trainable")
        assert months == 24

    def test_estimate_time_long_reskill(self):
        """long_reskill should return None (too uncertain)."""
        assert _estimate_time_to_ready(20, 80, "long_reskill") is None

    def test_bucket_order(self):
        """BUCKET_ORDER should rank ready_now < trainable < long_reskill."""
        assert BUCKET_ORDER["ready_now"] < BUCKET_ORDER["trainable"]
        assert BUCKET_ORDER["trainable"] < BUCKET_ORDER["long_reskill"]

    def test_skill_update_schema(self):
        """SkillUpdate should validate rating range."""
        upd = SkillUpdate(element_id="2.B.1.a", new_rating=3)
        assert upd.new_rating == 3

        with pytest.raises(Exception):
            SkillUpdate(element_id="x", new_rating=5)

    def test_skills_update_request_min_length(self):
        """SkillsUpdateRequest must have at least one update."""
        with pytest.raises(Exception):
            SkillsUpdateRequest(user_id=1, updates=[])

    def test_progress_metrics_schema(self):
        """ProgressMetrics should serialise correctly."""
        metrics = ProgressMetrics(
            skills_gained_count=3,
            bucket_improvements=[
                BucketImprovement(
                    onet_code="15-1252.00",
                    occupation_title="Software Developer",
                    old_bucket="trainable",
                    new_bucket="ready_now",
                )
            ],
            estimated_time_to_ready=6,
        )
        assert metrics.skills_gained_count == 3
        assert len(metrics.bucket_improvements) == 1
        assert metrics.estimated_time_to_ready == 6


# =========================================================================
# Return Engagement helpers
# =========================================================================

class TestReturnEngagementHelpers:
    """Tests for return engagement helper functions."""

    def test_count_bucket_improvements_none(self):
        """No improvements when buckets haven't changed."""
        items = [
            {"bucket_at_save": "trainable", "current_bucket": "trainable"},
            {"bucket_at_save": "long_reskill", "current_bucket": "long_reskill"},
        ]
        assert _count_bucket_improvements(items) == 0

    def test_count_bucket_improvements_one(self):
        """One improvement when a bucket has changed for the better."""
        items = [
            {"bucket_at_save": "trainable", "current_bucket": "ready_now"},
            {"bucket_at_save": "long_reskill", "current_bucket": "long_reskill"},
        ]
        assert _count_bucket_improvements(items) == 1

    def test_count_bucket_improvements_regression_not_counted(self):
        """Regressions should not be counted as improvements."""
        items = [
            {"bucket_at_save": "ready_now", "current_bucket": "trainable"},
        ]
        assert _count_bucket_improvements(items) == 0

    def test_count_bucket_improvements_multiple(self):
        """Multiple improvements should all be counted."""
        items = [
            {"bucket_at_save": "long_reskill", "current_bucket": "trainable"},
            {"bucket_at_save": "trainable", "current_bucket": "ready_now"},
            {"bucket_at_save": "long_reskill", "current_bucket": "ready_now"},
        ]
        assert _count_bucket_improvements(items) == 3

    def test_determine_next_milestone_no_skills(self):
        """With < 5 skills, milestone should be 'rate 5 skills'."""
        milestone = _determine_next_milestone(
            skills_developed=2,
            occupations_tracked=0,
            bucket_improvements=0,
            saved_items=[],
        )
        assert milestone is not None
        assert milestone.target_metric == "skills_developed"
        assert milestone.target_value == 5.0
        assert milestone.progress_pct == pytest.approx(40.0)

    def test_determine_next_milestone_no_occupations(self):
        """With >= 5 skills but < 3 occupations, milestone should be 'save 3'."""
        milestone = _determine_next_milestone(
            skills_developed=5,
            occupations_tracked=1,
            bucket_improvements=0,
            saved_items=[{"current_bucket": "trainable"}],
        )
        assert milestone is not None
        assert milestone.target_metric == "occupations_tracked"

    def test_determine_next_milestone_improve_bucket(self):
        """With enough skills and occupations but no improvements, suggest improvement."""
        items = [{"current_bucket": "trainable"}] * 3
        milestone = _determine_next_milestone(
            skills_developed=7,
            occupations_tracked=3,
            bucket_improvements=0,
            saved_items=items,
        )
        assert milestone is not None
        assert milestone.target_metric == "bucket_improvements"

    def test_determine_next_milestone_rate_10(self):
        """With >= 5 but < 10 skills, >= 3 occs, some improvements: 'rate 10'."""
        items = [{"current_bucket": "ready_now"}] * 3
        milestone = _determine_next_milestone(
            skills_developed=7,
            occupations_tracked=3,
            bucket_improvements=1,
            saved_items=items,
        )
        assert milestone is not None
        assert milestone.target_metric == "skills_developed"
        assert milestone.target_value == 10.0

    def test_determine_next_milestone_all_done(self):
        """With >= 10 skills and improvements, milestone should be None."""
        milestone = _determine_next_milestone(
            skills_developed=15,
            occupations_tracked=5,
            bucket_improvements=2,
            saved_items=[{"current_bucket": "ready_now"}] * 5,
        )
        assert milestone is None

    def test_build_summary_text_new_user(self):
        """New users should see a welcome message."""
        text = _build_summary_text(
            days_active=1,
            skills_developed=0,
            occupations_tracked=0,
            bucket_improvements=0,
        )
        assert "Welcome" in text
        assert "rating" in text.lower() or "start" in text.lower()

    def test_build_summary_text_active_user(self):
        """Active users should see their stats."""
        text = _build_summary_text(
            days_active=30,
            skills_developed=8,
            occupations_tracked=3,
            bucket_improvements=1,
        )
        assert "30 day" in text
        assert "8 skill" in text
        assert "3 occupation" in text
        assert "1 occupation" in text  # bucket improvement

    def test_build_summary_text_no_occupations(self):
        """Users with no tracked occupations should not see occupation text."""
        text = _build_summary_text(
            days_active=5,
            skills_developed=3,
            occupations_tracked=0,
            bucket_improvements=0,
        )
        assert "tracking" not in text.lower()

    def test_next_milestone_progress_pct_capped(self):
        """progress_pct should never exceed 100."""
        milestone = _determine_next_milestone(
            skills_developed=4,
            occupations_tracked=0,
            bucket_improvements=0,
            saved_items=[],
        )
        assert milestone is not None
        assert milestone.progress_pct <= 100.0

    def test_progress_summary_response_schema(self):
        """ProgressSummaryResponse should construct correctly."""
        resp = ProgressSummaryResponse(
            user_id=1,
            days_active=10,
            skills_developed=5,
            occupations_tracked=2,
            bucket_improvements=1,
            recommendation_events=3,
            next_milestone=NextMilestone(
                description="Save 3 occupations.",
                target_metric="occupations_tracked",
                current_value=2.0,
                target_value=3.0,
                progress_pct=66.7,
            ),
            summary_text="Good progress!",
            generated_at=datetime.utcnow(),
        )
        assert resp.days_active == 10
        assert resp.next_milestone.progress_pct == pytest.approx(66.7)
