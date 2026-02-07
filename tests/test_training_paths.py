"""Tests for training path system: catalog, filtering, and path generation.

Covers constraint scenarios including:
    - Zero budget (free resources only)
    - No computer access
    - Limited hours per week
    - Short timeline
    - Combined constraints leading to infeasibility
    - Prerequisite ordering
    - Catalog completeness
"""
import pytest
from typing import List

from app.features.training_paths.training_catalog import (
    CostTier,
    DeliveryFormat,
    ResourceCategory,
    TrainingResource,
    get_catalog,
    get_resource_by_id,
    get_resources_by_skill,
    get_resources_by_skill_name,
    get_resources_by_category,
    get_resources_by_cost_tier,
    get_no_computer_resources,
    get_catalog_stats,
    SKILL_CODES,
)
from app.features.training_paths.resource_filter import (
    UserConstraints,
    filter_resources,
    filter_by_budget,
    filter_by_hours,
    filter_by_computer_access,
    filter_by_internet_access,
    filter_by_duration,
    filter_by_skill_codes,
)
from app.features.training_paths.path_generator import (
    PathGenerator,
    SkillGap,
    TrainingPath,
    PREREQUISITE_GRAPH,
)


# ==================== Catalog Tests ====================

class TestTrainingCatalog:
    """Tests for the training resource catalog."""

    def test_catalog_has_minimum_resources(self):
        """Catalog should contain at least 30 resources."""
        catalog = get_catalog()
        assert len(catalog) >= 30, (
            f"Catalog has {len(catalog)} resources, expected at least 30"
        )

    def test_catalog_has_free_certificates(self):
        """Catalog should include free certificate programs."""
        free_certs = get_resources_by_category(ResourceCategory.FREE_CERTIFICATE)
        assert len(free_certs) >= 5, "Should have at least 5 free certificate programs"

    def test_catalog_has_google_certificates(self):
        """Catalog should include Google professional certificates."""
        catalog = get_catalog()
        google_resources = [r for r in catalog if "Google" in r.provider]
        assert len(google_resources) >= 3, "Should have at least 3 Google certificates"

    def test_catalog_has_ibm_certificates(self):
        """Catalog should include IBM professional certificates."""
        catalog = get_catalog()
        ibm_resources = [r for r in catalog if "IBM" in r.provider]
        assert len(ibm_resources) >= 1, "Should have at least 1 IBM certificate"

    def test_catalog_has_meta_certificates(self):
        """Catalog should include Meta professional certificates."""
        catalog = get_catalog()
        meta_resources = [r for r in catalog if "Meta" in r.provider]
        assert len(meta_resources) >= 1, "Should have at least 1 Meta certificate"

    def test_catalog_has_government_programs(self):
        """Catalog should include government programs."""
        gov_programs = get_resources_by_category(ResourceCategory.GOVERNMENT_PROGRAM)
        assert len(gov_programs) >= 3, "Should have at least 3 government programs"

    def test_catalog_has_wioa_program(self):
        """Catalog should include WIOA training program."""
        wioa = get_resource_by_id("wioa-adult")
        assert wioa is not None, "WIOA Adult program should be in catalog"
        assert wioa.cost_tier == CostTier.FREE.value
        assert wioa.requires_computer is False

    def test_catalog_has_taa_program(self):
        """Catalog should include Trade Adjustment Assistance."""
        taa = get_resource_by_id("taa-program")
        assert taa is not None, "TAA program should be in catalog"
        assert taa.cost_tier == CostTier.FREE.value

    def test_catalog_has_veteran_benefits(self):
        """Catalog should include veteran education benefits."""
        gi_bill = get_resource_by_id("gi-bill")
        assert gi_bill is not None, "GI Bill should be in catalog"

        vet_tec = get_resource_by_id("vet-tec")
        assert vet_tec is not None, "VET TEC should be in catalog"

    def test_catalog_has_community_college_programs(self):
        """Catalog should include community college programs."""
        cc_programs = get_resources_by_category(ResourceCategory.COMMUNITY_COLLEGE)
        assert len(cc_programs) >= 3, "Should have at least 3 community college programs"

    def test_catalog_has_bootcamps(self):
        """Catalog should include coding bootcamps."""
        bootcamps = get_resources_by_category(ResourceCategory.BOOTCAMP)
        assert len(bootcamps) >= 2, "Should have at least 2 bootcamps"

    def test_catalog_has_self_directed(self):
        """Catalog should include self-directed learning platforms."""
        self_directed = get_resources_by_category(ResourceCategory.SELF_DIRECTED)
        assert len(self_directed) >= 3, "Should have at least 3 self-directed resources"

    def test_catalog_has_freecodecamp(self):
        """Catalog should include freeCodeCamp."""
        fcc = get_resource_by_id("freecodecamp-web")
        assert fcc is not None, "freeCodeCamp should be in catalog"
        assert fcc.cost_tier == CostTier.FREE.value
        assert fcc.estimated_cost_usd == 0.0

    def test_catalog_has_khan_academy(self):
        """Catalog should include Khan Academy."""
        khan = get_resource_by_id("khan-academy-math")
        assert khan is not None, "Khan Academy Math should be in catalog"
        assert khan.cost_tier == CostTier.FREE.value

    def test_all_resources_have_skill_codes(self):
        """Every resource should map to at least one O*NET skill code."""
        catalog = get_catalog()
        for resource in catalog:
            assert len(resource.skill_codes) >= 1, (
                f"Resource '{resource.name}' has no skill codes"
            )

    def test_all_resources_have_valid_fields(self):
        """Every resource should have required fields populated."""
        catalog = get_catalog()
        for resource in catalog:
            assert resource.id, f"Resource missing id: {resource.name}"
            assert resource.name, f"Resource missing name: {resource.id}"
            assert resource.provider, f"Resource missing provider: {resource.id}"
            assert resource.description, f"Resource missing description: {resource.id}"
            assert resource.hours_per_week > 0, f"Invalid hours for {resource.id}"
            assert resource.total_weeks > 0, f"Invalid weeks for {resource.id}"

    def test_lookup_by_skill_code(self):
        """Should find resources by O*NET skill code."""
        programming_resources = get_resources_by_skill(SKILL_CODES["programming"])
        assert len(programming_resources) >= 5, (
            "Should have at least 5 resources for programming"
        )

    def test_lookup_by_skill_name(self):
        """Should find resources by skill name."""
        math_resources = get_resources_by_skill_name("Mathematics")
        assert len(math_resources) >= 2, (
            "Should have at least 2 resources for Mathematics"
        )

    def test_no_computer_resources_exist(self):
        """Should have resources for users without computers."""
        no_computer = get_no_computer_resources()
        assert len(no_computer) >= 5, (
            "Should have at least 5 no-computer resources"
        )

    def test_catalog_stats(self):
        """Catalog stats should summarize correctly."""
        stats = get_catalog_stats()
        assert stats["total_resources"] >= 30
        assert stats["no_computer_required"] >= 5
        assert len(stats["by_category"]) >= 4
        assert len(stats["by_cost_tier"]) >= 2


# ==================== Resource Filter Tests ====================

class TestResourceFilter:
    """Tests for constraint-aware resource filtering."""

    def test_zero_budget_returns_only_free(self):
        """Zero budget should return only free resources."""
        constraints = UserConstraints(budget_usd=0.0)
        result = filter_resources(constraints)

        for resource in result.matching_resources:
            assert resource.estimated_cost_usd == 0.0, (
                f"Resource '{resource.name}' costs ${resource.estimated_cost_usd} "
                f"but should be free with $0 budget"
            )
        assert len(result.matching_resources) >= 10, (
            "Should have at least 10 free resources"
        )

    def test_no_computer_returns_accessible_resources(self):
        """No computer should return only resources not requiring a computer."""
        constraints = UserConstraints(has_computer=False)
        result = filter_resources(constraints)

        for resource in result.matching_resources:
            assert resource.requires_computer is False, (
                f"Resource '{resource.name}' requires a computer"
            )
        assert len(result.matching_resources) >= 5

    def test_no_computer_includes_library_programs(self):
        """No computer filter should include library programs."""
        constraints = UserConstraints(has_computer=False)
        result = filter_resources(constraints)

        categories = {r.category for r in result.matching_resources}
        assert "library_community" in categories, (
            "No-computer results should include library/community resources"
        )

    def test_no_computer_includes_government_programs(self):
        """No computer filter should include government programs."""
        constraints = UserConstraints(has_computer=False)
        result = filter_resources(constraints)

        categories = {r.category for r in result.matching_resources}
        assert "government_program" in categories, (
            "No-computer results should include government programs"
        )

    def test_no_computer_suggestions(self):
        """No computer filter should suggest library computer access."""
        constraints = UserConstraints(has_computer=False)
        result = filter_resources(constraints)

        assert any("library" in s.lower() for s in result.suggestions), (
            "Should suggest library for computer access"
        )

    def test_no_internet_filtering(self):
        """No internet should filter out online-only resources."""
        constraints = UserConstraints(has_internet=False)
        result = filter_resources(constraints)

        for resource in result.matching_resources:
            assert resource.requires_internet is False, (
                f"Resource '{resource.name}' requires internet"
            )

    def test_limited_hours_filtering(self):
        """Limited hours should filter out intensive programs."""
        constraints = UserConstraints(hours_per_week=5.0)
        result = filter_resources(constraints)

        for resource in result.matching_resources:
            assert resource.hours_per_week <= 5.0, (
                f"Resource '{resource.name}' requires {resource.hours_per_week} "
                f"hours/week but limit is 5"
            )

    def test_short_timeline_filtering(self):
        """Short timeline should filter out long programs."""
        constraints = UserConstraints(max_weeks=12)
        result = filter_resources(constraints)

        for resource in result.matching_resources:
            assert resource.total_weeks <= 12, (
                f"Resource '{resource.name}' takes {resource.total_weeks} weeks "
                f"but limit is 12"
            )

    def test_skill_code_filtering(self):
        """Should filter by target skill codes."""
        constraints = UserConstraints(
            target_skill_codes=[SKILL_CODES["programming"]],
        )
        result = filter_resources(constraints)

        for resource in result.matching_resources:
            assert SKILL_CODES["programming"] in resource.skill_codes, (
                f"Resource '{resource.name}' does not address programming"
            )

    def test_combined_zero_budget_no_computer(self):
        """Zero budget + no computer should still return usable resources."""
        constraints = UserConstraints(
            budget_usd=0.0,
            has_computer=False,
        )
        result = filter_resources(constraints)

        assert len(result.matching_resources) >= 3, (
            "Should have resources for zero budget + no computer"
        )
        for resource in result.matching_resources:
            assert resource.estimated_cost_usd == 0.0
            assert resource.requires_computer is False

    def test_format_filtering(self):
        """Should filter by delivery format."""
        constraints = UserConstraints(
            preferred_formats=[DeliveryFormat.IN_PERSON.value],
        )
        result = filter_resources(constraints)

        for resource in result.matching_resources:
            assert resource.delivery_format == DeliveryFormat.IN_PERSON.value, (
                f"Resource '{resource.name}' is {resource.delivery_format}, "
                f"not in_person"
            )

    def test_filter_result_includes_diagnostics(self):
        """Filter result should include applied filters and warnings."""
        constraints = UserConstraints(
            budget_usd=0.0,
            has_computer=False,
            max_weeks=4,
        )
        result = filter_resources(constraints)

        assert len(result.filters_applied) >= 2
        assert result.total_catalog_size >= 30


# ==================== Path Generator Tests ====================

class TestPathGenerator:
    """Tests for personalized training path generation."""

    @pytest.fixture
    def generator(self):
        """Create a path generator with the default catalog."""
        return PathGenerator()

    @pytest.fixture
    def programming_gaps(self) -> List[SkillGap]:
        """Programming skill gaps for testing."""
        return [
            SkillGap(
                skill_code=SKILL_CODES["programming"],
                skill_name="Programming",
                current_level=0.0,
                required_level=0.75,
                gap_weight=0.8,
            ),
            SkillGap(
                skill_code=SKILL_CODES["mathematics"],
                skill_name="Mathematics",
                current_level=0.25,
                required_level=0.75,
                gap_weight=0.5,
            ),
        ]

    @pytest.fixture
    def diverse_gaps(self) -> List[SkillGap]:
        """Diverse skill gaps covering multiple domains."""
        return [
            SkillGap(
                skill_code=SKILL_CODES["programming"],
                skill_name="Programming",
                gap_weight=0.8,
            ),
            SkillGap(
                skill_code=SKILL_CODES["speaking"],
                skill_name="Speaking",
                gap_weight=0.5,
            ),
            SkillGap(
                skill_code=SKILL_CODES["management_financial"],
                skill_name="Management of Financial Resources",
                gap_weight=0.6,
            ),
        ]

    def test_basic_path_generation(self, generator, programming_gaps):
        """Should generate a path with at least one step."""
        constraints = UserConstraints()
        path = generator.generate(skill_gaps=programming_gaps, constraints=constraints)

        assert len(path.steps) >= 1, "Path should have at least one step"
        assert path.total_weeks > 0
        assert len(path.skills_covered) >= 1

    def test_zero_budget_path(self, generator, programming_gaps):
        """Zero budget path should use only free resources."""
        constraints = UserConstraints(budget_usd=0.0)
        path = generator.generate(skill_gaps=programming_gaps, constraints=constraints)

        assert path.total_cost_usd == 0.0, (
            f"Path cost should be $0 but is ${path.total_cost_usd}"
        )
        for step in path.steps:
            assert step.estimated_cost_usd == 0.0, (
                f"Step '{step.resource.name}' costs ${step.estimated_cost_usd}"
            )
        assert len(path.steps) >= 1, "Should find free resources"

    def test_no_computer_path(self, generator):
        """No computer path should find accessible resources."""
        gaps = [
            SkillGap(
                skill_code=SKILL_CODES["active_learning"],
                skill_name="Active Learning",
                gap_weight=0.7,
            ),
            SkillGap(
                skill_code=SKILL_CODES["speaking"],
                skill_name="Speaking",
                gap_weight=0.5,
            ),
        ]
        constraints = UserConstraints(has_computer=False)
        path = generator.generate(skill_gaps=gaps, constraints=constraints)

        for step in path.steps:
            assert step.resource.requires_computer is False, (
                f"Step '{step.resource.name}' requires a computer"
            )

    def test_limited_hours_path(self, generator, programming_gaps):
        """Limited hours path should respect weekly time constraint."""
        constraints = UserConstraints(hours_per_week=5.0)
        path = generator.generate(skill_gaps=programming_gaps, constraints=constraints)

        for step in path.steps:
            assert step.resource.hours_per_week <= 5.0, (
                f"Step '{step.resource.name}' requires "
                f"{step.resource.hours_per_week} hours/week"
            )

    def test_short_timeline_path(self, generator, programming_gaps):
        """Short timeline should either fit or report infeasibility."""
        constraints = UserConstraints(max_weeks=8)
        path = generator.generate(skill_gaps=programming_gaps, constraints=constraints)

        if path.steps:
            assert path.total_weeks <= 8, (
                f"Path takes {path.total_weeks} weeks but limit is 8"
            )

    def test_prerequisite_ordering(self, generator):
        """Prerequisites should come before dependent skills."""
        # Mathematics is a prerequisite for programming
        gaps = [
            SkillGap(
                skill_code=SKILL_CODES["programming"],
                skill_name="Programming",
                gap_weight=0.8,
            ),
            SkillGap(
                skill_code=SKILL_CODES["mathematics"],
                skill_name="Mathematics",
                gap_weight=0.3,
            ),
        ]
        constraints = UserConstraints()

        # Check that prioritization puts math before programming
        ordered = generator._prioritize_gaps(gaps)
        math_idx = next(
            i for i, g in enumerate(ordered)
            if g.skill_code == SKILL_CODES["mathematics"]
        )
        prog_idx = next(
            i for i, g in enumerate(ordered)
            if g.skill_code == SKILL_CODES["programming"]
        )
        assert math_idx < prog_idx, (
            "Mathematics should be prioritized before Programming "
            "because it is a prerequisite"
        )

    def test_cumulative_tracking(self, generator, diverse_gaps):
        """Steps should track cumulative cost and weeks."""
        constraints = UserConstraints()
        path = generator.generate(skill_gaps=diverse_gaps, constraints=constraints)

        if len(path.steps) >= 2:
            for i in range(1, len(path.steps)):
                assert (
                    path.steps[i].cumulative_weeks
                    >= path.steps[i - 1].cumulative_weeks
                ), "Cumulative weeks should be non-decreasing"
                assert (
                    path.steps[i].cumulative_cost_usd
                    >= path.steps[i - 1].cumulative_cost_usd
                ), "Cumulative cost should be non-decreasing"

    def test_empty_gaps_returns_complete(self, generator):
        """Empty skill gaps should return a complete, empty path."""
        constraints = UserConstraints()
        path = generator.generate(skill_gaps=[], constraints=constraints)

        assert path.is_complete is True
        assert len(path.steps) == 0

    def test_infeasible_constraints_reported(self, generator):
        """Extremely tight constraints should report infeasibility clearly."""
        gaps = [
            SkillGap(
                skill_code=SKILL_CODES["programming"],
                skill_name="Programming",
                gap_weight=0.8,
            ),
        ]
        constraints = UserConstraints(
            budget_usd=0.0,
            has_computer=False,
            has_internet=False,
            max_weeks=1,
            hours_per_week=1.0,
        )
        path = generator.generate(skill_gaps=gaps, constraints=constraints)

        # Should either find something or clearly state infeasibility
        if not path.steps:
            assert (
                len(path.infeasibility_reasons) > 0
                or len(path.warnings) > 0
                or len(path.skills_not_covered) > 0
            ), "Should explain why no path was generated"

    def test_path_steps_have_rationale(self, generator, programming_gaps):
        """Every step should have a rationale."""
        constraints = UserConstraints()
        path = generator.generate(skill_gaps=programming_gaps, constraints=constraints)

        for step in path.steps:
            assert step.rationale, (
                f"Step {step.step_number} missing rationale"
            )

    def test_path_no_duplicate_resources(self, generator, diverse_gaps):
        """Path should not use the same resource twice."""
        constraints = UserConstraints()
        path = generator.generate(skill_gaps=diverse_gaps, constraints=constraints)

        resource_ids = [step.resource.id for step in path.steps]
        assert len(resource_ids) == len(set(resource_ids)), (
            "Path contains duplicate resources"
        )

    def test_combined_zero_budget_limited_hours(self, generator, programming_gaps):
        """Zero budget + limited hours should still produce results."""
        constraints = UserConstraints(
            budget_usd=0.0,
            hours_per_week=10.0,
        )
        path = generator.generate(skill_gaps=programming_gaps, constraints=constraints)

        assert path.total_cost_usd == 0.0
        for step in path.steps:
            assert step.resource.hours_per_week <= 10.0

    def test_path_feasibility_flag(self, generator, programming_gaps):
        """Path should correctly set is_feasible flag."""
        # Generous constraints should be feasible
        constraints = UserConstraints()
        path = generator.generate(skill_gaps=programming_gaps, constraints=constraints)

        if path.steps and not path.infeasibility_reasons:
            assert path.is_feasible is True


# ==================== Prerequisite Graph Tests ====================

class TestPrerequisiteGraph:
    """Tests for the prerequisite dependency graph."""

    def test_programming_has_prerequisites(self):
        """Programming should list prerequisites."""
        prereqs = PREREQUISITE_GRAPH.get(SKILL_CODES["programming"], [])
        assert len(prereqs) >= 1, "Programming should have prerequisites"
        assert SKILL_CODES["mathematics"] in prereqs or SKILL_CODES["critical_thinking"] in prereqs

    def test_no_circular_dependencies(self):
        """Prerequisite graph should have no circular dependencies."""
        visited = set()
        path_stack = set()

        def has_cycle(node):
            if node in path_stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            path_stack.add(node)
            for prereq in PREREQUISITE_GRAPH.get(node, []):
                if has_cycle(prereq):
                    return True
            path_stack.discard(node)
            return False

        for skill_code in PREREQUISITE_GRAPH:
            visited.clear()
            path_stack.clear()
            assert not has_cycle(skill_code), (
                f"Circular dependency detected starting from {skill_code}"
            )
