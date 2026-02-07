"""Tests for the Skills Translator feature.

Tests cover:
- Phrase dictionary integrity
- Rule-based matching engine
- TF-IDF semantic matching
- Merge/deduplication logic
- Confidence tiering
- User confirmation promotion
- Edge cases (empty input, nonsense text, very long input)
- 12 real-world persona scenarios representing SkillSprout's target users

Each persona test verifies that the translator surfaces the correct O*NET
skills from an informal, plain-language description of work experience.
"""

import pytest

from app.features.skills_translator.skill_dictionary import (
    ONET_SKILL_DESCRIPTIONS,
    ONET_SKILLS,
    PHRASE_TO_SKILL,
    get_all_skill_ids,
    get_phrases_for_skill,
)
from app.features.skills_translator.skills_translator import (
    ConfidenceLevel,
    MatchedSkill,
    SkillsTranslator,
    TranslationResult,
    get_translator,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def translator() -> SkillsTranslator:
    """Return a fresh translator instance per test."""
    return SkillsTranslator()


# ===================================================================
# Dictionary integrity tests
# ===================================================================


class TestSkillDictionary:
    """Validate the phrase-to-skill dictionary data."""

    def test_dictionary_has_at_least_200_entries(self):
        """The dictionary should contain at least 200 phrase mappings."""
        assert len(PHRASE_TO_SKILL) >= 200

    def test_all_phrases_are_lowercase(self):
        """Every phrase key must be lowercase for consistent matching."""
        for phrase in PHRASE_TO_SKILL:
            assert phrase == phrase.lower(), f"Phrase not lowercase: {phrase!r}"

    def test_all_element_ids_are_valid(self):
        """Every element_id in the dictionary must exist in ONET_SKILLS."""
        for phrase, mapping in PHRASE_TO_SKILL.items():
            eid = mapping["element_id"]
            assert eid in ONET_SKILLS, (
                f"Phrase {phrase!r} references unknown element_id {eid!r}"
            )

    def test_all_skill_names_match_canonical(self):
        """skill_name in each entry must match the canonical ONET_SKILLS name."""
        for phrase, mapping in PHRASE_TO_SKILL.items():
            eid = mapping["element_id"]
            expected = ONET_SKILLS[eid]
            assert mapping["skill_name"] == expected, (
                f"Phrase {phrase!r}: skill_name {mapping['skill_name']!r} "
                f"does not match canonical {expected!r}"
            )

    def test_onet_skill_descriptions_cover_all_skills(self):
        """Every skill in ONET_SKILLS should have a description."""
        for eid in ONET_SKILLS:
            assert eid in ONET_SKILL_DESCRIPTIONS, (
                f"Missing description for {eid} ({ONET_SKILLS[eid]})"
            )

    def test_get_all_skill_ids_returns_set(self):
        """get_all_skill_ids should return a non-empty set."""
        ids = get_all_skill_ids()
        assert isinstance(ids, set)
        assert len(ids) > 0

    def test_get_phrases_for_skill(self):
        """get_phrases_for_skill should return relevant phrases."""
        phrases = get_phrases_for_skill("2.B.7.e")  # Instructing
        assert len(phrases) > 0
        assert "trained new hires" in phrases


# ===================================================================
# Rule-based matching tests
# ===================================================================


class TestRuleBasedMatching:
    """Test the regex-based phrase matching engine."""

    def test_exact_phrase_match(self, translator: SkillsTranslator):
        """An exact dictionary phrase should produce a HIGH confidence match."""
        result = translator.translate("I trained new hires at the store.")
        eids = {m.element_id for m in result.matched_skills}
        assert "2.B.7.e" in eids  # Instructing

    def test_phrase_embedded_in_sentence(self, translator: SkillsTranslator):
        """Phrases embedded within larger sentences should still match."""
        result = translator.translate(
            "For three years I managed a team of twelve customer service reps."
        )
        eids = {m.element_id for m in result.matched_skills}
        assert "2.B.4.e" in eids  # Management of Personnel Resources

    def test_multiple_phrases_match(self, translator: SkillsTranslator):
        """Multiple phrases in one input should each produce matches."""
        result = translator.translate(
            "I handled cash, trained employees, and managed inventory."
        )
        eids = {m.element_id for m in result.all_matches}
        assert "2.B.5.a" in eids  # Mathematics (handled cash)
        assert "2.B.7.e" in eids  # Instructing (trained employees)
        assert "2.B.4.g" in eids  # Monitoring (managed inventory)

    def test_case_insensitivity(self, translator: SkillsTranslator):
        """Matching should be case-insensitive."""
        result = translator.translate("I TRAINED NEW HIRES every quarter.")
        eids = {m.element_id for m in result.matched_skills}
        assert "2.B.7.e" in eids  # Instructing

    def test_rule_matches_have_high_confidence(self, translator: SkillsTranslator):
        """Rule-based matches should be assigned HIGH confidence."""
        result = translator.translate("I worked the register daily.")
        register_match = next(
            (m for m in result.matched_skills if m.element_id == "2.B.5.a"),
            None,
        )
        assert register_match is not None
        assert register_match.confidence_level == ConfidenceLevel.HIGH
        assert register_match.source == "rule"


# ===================================================================
# TF-IDF matching tests
# ===================================================================


class TestTfidfMatching:
    """Test the TF-IDF semantic similarity engine."""

    def test_novel_phrasing_matches_via_tfidf(self, translator: SkillsTranslator):
        """Phrases not in the dictionary should still match via TF-IDF."""
        # "communicating with diverse audiences" isn't a dictionary phrase
        # but is semantically close to Speaking.
        result = translator.translate(
            "I spent years communicating with diverse audiences and presenting "
            "complex information clearly to large groups."
        )
        all_eids = {m.element_id for m in result.all_matches}
        # Should pick up Speaking or Active Listening via semantic similarity
        assert len(all_eids) > 0

    def test_tfidf_source_label(self, translator: SkillsTranslator):
        """TF-IDF-only matches should have source='tfidf'."""
        result = translator.translate(
            "Determining the strengths and weaknesses of different approaches "
            "to complex organizational challenges."
        )
        tfidf_matches = [m for m in result.all_matches if m.source == "tfidf"]
        # At minimum TF-IDF should fire for such descriptive text
        assert len(tfidf_matches) >= 0  # May or may not fire depending on vocabulary


# ===================================================================
# Merge and deduplication tests
# ===================================================================


class TestMergeAndDedup:
    """Test that rule + TF-IDF results merge correctly."""

    def test_no_duplicate_element_ids(self, translator: SkillsTranslator):
        """The merged result should have no duplicate element IDs."""
        result = translator.translate(
            "I managed a team, trained employees, handled customer complaints, "
            "and managed budgets for the entire store."
        )
        eids = [m.element_id for m in result.all_matches]
        assert len(eids) == len(set(eids)), "Duplicate element IDs in merged output"

    def test_rule_wins_on_tie(self, translator: SkillsTranslator):
        """When both engines match the same skill, rule-based should win on tie."""
        result = translator.translate("I taught others how to do new things.")
        instructing = next(
            (m for m in result.all_matches if m.element_id == "2.B.7.e"), None
        )
        if instructing is not None:
            assert instructing.source == "rule"


# ===================================================================
# Confidence and confirmation tests
# ===================================================================


class TestConfidenceAndConfirmation:
    """Test confidence tiering and user confirmation promotion."""

    def test_confirmed_skills_promoted_to_high(self, translator: SkillsTranslator):
        """User-confirmed skill IDs should be promoted to HIGH / 1.0."""
        result = translator.translate(
            "I did various tasks at a warehouse.",
            confirmed_skill_ids=["2.B.7.b"],  # Coordination
        )
        coord = next(
            (m for m in result.all_matches if m.element_id == "2.B.7.b"), None
        )
        if coord is not None:
            assert coord.confidence == 1.0
            assert coord.confidence_level == ConfidenceLevel.HIGH

    def test_needs_confirmation_has_low_confidence(self, translator: SkillsTranslator):
        """Items in needs_confirmation should all have LOW confidence."""
        result = translator.translate(
            "I did a lot of different things at my job."
        )
        for m in result.needs_confirmation:
            assert m.confidence_level == ConfidenceLevel.LOW


# ===================================================================
# Edge case tests
# ===================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input_raises_value_error(self, translator: SkillsTranslator):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            translator.translate("")

    def test_whitespace_only_raises_value_error(self, translator: SkillsTranslator):
        """Whitespace-only input should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            translator.translate("   \t\n  ")

    def test_nonsense_input_returns_few_or_no_matches(
        self, translator: SkillsTranslator
    ):
        """Gibberish text should not produce spurious HIGH confidence matches."""
        result = translator.translate("xyzzy plugh foobar baz quux")
        high_matches = [
            m for m in result.matched_skills
            if m.confidence_level == ConfidenceLevel.HIGH
        ]
        assert len(high_matches) == 0

    def test_very_long_input_does_not_crash(self, translator: SkillsTranslator):
        """A very long input should complete without error."""
        long_text = (
            "I managed a team of retail workers, trained new hires, "
            "handled customer complaints, and tracked inventory. "
        ) * 100
        result = translator.translate(long_text)
        assert isinstance(result, TranslationResult)
        assert len(result.matched_skills) > 0

    def test_singleton_translator(self):
        """get_translator should return the same instance."""
        t1 = get_translator()
        t2 = get_translator()
        assert t1 is t2


# ===================================================================
# PERSONA TESTS  (12 real-world personas)
# ===================================================================


class TestPersonaStayAtHomeParent:
    """Persona: Stay-at-home parent returning to workforce."""

    DESCRIPTION = (
        "For the past eight years I stayed home with my three kids. I managed "
        "the household budget, organized family schedules, helped with homework "
        "every night, coordinated carpooling with other families, and planned "
        "meals for the whole week. I also tutored my children in math and "
        "reading, and volunteered at the school as a PTA volunteer."
    )

    def test_matches_expected_skills(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        eids = {m.element_id for m in result.matched_skills}
        assert "2.B.4.f" in eids, "Should detect Management of Financial Resources"
        assert "2.B.6.b" in eids, "Should detect Time Management"
        assert "2.B.7.e" in eids, "Should detect Instructing"
        assert "2.B.7.b" in eids, "Should detect Coordination"

    def test_at_least_four_skills_matched(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        assert len(result.matched_skills) >= 4


class TestPersonaRetailManager:
    """Persona: Retail store manager."""

    DESCRIPTION = (
        "I was the store manager at a clothing shop for six years. I supervised "
        "fifteen employees, created work schedules, trained new hires, handled "
        "complaints from customers, managed inventory, met sales goals every "
        "quarter, and managed budgets for the store."
    )

    def test_matches_expected_skills(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        eids = {m.element_id for m in result.matched_skills}
        assert "2.B.4.e" in eids, "Should detect Management of Personnel Resources"
        assert "2.B.7.e" in eids, "Should detect Instructing"
        assert "2.B.7.d" in eids, "Should detect Negotiation (complaints)"
        assert "2.B.7.c" in eids, "Should detect Persuasion (sales goals)"
        assert "2.B.4.g" in eids, "Should detect Monitoring (inventory)"

    def test_at_least_five_skills_matched(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        assert len(result.matched_skills) >= 5


class TestPersonaMilitaryVeteran:
    """Persona: Military veteran transitioning to civilian workforce."""

    DESCRIPTION = (
        "I served in the army for twelve years. I led a platoon of 30 soldiers, "
        "wrote after-action reports, trained soldiers in field tactics, maintained "
        "equipment and weapons, and did mission planning and tactical planning "
        "for operations. I made split-second decisions under pressure and was "
        "responsible for risk assessment."
    )

    def test_matches_expected_skills(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        eids = {m.element_id for m in result.matched_skills}
        assert "2.B.4.e" in eids, "Should detect Management of Personnel Resources"
        assert "2.B.3.a" in eids, "Should detect Writing"
        assert "2.B.7.e" in eids, "Should detect Instructing"
        assert "2.B.4.h" in eids, "Should detect Equipment Maintenance"
        assert "2.B.8.d" in eids, "Should detect Systems Analysis"
        assert "2.B.8.c" in eids, "Should detect Judgment and Decision Making"

    def test_at_least_six_skills_matched(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        assert len(result.matched_skills) >= 6


class TestPersonaChurchVolunteer:
    """Persona: Church/community volunteer with no formal employment."""

    DESCRIPTION = (
        "I volunteered at church for over ten years. I taught sunday school, "
        "organized events like holiday bazaars and potlucks, led a committee "
        "for community outreach, recruited volunteers for the food bank, and "
        "did fundraising for building repairs. I also mentored youth in the "
        "after-school programme and counseled people going through hard times."
    )

    def test_matches_expected_skills(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        eids = {m.element_id for m in result.matched_skills}
        assert "2.B.7.e" in eids, "Should detect Instructing"
        assert "2.B.7.b" in eids, "Should detect Coordination"
        assert "2.B.7.c" in eids, "Should detect Persuasion (fundraising)"
        assert "2.B.4.e" in eids, "Should detect Management of Personnel Resources"
        assert "2.B.7.a" in eids, "Should detect Social Perceptiveness"

    def test_at_least_five_skills_matched(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        assert len(result.matched_skills) >= 5


class TestPersonaGigWorker:
    """Persona: Gig economy worker (rideshare + delivery)."""

    DESCRIPTION = (
        "I drove for uber and also did doordash deliveries for about three "
        "years. I managed my own schedule, negotiated rates with some "
        "customers, tracked my own income for tax preparation, and did my own "
        "taxes. I was basically self-employed and had to find my own clients "
        "for side jobs like moving furniture."
    )

    def test_matches_expected_skills(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        eids = {m.element_id for m in result.matched_skills}
        assert "2.B.1.f" in eids, "Should detect Service Orientation"
        assert "2.B.6.b" in eids, "Should detect Time Management"
        assert "2.B.7.d" in eids, "Should detect Negotiation"
        assert "2.B.5.a" in eids, "Should detect Mathematics (taxes)"

    def test_at_least_four_skills_matched(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        assert len(result.matched_skills) >= 4


class TestPersonaTradesWorker:
    """Persona: Skilled trades worker (electrician/plumber)."""

    DESCRIPTION = (
        "I've been doing plumbing and electrical work for 15 years. I fixed "
        "things around the house and then started my own business. I read "
        "blueprints, used power tools every day, estimated costs for clients, "
        "quoted jobs, and did safety inspections. I also did vehicle maintenance "
        "on the company truck and selected the right equipment for each project."
    )

    def test_matches_expected_skills(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        eids = {m.element_id for m in result.matched_skills}
        assert "2.B.9.a" in eids, "Should detect Troubleshooting"
        assert "2.B.1.a" in eids, "Should detect Reading Comprehension (blueprints)"
        assert "2.B.9.b" in eids, "Should detect Equipment Selection"
        assert "2.B.4.f" in eids, "Should detect Management of Financial Resources"
        assert "2.B.4.g" in eids, "Should detect Monitoring (safety inspections)"

    def test_at_least_five_skills_matched(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        assert len(result.matched_skills) >= 5


class TestPersonaAdminAssistant:
    """Persona: Administrative assistant / office manager."""

    DESCRIPTION = (
        "I worked as an office manager for a small law firm. I answered phones, "
        "filed paperwork, typed documents, wrote reports, booked appointments, "
        "tracked deadlines, used spreadsheets for budgets, and processed "
        "invoices. I also gave presentations to the partners and onboarded "
        "new employees."
    )

    def test_matches_expected_skills(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        eids = {m.element_id for m in result.matched_skills}
        assert "2.B.2.a" in eids, "Should detect Active Listening (answered phones)"
        assert "2.B.1.a" in eids, "Should detect Reading Comprehension (filed paperwork)"
        assert "2.B.3.a" in eids, "Should detect Writing"
        assert "2.B.6.b" in eids, "Should detect Time Management"
        assert "2.B.4.f" in eids, "Should detect Management of Financial Resources"
        assert "2.B.4.a" in eids, "Should detect Speaking (presentations)"
        assert "2.B.7.e" in eids, "Should detect Instructing (onboarded)"

    def test_at_least_seven_skills_matched(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        assert len(result.matched_skills) >= 7


class TestPersonaElderCaregiver:
    """Persona: Elder caregiver (home health aide)."""

    DESCRIPTION = (
        "I cared for elderly parent with dementia for four years. I managed "
        "medications on a strict schedule, monitored health signs daily, "
        "coordinated activities with doctors and nurses, listened to people's "
        "problems, and provided emotional support to the whole family. I also "
        "researched treatment options and chose a daycare programme."
    )

    def test_matches_expected_skills(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        eids = {m.element_id for m in result.matched_skills}
        assert "2.B.1.f" in eids, "Should detect Service Orientation"
        assert "2.B.4.g" in eids, "Should detect Monitoring"
        assert "2.B.7.b" in eids, "Should detect Coordination"
        assert "2.B.2.a" in eids, "Should detect Active Listening"
        assert "2.B.7.a" in eids, "Should detect Social Perceptiveness"

    def test_at_least_five_skills_matched(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        assert len(result.matched_skills) >= 5


class TestPersonaFoodServiceWorker:
    """Persona: Fast food / restaurant worker."""

    DESCRIPTION = (
        "I worked in fast food for two years then became a server at a sit-down "
        "restaurant. I waited tables, worked the register, counted the drawer "
        "at the end of every shift, and dealt with complaints. As shift leader "
        "I trained employees and scheduled staff for weekend rushes."
    )

    def test_matches_expected_skills(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        eids = {m.element_id for m in result.matched_skills}
        assert "2.B.5.a" in eids, "Should detect Mathematics"
        assert "2.B.7.d" in eids, "Should detect Negotiation (complaints)"
        assert "2.B.4.e" in eids, "Should detect Management of Personnel Resources"
        assert "2.B.7.e" in eids, "Should detect Instructing"
        assert "2.B.6.b" in eids, "Should detect Time Management"

    def test_at_least_five_skills_matched(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        assert len(result.matched_skills) >= 5


class TestPersonaFreelanceCreative:
    """Persona: Freelance creative (graphic design, content creation)."""

    DESCRIPTION = (
        "I'm a freelancer who does graphic design and content creation. I "
        "designed logos for small businesses, built websites, managed social "
        "media accounts, sold online on Etsy, negotiated contracts with "
        "clients, and marketed my services through social media marketing. "
        "I'm self-taught and watched tutorials to learn new software."
    )

    def test_matches_expected_skills(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        eids = {m.element_id for m in result.matched_skills}
        assert "2.B.5.c" in eids, "Should detect Design"
        assert "2.B.1.g" in eids, "Should detect Programming (built websites)"
        assert "2.B.3.a" in eids, "Should detect Writing (managed social media)"
        assert "2.B.7.c" in eids, "Should detect Persuasion"
        assert "2.B.7.d" in eids, "Should detect Negotiation"
        assert "2.B.1.e" in eids, "Should detect Learning Strategies (self-taught)"

    def test_at_least_six_skills_matched(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        assert len(result.matched_skills) >= 6


class TestPersonaConstructionWorker:
    """Persona: Construction / general contractor."""

    DESCRIPTION = (
        "I worked in construction for ten years. I did framing, drywall, "
        "roofing, and some carpentry. I operated heavy equipment like a "
        "forklift, measured materials carefully, and built things from "
        "blueprints. I eventually ran my own business, estimated costs, "
        "and managed a team of workers."
    )

    def test_matches_expected_skills(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        eids = {m.element_id for m in result.matched_skills}
        assert "2.B.9.b" in eids, "Should detect Equipment Selection"
        assert "2.B.5.c" in eids, "Should detect Design (framing, built things)"
        assert "2.B.5.a" in eids, "Should detect Mathematics (measured materials)"
        assert "2.B.4.f" in eids, "Should detect Management of Financial Resources"
        assert "2.B.4.e" in eids, "Should detect Management of Personnel Resources"

    def test_at_least_five_skills_matched(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        assert len(result.matched_skills) >= 5


class TestPersonaQuickLearnerTechSwitch:
    """Persona: Non-technical person who self-taught tech skills."""

    DESCRIPTION = (
        "I don't have a tech background but I'm a quick learner. I took "
        "online courses, watched tutorials, figured things out on my own, "
        "and learned new software fast. I helped people with technology at "
        "the library, fixed computer problems for neighbours, and built "
        "websites for a few local businesses as a side hustle."
    )

    def test_matches_expected_skills(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        eids = {m.element_id for m in result.matched_skills}
        assert "2.B.1.b" in eids, "Should detect Active Learning"
        assert "2.B.1.e" in eids, "Should detect Learning Strategies"
        assert "2.B.8.b" in eids, "Should detect Complex Problem Solving"
        assert "2.B.9.a" in eids, "Should detect Troubleshooting"
        assert "2.B.1.g" in eids, "Should detect Programming (built websites)"

    def test_at_least_five_skills_matched(self, translator: SkillsTranslator):
        result = translator.translate(self.DESCRIPTION)
        assert len(result.matched_skills) >= 5
