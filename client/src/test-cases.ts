/**
 * Test cases for the SkillSprout career trajectory engine.
 *
 * Healthcare & Data Edition — tests cover transitions within the filtered
 * dataset of ~256 healthcare, medical, data, and science occupations from
 * the O*NET 28.3 database.
 *
 * Themes:
 *   1. Healthcare career ladders
 *   2. Data & informatics transitions
 *   3. Clinical cross-specialty pivots
 *   4. Healthcare support upskilling
 *   5. Science-to-healthcare bridges
 *   6. Mental health & social service paths
 *   7. Healthcare management transitions
 *   8. Iterative refinement (multi-turn API sessions)
 *   9. Edge cases
 *  10. Skill augmentation flows
 *  11. Zone ladder tests (healthcare focus)
 */

import { callApi, type ApiRequest } from "./api";
import type { TrajectoryResponse, TransitionMatch } from "./engine";

export interface TestCase {
  id: string;
  name: string;
  description: string;
  theme: string;
  steps: ApiRequest[];
  expect: TestExpectation;
}

interface TestExpectation {
  success: boolean;
  minReadyNow?: number;
  minTrainable?: number;
  minLongTermReskill?: number;
  /** Total matches across all categories */
  minTotal?: number;
  shouldContain?: string[];
  shouldNotContain?: string[];
  sourceTitle?: string;
  errorContains?: string;
}

// ────────────────────────────────────────────────────────────────────

export const testCases: TestCase[] = [
  // ── Theme 1: Healthcare Career Ladders ──────────────────────────

  {
    id: "hcl-01",
    name: "LPN → Registered Nurse → Physician Assistant chain",
    description: "Classic healthcare ladder. LPN shares nursing, patient care, and medical knowledge with RN.",
    theme: "Healthcare Career Ladder",
    steps: [{ action: "search", occupation: "Licensed Practical" }],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "hcl-02",
    name: "Nursing Assistants → LPN/RN ladder",
    description: "Nursing assistants share medicine, psychology, and patient care with nursing roles.",
    theme: "Healthcare Career Ladder",
    steps: [{ action: "search", occupation: "Nursing Assistants" }],
    expect: { success: true, minReadyNow: 2, minTotal: 10 },
  },
  {
    id: "hcl-03",
    name: "Medical Assistants → broader healthcare roles",
    description: "Medical assistants have medicine, clinical, and administrative skills.",
    theme: "Healthcare Career Ladder",
    steps: [{ action: "search", occupation: "Medical Assistants" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "hcl-04",
    name: "Pharmacy Technicians → upward paths",
    description: "Pharmacy techs have medicine, customer service, and clerical skills. Tests healthcare ladder.",
    theme: "Healthcare Career Ladder",
    steps: [{ action: "search", occupation: "Pharmacy Technicians" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "hcl-05",
    name: "Medical Assistant → LPN → Registered Nurse chain",
    description: "Classic healthcare ladder. Each step is a natural progression.",
    theme: "Healthcare Career Ladder",
    steps: [
      { action: "search", occupation: "Medical Assistants", maxPerCategory: 3 },
      { action: "reset" },
      { action: "search", occupation: "Licensed Practical and Licensed Vocational Nurses" },
    ],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "hcl-06",
    name: "Home Health Aides → healthcare support transitions",
    description: "Home health aides share customer service, psychology, and medicine. Zone 2 starting point.",
    theme: "Healthcare Career Ladder",
    steps: [{ action: "search", occupation: "Home Health Aides" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "hcl-07",
    name: "Registered Nurses → Health Services Manager",
    description: "Nurses share patient care, administration, and personnel management with health managers.",
    theme: "Healthcare Career Ladder",
    steps: [{ action: "search", occupation: "Registered Nurses" }],
    expect: { success: true, minTrainable: 2, minTotal: 10 },
  },
  {
    id: "hcl-08",
    name: "Physicians → lateral specialist transitions",
    description: "Physicians should see many ready-now transitions to other physician specialties.",
    theme: "Healthcare Career Ladder",
    steps: [{ action: "search", occupation: "Family Medicine Physicians" }],
    expect: { success: true, minReadyNow: 5, minTotal: 10 },
  },

  // ── Theme 2: Data & Informatics Transitions ─────────────────────

  {
    id: "data-01",
    name: "Data Scientists → nearby data/analytics roles",
    description: "Data Scientists should see many Ready Now and Trainable matches in Computer & Mathematical occupations.",
    theme: "Data & Informatics",
    steps: [{ action: "search", occupation: "Data Scientists" }],
    expect: { success: true, minReadyNow: 2, minTrainable: 2 },
  },
  {
    id: "data-02",
    name: "Software Developers → health informatics tech roles",
    description: "Software Developers should see many Ready Now matches among Computer & Mathematical occupations.",
    theme: "Data & Informatics",
    steps: [{ action: "search", occupation: "Software Developers" }],
    expect: { success: true, minReadyNow: 2, minTrainable: 2 },
  },
  {
    id: "data-03",
    name: "Database Architects → Systems Analyst path",
    description: "Database Architects share computers/electronics and systems analysis with Computer Systems Analysts.",
    theme: "Data & Informatics",
    steps: [{ action: "search", occupation: "Database Architects" }],
    expect: { success: true, minReadyNow: 1, minTrainable: 2 },
  },
  {
    id: "data-04",
    name: "Statisticians → Data Scientist",
    description: "Statisticians share mathematics, critical thinking, and complex problem solving with Data Scientists.",
    theme: "Data & Informatics",
    steps: [{ action: "search", occupation: "Statisticians" }],
    expect: { success: true, minReadyNow: 1, minTotal: 5 },
  },
  {
    id: "data-05",
    name: "Operations Research Analyst → Data Scientist",
    description: "Operations Research shares mathematics, statistics, and analytical methods with Data Science.",
    theme: "Data & Informatics",
    steps: [{ action: "search", occupation: "Operations Research Analysts" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "data-06",
    name: "Computer Programmer → Software Developer",
    description: "Near-identical skill profiles. Should be Ready Now.",
    theme: "Data & Informatics",
    steps: [{ action: "search", occupation: "Computer Programmers" }],
    expect: { success: true, minReadyNow: 1 },
  },
  {
    id: "data-07",
    name: "Medical Records Specialists → health-tech paths",
    description: "Medical records has high AI exposure. Should have paths to health informatics and management.",
    theme: "Data & Informatics",
    steps: [{ action: "search", occupation: "Medical Records Specialists" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "data-08",
    name: "Computer Network Architects → tech cluster",
    description: "Network architects share computers/electronics, engineering/technology, and telecommunications.",
    theme: "Data & Informatics",
    steps: [{ action: "search", occupation: "Computer Network Architects" }],
    expect: { success: true, minReadyNow: 2, minTotal: 5 },
  },

  // ── Theme 3: Clinical Cross-Specialty Pivots ────────────────────

  {
    id: "clin-01",
    name: "Dental Hygienists → lateral health paths",
    description: "Dental hygienists share patient care and medical knowledge. Tests cross-specialty transitions.",
    theme: "Clinical Cross-Specialty",
    steps: [{ action: "search", occupation: "Dental Hygienists" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "clin-02",
    name: "Physical Therapists → adjacent health professions",
    description: "PT shares therapy/counseling, medicine, and education with related health roles.",
    theme: "Clinical Cross-Specialty",
    steps: [{ action: "search", occupation: "Physical Therapists" }],
    expect: { success: true, minReadyNow: 1 },
  },
  {
    id: "clin-03",
    name: "Occupational Therapists → health practitioner cluster",
    description: "OTs share therapy/counseling, psychology, medicine, and education.",
    theme: "Clinical Cross-Specialty",
    steps: [{ action: "search", occupation: "Occupational Therapists" }],
    expect: { success: true, minReadyNow: 2, minTotal: 10 },
  },
  {
    id: "clin-04",
    name: "Radiologic Technologists → imaging/health-tech paths",
    description: "Imaging is augmented by AI but technologists have patient care skills. Tests lateral health transitions.",
    theme: "Clinical Cross-Specialty",
    steps: [{ action: "search", occupation: "Radiologic Technologists" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "clin-05",
    name: "Respiratory Therapists → adjacent therapy roles",
    description: "Respiratory therapists share medicine, therapy/counseling, and biology with other health roles.",
    theme: "Clinical Cross-Specialty",
    steps: [{ action: "search", occupation: "Respiratory Therapists" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "clin-06",
    name: "Pharmacists → specialized health transitions",
    description: "Pharmacists share medicine/dentistry, chemistry, biology with many health practitioners.",
    theme: "Clinical Cross-Specialty",
    steps: [{ action: "search", occupation: "Pharmacists", rarityWeight: 3.0 }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "clin-07",
    name: "Veterinarians → related science/health paths",
    description: "Veterinarians have biology, medicine, and science skills that transfer to human health roles.",
    theme: "Clinical Cross-Specialty",
    steps: [{ action: "search", occupation: "Veterinarians" }],
    expect: { success: true, minTotal: 5 },
  },

  // ── Theme 4: Healthcare Support Upskilling ──────────────────────

  {
    id: "supp-01",
    name: "Medical Secretaries → health admin paths",
    description: "Medical secretaries share clerical, customer service, and medical knowledge.",
    theme: "Healthcare Support Upskilling",
    steps: [{ action: "search", occupation: "Medical Secretaries" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "supp-02",
    name: "Data Entry Keyers → upskill paths with added skills",
    description: "Data entry is highly automatable. With added skills, should unlock analyst and health data roles.",
    theme: "Healthcare Support Upskilling",
    steps: [
      { action: "search", occupation: "Data Entry Keyers" },
      { action: "add_skills", additionalSkills: ["mathematics", "computers and electronics"] },
    ],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "supp-03",
    name: "Psychiatric Technicians → mental health paths",
    description: "Psychiatric techs share psychology, therapy/counseling, and medicine with counseling roles.",
    theme: "Healthcare Support Upskilling",
    steps: [{ action: "search", occupation: "Psychiatric Technicians" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "supp-04",
    name: "Dental Assistants → upward dental/health paths",
    description: "Dental assistants share medicine/dentistry and customer service with broader health roles.",
    theme: "Healthcare Support Upskilling",
    steps: [{ action: "search", occupation: "Dental Assistants" }],
    expect: { success: true, minTotal: 5 },
  },

  // ── Theme 5: Science-to-Healthcare Bridges ──────────────────────

  {
    id: "sci-01",
    name: "Biological Scientists → Epidemiologist",
    description: "Bio scientists share biology, research methods, and statistics with epidemiologists.",
    theme: "Science-to-Healthcare Bridge",
    steps: [{ action: "search", occupation: "Biologists" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "sci-02",
    name: "Microbiologists → science cluster",
    description: "Microbiologists share biology, chemistry, and research skills with many science and health roles.",
    theme: "Science-to-Healthcare Bridge",
    steps: [{ action: "search", occupation: "Microbiologists" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "sci-03",
    name: "Environmental Scientists → health/data cross-domain",
    description: "Environmental scientists have statistics, mathematics, and analytical skills transferable to health data.",
    theme: "Science-to-Healthcare Bridge",
    steps: [{ action: "search", occupation: "Environmental Scientists" }],
    expect: { success: true, minTrainable: 2 },
  },
  {
    id: "sci-04",
    name: "Epidemiologists → public health cluster",
    description: "Epidemiologists share biology, mathematics, and sociology with public health roles.",
    theme: "Science-to-Healthcare Bridge",
    steps: [{ action: "search", occupation: "Epidemiologists" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "sci-05",
    name: "Biochemists → pharma/clinical research paths",
    description: "Biochemists share biology, chemistry, and mathematics with pharmaceutical and clinical roles.",
    theme: "Science-to-Healthcare Bridge",
    steps: [{ action: "search", occupation: "Biochemists and Biophysicists" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "sci-06",
    name: "Social Workers → Epidemiologist (public health bridge)",
    description: "Social workers share psychology, sociology, and counseling. Epi requires biostatistics — a long-term reskill.",
    theme: "Science-to-Healthcare Bridge",
    steps: [{ action: "search", occupation: "Healthcare Social Workers", categoryFilter: "long_term_reskill", maxPerCategory: 10 }],
    expect: { success: true, minLongTermReskill: 2 },
  },

  // ── Theme 6: Mental Health & Social Service Paths ───────────────

  {
    id: "mh-01",
    name: "Mental Health Counselors → counseling/social service cluster",
    description: "Mental health counselors share psychology, therapy/counseling, and sociology.",
    theme: "Mental Health & Social Service",
    steps: [{ action: "search", occupation: "Mental Health Counselors" }],
    expect: { success: true, minReadyNow: 2, minTotal: 5 },
  },
  {
    id: "mh-02",
    name: "Substance Abuse Counselors → broader counseling paths",
    description: "Substance abuse counselors share therapy/counseling, psychology, and sociology.",
    theme: "Mental Health & Social Service",
    steps: [{ action: "search", occupation: "Substance Abuse and Behavioral Disorder Counselors" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "mh-03",
    name: "Rehabilitation Counselors → health/social paths",
    description: "Rehab counselors share therapy/counseling, psychology, and education with related roles.",
    theme: "Mental Health & Social Service",
    steps: [{ action: "search", occupation: "Rehabilitation Counselors" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "mh-04",
    name: "Community Health Workers → health education paths",
    description: "Community health workers share education, psychology, and sociology.",
    theme: "Mental Health & Social Service",
    steps: [{ action: "search", occupation: "Community Health Workers" }],
    expect: { success: true, minTotal: 5 },
  },

  // ── Theme 7: Healthcare Management Transitions ──────────────────

  {
    id: "mgmt-01",
    name: "Medical and Health Services Managers → leadership paths",
    description: "Health services managers share administration, management, and personnel skills broadly.",
    theme: "Healthcare Management",
    steps: [{ action: "search", occupation: "Medical and Health Services Managers" }],
    expect: { success: true, minReadyNow: 1, minTotal: 5 },
  },
  {
    id: "mgmt-02",
    name: "Clinical Research Coordinators → science management",
    description: "CRCs share biology, administration, and English language with research and management roles.",
    theme: "Healthcare Management",
    steps: [{ action: "search", occupation: "Clinical Research Coordinators" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "mgmt-03",
    name: "Computer and Information Systems Managers → data leadership",
    description: "CIS Managers share computers/electronics, administration, and engineering/technology.",
    theme: "Healthcare Management",
    steps: [{ action: "search", occupation: "Computer and Information Systems Managers" }],
    expect: { success: true, minReadyNow: 1, minTotal: 5 },
  },
  {
    id: "mgmt-04",
    name: "Quality Control Systems Managers → compliance/management paths",
    description: "QC managers share administration, management, and engineering/technology.",
    theme: "Healthcare Management",
    steps: [{ action: "search", occupation: "Quality Control Systems Managers" }],
    expect: { success: true, minTotal: 5 },
  },

  // ── Theme 8: Iterative Refinement ───────────────────────────────

  {
    id: "iter-01",
    name: "Multi-turn: search → filter → more → add skills",
    description: "Full iterative flow: search, filter to trainable, paginate, add skills.",
    theme: "Iterative Refinement",
    steps: [
      { action: "search", occupation: "Registered Nurses", maxPerCategory: 3 },
      { action: "filter", categoryFilter: "trainable" },
      { action: "more", categoryFilter: "trainable" },
      { action: "add_skills", additionalSkills: ["computers and electronics", "programming"] },
    ],
    expect: { success: true, minTrainable: 1 },
  },
  {
    id: "iter-02",
    name: "Multi-turn: search → reset → new search",
    description: "Session reset should produce clean results for the new occupation.",
    theme: "Iterative Refinement",
    steps: [
      { action: "search", occupation: "Registered Nurses" },
      { action: "reset" },
      { action: "search", occupation: "Epidemiologists" },
    ],
    expect: { success: true, sourceTitle: "Epidemiologists" },
  },
  {
    id: "iter-03",
    name: "Pagination: exhaust results across turns",
    description: "Request maxPerCategory=2 then 'more' repeatedly. Tests excludeCodes prevents duplicates.",
    theme: "Iterative Refinement",
    steps: [
      { action: "search", occupation: "Data Scientists", maxPerCategory: 2 },
      { action: "more", maxPerCategory: 2 },
      { action: "more", maxPerCategory: 2 },
    ],
    expect: { success: true },
  },
  {
    id: "iter-04",
    name: "Category filter isolation",
    description: "Filter to long_term_reskill only for Data Scientists.",
    theme: "Iterative Refinement",
    steps: [{ action: "search", occupation: "Data Scientists", categoryFilter: "long_term_reskill", maxPerCategory: 10 }],
    expect: { success: true, minLongTermReskill: 3 },
  },
  {
    id: "iter-05",
    name: "Preferred categories boost",
    description: "Search with preferred Healthcare categories from a Software Developer.",
    theme: "Iterative Refinement",
    steps: [{ action: "search", occupation: "Software Developers", preferredCategories: ["Healthcare Practitioners"], categoryFilter: "long_term_reskill", maxPerCategory: 10 }],
    expect: { success: true, minLongTermReskill: 2 },
  },

  // ── Theme 9: Edge Cases ─────────────────────────────────────────

  {
    id: "edge-01",
    name: "Unknown occupation returns error + suggestions",
    description: "Non-existent occupation should fail gracefully with 'did you mean' suggestions.",
    theme: "Edge Case",
    steps: [{ action: "search", occupation: "Underwater Basket Weaver" }],
    expect: { success: false, errorContains: "Could not find occupation" },
  },
  {
    id: "edge-02",
    name: "Empty string fails gracefully",
    description: "Empty input should return error, not crash.",
    theme: "Edge Case",
    steps: [{ action: "search", occupation: "" }],
    expect: { success: false, errorContains: "Could not find occupation" },
  },
  {
    id: "edge-03",
    name: "Partial title match resolves correctly",
    description: "Searching 'nurse' should resolve to a nursing occupation.",
    theme: "Edge Case",
    steps: [{ action: "search", occupation: "nurse" }],
    expect: { success: true },
  },
  {
    id: "edge-04",
    name: "O*NET code-based search",
    description: "Searching by code '29-1141.00' should resolve to Registered Nurses.",
    theme: "Edge Case",
    steps: [{ action: "search", occupation: "29-1141.00" }],
    expect: { success: true, sourceTitle: "Registered Nurses" },
  },
  {
    id: "edge-05",
    name: "Very high minScore returns zero matches",
    description: "minScore=0.95 should filter almost everything.",
    theme: "Edge Case",
    steps: [{ action: "search", occupation: "Registered Nurses", minScore: 0.95 }],
    expect: { success: true },
  },
  {
    id: "edge-06",
    name: "List all occupations returns healthcare/data set",
    description: "list_occupations should return the filtered healthcare & data catalog.",
    theme: "Edge Case",
    steps: [{ action: "list_occupations" }],
    expect: { success: true },
  },
  {
    id: "edge-07",
    name: "Case insensitive search",
    description: "Searching 'REGISTERED NURSES' should work the same as 'Registered Nurses'.",
    theme: "Edge Case",
    steps: [{ action: "search", occupation: "REGISTERED NURSES" }],
    expect: { success: true, sourceTitle: "Registered Nurses" },
  },
  {
    id: "edge-08",
    name: "Non-healthcare occupation not found",
    description: "Searching for an occupation outside the healthcare/data filter should fail.",
    theme: "Edge Case",
    steps: [{ action: "search", occupation: "Carpenters" }],
    expect: { success: false, errorContains: "Could not find occupation" },
  },
  {
    id: "edge-09",
    name: "Partial code match resolves correctly",
    description: "Searching '29-1141' should resolve to Registered Nurses.",
    theme: "Edge Case",
    steps: [{ action: "search", occupation: "29-1141" }],
    expect: { success: true, sourceTitle: "Registered Nurses" },
  },

  // ── Theme 10: Skill Augmentation Flows ──────────────────────────

  {
    id: "aug-01",
    name: "Nurse + computers/electronics → Health Informatics",
    description: "Adding tech skills to a nursing background should surface health informatics and clinical data roles.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Registered Nurses" },
      { action: "add_skills", additionalSkills: ["computers and electronics", "telecommunications"] },
    ],
    expect: { success: true, minTrainable: 3 },
  },
  {
    id: "aug-02",
    name: "Medical Assistant + programming → health data roles",
    description: "Adding programming to medical background should unlock health informatics paths.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Medical Assistants" },
      { action: "add_skills", additionalSkills: ["programming", "computers and electronics", "mathematics"] },
    ],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "aug-03",
    name: "Data Scientist + biology → bioinformatics paths",
    description: "Adding biology to data science should surface bioinformatics and epidemiological roles.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Data Scientists" },
      { action: "add_skills", additionalSkills: ["biology", "chemistry"] },
    ],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "aug-04",
    name: "Social Worker + biology/chemistry → Public Health path",
    description: "Adding science skills to social work should unlock epidemiology and health education roles.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Healthcare Social Workers" },
      { action: "add_skills", additionalSkills: ["biology", "chemistry", "mathematics"] },
    ],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "aug-05",
    name: "Software Developer + medicine → health informatics",
    description: "Adding medical knowledge to tech skills should surface clinical informatics and health IT roles.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Software Developers" },
      { action: "add_skills", additionalSkills: ["medicine and dentistry", "biology"] },
    ],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "aug-06",
    name: "Epidemiologist + programming → data science bridge",
    description: "Adding tech skills to epidemiology should strengthen matches with data and informatics roles.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Epidemiologists" },
      { action: "add_skills", additionalSkills: ["programming", "computers and electronics"] },
    ],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "aug-07",
    name: "Adding nonsense skill doesn't crash",
    description: "An unrecognized skill should be accepted without error — just won't match anything.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Registered Nurses" },
      { action: "add_skills", additionalSkills: ["underwater basket weaving", "quantum yodeling"] },
    ],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "aug-08",
    name: "Multiple add_skills calls accumulate",
    description: "Skills added in separate calls should stack, not replace.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Medical Records Specialists" },
      { action: "add_skills", additionalSkills: ["mathematics"] },
      { action: "add_skills", additionalSkills: ["programming"] },
      { action: "add_skills", additionalSkills: ["computers and electronics"] },
    ],
    expect: { success: true, minTotal: 5 },
  },

  // ── Theme 11: Zone Ladder Tests (Healthcare Focus) ──────────────

  {
    id: "zone-2a",
    name: "Zone 2: Nursing Assistants → zone 3/4 health paths",
    description: "Nursing assistants share medicine, psychology, and patient care with nursing roles.",
    theme: "Zone Ladder",
    steps: [{ action: "search", occupation: "Nursing Assistants" }],
    expect: { success: true, minReadyNow: 2, minTotal: 10 },
  },
  {
    id: "zone-2b",
    name: "Zone 2: Home Health Aides → upward paths",
    description: "Home health aides share customer service, psychology, and medicine.",
    theme: "Zone Ladder",
    steps: [{ action: "search", occupation: "Home Health Aides" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "zone-3a",
    name: "Zone 3: Dental Hygienists → zone 4 health paths",
    description: "Dental hygienists share medicine, biology, and chemistry with broader health roles.",
    theme: "Zone Ladder",
    steps: [{ action: "search", occupation: "Dental Hygienists" }],
    expect: { success: true, minTrainable: 2, minTotal: 5 },
  },
  {
    id: "zone-3b",
    name: "Zone 3: Web Developers → zone 4 data/tech paths",
    description: "Web developers should see software developer and data analyst roles as trainable transitions.",
    theme: "Zone Ladder",
    steps: [{ action: "search", occupation: "Web Developers" }],
    expect: { success: true, minReadyNow: 1, minTotal: 5 },
  },
  {
    id: "zone-4a",
    name: "Zone 4: Biostatisticians → zone 5 paths",
    description: "Biostatisticians should see science management and advanced research as trainable/long-term.",
    theme: "Zone Ladder",
    steps: [{ action: "search", occupation: "Biostatisticians" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "zone-5a",
    name: "Zone 5: Physicians → lateral specialist transitions",
    description: "Physicians should see many ready-now transitions to other physician specialties.",
    theme: "Zone Ladder",
    steps: [{ action: "search", occupation: "Family Medicine Physicians" }],
    expect: { success: true, minReadyNow: 5, minTotal: 10 },
  },

  // ── Theme 12: Advanced Edge Cases ───────────────────────────────

  {
    id: "adv-01",
    name: "Very low minScore returns maximum matches",
    description: "minScore=0.01 should return near-maximum matches for a well-connected occupation.",
    theme: "Advanced Edge Case",
    steps: [{ action: "search", occupation: "Registered Nurses", minScore: 0.01, maxPerCategory: 5 }],
    expect: { success: true, minReadyNow: 5, minTrainable: 5, minLongTermReskill: 5 },
  },
  {
    id: "adv-02",
    name: "Same occupation searched twice resets seen codes",
    description: "Searching the same occupation again should reset exclusions and return full results.",
    theme: "Advanced Edge Case",
    steps: [
      { action: "search", occupation: "Data Scientists", maxPerCategory: 2 },
      { action: "more", maxPerCategory: 2 },
      { action: "search", occupation: "Data Scientists", maxPerCategory: 5 },
    ],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "adv-03",
    name: "High rarityWeight emphasizes specialist matches",
    description: "rarityWeight=3.0 should boost occupations sharing rare specialized skills.",
    theme: "Advanced Edge Case",
    steps: [{ action: "search", occupation: "Pharmacists", rarityWeight: 3.0 }],
    expect: { success: true, minTotal: 5 },
  },
];

// ── Test Runner ────────────────────────────────────────────────────

export interface TestResult {
  id: string;
  name: string;
  passed: boolean;
  failures: string[];
  response?: TrajectoryResponse;
}

export function runTestCase(tc: TestCase): TestResult {
  const failures: string[] = [];
  let lastApiResult: any = null;
  let sessionId: string | undefined;

  for (const step of tc.steps) {
    const req = { ...step, sessionId };
    const result = callApi(req);
    sessionId = result.sessionId;
    lastApiResult = result;
  }

  if (!lastApiResult) {
    return { id: tc.id, name: tc.name, passed: false, failures: ["No API result"] };
  }

  const data = lastApiResult.data;

  if ("occupations" in data) {
    if (tc.expect.success && data.occupations.length === 0) {
      failures.push("Expected occupations list but got empty");
    }
    return { id: tc.id, name: tc.name, passed: failures.length === 0, failures };
  }

  if ("message" in data) {
    return { id: tc.id, name: tc.name, passed: true, failures: [] };
  }

  const resp = data as TrajectoryResponse;

  if (resp.success !== tc.expect.success) {
    failures.push(`Expected success=${tc.expect.success}, got ${resp.success}`);
  }

  if (!resp.success && tc.expect.errorContains) {
    if (!resp.error?.includes(tc.expect.errorContains)) {
      failures.push(`Expected error containing "${tc.expect.errorContains}", got "${resp.error}"`);
    }
  }

  if (resp.success) {
    if (tc.expect.sourceTitle && resp.source?.title !== tc.expect.sourceTitle) {
      failures.push(`Expected source "${tc.expect.sourceTitle}", got "${resp.source?.title}"`);
    }

    if (tc.expect.minReadyNow !== undefined && resp.readyNow.length < tc.expect.minReadyNow) {
      failures.push(`Expected ≥${tc.expect.minReadyNow} Ready Now, got ${resp.readyNow.length}`);
    }
    if (tc.expect.minTrainable !== undefined && resp.trainable.length < tc.expect.minTrainable) {
      failures.push(`Expected ≥${tc.expect.minTrainable} Trainable, got ${resp.trainable.length}`);
    }
    if (tc.expect.minLongTermReskill !== undefined && resp.longTermReskill.length < tc.expect.minLongTermReskill) {
      failures.push(`Expected ≥${tc.expect.minLongTermReskill} Long-Term Reskill, got ${resp.longTermReskill.length}`);
    }

    if (tc.expect.minTotal !== undefined) {
      const total = resp.readyNow.length + resp.trainable.length + resp.longTermReskill.length;
      if (total < tc.expect.minTotal) {
        failures.push(`Expected ≥${tc.expect.minTotal} total matches, got ${total}`);
      }
    }

    if (tc.expect.shouldContain) {
      const allTitles = [...resp.readyNow, ...resp.trainable, ...resp.longTermReskill].map(m => m.occupation.title);
      for (const expected of tc.expect.shouldContain) {
        if (!allTitles.includes(expected)) {
          failures.push(`Expected results to contain "${expected}"`);
        }
      }
    }

    if (tc.expect.shouldNotContain) {
      const allTitles = [...resp.readyNow, ...resp.trainable, ...resp.longTermReskill].map(m => m.occupation.title);
      for (const excluded of tc.expect.shouldNotContain) {
        if (allTitles.includes(excluded)) {
          failures.push(`Expected results NOT to contain "${excluded}"`);
        }
      }
    }
  }

  return { id: tc.id, name: tc.name, passed: failures.length === 0, failures, response: resp };
}

export function runAllTests(): TestResult[] {
  return testCases.map(runTestCase);
}
