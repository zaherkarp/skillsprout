/**
 * Comprehensive test cases for the SkillSprout career trajectory engine.
 *
 * Now uses the full O*NET 28.3 database (1,016 occupations) with canonical
 * O*NET skill and knowledge names.
 *
 * Organized around themes from Anthropic's "The Macroeconomic Impact of
 * Artificial Intelligence" (2025), which uses O*NET task and skill data to
 * model AI-driven labor market transitions:
 *
 *   1. AI-adjacent upskilling
 *   2. Cross-domain pivots
 *   3. Declining-role escape paths
 *   4. Trades-to-tech transitions
 *   5. Long-term reskill with significant zone jumps
 *   6. Iterative refinement (multi-turn API sessions)
 *   7. Edge cases
 *   8. Anthropic paper-specific scenarios
 *   9. Healthcare transitions
 *  10. Full-dataset coverage (any-occupation searches)
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
  // ── Theme 1: AI-Adjacent Upskilling ──────────────────────────────

  {
    id: "ai-01",
    name: "Business Intelligence Analyst → Data Scientist path",
    description: "BI Analysts share statistics, data analysis, and programming skills with Data Scientists. A primary AI upskilling corridor.",
    theme: "AI-Adjacent Upskilling",
    steps: [{ action: "search", occupation: "Business Intelligence Analysts" }],
    expect: { success: true, minTrainable: 2, minTotal: 5 },
  },
  {
    id: "ai-02",
    name: "Software Developer → nearby tech roles",
    description: "Software Developers should see many Ready Now and Trainable matches in Computer & Mathematical occupations.",
    theme: "AI-Adjacent Upskilling",
    steps: [{ action: "search", occupation: "Software Developers" }],
    expect: { success: true, minReadyNow: 2, minTrainable: 3 },
  },
  {
    id: "ai-03",
    name: "Software Developer + added ML skills",
    description: "Adding machine learning and mathematics skills should shift more roles into Trainable from Long-Term.",
    theme: "AI-Adjacent Upskilling",
    steps: [
      { action: "search", occupation: "Software Developers" },
      { action: "add_skills", additionalSkills: ["mathematics", "physics"] },
    ],
    expect: { success: true, minTrainable: 2 },
  },
  {
    id: "ai-04",
    name: "Statistician → Data Scientist",
    description: "Statisticians share mathematics, critical thinking, and complex problem solving with Data Scientists. Should be Ready Now or Trainable.",
    theme: "AI-Adjacent Upskilling",
    steps: [{ action: "search", occupation: "Statisticians" }],
    expect: { success: true, minReadyNow: 1, minTotal: 10 },
  },
  {
    id: "ai-05",
    name: "Database Architects → Systems Analyst path",
    description: "Database Architects share computers/electronics and systems analysis with Computer Systems Analysts.",
    theme: "AI-Adjacent Upskilling",
    steps: [{ action: "search", occupation: "Database Architects" }],
    expect: { success: true, minReadyNow: 1, minTrainable: 2 },
  },
  {
    id: "ai-06",
    name: "QA Analysts → Software Developer",
    description: "QA Analysts share programming, problem solving, and computers/electronics with developers.",
    theme: "AI-Adjacent Upskilling",
    steps: [{ action: "search", occupation: "Software Quality Assurance" }],
    expect: { success: true, minTrainable: 2 },
  },
  {
    id: "ai-07",
    name: "Operations Research Analyst → Data Scientist",
    description: "Operations Research shares mathematics, statistics, and analytical methods with Data Science.",
    theme: "AI-Adjacent Upskilling",
    steps: [{ action: "search", occupation: "Operations Research Analysts" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "ai-08",
    name: "Computer Programmer → Software Developer",
    description: "Near-identical skill profiles. Should be Ready Now.",
    theme: "AI-Adjacent Upskilling",
    steps: [{ action: "search", occupation: "Computer Programmers" }],
    expect: { success: true, minReadyNow: 1 },
  },

  // ── Theme 2: Cross-Domain Pivots ─────────────────────────────────

  {
    id: "cross-01",
    name: "Registered Nurses → Health Services Manager",
    description: "Nurses share patient care, administration, and personnel management with health managers. Well-documented O*NET transition.",
    theme: "Cross-Domain Pivot",
    steps: [{ action: "search", occupation: "Registered Nurses" }],
    expect: { success: true, minTrainable: 2, minTotal: 10 },
  },
  {
    id: "cross-02",
    name: "Accountants → Financial Analysts",
    description: "High overlap in economics/accounting, mathematics, and analytical skills.",
    theme: "Cross-Domain Pivot",
    steps: [{ action: "search", occupation: "Accountants and Auditors" }],
    expect: { success: true, minReadyNow: 1, minTotal: 10 },
  },
  {
    id: "cross-03",
    name: "Mechanical Engineer → Industrial Engineer",
    description: "Shared engineering, math, and design skills. Same zone level. Should be Ready Now or Trainable.",
    theme: "Cross-Domain Pivot",
    steps: [{ action: "search", occupation: "Mechanical Engineers" }],
    expect: { success: true, minReadyNow: 1 },
  },
  {
    id: "cross-04",
    name: "Paralegal → Compliance Officer",
    description: "Paralegals have law/government, reading comprehension, and writing — overlaps with compliance roles.",
    theme: "Cross-Domain Pivot",
    steps: [{ action: "search", occupation: "Paralegals" }],
    expect: { success: true, minTrainable: 1 },
  },
  {
    id: "cross-05",
    name: "Graphic Designer → Web Developer",
    description: "Designers share design, communications/media, and computers/electronics with web developers.",
    theme: "Cross-Domain Pivot",
    steps: [{ action: "search", occupation: "Graphic Designers" }],
    expect: { success: true, minTrainable: 2 },
  },
  {
    id: "cross-06",
    name: "Sales Rep → Market Research Analyst",
    description: "Sales and marketing overlap via customer service, sales/marketing knowledge, and communication.",
    theme: "Cross-Domain Pivot",
    steps: [{ action: "search", occupation: "Sales Representatives, Wholesale" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "cross-07",
    name: "Civil Engineer → Construction Manager",
    description: "Civil engineers share building/construction, engineering, and mathematics with construction managers.",
    theme: "Cross-Domain Pivot",
    steps: [{ action: "search", occupation: "Civil Engineers" }],
    expect: { success: true, minTotal: 5 },
  },

  // ── Theme 3: Declining-Role Escape Paths ─────────────────────────

  {
    id: "decline-01",
    name: "File Clerks → multiple escape paths",
    description: "Clerical roles face high automation risk. Should see paths to administrative and support roles.",
    theme: "Declining-Role Escape",
    steps: [{ action: "search", occupation: "File Clerks" }],
    expect: { success: true, minTrainable: 1, minLongTermReskill: 2 },
  },
  {
    id: "decline-02",
    name: "Data Entry Keyers → upskill paths",
    description: "Data entry is highly automatable. With added skills, should unlock analyst roles.",
    theme: "Declining-Role Escape",
    steps: [
      { action: "search", occupation: "Data Entry Keyers" },
      { action: "add_skills", additionalSkills: ["mathematics", "computers and electronics"] },
    ],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "decline-03",
    name: "Tellers → multiple paths",
    description: "Bank tellers face declining demand. Should see customer service and finance-adjacent paths.",
    theme: "Declining-Role Escape",
    steps: [{ action: "search", occupation: "Tellers" }],
    expect: { success: true, minTrainable: 1, minLongTermReskill: 2 },
  },
  {
    id: "decline-04",
    name: "Word Processors → office transition paths",
    description: "Word processors and typists should see administrative assistant and secretarial paths.",
    theme: "Declining-Role Escape",
    steps: [{ action: "search", occupation: "Word Processors" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "decline-05",
    name: "Switchboard Operators → customer service paths",
    description: "Switchboard operators share customer service and communication skills with many roles.",
    theme: "Declining-Role Escape",
    steps: [{ action: "search", occupation: "Switchboard Operators" }],
    expect: { success: true, minTotal: 3 },
  },

  // ── Theme 4: Trades-to-Tech ──────────────────────────────────────

  {
    id: "trades-01",
    name: "Electricians → Engineering Technician",
    description: "Electricians share mechanical, engineering, and math knowledge with engineering technicians. Tests zone 3→3/4 transitions.",
    theme: "Trades-to-Tech",
    steps: [{ action: "search", occupation: "Electricians" }],
    expect: { success: true, minReadyNow: 1, minTotal: 10 },
  },
  {
    id: "trades-02",
    name: "HVAC Mechanics → multiple paths",
    description: "HVAC technicians share mechanical, troubleshooting, and engineering skills. Many lateral and upward paths.",
    theme: "Trades-to-Tech",
    steps: [{ action: "search", occupation: "Heating, Air Conditioning" }],
    expect: { success: true, minReadyNow: 1, minLongTermReskill: 2 },
  },
  {
    id: "trades-03",
    name: "Machinists → Industrial Engineer path",
    description: "Machinists share production, mechanical, and mathematics with industrial engineers.",
    theme: "Trades-to-Tech",
    steps: [{ action: "search", occupation: "Machinists" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "trades-04",
    name: "Welders → multiple paths",
    description: "Welders have production, mechanical, and engineering skills that transfer to manufacturing roles.",
    theme: "Trades-to-Tech",
    steps: [{ action: "search", occupation: "Welders" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "trades-05",
    name: "Plumbers → construction transition paths",
    description: "Plumbers share building/construction, mechanical, and troubleshooting with multiple trades.",
    theme: "Trades-to-Tech",
    steps: [{ action: "search", occupation: "Plumbers" }],
    expect: { success: true, minReadyNow: 1 },
  },
  {
    id: "trades-06",
    name: "Automotive Technicians → Robotics Technician",
    description: "Auto technicians share mechanical, troubleshooting, and electronics skills with robotics roles.",
    theme: "Trades-to-Tech",
    steps: [{ action: "search", occupation: "Automotive Service Technicians" }],
    expect: { success: true, minTotal: 5 },
  },

  // ── Theme 5: Long-Term Reskill ───────────────────────────────────

  {
    id: "reskill-01",
    name: "Stock Clerks → long-term paths (Zone 2→4+)",
    description: "Zone 2 warehouse/stock roles should have numerous long-term reskill options across domains.",
    theme: "Long-Term Reskill",
    steps: [{ action: "search", occupation: "Stock Clerks", categoryFilter: "long_term_reskill", maxPerCategory: 10 }],
    expect: { success: true, minLongTermReskill: 5 },
  },
  {
    id: "reskill-02",
    name: "Fast Food Cooks → long-term paths (Zone 1→4+)",
    description: "Zone 1 food prep roles should still have bridging skills (customer service, active listening) enabling long-term transitions.",
    theme: "Long-Term Reskill",
    steps: [{ action: "search", occupation: "Cooks, Fast Food", categoryFilter: "long_term_reskill", maxPerCategory: 10 }],
    expect: { success: true, minLongTermReskill: 3 },
  },
  {
    id: "reskill-03",
    name: "Cashiers → multiple long-term options",
    description: "Cashiers have customer service and communication skills. Should see many Long-Term paths.",
    theme: "Long-Term Reskill",
    steps: [{ action: "search", occupation: "Cashiers", categoryFilter: "long_term_reskill", maxPerCategory: 10 }],
    expect: { success: true, minLongTermReskill: 5 },
  },
  {
    id: "reskill-04",
    name: "Janitors → escape paths with low overlap",
    description: "Zone 1 cleaning roles have minimal skill overlap. Tests engine's ability to find creative paths via generic skills.",
    theme: "Long-Term Reskill",
    steps: [{ action: "search", occupation: "Janitors", maxPerCategory: 10, minScore: 0.08 }],
    expect: { success: true, minLongTermReskill: 3 },
  },
  {
    id: "reskill-05",
    name: "LPN → Registered Nurse → Physician Assistant chain",
    description: "Classic healthcare ladder. LPN shares nursing, patient care, and medical knowledge with RN.",
    theme: "Long-Term Reskill",
    steps: [{ action: "search", occupation: "Licensed Practical" }],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "reskill-06",
    name: "Social Workers → Epidemiologist (public health bridge)",
    description: "Social workers share psychology, sociology, and counseling. Epi requires biostatistics — a long-term reskill.",
    theme: "Long-Term Reskill",
    steps: [{ action: "search", occupation: "Child, Family, and School Social Workers", categoryFilter: "long_term_reskill", maxPerCategory: 10 }],
    expect: { success: true, minLongTermReskill: 3 },
  },
  {
    id: "reskill-07",
    name: "Taxi Drivers → long-term upskill options",
    description: "Zone 2 transportation roles. Tests that generic skills (customer service, geography) still produce matches.",
    theme: "Long-Term Reskill",
    steps: [{ action: "search", occupation: "Taxi Drivers", categoryFilter: "long_term_reskill", maxPerCategory: 10 }],
    expect: { success: true, minLongTermReskill: 3 },
  },
  {
    id: "reskill-08",
    name: "Security Guards → law enforcement/management paths",
    description: "Security guards share public safety and security knowledge. Tests protective service → management transitions.",
    theme: "Long-Term Reskill",
    steps: [{ action: "search", occupation: "Security Guards" }],
    expect: { success: true, minTotal: 5 },
  },

  // ── Theme 6: Iterative Refinement ────────────────────────────────

  {
    id: "iter-01",
    name: "Multi-turn: search → filter → more → add skills",
    description: "Full iterative flow: search, filter to trainable, paginate, add skills.",
    theme: "Iterative Refinement",
    steps: [
      { action: "search", occupation: "Software Developers", maxPerCategory: 3 },
      { action: "filter", categoryFilter: "trainable" },
      { action: "more", categoryFilter: "trainable" },
      { action: "add_skills", additionalSkills: ["mathematics", "engineering and technology"] },
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
      { action: "search", occupation: "Lawyers" },
    ],
    expect: { success: true, sourceTitle: "Lawyers" },
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
    description: "Filter to long_term_reskill only for Software Developers.",
    theme: "Iterative Refinement",
    steps: [{ action: "search", occupation: "Software Developers", categoryFilter: "long_term_reskill", maxPerCategory: 10 }],
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

  // ── Theme 7: Edge Cases ──────────────────────────────────────────

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
    description: "Searching by code '15-1252.00' should resolve to Software Developers.",
    theme: "Edge Case",
    steps: [{ action: "search", occupation: "15-1252.00" }],
    expect: { success: true, sourceTitle: "Software Developers" },
  },
  {
    id: "edge-05",
    name: "Very high minScore returns zero matches",
    description: "minScore=0.95 should filter almost everything.",
    theme: "Edge Case",
    steps: [{ action: "search", occupation: "Software Developers", minScore: 0.95 }],
    expect: { success: true },
  },
  {
    id: "edge-06",
    name: "List all occupations returns 1000+",
    description: "list_occupations should return the full O*NET catalog.",
    theme: "Edge Case",
    steps: [{ action: "list_occupations" }],
    expect: { success: true },
  },
  {
    id: "edge-07",
    name: "Case insensitive search",
    description: "Searching 'SOFTWARE DEVELOPERS' should work the same as 'Software Developers'.",
    theme: "Edge Case",
    steps: [{ action: "search", occupation: "SOFTWARE DEVELOPERS" }],
    expect: { success: true, sourceTitle: "Software Developers" },
  },

  // ── Theme 8: Anthropic Paper Scenarios ───────────────────────────

  {
    id: "anthro-01",
    name: "Computer User Support → System Admin/Security path",
    description: "IT support has high AI augmentation potential per Anthropic's paper. Should see lateral tech transitions.",
    theme: "Anthropic Paper Scenario",
    steps: [{ action: "search", occupation: "Computer User Support Specialists" }],
    expect: { success: true, minTrainable: 2 },
  },
  {
    id: "anthro-02",
    name: "Medical Records Specialists → health-tech paths",
    description: "Medical records is flagged as high AI exposure. Should have escape paths to health informatics and management.",
    theme: "Anthropic Paper Scenario",
    steps: [{ action: "search", occupation: "Medical Records Specialists" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "anthro-03",
    name: "Market Research → Data Scientist path",
    description: "Market research shares statistics, data analysis, and mathematics with data science.",
    theme: "Anthropic Paper Scenario",
    steps: [{ action: "search", occupation: "Market Research Analysts" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "anthro-04",
    name: "Technical Writers → adjacent creative/analytical roles",
    description: "Technical writing has high AI exposure. Should see transitions to instructional design and communications.",
    theme: "Anthropic Paper Scenario",
    steps: [{ action: "search", occupation: "Technical Writers" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "anthro-05",
    name: "Financial Analysts → Actuary (long-term)",
    description: "Finance roles face AI augmentation. Actuary requires specialized math but shares economics/accounting.",
    theme: "Anthropic Paper Scenario",
    steps: [{ action: "search", occupation: "Financial and Investment Analysts" }],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "anthro-06",
    name: "Environmental Scientists → Data Scientist (cross-domain)",
    description: "Environmental scientists have statistics, mathematics, and analytical skills that transfer to data science.",
    theme: "Anthropic Paper Scenario",
    steps: [{ action: "search", occupation: "Environmental Scientists" }],
    expect: { success: true, minTrainable: 2 },
  },
  {
    id: "anthro-07",
    name: "Biological Scientists → Epidemiologist",
    description: "Bio scientists share biology, research methods, and statistics with epidemiologists.",
    theme: "Anthropic Paper Scenario",
    steps: [{ action: "search", occupation: "Biologists" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "anthro-08",
    name: "Economists → Management Analyst",
    description: "Economists share analytical, mathematical, and communication skills with management analysts.",
    theme: "Anthropic Paper Scenario",
    steps: [{ action: "search", occupation: "Economists" }],
    expect: { success: true, minTotal: 5 },
  },

  // ── Theme 9: Healthcare Transitions ──────────────────────────────

  {
    id: "health-01",
    name: "Pharmacy Technicians → upward paths",
    description: "Pharmacy techs have medicine, customer service, and clerical skills. Tests healthcare ladder.",
    theme: "Healthcare Transition",
    steps: [{ action: "search", occupation: "Pharmacy Technicians" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "health-02",
    name: "Medical Assistants → broader healthcare roles",
    description: "Medical assistants have medicine, clinical, and administrative skills.",
    theme: "Healthcare Transition",
    steps: [{ action: "search", occupation: "Medical Assistants" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "health-03",
    name: "Dental Hygienists → lateral health paths",
    description: "Dental hygienists share patient care and medical knowledge. Tests cross-specialty transitions.",
    theme: "Healthcare Transition",
    steps: [{ action: "search", occupation: "Dental Hygienists" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "health-04",
    name: "Physical Therapists → adjacent health professions",
    description: "PT shares therapy/counseling, medicine, and education with related health roles.",
    theme: "Healthcare Transition",
    steps: [{ action: "search", occupation: "Physical Therapists" }],
    expect: { success: true, minReadyNow: 1 },
  },

  // ── Theme 10: Full-Dataset Coverage ──────────────────────────────

  {
    id: "full-01",
    name: "Farmers → diverse long-term options",
    description: "Tests agricultural occupations with unique skill profiles (food production, biology).",
    theme: "Full-Dataset Coverage",
    steps: [{ action: "search", occupation: "Farmers, Ranchers" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "full-02",
    name: "Police Officers → transition paths",
    description: "Protective service workers share public safety, law/government, and psychology.",
    theme: "Full-Dataset Coverage",
    steps: [{ action: "search", occupation: "Police and Sheriff" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "full-03",
    name: "Architects → related design/engineering",
    description: "Architects share design, engineering, and building/construction knowledge.",
    theme: "Full-Dataset Coverage",
    steps: [{ action: "search", occupation: "Architects, Except Landscape" }],
    expect: { success: true, minReadyNow: 1 },
  },
  {
    id: "full-04",
    name: "Clergy → community service transitions",
    description: "Clergy share psychology, therapy/counseling, and education. Tests social service transitions.",
    theme: "Full-Dataset Coverage",
    steps: [{ action: "search", occupation: "Clergy" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "full-05",
    name: "Chefs → food service management",
    description: "Head cooks share food production, administration, and personnel management.",
    theme: "Full-Dataset Coverage",
    steps: [{ action: "search", occupation: "Chefs and Head Cooks" }],
    expect: { success: true, minTotal: 3 },
  },
  {
    id: "full-06",
    name: "Veterinarians → related science paths",
    description: "Veterinarians have biology, medicine, and science skills that transfer broadly.",
    theme: "Full-Dataset Coverage",
    steps: [{ action: "search", occupation: "Veterinarians" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "full-07",
    name: "Pilots → aviation and management paths",
    description: "Airline pilots share transportation, physics, and geography knowledge.",
    theme: "Full-Dataset Coverage",
    steps: [{ action: "search", occupation: "Airline Pilots" }],
    expect: { success: true, minTotal: 3 },
  },
  {
    id: "full-08",
    name: "Fitness Trainers → health education paths",
    description: "Fitness trainers share education, psychology, and customer service skills.",
    theme: "Full-Dataset Coverage",
    steps: [{ action: "search", occupation: "Fitness Trainers" }],
    expect: { success: true, minTotal: 5 },
  },

  // ── Theme 11: Every SOC Major Group ──────────────────────────────
  // At least one test per SOC major group to ensure full coverage.

  {
    id: "soc-11",
    name: "Chief Executives (SOC 11) → management transitions",
    description: "Top-level managers share administration, judgment, and personnel skills broadly.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Chief Executives" }],
    expect: { success: true, minReadyNow: 3, minTotal: 10 },
  },
  {
    id: "soc-13",
    name: "Budget Analysts (SOC 13) → finance cluster",
    description: "Budget analysts share economics/accounting, mathematics, and English language skills.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Budget Analysts" }],
    expect: { success: true, minReadyNow: 2, minTotal: 10 },
  },
  {
    id: "soc-15",
    name: "Computer Network Architects (SOC 15) → tech cluster",
    description: "Network architects share computers/electronics, engineering/technology, and telecommunications.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Computer Network Architects" }],
    expect: { success: true, minReadyNow: 2, minTotal: 10 },
  },
  {
    id: "soc-17",
    name: "Chemical Engineers (SOC 17) → engineering cluster",
    description: "Chemical engineers share engineering/technology, chemistry, mathematics, and physics.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Chemical Engineers" }],
    expect: { success: true, minReadyNow: 1, minTotal: 10 },
  },
  {
    id: "soc-19",
    name: "Microbiologists (SOC 19) → science cluster",
    description: "Microbiologists share biology, chemistry, and research skills with many science roles.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Microbiologists" }],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "soc-21",
    name: "Mental Health Counselors (SOC 21) → counseling/social service cluster",
    description: "Mental health counselors share psychology, therapy/counseling, and sociology.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Mental Health Counselors" }],
    expect: { success: true, minReadyNow: 2, minTotal: 10 },
  },
  {
    id: "soc-23",
    name: "Lawyers (SOC 23) → legal cluster",
    description: "Lawyers share law/government, English language, and reading comprehension.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Lawyers" }],
    expect: { success: true, minReadyNow: 1, minTotal: 5 },
  },
  {
    id: "soc-25",
    name: "Elementary School Teachers (SOC 25) → education cluster",
    description: "Elementary teachers share education/training, psychology, English language, and sociology.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Elementary School Teachers" }],
    expect: { success: true, minReadyNow: 2, minTotal: 10 },
  },
  {
    id: "soc-27",
    name: "Sound Engineering Technicians (SOC 27) → media/arts cluster",
    description: "Sound engineers share communications/media, computers/electronics, and fine arts.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Sound Engineering Technicians" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "soc-29",
    name: "Occupational Therapists (SOC 29) → health practitioner cluster",
    description: "OTs share therapy/counseling, psychology, medicine, and education.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Occupational Therapists" }],
    expect: { success: true, minReadyNow: 2, minTotal: 10 },
  },
  {
    id: "soc-31",
    name: "Home Health Aides (SOC 31) → healthcare support transitions",
    description: "Home health aides share customer service, psychology, and medicine. Zone 2 starting point.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Home Health Aides" }],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "soc-33",
    name: "Firefighters (SOC 33) → protective service cluster",
    description: "Firefighters share public safety, medicine, mechanical, and building/construction knowledge.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Firefighters" }],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "soc-35",
    name: "Bartenders (SOC 35) → food service transitions",
    description: "Bartenders share customer service, sales/marketing, and food production skills.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Bartenders" }],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "soc-37",
    name: "Pest Control Workers (SOC 37) → building/grounds cluster",
    description: "Pest control workers share chemistry, biology, and customer service.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Pest Control Workers" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "soc-39",
    name: "Hairdressers (SOC 39) → personal care transitions",
    description: "Hairdressers share customer service, active listening, and service orientation.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Hairdressers" }],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "soc-41",
    name: "Insurance Sales Agents (SOC 41) → sales cluster",
    description: "Insurance agents share sales/marketing, customer service, economics/accounting, and law/government.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Insurance Sales Agents" }],
    expect: { success: true, minReadyNow: 1, minTotal: 10 },
  },
  {
    id: "soc-43",
    name: "Bookkeeping Clerks (SOC 43) → office/admin cluster",
    description: "Bookkeeping clerks share economics/accounting, mathematics, and clerical skills.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Bookkeeping" }],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "soc-45",
    name: "Agricultural Inspectors (SOC 45) → farming cluster",
    description: "Agricultural inspectors share food production, biology, and law/government.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Agricultural Inspectors" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "soc-47",
    name: "Carpenters (SOC 47) → construction cluster",
    description: "Carpenters share building/construction, mechanical, mathematics, and design skills.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Carpenters" }],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "soc-49",
    name: "Automotive Technicians (SOC 49) → maintenance/repair cluster",
    description: "Auto technicians share mechanical, engineering, and troubleshooting skills.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Automotive Service Technicians" }],
    expect: { success: true, minReadyNow: 1, minTotal: 10 },
  },
  {
    id: "soc-51",
    name: "Tool and Die Makers (SOC 51) → production cluster",
    description: "Tool and die makers share mechanical, mathematics, design, and production skills.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Tool and Die Makers" }],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "soc-53",
    name: "Airline Pilots (SOC 53) → transportation cluster",
    description: "Pilots share transportation, physics, geography, and public safety knowledge.",
    theme: "SOC Major Group Coverage",
    steps: [{ action: "search", occupation: "Airline Pilots" }],
    expect: { success: true, minTotal: 5 },
  },

  // ── Theme 12: Skill Augmentation Flows ───────────────────────────
  // Test that adding specific skills meaningfully changes results.

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
    name: "Accountant + programming → Quantitative Analyst path",
    description: "Adding programming to accounting should unlock fintech/quantitative roles.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Accountants and Auditors" },
      { action: "add_skills", additionalSkills: ["programming", "computers and electronics", "mathematics"] },
    ],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "aug-03",
    name: "Teacher + computers → Instructional Coordinator path",
    description: "Adding tech skills to teaching should surface instructional technology and EdTech roles.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Elementary School Teachers" },
      { action: "add_skills", additionalSkills: ["computers and electronics", "design"] },
    ],
    expect: { success: true, minTrainable: 3 },
  },
  {
    id: "aug-04",
    name: "Truck Driver + mechanical → Equipment Operator path",
    description: "Adding mechanical knowledge to driving should surface construction equipment and maintenance roles.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Heavy and Tractor-Trailer Truck Drivers" },
      { action: "add_skills", additionalSkills: ["mechanical", "engineering and technology"] },
    ],
    expect: { success: true, minTrainable: 3 },
  },
  {
    id: "aug-05",
    name: "Cashier + economics/accounting → Bookkeeping/Teller path",
    description: "Adding finance skills to cashier should surface financial clerk roles.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Cashiers" },
      { action: "add_skills", additionalSkills: ["economics and accounting", "mathematics"] },
    ],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "aug-06",
    name: "Graphic Designer + programming → Web Developer path",
    description: "Adding programming to design should unlock web development and UI engineering roles.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Graphic Designers" },
      { action: "add_skills", additionalSkills: ["programming", "computers and electronics"] },
    ],
    expect: { success: true, minTrainable: 3 },
  },
  {
    id: "aug-07",
    name: "Social Worker + biology/chemistry → Public Health path",
    description: "Adding science skills to social work should unlock epidemiology and health education roles.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Child, Family, and School Social Workers" },
      { action: "add_skills", additionalSkills: ["biology", "chemistry", "mathematics"] },
    ],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "aug-08",
    name: "Security Guard + law/government → Corrections/Probation path",
    description: "Adding legal knowledge to security should unlock criminal justice roles.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Security Guards" },
      { action: "add_skills", additionalSkills: ["law and government", "psychology"] },
    ],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "aug-09",
    name: "Cook + administration/management → Food Service Manager",
    description: "Adding management skills to cooking should surface supervisor and manager roles.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Cooks, Restaurant" },
      { action: "add_skills", additionalSkills: ["administration and management", "personnel and human resources"] },
    ],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "aug-10",
    name: "Paralegal + economics → Compliance Officer path",
    description: "Adding finance knowledge to legal background should unlock compliance and regulatory roles.",
    theme: "Skill Augmentation",
    steps: [
      { action: "search", occupation: "Paralegals" },
      { action: "add_skills", additionalSkills: ["economics and accounting", "administration and management"] },
    ],
    expect: { success: true, minTotal: 10 },
  },

  // ── Theme 13: Zone Ladder Tests ──────────────────────────────────
  // Verify that each Job Zone level produces appropriate transitions.

  {
    id: "zone-1a",
    name: "Zone 1: Dishwashers → zone 2 paths",
    description: "Dishwashers (zone 1) should have mostly trainable and long-term reskill options.",
    theme: "Zone Ladder",
    steps: [{ action: "search", occupation: "Dishwashers" }],
    expect: { success: true, minTrainable: 2, minLongTermReskill: 5 },
  },
  {
    id: "zone-1b",
    name: "Zone 1: Farmworkers → agricultural ladder",
    description: "Farmworkers should see agricultural inspectors, graders, and farm supervisors.",
    theme: "Zone Ladder",
    steps: [{ action: "search", occupation: "Farmworkers and Laborers" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "zone-2a",
    name: "Zone 2: Nursing Assistants → LPN/RN ladder",
    description: "Nursing assistants share medicine, psychology, and patient care with nursing roles.",
    theme: "Zone Ladder",
    steps: [{ action: "search", occupation: "Nursing Assistants" }],
    expect: { success: true, minReadyNow: 2, minTotal: 10 },
  },
  {
    id: "zone-2b",
    name: "Zone 2: Retail Salespersons → sales ladder",
    description: "Retail salespersons should see paths to first-line supervisors and specialized sales.",
    theme: "Zone Ladder",
    steps: [{ action: "search", occupation: "Retail Salespersons" }],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "zone-3a",
    name: "Zone 3: Dental Hygienists → zone 4 health paths",
    description: "Dental hygienists share medicine, biology, and chemistry with broader health roles.",
    theme: "Zone Ladder",
    steps: [{ action: "search", occupation: "Dental Hygienists" }],
    expect: { success: true, minTrainable: 3, minTotal: 10 },
  },
  {
    id: "zone-3b",
    name: "Zone 3: Web Developers → zone 4 tech paths",
    description: "Web developers should see software developer and designer roles as trainable transitions.",
    theme: "Zone Ladder",
    steps: [{ action: "search", occupation: "Web Developers" }],
    expect: { success: true, minReadyNow: 1, minTotal: 10 },
  },
  {
    id: "zone-4a",
    name: "Zone 4: Industrial Engineers → zone 5 paths",
    description: "Industrial engineers should see management and advanced engineering as trainable/long-term.",
    theme: "Zone Ladder",
    steps: [{ action: "search", occupation: "Industrial Engineers" }],
    expect: { success: true, minReadyNow: 2, minTotal: 10 },
  },
  {
    id: "zone-5a",
    name: "Zone 5: Physicians → lateral specialist transitions",
    description: "Physicians should see many ready-now transitions to other physician specialties.",
    theme: "Zone Ladder",
    steps: [{ action: "search", occupation: "Family Medicine Physicians" }],
    expect: { success: true, minReadyNow: 5, minTotal: 10 },
  },

  // ── Theme 14: Anthropic Paper Extended Scenarios ─────────────────
  // Additional scenarios modeling AI automation exposure and escape paths.

  {
    id: "anthro-09",
    name: "Bookkeeping Clerks → automation escape (high AI exposure)",
    description: "Bookkeeping is one of the most AI-exposed occupations per Anthropic's task analysis. Tests escape paths to analyst roles.",
    theme: "Anthropic Paper Extended",
    steps: [{ action: "search", occupation: "Bookkeeping" }],
    expect: { success: true, minTrainable: 3, minLongTermReskill: 3 },
  },
  {
    id: "anthro-10",
    name: "Paralegals → automation escape (high AI exposure)",
    description: "Legal research and document review are highly automatable. Tests paths to compliance, management.",
    theme: "Anthropic Paper Extended",
    steps: [{ action: "search", occupation: "Paralegals" }],
    expect: { success: true, minTrainable: 3 },
  },
  {
    id: "anthro-11",
    name: "Radiologic Technologists → automation-resistant health paths",
    description: "Imaging is augmented by AI but technologists have patient care skills. Tests lateral health transitions.",
    theme: "Anthropic Paper Extended",
    steps: [{ action: "search", occupation: "Radiologic Technologists" }],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "anthro-12",
    name: "Loan Officers → AI-augmented finance transitions",
    description: "Loan decisioning is highly AI-augmented. Tests transitions to advisory and analyst roles.",
    theme: "Anthropic Paper Extended",
    steps: [{ action: "search", occupation: "Loan Officers" }],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "anthro-13",
    name: "Claims Adjusters → insurance/compliance transitions",
    description: "Claims processing faces AI augmentation. Adjusters share law/government and customer service.",
    theme: "Anthropic Paper Extended",
    steps: [{ action: "search", occupation: "Claims Adjusters" }],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "anthro-14",
    name: "Tax Preparers → accounting ladder (seasonal automation risk)",
    description: "Tax preparation is heavily AI-impacted. Tests upward mobility to accountant and auditor roles.",
    theme: "Anthropic Paper Extended",
    steps: [{ action: "search", occupation: "Tax Preparers" }],
    expect: { success: true, minTrainable: 2, minLongTermReskill: 3 },
  },
  {
    id: "anthro-15",
    name: "Proofreaders → writing/communications escape",
    description: "Proofreading is highly AI-automatable. Tests paths to editing, communications, and technical writing.",
    theme: "Anthropic Paper Extended",
    steps: [{ action: "search", occupation: "Proofreaders" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "anthro-16",
    name: "Interpreters and Translators → AI disruption escape",
    description: "Translation is heavily impacted by LLMs. Tests paths leveraging foreign language + communication skills.",
    theme: "Anthropic Paper Extended",
    steps: [{ action: "search", occupation: "Interpreters and Translators" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "anthro-17",
    name: "Computer Programmers → developer evolution",
    description: "Code generation AI augments but doesn't eliminate programming. Tests evolution to architecture and engineering.",
    theme: "Anthropic Paper Extended",
    steps: [{ action: "search", occupation: "Computer Programmers" }],
    expect: { success: true, minReadyNow: 2, minTotal: 10 },
  },
  {
    id: "anthro-18",
    name: "Actuaries → low AI exposure but narrow",
    description: "Actuaries are AI-augmented but require deep mathematical expertise. Tests lateral paths.",
    theme: "Anthropic Paper Extended",
    steps: [{ action: "search", occupation: "Actuaries" }],
    expect: { success: true, minTotal: 5 },
  },

  // ── Theme 15: Multi-Step Career Chains ───────────────────────────
  // Test sequential career progressions — search first hop, then pivot.

  {
    id: "chain-01",
    name: "Warehouse Worker → Truck Driver → Logistics Analyst",
    description: "Three-step progression from zone 1 to zone 4. Tests two separate searches simulating career chain.",
    theme: "Career Chain",
    steps: [
      { action: "search", occupation: "Stockers and Order Fillers", maxPerCategory: 3 },
      { action: "reset" },
      { action: "search", occupation: "Heavy and Tractor-Trailer Truck Drivers", maxPerCategory: 3 },
      { action: "reset" },
      { action: "search", occupation: "Logisticians" },
    ],
    expect: { success: true, sourceTitle: "Logisticians", minTotal: 10 },
  },
  {
    id: "chain-02",
    name: "Medical Assistant → LPN → Registered Nurse",
    description: "Classic healthcare ladder. Each step is a natural progression.",
    theme: "Career Chain",
    steps: [
      { action: "search", occupation: "Medical Assistants", maxPerCategory: 3 },
      { action: "reset" },
      { action: "search", occupation: "Licensed Practical and Licensed Vocational Nurses" },
    ],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "chain-03",
    name: "Retail Sales → Insurance Sales → Financial Advisor",
    description: "Sales career ladder from retail to specialized advisory.",
    theme: "Career Chain",
    steps: [
      { action: "search", occupation: "Retail Salespersons", maxPerCategory: 3 },
      { action: "reset" },
      { action: "search", occupation: "Insurance Sales Agents", maxPerCategory: 3 },
      { action: "reset" },
      { action: "search", occupation: "Personal Financial Advisors" },
    ],
    expect: { success: true, sourceTitle: "Personal Financial Advisors", minTotal: 5 },
  },
  {
    id: "chain-04",
    name: "Construction Laborer → Carpenter → Construction Manager",
    description: "Construction career ladder from zone 1 to zone 4.",
    theme: "Career Chain",
    steps: [
      { action: "search", occupation: "Construction Laborers", maxPerCategory: 3 },
      { action: "reset" },
      { action: "search", occupation: "Carpenters", maxPerCategory: 3 },
      { action: "reset" },
      { action: "search", occupation: "Construction Managers" },
    ],
    expect: { success: true, sourceTitle: "Construction Managers", minTotal: 10 },
  },

  // ── Theme 16: Advanced Edge Cases ────────────────────────────────

  {
    id: "adv-edge-01",
    name: "Search by partial code prefix",
    description: "Searching '29-1141' should resolve to Registered Nurses.",
    theme: "Advanced Edge Case",
    steps: [{ action: "search", occupation: "29-1141" }],
    expect: { success: true, sourceTitle: "Registered Nurses" },
  },
  {
    id: "adv-edge-02",
    name: "Very low minScore returns massive results",
    description: "minScore=0.01 should return near-maximum matches for a well-connected occupation.",
    theme: "Advanced Edge Case",
    steps: [{ action: "search", occupation: "General and Operations Managers", minScore: 0.01, maxPerCategory: 5 }],
    expect: { success: true, minReadyNow: 5, minTrainable: 5, minLongTermReskill: 5 },
  },
  {
    id: "adv-edge-03",
    name: "High rarityWeight emphasizes specialist matches",
    description: "rarityWeight=3.0 should boost occupations sharing rare specialized skills.",
    theme: "Advanced Edge Case",
    steps: [{ action: "search", occupation: "Pharmacists", rarityWeight: 3.0 }],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "adv-edge-04",
    name: "Exclude all codes returns empty",
    description: "Excluding every code should produce zero results.",
    theme: "Advanced Edge Case",
    steps: [{ action: "search", occupation: "Software Developers", maxPerCategory: 3 },
            { action: "more", maxPerCategory: 3 },
            { action: "more", maxPerCategory: 3 },
            { action: "more", maxPerCategory: 3 },
            { action: "more", maxPerCategory: 3 }],
    expect: { success: true },
  },
  {
    id: "adv-edge-05",
    name: "Adding nonsense skill doesn't crash",
    description: "An unrecognized skill should be accepted without error — just won't match anything.",
    theme: "Advanced Edge Case",
    steps: [
      { action: "search", occupation: "Cashiers" },
      { action: "add_skills", additionalSkills: ["underwater basket weaving", "quantum yodeling"] },
    ],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "adv-edge-06",
    name: "Multiple add_skills calls accumulate",
    description: "Skills added in separate calls should stack, not replace.",
    theme: "Advanced Edge Case",
    steps: [
      { action: "search", occupation: "Electricians" },
      { action: "add_skills", additionalSkills: ["mathematics"] },
      { action: "add_skills", additionalSkills: ["engineering and technology"] },
      { action: "add_skills", additionalSkills: ["computers and electronics"] },
    ],
    expect: { success: true, minTotal: 10 },
  },
  {
    id: "adv-edge-07",
    name: "Same occupation searched twice resets seen codes",
    description: "Searching the same occupation again should reset exclusions and return full results.",
    theme: "Advanced Edge Case",
    steps: [
      { action: "search", occupation: "Accountants and Auditors", maxPerCategory: 2 },
      { action: "more", maxPerCategory: 2 },
      { action: "search", occupation: "Accountants and Auditors", maxPerCategory: 5 },
    ],
    expect: { success: true, minTotal: 10 },
  },

  // ── Theme 17: Niche and Uncommon Occupations ─────────────────────

  {
    id: "niche-01",
    name: "Funeral Home Managers → management transitions",
    description: "Niche management role shares administration, customer service, and psychology.",
    theme: "Niche Occupation",
    steps: [{ action: "search", occupation: "Funeral Home Managers" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "niche-02",
    name: "Gambling Dealers → service industry transitions",
    description: "Gambling dealers share customer service, mathematics, and service orientation.",
    theme: "Niche Occupation",
    steps: [{ action: "search", occupation: "Gambling Dealers" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "niche-03",
    name: "Embalmers → unique healthcare-adjacent transitions",
    description: "Embalmers share chemistry, biology, and medicine knowledge.",
    theme: "Niche Occupation",
    steps: [{ action: "search", occupation: "Embalmers" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "niche-04",
    name: "Astronomers → physics/math cluster",
    description: "Astronomers share physics, mathematics, and computers/electronics.",
    theme: "Niche Occupation",
    steps: [{ action: "search", occupation: "Astronomers" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "niche-05",
    name: "Foresters → environmental/agricultural cluster",
    description: "Foresters share biology, geography, and food production knowledge.",
    theme: "Niche Occupation",
    steps: [{ action: "search", occupation: "Foresters" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "niche-06",
    name: "Ship Engineers → maritime/mechanical transitions",
    description: "Ship engineers share mechanical, engineering, and transportation knowledge.",
    theme: "Niche Occupation",
    steps: [{ action: "search", occupation: "Ship Engineers" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "niche-07",
    name: "Nuclear Engineers → specialized engineering transitions",
    description: "Nuclear engineers share physics, engineering, mathematics, and chemistry.",
    theme: "Niche Occupation",
    steps: [{ action: "search", occupation: "Nuclear Engineers" }],
    expect: { success: true, minTotal: 5 },
  },
  {
    id: "niche-08",
    name: "Urban Planners → policy/design transitions",
    description: "Urban planners share geography, sociology, law/government, and design.",
    theme: "Niche Occupation",
    steps: [{ action: "search", occupation: "Urban and Regional Planners" }],
    expect: { success: true, minTotal: 10 },
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
