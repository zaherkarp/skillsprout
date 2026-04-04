/**
 * O*NET occupation and skill dataset for the SkillSprout career trajectory engine.
 *
 * Filtered to healthcare, medical, data, and closely related occupations from the
 * O*NET 28.3 database (~256 occupations from the full 1,016). Includes:
 *   - Healthcare Practitioners & Healthcare Support (full categories)
 *   - Life, Physical & Social Science (full category)
 *   - Computer & Mathematical (full category)
 *   - Healthcare-adjacent Management, Community & Social Service, Education,
 *     Business & Finance, and Office & Admin roles (cherry-picked)
 *
 * Data source: O*NET Database 28.3
 *   https://www.onetcenter.org/database.html
 *   License: CC BY 4.0
 */

import onetData from "../data/onet-full.json";

// ── Types ──────────────────────────────────────────────────────────

export interface OnetSkill {
  name: string;        // O*NET canonical skill/knowledge name
  importance: number;  // Importance rating (1.0–5.0)
}

export interface OnetOccupation {
  code: string;        // O*NET-SOC code (e.g., "15-1252.00")
  title: string;       // Standard Occupational Title
  zone: number;        // Job Zone 1-5 (education/training proxy)
  category: string;    // SOC major group label
  skills: OnetSkill[]; // Top 15 skills by importance
}

// ── Healthcare / Data filter ──────────────────────────────────────

/** SOC major groups included in full. */
const INCLUDED_CATEGORIES = new Set([
  "Healthcare Practitioners",
  "Healthcare Support",
  "Life, Physical & Social Science",
  "Computer & Mathematical",
]);

/** Individual codes cherry-picked from other categories. */
const INCLUDED_CODES = new Set([
  // Management — healthcare / data adjacent
  "11-1011.00", // Chief Executives
  "11-1021.00", // General and Operations Managers
  "11-3021.00", // Computer and Information Systems Managers
  "11-3051.01", // Quality Control Systems Managers
  "11-3131.00", // Training and Development Managers
  "11-9111.00", // Medical and Health Services Managers
  "11-9121.00", // Natural Sciences Managers
  "11-9121.01", // Clinical Research Coordinators
  "11-9151.00", // Social and Community Service Managers
  "11-9179.01", // Fitness and Wellness Coordinators
  "11-9199.01", // Regulatory Affairs Managers
  "11-9199.02", // Compliance Managers

  // Community & Social Service — health-related
  "21-1011.00", // Substance Abuse and Behavioral Disorder Counselors
  "21-1013.00", // Marriage and Family Therapists
  "21-1014.00", // Mental Health Counselors
  "21-1015.00", // Rehabilitation Counselors
  "21-1022.00", // Healthcare Social Workers
  "21-1023.00", // Mental Health and Substance Abuse Social Workers
  "21-1091.00", // Health Education Specialists
  "21-1094.00", // Community Health Workers

  // Education — health / science / CS postsecondary
  "25-1021.00", // Computer Science Teachers, Postsecondary
  "25-1022.00", // Mathematical Science Teachers, Postsecondary
  "25-1042.00", // Biological Science Teachers, Postsecondary
  "25-1052.00", // Chemistry Teachers, Postsecondary
  "25-1066.00", // Psychology Teachers, Postsecondary
  "25-1071.00", // Health Specialties Teachers, Postsecondary
  "25-1072.00", // Nursing Instructors and Teachers, Postsecondary
  "25-1113.00", // Social Work Teachers, Postsecondary

  // Office & Admin — healthcare
  "43-6013.00", // Medical Secretaries and Administrative Assistants
  "43-9021.00", // Data Entry Keyers

  // Business & Finance — healthcare adjacent
  "13-1041.00", // Compliance Officers
  "13-1041.06", // Coroners
  "13-1041.07", // Regulatory Affairs Specialists
  "13-1082.00", // Project Management Specialists
  "13-1111.00", // Management Analysts
  "13-2031.00", // Budget Analysts
]);

// ── Data ───────────────────────────────────────────────────────────

const typedData = onetData as {
  version: string;
  generated: string;
  source: string;
  license: string;
  stats: { occupations: number; uniqueSkills: number; categories: number };
  occupations: OnetOccupation[];
};

/** Filtered to healthcare, data, and closely related occupations. */
export const occupations: OnetOccupation[] = typedData.occupations.filter(
  (o) => INCLUDED_CATEGORIES.has(o.category) || INCLUDED_CODES.has(o.code),
);

export const dataInfo = {
  version: typedData.version,
  source: typedData.source,
  license: typedData.license,
  stats: {
    occupations: occupations.length,
    uniqueSkills: (() => {
      const s = new Set<string>();
      for (const occ of occupations) for (const sk of occ.skills) s.add(sk.name);
      return s.size;
    })(),
    categories: (() => {
      const s = new Set<string>();
      for (const occ of occupations) s.add(occ.category);
      return s.size;
    })(),
  },
};

// ── Derived catalogs ───────────────────────────────────────────────

/** All unique skill names across the filtered dataset. */
export const SKILL_CATALOG: string[] = (() => {
  const set = new Set<string>();
  for (const occ of occupations) {
    for (const s of occ.skills) {
      set.add(s.name);
    }
  }
  return [...set].sort();
})();

/** All unique occupational categories in the filtered dataset. */
export const CATEGORIES: string[] = (() => {
  const set = new Set<string>();
  for (const occ of occupations) set.add(occ.category);
  return [...set].sort();
})();
