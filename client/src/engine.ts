/**
 * SkillSprout Career Trajectory Engine
 *
 * Client-side matching engine that computes career transition recommendations
 * using the full O*NET 28.3 database (1,016 occupations, 65 skill/knowledge
 * dimensions). Exposes a request/response API supporting iterative refinement.
 *
 * Transition categories (per Anthropic's AI economic-impact framework):
 *   - Ready Now:        ≥55% weighted skill overlap, same or lower Job Zone
 *   - Trainable:        30-54% overlap, Zone delta ≤ 1
 *   - Long-Term Reskill: 10-29% overlap OR Zone delta ≥ 2
 *
 * Matching uses IDF-weighted Jaccard overlap so that ubiquitous skills
 * (e.g., "Speaking", "Active Listening") don't dominate the score while
 * specialized skills (e.g., "Medicine and Dentistry", "Physics") carry
 * more weight.
 */

import { occupations, type OnetOccupation } from "./occupations";

// ── Types ──────────────────────────────────────────────────────────

export interface TrajectoryRequest {
  /** O*NET code or title (case-insensitive partial match) */
  sourceOccupation: string;
  /** Optional additional skills the user has beyond their occupation's defaults */
  additionalSkills?: string[];
  /** Filter results to a specific category */
  categoryFilter?: "all" | "ready_now" | "trainable" | "long_term_reskill";
  /** Occupation codes to exclude (for pagination / "show me more") */
  excludeCodes?: string[];
  /** Max results per category (default 5) */
  maxPerCategory?: number;
  /** Minimum overlap score 0-1 to include (default 0.10) */
  minScore?: number;
  /** Weight multiplier for skill rarity (IDF). Higher = rarer skills matter more */
  rarityWeight?: number;
  /** Prefer occupations in these categories */
  preferredCategories?: string[];
  /** Conversation turn number for iterative refinement */
  turn?: number;
}

export interface SkillGap {
  skill: string;
  importance: "critical" | "important" | "nice-to-have";
}

export interface TrainingPath {
  skill: string;
  estimatedMonths: number;
  resources: string[];
}

export interface TransitionMatch {
  occupation: OnetOccupation;
  overlapScore: number;           // 0-1
  sharedSkills: string[];
  missingSkills: SkillGap[];
  trainingPaths: TrainingPath[];
  category: "ready_now" | "trainable" | "long_term_reskill";
  zoneDelta: number;
}

export interface TrajectoryResponse {
  success: boolean;
  source: OnetOccupation | null;
  sourceSkills: string[];
  readyNow: TransitionMatch[];
  trainable: TransitionMatch[];
  longTermReskill: TransitionMatch[];
  totalMatches: number;
  turn: number;
  suggestions: string[];
  error?: string;
}

// ── Helpers ────────────────────────────────────────────────────────

/** Extract skill names from an occupation as a lowercase Set. */
function skillNamesOf(occ: OnetOccupation): Set<string> {
  return new Set(occ.skills.map(s => s.name.toLowerCase()));
}

/** Compute inverse document frequency for each skill across the corpus. */
function computeIdf(): Map<string, number> {
  const docCount = occupations.length;
  const skillDocFreq = new Map<string, number>();
  for (const occ of occupations) {
    const seen = new Set<string>();
    for (const s of occ.skills) {
      const key = s.name.toLowerCase();
      if (!seen.has(key)) {
        skillDocFreq.set(key, (skillDocFreq.get(key) || 0) + 1);
        seen.add(key);
      }
    }
  }
  const idf = new Map<string, number>();
  for (const [skill, df] of skillDocFreq) {
    idf.set(skill, Math.log(docCount / df));
  }
  return idf;
}

const IDF = computeIdf();

function idfWeight(skill: string, rarityMultiplier: number): number {
  const base = IDF.get(skill.toLowerCase()) ?? Math.log(occupations.length);
  const maxIdf = Math.log(occupations.length);
  return 1 + (base - 1) * (rarityMultiplier - 1) / (maxIdf - 1 || 1);
}

function resolveOccupation(query: string): OnetOccupation | null {
  const q = query.toLowerCase().trim();
  if (!q) return null;
  // Exact code match
  const byCode = occupations.find((o) => o.code.toLowerCase() === q);
  if (byCode) return byCode;
  // Exact title match
  const byTitle = occupations.find((o) => o.title.toLowerCase() === q);
  if (byTitle) return byTitle;
  // Partial title match (prefer shorter titles = more specific)
  const partials = occupations
    .filter((o) => o.title.toLowerCase().includes(q))
    .sort((a, b) => a.title.length - b.title.length);
  if (partials.length > 0) return partials[0];
  // Partial code match
  const partialCode = occupations.find((o) => o.code.includes(q));
  return partialCode ?? null;
}

function classifyImportance(idfScore: number): "critical" | "important" | "nice-to-have" {
  if (idfScore > 2.5) return "critical";
  if (idfScore > 1.5) return "important";
  return "nice-to-have";
}

function estimateTrainingMonths(skill: string, zoneDelta: number): number {
  const base = IDF.get(skill.toLowerCase()) ?? 2;
  const months = Math.max(1, Math.round(base * 1.5 + Math.max(0, zoneDelta) * 3));
  return Math.min(months, 36);
}

function suggestResources(skill: string): string[] {
  const s = skill.toLowerCase();
  const resources: string[] = [];

  // Map O*NET canonical skill/knowledge names to learning resources
  const resourceMap: [string[], string[]][] = [
    [
      ["computers and electronics", "programming", "telecommunications"],
      ["freeCodeCamp", "Codecademy", "MIT OpenCourseWare"],
    ],
    [
      ["mathematics", "physics", "chemistry"],
      ["Khan Academy", "MIT OpenCourseWare", "Coursera"],
    ],
    [
      ["engineering and technology", "design", "mechanical"],
      ["Coursera Engineering", "edX", "LinkedIn Learning"],
    ],
    [
      ["medicine and dentistry", "therapy and counseling", "biology"],
      ["Coursera Health", "CDC TRAIN", "Khan Academy Biology"],
    ],
    [
      ["administration and management", "personnel and human resources"],
      ["Google Project Management Certificate", "LinkedIn Learning", "edX MicroMasters"],
    ],
    [
      ["economics and accounting", "management of financial resources"],
      ["Khan Academy Finance", "Coursera Finance", "edX"],
    ],
    [
      ["law and government", "public safety and security"],
      ["Coursera Law", "edX Legal Studies", "LinkedIn Learning"],
    ],
    [
      ["education and training", "sociology and anthropology", "psychology"],
      ["Coursera Education", "edX Social Science", "Khan Academy"],
    ],
    [
      ["building and construction", "production and processing"],
      ["Trade school / apprenticeship", "Union training programs", "OSHA certification"],
    ],
    [
      ["fine arts", "communications and media"],
      ["Skillshare", "MasterClass", "LinkedIn Learning"],
    ],
    [
      ["customer and personal service", "sales and marketing"],
      ["HubSpot Academy", "Google Digital Marketing", "LinkedIn Learning"],
    ],
    [
      ["food production", "transportation"],
      ["Industry certification programs", "Community college courses", "Online training"],
    ],
    [
      ["systems analysis", "systems evaluation", "operations analysis", "technology design"],
      ["Coursera Systems Engineering", "MIT OpenCourseWare", "edX"],
    ],
    [
      ["complex problem solving", "critical thinking", "judgment and decision making"],
      ["Coursera Critical Thinking", "edX", "LinkedIn Learning"],
    ],
  ];

  for (const [keywords, recs] of resourceMap) {
    if (keywords.some(k => s.includes(k) || k.includes(s))) {
      resources.push(...recs);
      break;
    }
  }

  if (resources.length === 0) {
    resources.push("Coursera", "LinkedIn Learning", "edX");
  }

  return [...new Set(resources)].slice(0, 3);
}

// ── Core Engine ────────────────────────────────────────────────────

function computeMatch(
  source: OnetOccupation,
  sourceSkills: Set<string>,
  target: OnetOccupation,
  rarityWeight: number,
): TransitionMatch | null {
  if (source.code === target.code) return null;

  const targetSkills = skillNamesOf(target);
  const shared: string[] = [];
  const missing: SkillGap[] = [];

  let sharedWeightedSum = 0;
  let unionWeightedSum = 0;

  const allSkills = new Set([...sourceSkills, ...targetSkills]);

  for (const skill of allSkills) {
    const w = idfWeight(skill, rarityWeight);
    unionWeightedSum += w;
    if (sourceSkills.has(skill) && targetSkills.has(skill)) {
      sharedWeightedSum += w;
      shared.push(skill);
    }
  }

  for (const skill of targetSkills) {
    if (!sourceSkills.has(skill)) {
      const idfVal = IDF.get(skill) ?? 2;
      missing.push({ skill, importance: classifyImportance(idfVal) });
    }
  }

  const overlapScore = unionWeightedSum > 0 ? sharedWeightedSum / unionWeightedSum : 0;
  const zoneDelta = target.zone - source.zone;

  let category: "ready_now" | "trainable" | "long_term_reskill";
  if (overlapScore >= 0.55 && zoneDelta <= 0) {
    category = "ready_now";
  } else if (overlapScore >= 0.30 && zoneDelta <= 1) {
    category = "trainable";
  } else if (overlapScore >= 0.10) {
    category = "long_term_reskill";
  } else {
    return null;
  }

  missing.sort((a, b) => {
    const order = { critical: 0, important: 1, "nice-to-have": 2 };
    return order[a.importance] - order[b.importance];
  });

  const trainingPaths: TrainingPath[] = missing
    .filter(g => g.importance !== "nice-to-have")
    .slice(0, 5)
    .map(g => ({
      skill: g.skill,
      estimatedMonths: estimateTrainingMonths(g.skill, Math.max(0, zoneDelta)),
      resources: suggestResources(g.skill),
    }));

  return {
    occupation: target,
    overlapScore,
    sharedSkills: shared,
    missingSkills: missing,
    trainingPaths,
    category,
    zoneDelta,
  };
}

// ── Public API ─────────────────────────────────────────────────────

export function getCareerTrajectories(req: TrajectoryRequest): TrajectoryResponse {
  const turn = req.turn ?? 1;

  const source = resolveOccupation(req.sourceOccupation);
  if (!source) {
    // Suggest closest title matches
    const q = req.sourceOccupation.toLowerCase();
    const suggestions = occupations
      .map(o => ({ title: o.title, dist: levenshteinLike(q, o.title.toLowerCase()) }))
      .sort((a, b) => a.dist - b.dist)
      .slice(0, 5)
      .map(s => s.title);

    return {
      success: false,
      source: null,
      sourceSkills: [],
      readyNow: [],
      trainable: [],
      longTermReskill: [],
      totalMatches: 0,
      turn,
      suggestions: [
        `Did you mean: ${suggestions.join(", ")}?`,
        "Try a standard O*NET job title or SOC code (e.g., '15-1252.00')",
      ],
      error: `Could not find occupation matching "${req.sourceOccupation}". 1,016 O*NET occupations are available.`,
    };
  }

  const additionalSkills = (req.additionalSkills ?? []).map(s => s.toLowerCase());
  const sourceSkills = new Set([
    ...source.skills.map(s => s.name.toLowerCase()),
    ...additionalSkills,
  ]);

  const excludeSet = new Set(req.excludeCodes ?? []);
  const maxPer = req.maxPerCategory ?? 5;
  const minScore = req.minScore ?? 0.10;
  const rarityWeight = req.rarityWeight ?? 1.5;
  const categoryFilter = req.categoryFilter ?? "all";
  const preferredCategories = new Set(
    (req.preferredCategories ?? []).map(c => c.toLowerCase())
  );

  const readyNow: TransitionMatch[] = [];
  const trainable: TransitionMatch[] = [];
  const longTermReskill: TransitionMatch[] = [];

  for (const target of occupations) {
    if (excludeSet.has(target.code)) continue;

    const match = computeMatch(source, sourceSkills, target, rarityWeight);
    if (!match || match.overlapScore < minScore) continue;

    if (categoryFilter !== "all" && match.category !== categoryFilter) continue;

    switch (match.category) {
      case "ready_now":
        readyNow.push(match);
        break;
      case "trainable":
        trainable.push(match);
        break;
      case "long_term_reskill":
        longTermReskill.push(match);
        break;
    }
  }

  const sortFn = (a: TransitionMatch, b: TransitionMatch) => {
    const aBoost = preferredCategories.has(a.occupation.category.toLowerCase()) ? 0.1 : 0;
    const bBoost = preferredCategories.has(b.occupation.category.toLowerCase()) ? 0.1 : 0;
    return (b.overlapScore + bBoost) - (a.overlapScore + aBoost);
  };

  readyNow.sort(sortFn);
  trainable.sort(sortFn);
  longTermReskill.sort(sortFn);

  const trimmed = {
    readyNow: readyNow.slice(0, maxPer),
    trainable: trainable.slice(0, maxPer),
    longTermReskill: longTermReskill.slice(0, maxPer),
  };

  const totalMatches = readyNow.length + trainable.length + longTermReskill.length;

  const suggestions: string[] = [];
  if (readyNow.length > maxPer) {
    suggestions.push(`${readyNow.length - maxPer} more Ready Now matches available`);
  }
  if (trainable.length > maxPer) {
    suggestions.push(`${trainable.length - maxPer} more Trainable matches available`);
  }
  if (longTermReskill.length > maxPer) {
    suggestions.push(`${longTermReskill.length - maxPer} more Long-Term Reskill matches available`);
  }
  if (additionalSkills.length === 0) {
    suggestions.push("Add your additional skills to improve matching accuracy");
  }

  return {
    success: true,
    source,
    sourceSkills: [...sourceSkills],
    ...trimmed,
    totalMatches,
    turn,
    suggestions,
  };
}

/** List all available occupations (for autocomplete). */
export function listOccupations(): { code: string; title: string; category: string }[] {
  return occupations.map(o => ({ code: o.code, title: o.title, category: o.category }));
}

/** Get a single occupation by code or title. */
export function getOccupation(query: string): OnetOccupation | null {
  return resolveOccupation(query);
}

// ── Utility ────────────────────────────────────────────────────────

/** Simple substring-based distance for "did you mean" suggestions. */
function levenshteinLike(a: string, b: string): number {
  if (b.includes(a)) return 0;
  if (a.includes(b)) return 0;
  // Count shared words
  const aWords = new Set(a.split(/\s+/));
  const bWords = new Set(b.split(/\s+/));
  let shared = 0;
  for (const w of aWords) if (bWords.has(w)) shared++;
  return -shared + Math.abs(a.length - b.length) * 0.01;
}
