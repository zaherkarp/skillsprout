/**
 * SkillSprout API — stateful, iterable career trajectory API.
 *
 * This module wraps the trajectory engine in a session-based API that tracks
 * conversation state, allowing callers to iteratively refine results:
 *
 *   1. Initial request: provide source occupation
 *   2. Refine: filter by category, add skills, exclude already-seen occupations
 *   3. Paginate: request more results by passing excludeCodes
 *   4. Pivot: change source occupation entirely
 *
 * Designed so an external client (or LLM agent) can call it in a loop.
 */

import {
  getCareerTrajectories,
  listOccupations,
  getOccupation,
  type TrajectoryRequest,
  type TrajectoryResponse,
  type TransitionMatch,
} from "./engine";

// ── Session state ──────────────────────────────────────────────────

export interface ApiSession {
  id: string;
  turn: number;
  history: TrajectoryResponse[];
  seenCodes: Set<string>;
  currentSource: string;
  additionalSkills: string[];
}

const sessions = new Map<string, ApiSession>();

function newSessionId(): string {
  return `ss-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

// ── API request/response types ─────────────────────────────────────

export type ApiAction =
  | "search"           // New or refined trajectory search
  | "more"             // Get more results (paginate)
  | "filter"           // Filter current results by category
  | "add_skills"       // Add additional skills and re-run
  | "list_occupations" // List available occupations
  | "reset";           // Clear session state

export interface ApiRequest {
  action: ApiAction;
  sessionId?: string;
  occupation?: string;
  additionalSkills?: string[];
  categoryFilter?: "all" | "ready_now" | "trainable" | "long_term_reskill";
  maxPerCategory?: number;
  minScore?: number;
  rarityWeight?: number;
  preferredCategories?: string[];
}

export interface ApiResult {
  sessionId: string;
  turn: number;
  data: TrajectoryResponse | { occupations: { code: string; title: string; category: string }[] } | { message: string };
  meta: {
    action: ApiAction;
    hasMore: boolean;
    totalSeenCodes: number;
    suggestions: string[];
  };
}

// ── API implementation ─────────────────────────────────────────────

export function callApi(req: ApiRequest): ApiResult {
  // Resolve or create session
  let session: ApiSession;
  if (req.sessionId && sessions.has(req.sessionId)) {
    session = sessions.get(req.sessionId)!;
  } else {
    const id = req.sessionId || newSessionId();
    session = {
      id,
      turn: 0,
      history: [],
      seenCodes: new Set(),
      currentSource: "",
      additionalSkills: [],
    };
    sessions.set(id, session);
  }

  switch (req.action) {
    case "list_occupations": {
      const occs = listOccupations();
      return {
        sessionId: session.id,
        turn: session.turn,
        data: { occupations: occs },
        meta: {
          action: "list_occupations",
          hasMore: false,
          totalSeenCodes: session.seenCodes.size,
          suggestions: ["Pick an occupation and use action: 'search' with the occupation field"],
        },
      };
    }

    case "reset": {
      session.turn = 0;
      session.history = [];
      session.seenCodes.clear();
      session.currentSource = "";
      session.additionalSkills = [];
      return {
        sessionId: session.id,
        turn: 0,
        data: { message: "Session reset. Ready for a new search." },
        meta: {
          action: "reset",
          hasMore: false,
          totalSeenCodes: 0,
          suggestions: ["Start fresh with action: 'search'"],
        },
      };
    }

    case "add_skills": {
      if (req.additionalSkills) {
        session.additionalSkills = [
          ...new Set([...session.additionalSkills, ...req.additionalSkills]),
        ];
      }
      // Re-run search with updated skills
      return runSearch(session, req);
    }

    case "more": {
      // Paginate by excluding all previously seen codes
      return runSearch(session, req);
    }

    case "filter": {
      // Re-run with category filter, but don't exclude seen (let user re-see filtered view)
      const filterReq = { ...req };
      // Don't add to seenCodes for filter operations
      return runSearch(session, filterReq, false);
    }

    case "search":
    default: {
      if (req.occupation) {
        // New occupation = partial reset
        if (req.occupation.toLowerCase() !== session.currentSource.toLowerCase()) {
          session.seenCodes.clear();
          session.currentSource = req.occupation;
        }
      }
      return runSearch(session, req);
    }
  }
}

function runSearch(session: ApiSession, req: ApiRequest, trackSeen = true): ApiResult {
  session.turn++;

  const trajectoryReq: TrajectoryRequest = {
    sourceOccupation: req.occupation || session.currentSource,
    additionalSkills: session.additionalSkills,
    categoryFilter: req.categoryFilter,
    excludeCodes: [...session.seenCodes],
    maxPerCategory: req.maxPerCategory ?? 5,
    minScore: req.minScore ?? 0.10,
    rarityWeight: req.rarityWeight ?? 1.5,
    preferredCategories: req.preferredCategories,
    turn: session.turn,
  };

  const result = getCareerTrajectories(trajectoryReq);
  session.history.push(result);

  // Track seen occupation codes
  if (trackSeen && result.success) {
    const allMatches: TransitionMatch[] = [
      ...result.readyNow,
      ...result.trainable,
      ...result.longTermReskill,
    ];
    for (const m of allMatches) {
      session.seenCodes.add(m.occupation.code);
    }
  }

  const hasMore =
    result.totalMatches >
    (result.readyNow.length + result.trainable.length + result.longTermReskill.length);

  return {
    sessionId: session.id,
    turn: session.turn,
    data: result,
    meta: {
      action: req.action,
      hasMore,
      totalSeenCodes: session.seenCodes.size,
      suggestions: result.suggestions,
    },
  };
}

/** Convenience: one-shot search without session management. */
export function quickSearch(
  occupation: string,
  additionalSkills?: string[],
): TrajectoryResponse {
  return getCareerTrajectories({
    sourceOccupation: occupation,
    additionalSkills,
    maxPerCategory: 5,
  });
}
