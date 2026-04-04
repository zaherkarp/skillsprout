/**
 * @zaherkarp/skillsprout-client
 *
 * Client-side career trajectory engine powered by the full O*NET 28.3 database.
 * Runs entirely in the browser — no data leaves the client.
 */

// Engine
export {
  getCareerTrajectories,
  listOccupations,
  getOccupation,
} from "./engine";
export type {
  TrajectoryRequest,
  SkillGap,
  TrainingPath,
  TransitionMatch,
  TrajectoryResponse,
} from "./engine";

// Session API
export { callApi, quickSearch } from "./api";
export type { ApiSession, ApiAction, ApiRequest, ApiResult } from "./api";

// O*NET data
export { occupations, dataInfo, SKILL_CATALOG, CATEGORIES } from "./occupations";
export type { OnetSkill, OnetOccupation } from "./occupations";

// Test utilities
export { testCases, runTestCase, runAllTests } from "./test-cases";
export type { TestCase, TestResult } from "./test-cases";
