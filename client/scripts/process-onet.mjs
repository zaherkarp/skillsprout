#!/usr/bin/env node
/**
 * O*NET Data Processor
 *
 * Transforms raw O*NET database text files into the compact JSON dataset
 * used by the SkillSprout career trajectory engine.
 *
 * Input files (from https://www.onetcenter.org/database.html):
 *   - Occupation Data.txt   — occupation codes, titles, descriptions
 *   - Skills.txt            — 35 skills rated by importance (IM) per occupation
 *   - Knowledge.txt         — 33 knowledge areas rated by importance per occupation
 *   - Job Zones.txt         — job zone (1-5) per occupation
 *
 * Output: src/data/onet-full.json
 *
 * Usage:
 *   node scripts/process-onet.mjs /path/to/db_28_3_text/
 */

import { readFileSync, writeFileSync } from "fs";
import { join } from "path";

const DATA_DIR = process.argv[2] || "/tmp/db_28_3_text";
const OUT_FILE = join(import.meta.dirname, "..", "data", "onet-full.json");

// ── Parse TSV ──────────────────────────────────────────────────────

function parseTsv(filePath) {
  const raw = readFileSync(filePath, "utf-8");
  const lines = raw.split("\n").filter(l => l.trim());
  const headers = lines[0].split("\t");
  return lines.slice(1).map(line => {
    const vals = line.split("\t");
    const obj = {};
    headers.forEach((h, i) => { obj[h.trim()] = (vals[i] || "").trim(); });
    return obj;
  });
}

// ── Load raw data ──────────────────────────────────────────────────

console.log("Loading O*NET data from", DATA_DIR);

const occupationRows = parseTsv(join(DATA_DIR, "Occupation Data.txt"));
const skillRows = parseTsv(join(DATA_DIR, "Skills.txt"));
const knowledgeRows = parseTsv(join(DATA_DIR, "Knowledge.txt"));
const jobZoneRows = parseTsv(join(DATA_DIR, "Job Zones.txt"));

console.log(`  Occupations: ${occupationRows.length}`);
console.log(`  Skill rows: ${skillRows.length}`);
console.log(`  Knowledge rows: ${knowledgeRows.length}`);
console.log(`  Job zone rows: ${jobZoneRows.length}`);

// ── Build job zone map ─────────────────────────────────────────────

const jobZoneMap = new Map();
for (const row of jobZoneRows) {
  jobZoneMap.set(row["O*NET-SOC Code"], parseInt(row["Job Zone"]) || 3);
}

// ── Build skill/knowledge maps ─────────────────────────────────────
// For each occupation, keep skills/knowledge with Importance (IM) scale
// and filter to those with data value >= 3.0 (moderately important+)

function buildTopItems(rows, minImportance = 2.75) {
  const map = new Map(); // code -> [{name, importance}]
  for (const row of rows) {
    if (row["Scale ID"] !== "IM") continue;
    if (row["Recommend Suppress"] === "Y") continue;
    const code = row["O*NET-SOC Code"];
    const name = row["Element Name"];
    const importance = parseFloat(row["Data Value"]);
    if (isNaN(importance) || importance < minImportance) continue;

    if (!map.has(code)) map.set(code, []);
    map.get(code).push({ name, importance });
  }

  // Sort each occupation's items by importance descending, keep top 12
  for (const [code, items] of map) {
    items.sort((a, b) => b.importance - a.importance);
    map.set(code, items.slice(0, 12));
  }

  return map;
}

const skillMap = buildTopItems(skillRows, 2.75);
const knowledgeMap = buildTopItems(knowledgeRows, 2.75);

// ── Determine occupation category from SOC code ────────────────────

function categoryFromCode(code) {
  const major = code.substring(0, 2);
  const categories = {
    "11": "Management",
    "13": "Business & Finance",
    "15": "Computer & Mathematical",
    "17": "Architecture & Engineering",
    "19": "Life, Physical & Social Science",
    "21": "Community & Social Service",
    "23": "Legal",
    "25": "Education & Training",
    "27": "Arts, Design & Media",
    "29": "Healthcare Practitioners",
    "31": "Healthcare Support",
    "33": "Protective Service",
    "35": "Food Preparation & Serving",
    "37": "Building & Grounds Cleaning",
    "39": "Personal Care & Service",
    "41": "Sales",
    "43": "Office & Administrative Support",
    "45": "Farming, Fishing & Forestry",
    "47": "Construction & Extraction",
    "49": "Installation, Maintenance & Repair",
    "51": "Production",
    "53": "Transportation & Material Moving",
    "55": "Military",
  };
  return categories[major] || "Other";
}

// ── Build final dataset ────────────────────────────────────────────

// First pass: build all occupations, some will have empty skills
const occupationsRaw = [];

for (const occ of occupationRows) {
  const code = occ["O*NET-SOC Code"];
  const title = occ["Title"];
  const zone = jobZoneMap.get(code) || 3;
  const category = categoryFromCode(code);

  // Merge skills + knowledge into a single skill list
  const skills = [];
  const seen = new Set();

  const occSkills = skillMap.get(code) || [];
  const occKnowledge = knowledgeMap.get(code) || [];

  for (const item of [...occSkills, ...occKnowledge]) {
    const normalized = item.name.toLowerCase();
    if (!seen.has(normalized)) {
      seen.add(normalized);
      skills.push({
        name: item.name,
        importance: Math.round(item.importance * 100) / 100,
      });
    }
  }

  // Sort by importance descending and keep top 15
  skills.sort((a, b) => b.importance - a.importance);
  const topSkills = skills.slice(0, 15);

  occupationsRaw.push({
    code,
    title,
    zone,
    category,
    skills: topSkills,
  });
}

// ── Fill empty skills via SOC family inheritance ───────────────────
// Many O*NET occupations (especially newer SOC codes and "All Other"
// catch-all codes) lack direct skill/knowledge survey data. For these,
// inherit skills from the closest related occupation in the same SOC
// family. Fallback order:
//   1. Same 6-digit broad occupation (e.g., 15-1252.xx → 15-1251.00)
//   2. Same 4-digit minor group (e.g., 15-12xx → other 15-12xx)
//   3. Same 2-digit major group (e.g., 15-xxxx → other 15-xxxx)

const codeIndex = new Map(occupationsRaw.map(o => [o.code, o]));

function findDonor(code) {
  const prefix6 = code.substring(0, 7); // e.g., "15-1252"
  const prefix4 = code.substring(0, 5); // e.g., "15-12"
  const prefix2 = code.substring(0, 2); // e.g., "15"

  // Try same broad occupation first
  for (const o of occupationsRaw) {
    if (o.code !== code && o.code.startsWith(prefix6) && o.skills.length > 0) return o;
  }
  // Try same minor group
  for (const o of occupationsRaw) {
    if (o.code !== code && o.code.startsWith(prefix4) && o.skills.length > 0) return o;
  }
  // Try same major group
  for (const o of occupationsRaw) {
    if (o.code !== code && o.code.startsWith(prefix2) && o.skills.length > 0) return o;
  }
  return null;
}

let inheritCount = 0;
for (const occ of occupationsRaw) {
  if (occ.skills.length === 0) {
    const donor = findDonor(occ.code);
    if (donor) {
      occ.skills = [...donor.skills]; // Copy skills from donor
      inheritCount++;
    }
  }
}

const stillEmpty = occupationsRaw.filter(o => o.skills.length === 0);
console.log(`\nSkill inheritance: ${inheritCount} occupations inherited from SOC family`);
if (stillEmpty.length > 0) {
  console.log(`  Still empty: ${stillEmpty.length} (${stillEmpty.map(o => o.code).join(", ")})`);
}

const occupations = occupationsRaw;

console.log(`\nProcessed ${occupations.length} occupations`);

// ── Compute stats ──────────────────────────────────────────────────

const categoryCount = {};
const zoneCount = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
const allSkillNames = new Set();

for (const occ of occupations) {
  categoryCount[occ.category] = (categoryCount[occ.category] || 0) + 1;
  zoneCount[occ.zone] = (zoneCount[occ.zone] || 0) + 1;
  for (const s of occ.skills) allSkillNames.add(s.name);
}

console.log("\nBy category:");
for (const [cat, count] of Object.entries(categoryCount).sort((a, b) => b[1] - a[1])) {
  console.log(`  ${cat}: ${count}`);
}

console.log("\nBy job zone:");
for (const [zone, count] of Object.entries(zoneCount)) {
  console.log(`  Zone ${zone}: ${count}`);
}

console.log(`\nUnique skill/knowledge names: ${allSkillNames.size}`);

// ── Write output ───────────────────────────────────────────────────

const output = {
  version: "28.3",
  generated: new Date().toISOString(),
  source: "O*NET Database 28.3 (https://www.onetcenter.org/database.html)",
  license: "Creative Commons Attribution 4.0 International (CC BY 4.0)",
  stats: {
    occupations: occupations.length,
    uniqueSkills: allSkillNames.size,
    categories: Object.keys(categoryCount).length,
  },
  occupations,
};

writeFileSync(OUT_FILE, JSON.stringify(output));
const sizeKB = Math.round(readFileSync(OUT_FILE).length / 1024);
console.log(`\nWrote ${OUT_FILE} (${sizeKB} KB)`);
