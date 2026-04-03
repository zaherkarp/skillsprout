// =============================================================================
// SkillSprout — Client-side career transition scoring engine
// Enhanced with interactive features: radar chart, expandable cards,
// skill gap markers, quick wins advisor, progress rings, URL persistence
// =============================================================================

const catalog = {
  skills: [
    "communication",
    "customer_service",
    "critical_thinking",
    "programming",
    "data_analysis",
    "operations",
  ],

  skillLabels: {
    communication: "Communication",
    customer_service: "Customer Service",
    critical_thinking: "Critical Thinking",
    programming: "Programming",
    data_analysis: "Data Analysis",
    operations: "Operations",
  },

  levelLabels: ["None", "Basic", "Intermediate", "Advanced", "Expert"],

  roles: {
    "Fashion Sales Associate": {
      communication: 4, customer_service: 4, critical_thinking: 2,
      programming: 0, data_analysis: 1, operations: 2,
    },
    "Retail Supervisor": {
      communication: 4, customer_service: 4, critical_thinking: 2,
      programming: 0, data_analysis: 1, operations: 3,
    },
    "Junior Developer": {
      communication: 2, customer_service: 1, critical_thinking: 3,
      programming: 4, data_analysis: 3, operations: 2,
    },
    "Marketing Coordinator": {
      communication: 4, customer_service: 3, critical_thinking: 3,
      programming: 1, data_analysis: 3, operations: 2,
    },
    "Administrative Assistant": {
      communication: 3, customer_service: 3, critical_thinking: 2,
      programming: 0, data_analysis: 1, operations: 4,
    },
  },

  personas: {
    "Nia (fashion -> new field)": {
      role: "Fashion Sales Associate",
      emoji: "👗",
      shortName: "Nia",
      shortRole: "Fashion retail, 6 yrs",
      description:
        "Nia spent 6 years in fashion retail and wants to move into a more stable career path. She's great with people, organized, and picks things up fast.",
      focusFields: ["Project Coordination", "Operations", "Data Analysis"],
      profile: {
        communication: 4, customer_service: 4, critical_thinking: 3,
        programming: 1, data_analysis: 2, operations: 3,
      },
    },
    "Marcus (retail -> tech)": {
      role: "Retail Supervisor",
      emoji: "🏪",
      shortName: "Marcus",
      shortRole: "Retail supervisor, 4 yrs",
      description:
        "Marcus has managed a retail team for 4 years and wants to break into the tech industry. He's a strong leader and communicator, but has limited technical skills.",
      focusFields: ["Project Coordination", "Data Analysis", "Engineering"],
      profile: {
        communication: 4, customer_service: 4, critical_thinking: 3,
        programming: 0, data_analysis: 1, operations: 3,
      },
    },
    "Priya (admin -> analytics)": {
      role: "Administrative Assistant",
      emoji: "📋",
      shortName: "Priya",
      shortRole: "Admin assistant, 3 yrs",
      description:
        "Priya has been an administrative assistant for 3 years and has been teaching herself Excel and SQL on the side. She wants to move into a data-focused role.",
      focusFields: ["Data Analysis", "Operations", "Project Coordination"],
      profile: {
        communication: 3, customer_service: 3, critical_thinking: 3,
        programming: 1, data_analysis: 3, operations: 4,
      },
    },
  },

  targets: [
    {
      title: "Project Coordinator", field: "Project Coordination",
      requirements: { communication: 4, customer_service: 3, critical_thinking: 3, programming: 0, data_analysis: 2, operations: 3 },
      trainingTip: "Look into PMP or CAPM certifications. Many online programs take 2-3 months.",
    },
    {
      title: "Operations Specialist", field: "Operations",
      requirements: { communication: 3, customer_service: 3, critical_thinking: 3, programming: 1, data_analysis: 2, operations: 4 },
      trainingTip: "Six Sigma or Lean certifications are valued. Free courses available on Coursera.",
    },
    {
      title: "Junior Data Analyst", field: "Data Analysis",
      requirements: { communication: 2, customer_service: 1, critical_thinking: 4, programming: 2, data_analysis: 4, operations: 2 },
      trainingTip: "Start with Google Data Analytics Certificate (~6 months). Learn SQL and Python basics.",
    },
    {
      title: "Software Developer", field: "Engineering",
      requirements: { communication: 2, customer_service: 1, critical_thinking: 3, programming: 4, data_analysis: 3, operations: 1 },
      trainingTip: "Consider a coding bootcamp (12-16 weeks) or freeCodeCamp's curriculum.",
    },
    {
      title: "Customer Success Manager", field: "Customer Success",
      requirements: { communication: 4, customer_service: 4, critical_thinking: 3, programming: 0, data_analysis: 2, operations: 3 },
      trainingTip: "Leverage your people skills. HubSpot Academy offers free CS certifications.",
    },
    {
      title: "Business Analyst", field: "Business Analysis",
      requirements: { communication: 3, customer_service: 2, critical_thinking: 4, programming: 1, data_analysis: 4, operations: 3 },
      trainingTip: "IIBA certifications (ECBA/CCBA) are industry standard. Strong Excel/SQL skills are key.",
    },
  ],
};

// =============================================================================
// State
// =============================================================================
let selectedTarget = null; // currently focused target for radar chart comparison
let lastResults = [];
let savedCareers = JSON.parse(localStorage.getItem("skillsprout_shortlist") || "[]");
let currentStep = 1;
let hasAdjustedSliders = false;
let prevReadyCount = 0;

// =============================================================================
// DOM refs
// =============================================================================
const currentRole = document.getElementById("current-role");
const personaPicker = document.getElementById("persona-picker");
const personaDescription = document.getElementById("persona-description");
const qaTableBody = document.querySelector("#qa-table tbody");
const skillsForm = document.getElementById("skills-form");
const resetButton = document.getElementById("reset-btn");

// =============================================================================
// Scoring engine
// =============================================================================
function normalize(value, max) {
  return max === 0 ? 0 : (value / max) * 100;
}

function scoreTarget(user, target) {
  let matchEarned = 0;
  let matchPossible = 0;
  let gapTotal = 0;
  let gapPossible = 0;
  const gapSkills = [];
  const skillComparison = {};

  for (const skill of catalog.skills) {
    const required = target.requirements[skill];
    const actual = user[skill] || 0;
    matchEarned += Math.min(actual, required);
    matchPossible += required;

    skillComparison[skill] = { actual, required, gap: Math.max(0, required - actual) };

    if (actual < required) {
      gapTotal += required - actual;
      gapSkills.push({ name: catalog.skillLabels[skill], gap: required - actual, required, actual });
    }
    gapPossible += required;
  }

  const match = normalize(matchEarned, matchPossible);
  const gap = normalize(gapTotal, gapPossible);

  let bucket = "long";
  if (match >= 75 && gap <= 25) {
    bucket = "ready";
  } else if (match >= 50 || (gap >= 26 && gap <= 55)) {
    bucket = "trainable";
  }

  return {
    bucket, match: Math.round(match), gap: Math.round(gap),
    title: target.title, field: target.field, gapSkills,
    skillComparison, trainingTip: target.trainingTip,
    requirements: target.requirements,
  };
}

function bucketLabel(bucket) {
  if (bucket === "ready") return "Ready Now";
  if (bucket === "trainable") return "Trainable";
  return "Long Reskill";
}

// =============================================================================
// Radar Chart (SVG)
// =============================================================================
function renderRadarChart(userProfile, targetReqs) {
  const canvas = document.getElementById("radar-chart");
  if (!canvas) return;

  const skills = catalog.skills;
  const n = skills.length;
  const cx = 150, cy = 150, maxR = 120;
  const angleStep = (2 * Math.PI) / n;
  const startAngle = -Math.PI / 2;

  function polarToXY(i, value) {
    const angle = startAngle + i * angleStep;
    const r = (value / 4) * maxR;
    return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
  }

  function makePolygon(values, cls) {
    const points = values.map((v, i) => {
      const p = polarToXY(i, v);
      return `${p.x},${p.y}`;
    }).join(" ");
    return `<polygon points="${points}" class="${cls}" />`;
  }

  // Grid rings
  let svg = "";
  for (let ring = 1; ring <= 4; ring++) {
    const pts = skills.map((_, i) => {
      const p = polarToXY(i, ring);
      return `${p.x},${p.y}`;
    }).join(" ");
    svg += `<polygon points="${pts}" class="radar-grid" />`;
  }

  // Axis lines + labels
  skills.forEach((skill, i) => {
    const p = polarToXY(i, 4.6);
    const pLine = polarToXY(i, 4);
    svg += `<line x1="${cx}" y1="${cy}" x2="${pLine.x}" y2="${pLine.y}" class="radar-axis" />`;
    const anchor = p.x < cx - 5 ? "end" : p.x > cx + 5 ? "start" : "middle";
    svg += `<text x="${p.x}" y="${p.y}" text-anchor="${anchor}" dominant-baseline="middle" class="radar-label">${catalog.skillLabels[skill]}</text>`;
  });

  // Target polygon (if selected)
  if (targetReqs) {
    const targetVals = skills.map(s => targetReqs[s] || 0);
    svg += makePolygon(targetVals, "radar-target");
  }

  // User polygon
  const userVals = skills.map(s => userProfile[s] || 0);
  svg += makePolygon(userVals, "radar-user");

  // User dots
  userVals.forEach((v, i) => {
    const p = polarToXY(i, v);
    svg += `<circle cx="${p.x}" cy="${p.y}" r="4" class="radar-dot" />`;
  });

  canvas.innerHTML = svg;
}

// =============================================================================
// Progress ring SVG helper
// =============================================================================
function progressRingSVG(pct, size, colorVar) {
  const r = (size - 6) / 2;
  const c = Math.PI * 2 * r;
  const offset = c - (pct / 100) * c;
  return `<svg class="progress-ring" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
    <circle cx="${size/2}" cy="${size/2}" r="${r}" class="progress-ring-bg" />
    <circle cx="${size/2}" cy="${size/2}" r="${r}" class="progress-ring-fill" style="stroke-dasharray:${c};stroke-dashoffset:${offset};stroke:var(${colorVar})" />
    <text x="${size/2}" y="${size/2}" text-anchor="middle" dominant-baseline="central" class="progress-ring-text">${pct}%</text>
  </svg>`;
}

// =============================================================================
// Read current skill values from sliders
// =============================================================================
function readSkills() {
  const values = {};
  for (const skill of catalog.skills) {
    const input = document.querySelector(`[data-skill="${skill}"]`);
    values[skill] = Number(input.value);
  }
  return values;
}

// =============================================================================
// Render skill sliders (with target gap markers)
// =============================================================================
function renderSkillInputs(profile) {
  skillsForm.innerHTML = "";
  for (const skill of catalog.skills) {
    const block = document.createElement("div");
    block.className = "skill-item";
    const value = profile[skill] ?? 0;

    block.innerHTML = `
      <label>${catalog.skillLabels[skill]}</label>
      <div class="slider-wrap">
        <input data-skill="${skill}" type="range" min="0" max="4" step="1" value="${value}" aria-label="${catalog.skillLabels[skill]} skill level" />
        <div class="target-marker" data-skill-target="${skill}" style="display:none"></div>
      </div>
      <div class="skill-level">
        <span class="skill-level-label">${catalog.levelLabels[value]}</span>
        <output>${value}</output>/4
      </div>
    `;
    skillsForm.appendChild(block);

    const slider = block.querySelector("input");
    const output = block.querySelector("output");
    const levelLabel = block.querySelector(".skill-level-label");

    slider.addEventListener("input", () => {
      const val = Number(slider.value);
      output.value = val;
      levelLabel.textContent = catalog.levelLabels[val];
      hasAdjustedSliders = true;
      updateStepProgress(2);
      runRecommendations();
      saveStateToURL();
    });
  }
}

// =============================================================================
// Update target markers on sliders
// =============================================================================
function updateTargetMarkers(targetReqs) {
  for (const skill of catalog.skills) {
    const marker = document.querySelector(`[data-skill-target="${skill}"]`);
    if (!marker) continue;
    if (!targetReqs) {
      marker.style.display = "none";
      continue;
    }
    const req = targetReqs[skill] || 0;
    marker.style.display = "block";
    marker.style.left = `${(req / 4) * 100}%`;
    marker.title = `Target needs: ${catalog.levelLabels[req]} (${req})`;

    const currentVal = Number(document.querySelector(`[data-skill="${skill}"]`).value);
    marker.className = "target-marker" + (currentVal >= req ? " target-met" : " target-gap");
  }
}

// =============================================================================
// Render recommendation results (with progress rings + expandable cards)
// =============================================================================
function renderResults(results) {
  const bucketEls = {
    ready: document.getElementById("bucket-ready"),
    trainable: document.getElementById("bucket-trainable"),
    long: document.getElementById("bucket-long"),
  };
  const emptyEls = {
    ready: document.getElementById("empty-ready"),
    trainable: document.getElementById("empty-trainable"),
    long: document.getElementById("empty-long"),
  };

  for (const list of Object.values(bucketEls)) {
    list.innerHTML = "";
  }

  const counts = { ready: 0, trainable: 0, long: 0 };
  const sorted = [...results].sort((a, b) => b.match - a.match);

  for (const result of sorted) {
    counts[result.bucket]++;
    const item = document.createElement("li");
    item.className = "result-item";
    const isSelected = selectedTarget === result.title;
    if (isSelected) item.classList.add("result-selected");

    const colorVar = result.bucket === "ready" ? "--ready" : result.bucket === "trainable" ? "--trainable" : "--long";

    const gapDetails = result.gapSkills.length > 0
      ? result.gapSkills.map(g =>
          `<div class="gap-detail-row">
            <span class="gap-skill-name">${g.name}</span>
            <span class="gap-skill-bar-wrap">
              <span class="gap-skill-bar-current" style="width:${(g.actual/4)*100}%"></span>
              <span class="gap-skill-bar-needed" style="width:${(g.required/4)*100}%"></span>
            </span>
            <span class="gap-skill-vals">${g.actual} &rarr; ${g.required}</span>
          </div>`
        ).join("")
      : '<div class="gap-none">No skill gaps - you meet all requirements!</div>';

    const trainingHTML = result.trainingTip
      ? `<div class="training-tip"><strong>Training tip:</strong> ${result.trainingTip}</div>`
      : "";

    item.innerHTML = `
      <div class="result-header" data-target="${result.title}">
        <div class="result-ring">${progressRingSVG(result.match, 48, colorVar)}</div>
        <div class="result-info">
          <div class="result-title">${result.title}</div>
          <div class="result-meta">
            <span>${result.field}</span>
            <span>Gap: <span class="gap-pct">${result.gap}%</span></span>
          </div>
        </div>
        <button class="btn-bookmark ${savedCareers.includes(result.title) ? "bookmarked" : ""}" data-bookmark="${result.title}" title="${savedCareers.includes(result.title) ? "Remove from shortlist" : "Add to shortlist"}">
          ${savedCareers.includes(result.title) ? "&#9733;" : "&#9734;"}
        </button>
        <span class="expand-icon">${isSelected ? "&#9650;" : "&#9660;"}</span>
      </div>
      <div class="result-details ${isSelected ? "open" : ""}">
        <div class="gap-breakdown">
          <h4>Skill Gap Breakdown</h4>
          ${gapDetails}
        </div>
        ${trainingHTML}
        <button class="btn-compare" data-compare="${result.title}">
          ${isSelected ? "Hide on Radar" : "Show on Radar Chart"}
        </button>
      </div>
    `;

    // Click to expand
    item.querySelector(".result-header").addEventListener("click", () => {
      const details = item.querySelector(".result-details");
      const icon = item.querySelector(".expand-icon");
      const isOpen = details.classList.contains("open");
      details.classList.toggle("open");
      icon.innerHTML = isOpen ? "&#9660;" : "&#9650;";
    });

    // Compare button
    item.querySelector(".btn-compare").addEventListener("click", (e) => {
      e.stopPropagation();
      if (selectedTarget === result.title) {
        selectedTarget = null;
        updateTargetMarkers(null);
      } else {
        selectedTarget = result.title;
        updateTargetMarkers(result.requirements);
      }
      runRecommendations();
      renderRadarChart(readSkills(), selectedTarget ? result.requirements : null);
    });

    // Bookmark button
    item.querySelector(".btn-bookmark").addEventListener("click", (e) => {
      e.stopPropagation();
      toggleBookmark(result.title);
    });

    bucketEls[result.bucket].appendChild(item);
  }

  // Toggle empty states
  for (const key of Object.keys(emptyEls)) {
    emptyEls[key].style.display = counts[key] > 0 ? "none" : "block";
  }

  // Update radar legend
  const radarLegend = document.getElementById("radar-legend-target");
  if (radarLegend) {
    radarLegend.style.display = selectedTarget ? "inline-flex" : "none";
    const radarTargetName = document.getElementById("radar-target-name");
    if (radarTargetName) radarTargetName.textContent = selectedTarget || "";
  }
}

// =============================================================================
// Quick Wins Advisor
// =============================================================================
function renderQuickWins(results, userProfile) {
  const container = document.getElementById("quick-wins");
  if (!container) return;

  // For each skill, calculate how many careers would improve if we +1 that skill
  const skillImpact = {};
  for (const skill of catalog.skills) {
    const boostedProfile = { ...userProfile, [skill]: Math.min(4, userProfile[skill] + 1) };
    let careersImproved = 0;
    let totalMatchGain = 0;

    for (const target of catalog.targets) {
      const current = scoreTarget(userProfile, target);
      const boosted = scoreTarget(boostedProfile, target);

      if (boosted.bucket !== current.bucket && (
        (boosted.bucket === "ready" && current.bucket !== "ready") ||
        (boosted.bucket === "trainable" && current.bucket === "long")
      )) {
        careersImproved++;
      }
      totalMatchGain += boosted.match - current.match;
    }

    if (userProfile[skill] < 4) {
      skillImpact[skill] = { careersImproved, totalMatchGain, currentLevel: userProfile[skill] };
    }
  }

  // Sort by impact
  const ranked = Object.entries(skillImpact)
    .sort((a, b) => {
      if (b[1].careersImproved !== a[1].careersImproved) return b[1].careersImproved - a[1].careersImproved;
      return b[1].totalMatchGain - a[1].totalMatchGain;
    })
    .slice(0, 3);

  if (ranked.length === 0 || ranked[0][1].totalMatchGain === 0) {
    container.innerHTML = '<div class="quick-win-empty">Your skills are maxed out across the board!</div>';
    return;
  }

  container.innerHTML = ranked.map(([skill, impact], i) => {
    const label = catalog.skillLabels[skill];
    const levelNow = catalog.levelLabels[impact.currentLevel];
    const levelNext = catalog.levelLabels[impact.currentLevel + 1];
    const unlockText = impact.careersImproved > 0
      ? `<span class="qw-unlock">Unlocks ${impact.careersImproved} career${impact.careersImproved > 1 ? "s" : ""} to a better tier</span>`
      : "";

    return `<div class="quick-win-item ${i === 0 ? "qw-top" : ""}">
      <div class="qw-rank">#${i + 1}</div>
      <div class="qw-body">
        <div class="qw-skill">${label}: ${levelNow} &rarr; ${levelNext}</div>
        <div class="qw-impact">+${impact.totalMatchGain}% total match improvement across all careers</div>
        ${unlockText}
      </div>
    </div>`;
  }).join("");
}

// =============================================================================
// Render persona QA matrix
// =============================================================================
function renderPersonaQaMatrix(results) {
  const personaKey = getActivePersonaKey();
  if (!personaKey) {
    qaTableBody.innerHTML = "";
    return;
  }
  const persona = catalog.personas[personaKey];
  const chosenFields = new Set(persona.focusFields);
  const rows = results
    .filter((result) => chosenFields.has(result.field))
    .sort((a, b) => b.match - a.match)
    .map(
      (result) => `
      <tr>
        <td>${result.field}</td>
        <td>${result.title}</td>
        <td><span class="badge badge-${result.bucket}">${bucketLabel(result.bucket)}</span></td>
        <td>${result.match}%</td>
        <td>${result.gap}%</td>
      </tr>
    `
    )
    .join("");

  qaTableBody.innerHTML = rows;
}

// =============================================================================
// URL State Persistence
// =============================================================================
function saveStateToURL() {
  const skills = readSkills();
  const params = new URLSearchParams();
  for (const [key, val] of Object.entries(skills)) {
    params.set(key, val);
  }
  const persona = getActivePersonaKey();
  if (persona) params.set("persona", persona);
  history.replaceState(null, "", `#${params.toString()}`);

  // Update share button
  const shareBtn = document.getElementById("share-btn");
  if (shareBtn) shareBtn.classList.remove("copied");
}

function loadStateFromURL() {
  const hash = window.location.hash.slice(1);
  if (!hash) return false;
  const params = new URLSearchParams(hash);

  // Check if we have skill data
  const hasSkills = catalog.skills.some(s => params.has(s));
  if (!hasSkills) return false;

  // Apply persona if present
  const persona = params.get("persona");
  if (persona && catalog.personas[persona]) {
    applyPersona(persona, true); // skipURL = true to avoid overwriting
  }

  // Override with URL skill values
  for (const skill of catalog.skills) {
    if (params.has(skill)) {
      const slider = document.querySelector(`[data-skill="${skill}"]`);
      if (slider) {
        const val = Number(params.get(skill));
        slider.value = val;
        const item = slider.closest(".skill-item");
        item.querySelector("output").value = val;
        item.querySelector(".skill-level-label").textContent = catalog.levelLabels[val];
      }
    }
  }

  return true;
}

// =============================================================================
// Step Progress Indicator
// =============================================================================
function updateStepProgress(step) {
  currentStep = Math.max(currentStep, step);
  const nodes = document.querySelectorAll(".step-node");
  const connectors = document.querySelectorAll(".step-connector");
  nodes.forEach((node, i) => {
    const s = i + 1;
    node.classList.toggle("active", s <= currentStep);
    node.classList.toggle("completed", s < currentStep);
  });
  connectors.forEach((conn, i) => {
    conn.classList.toggle("filled", i + 1 < currentStep);
  });
}

// =============================================================================
// Celebration Animation (confetti burst for Ready Now)
// =============================================================================
function celebrate() {
  const container = document.getElementById("results-section");
  if (!container) return;
  const rect = container.getBoundingClientRect();
  const colors = ["#15803d", "#0d9488", "#f59e0b", "#10b981", "#34d399"];
  for (let i = 0; i < 24; i++) {
    const particle = document.createElement("div");
    particle.className = "confetti";
    particle.style.left = `${rect.left + rect.width * Math.random()}px`;
    particle.style.top = `${rect.top + window.scrollY - 10}px`;
    particle.style.background = colors[i % colors.length];
    particle.style.animationDelay = `${Math.random() * 0.3}s`;
    particle.style.setProperty("--dx", `${(Math.random() - 0.5) * 200}px`);
    document.body.appendChild(particle);
    particle.addEventListener("animationend", () => particle.remove());
  }
}

// =============================================================================
// Bookmark / Shortlist
// =============================================================================
function toggleBookmark(title) {
  const idx = savedCareers.indexOf(title);
  if (idx >= 0) {
    savedCareers.splice(idx, 1);
  } else {
    savedCareers.push(title);
  }
  localStorage.setItem("skillsprout_shortlist", JSON.stringify(savedCareers));
  renderShortlist();
  // Re-render results to update bookmark icons
  if (lastResults.length) renderResults(lastResults);
}

function renderShortlist() {
  const section = document.getElementById("shortlist-section");
  const list = document.getElementById("shortlist");
  if (!section || !list) return;

  if (savedCareers.length === 0) {
    section.style.display = "none";
    return;
  }

  section.style.display = "block";
  updateStepProgress(4);

  list.innerHTML = savedCareers.map(title => {
    const result = lastResults.find(r => r.title === title);
    if (!result) return "";
    const colorVar = result.bucket === "ready" ? "--ready" : result.bucket === "trainable" ? "--trainable" : "--long";
    return `<li class="shortlist-item">
      <div class="shortlist-info">
        ${progressRingSVG(result.match, 36, colorVar)}
        <div>
          <span class="shortlist-title">${result.title}</span>
          <span class="shortlist-meta">${result.field} &middot; <span class="badge badge-${result.bucket}">${bucketLabel(result.bucket)}</span></span>
        </div>
      </div>
      <button class="btn-bookmark bookmarked" data-bookmark="${result.title}" title="Remove from shortlist">&#9733;</button>
    </li>`;
  }).join("");

  // Bind remove buttons
  list.querySelectorAll(".btn-bookmark").forEach(btn => {
    btn.addEventListener("click", () => toggleBookmark(btn.dataset.bookmark));
  });
}

// =============================================================================
// Run scoring + render
// =============================================================================
function runRecommendations() {
  const userProfile = readSkills();
  const scored = catalog.targets.map((target) =>
    scoreTarget(userProfile, target)
  );
  lastResults = scored;

  // Check for celebration: new Ready Now careers appearing
  const newReadyCount = scored.filter(r => r.bucket === "ready").length;
  if (newReadyCount > prevReadyCount && prevReadyCount >= 0 && hasAdjustedSliders) {
    celebrate();
  }
  prevReadyCount = newReadyCount;

  renderResults(scored);
  renderPersonaQaMatrix(scored);
  renderQuickWins(scored, userProfile);
  renderShortlist();
  renderRadarChart(userProfile, selectedTarget
    ? catalog.targets.find(t => t.title === selectedTarget)?.requirements
    : null
  );

  // Update step progress
  if (hasAdjustedSliders) updateStepProgress(3);
}

// =============================================================================
// Persona management
// =============================================================================
function getActivePersonaKey() {
  const active = personaPicker.querySelector(".persona-btn.active");
  return active ? active.dataset.persona : null;
}

function applyPersona(personaKey, skipURL) {
  const persona = catalog.personas[personaKey];

  personaPicker.querySelectorAll(".persona-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.persona === personaKey);
  });

  personaDescription.textContent = persona.description;
  currentRole.value = persona.role;
  selectedTarget = null;
  hasAdjustedSliders = false;
  prevReadyCount = -1; // prevent celebration on persona switch
  renderSkillInputs(persona.profile);
  updateStepProgress(1);
  runRecommendations();
  prevReadyCount = lastResults.filter(r => r.bucket === "ready").length;
  if (!skipURL) saveStateToURL();
}

// =============================================================================
// Share button handler
// =============================================================================
function handleShare() {
  saveStateToURL();
  const url = window.location.href;
  navigator.clipboard.writeText(url).then(() => {
    const btn = document.getElementById("share-btn");
    btn.classList.add("copied");
    btn.querySelector(".share-label").textContent = "Link Copied!";
    setTimeout(() => {
      btn.classList.remove("copied");
      btn.querySelector(".share-label").textContent = "Share Profile";
    }, 2000);
  }).catch(() => {
    // Fallback: select text in a temporary input
    const tmp = document.createElement("input");
    tmp.value = url;
    document.body.appendChild(tmp);
    tmp.select();
    document.execCommand("copy");
    document.body.removeChild(tmp);
  });
}

// =============================================================================
// Init: populate UI
// =============================================================================

// Roles dropdown
for (const roleName of Object.keys(catalog.roles)) {
  const option = document.createElement("option");
  option.value = roleName;
  option.textContent = roleName;
  currentRole.appendChild(option);
}

// Persona buttons
for (const [key, persona] of Object.entries(catalog.personas)) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "persona-btn";
  btn.dataset.persona = key;
  btn.innerHTML = `
    <span>${persona.emoji} ${persona.shortName}</span>
    <span class="persona-btn-role">${persona.shortRole}</span>
  `;
  btn.addEventListener("click", () => applyPersona(key));
  personaPicker.appendChild(btn);
}

// Event listeners
currentRole.addEventListener("change", () => {
  personaPicker.querySelectorAll(".persona-btn").forEach((btn) => btn.classList.remove("active"));
  personaDescription.textContent = "";
  selectedTarget = null;
  renderSkillInputs(catalog.roles[currentRole.value]);
  runRecommendations();
  saveStateToURL();
});

resetButton.addEventListener("click", () => {
  const personaKey = getActivePersonaKey();
  selectedTarget = null;
  if (personaKey) {
    applyPersona(personaKey);
  } else {
    renderSkillInputs(catalog.roles[currentRole.value]);
    runRecommendations();
  }
  saveStateToURL();
});

// Share button
const shareBtn = document.getElementById("share-btn");
if (shareBtn) shareBtn.addEventListener("click", handleShare);

// Clear shortlist button
const clearShortlist = document.getElementById("clear-shortlist");
if (clearShortlist) clearShortlist.addEventListener("click", () => {
  savedCareers = [];
  localStorage.setItem("skillsprout_shortlist", JSON.stringify(savedCareers));
  renderShortlist();
  runRecommendations();
});

// Boot: try loading from URL, else use first persona
const firstPersonaKey = Object.keys(catalog.personas)[0];
const loaded = loadStateFromURL();
if (!loaded) {
  applyPersona(firstPersonaKey);
} else {
  runRecommendations();
}
