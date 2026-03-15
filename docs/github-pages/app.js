// =============================================================================
// SkillSprout — Client-side career transition scoring engine
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
      communication: 4,
      customer_service: 4,
      critical_thinking: 2,
      programming: 0,
      data_analysis: 1,
      operations: 2,
    },
    "Retail Supervisor": {
      communication: 4,
      customer_service: 4,
      critical_thinking: 2,
      programming: 0,
      data_analysis: 1,
      operations: 3,
    },
    "Junior Developer": {
      communication: 2,
      customer_service: 1,
      critical_thinking: 3,
      programming: 4,
      data_analysis: 3,
      operations: 2,
    },
    "Marketing Coordinator": {
      communication: 4,
      customer_service: 3,
      critical_thinking: 3,
      programming: 1,
      data_analysis: 3,
      operations: 2,
    },
    "Administrative Assistant": {
      communication: 3,
      customer_service: 3,
      critical_thinking: 2,
      programming: 0,
      data_analysis: 1,
      operations: 4,
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
        communication: 4,
        customer_service: 4,
        critical_thinking: 3,
        programming: 1,
        data_analysis: 2,
        operations: 3,
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
        communication: 4,
        customer_service: 4,
        critical_thinking: 3,
        programming: 0,
        data_analysis: 1,
        operations: 3,
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
        communication: 3,
        customer_service: 3,
        critical_thinking: 3,
        programming: 1,
        data_analysis: 3,
        operations: 4,
      },
    },
  },

  targets: [
    {
      title: "Project Coordinator",
      field: "Project Coordination",
      requirements: {
        communication: 4,
        customer_service: 3,
        critical_thinking: 3,
        programming: 0,
        data_analysis: 2,
        operations: 3,
      },
    },
    {
      title: "Operations Specialist",
      field: "Operations",
      requirements: {
        communication: 3,
        customer_service: 3,
        critical_thinking: 3,
        programming: 1,
        data_analysis: 2,
        operations: 4,
      },
    },
    {
      title: "Junior Data Analyst",
      field: "Data Analysis",
      requirements: {
        communication: 2,
        customer_service: 1,
        critical_thinking: 4,
        programming: 2,
        data_analysis: 4,
        operations: 2,
      },
    },
    {
      title: "Software Developer",
      field: "Engineering",
      requirements: {
        communication: 2,
        customer_service: 1,
        critical_thinking: 3,
        programming: 4,
        data_analysis: 3,
        operations: 1,
      },
    },
    {
      title: "Customer Success Manager",
      field: "Customer Success",
      requirements: {
        communication: 4,
        customer_service: 4,
        critical_thinking: 3,
        programming: 0,
        data_analysis: 2,
        operations: 3,
      },
    },
    {
      title: "Business Analyst",
      field: "Business Analysis",
      requirements: {
        communication: 3,
        customer_service: 2,
        critical_thinking: 4,
        programming: 1,
        data_analysis: 4,
        operations: 3,
      },
    },
  ],
};

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

  for (const skill of catalog.skills) {
    const required = target.requirements[skill];
    const actual = user[skill] || 0;
    matchEarned += Math.min(actual, required);
    matchPossible += required;

    if (actual < required) {
      gapTotal += required - actual;
      gapSkills.push(catalog.skillLabels[skill]);
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
    bucket,
    match: Math.round(match),
    gap: Math.round(gap),
    title: target.title,
    field: target.field,
    gapSkills,
  };
}

function bucketLabel(bucket) {
  if (bucket === "ready") return "Ready Now";
  if (bucket === "trainable") return "Trainable";
  return "Long Reskill";
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
// Render skill sliders
// =============================================================================
function renderSkillInputs(profile) {
  skillsForm.innerHTML = "";
  for (const skill of catalog.skills) {
    const block = document.createElement("div");
    block.className = "skill-item";
    const value = profile[skill] ?? 0;
    block.innerHTML = `
      <label>${catalog.skillLabels[skill]}</label>
      <input data-skill="${skill}" type="range" min="0" max="4" step="1" value="${value}" aria-label="${catalog.skillLabels[skill]} skill level" />
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
      runRecommendations();
    });
  }
}

// =============================================================================
// Render recommendation results
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

  // Sort by match score descending within each bucket
  const sorted = [...results].sort((a, b) => b.match - a.match);

  for (const result of sorted) {
    counts[result.bucket]++;
    const item = document.createElement("li");

    const gapDetail =
      result.gapSkills.length > 0
        ? `<span class="result-gaps">Gaps: ${result.gapSkills.join(", ")}</span>`
        : "";

    item.innerHTML = `
      <div class="result-title">${result.title}</div>
      <div class="result-meta">
        <span>Match: <span class="match-pct">${result.match}%</span></span>
        <span>Gap: <span class="gap-pct">${result.gap}%</span></span>
        <span>${result.field}</span>
      </div>
      ${gapDetail}
    `;
    bucketEls[result.bucket].appendChild(item);
  }

  // Toggle empty states
  for (const key of Object.keys(emptyEls)) {
    emptyEls[key].style.display = counts[key] > 0 ? "none" : "block";
  }
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
// Run scoring + render
// =============================================================================
function runRecommendations() {
  const userProfile = readSkills();
  const scored = catalog.targets.map((target) =>
    scoreTarget(userProfile, target)
  );
  renderResults(scored);
  renderPersonaQaMatrix(scored);
}

// =============================================================================
// Persona management
// =============================================================================
function getActivePersonaKey() {
  const active = personaPicker.querySelector(".persona-btn.active");
  return active ? active.dataset.persona : null;
}

function applyPersona(personaKey) {
  const persona = catalog.personas[personaKey];

  // Update active button state
  personaPicker.querySelectorAll(".persona-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.persona === personaKey);
  });

  personaDescription.textContent = persona.description;
  currentRole.value = persona.role;
  renderSkillInputs(persona.profile);
  runRecommendations();
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
  // Deselect persona buttons when manually changing role
  personaPicker
    .querySelectorAll(".persona-btn")
    .forEach((btn) => btn.classList.remove("active"));
  personaDescription.textContent = "";
  renderSkillInputs(catalog.roles[currentRole.value]);
  runRecommendations();
});

resetButton.addEventListener("click", () => {
  const personaKey = getActivePersonaKey();
  if (personaKey) {
    applyPersona(personaKey);
  } else {
    renderSkillInputs(catalog.roles[currentRole.value]);
    runRecommendations();
  }
});

// Boot with first persona selected
const firstPersonaKey = Object.keys(catalog.personas)[0];
applyPersona(firstPersonaKey);
