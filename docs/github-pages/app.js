const catalog = {
  skills: ["communication", "customer_service", "critical_thinking", "programming", "data_analysis", "operations"],
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
  },
  personas: {
    "Nia (fashion -> new field)": {
      role: "Fashion Sales Associate",
      description: "Nia spent 6 years in fashion retail and wants to move into a more stable career path.",
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
  ],
};

const currentRole = document.getElementById("current-role");
const personaSelect = document.getElementById("persona-select");
const personaDescription = document.getElementById("persona-description");
const qaTableBody = document.querySelector("#qa-table tbody");
const skillsForm = document.getElementById("skills-form");
const runButton = document.getElementById("run-btn");
const resetButton = document.getElementById("reset-btn");

function normalize(value, max) {
  return max === 0 ? 0 : (value / max) * 100;
}

function scoreTarget(user, target) {
  let matchEarned = 0;
  let matchPossible = 0;
  let gapTotal = 0;
  let gapPossible = 0;

  for (const skill of catalog.skills) {
    const required = target.requirements[skill];
    const actual = user[skill] || 0;
    matchEarned += Math.min(actual, required);
    matchPossible += required;

    if (actual < required) {
      gapTotal += required - actual;
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
  };
}

function bucketLabel(bucket) {
  if (bucket === "ready") return "Ready Now";
  if (bucket === "trainable") return "Trainable";
  return "Long Reskill";
}

function readSkills() {
  const values = {};
  for (const skill of catalog.skills) {
    const input = document.querySelector(`[data-skill="${skill}"]`);
    values[skill] = Number(input.value);
  }
  return values;
}

function renderSkillInputs(profile) {
  skillsForm.innerHTML = "";
  for (const skill of catalog.skills) {
    const block = document.createElement("div");
    block.className = "skill-item";
    const value = profile[skill] ?? 0;
    block.innerHTML = `
      <label>${skill.replaceAll("_", " ")}</label>
      <input data-skill="${skill}" type="range" min="0" max="4" value="${value}" />
      <div>Level: <output>${value}</output></div>
    `;
    skillsForm.appendChild(block);

    const slider = block.querySelector("input");
    const output = block.querySelector("output");
    slider.addEventListener("input", () => {
      output.value = slider.value;
    });
  }
}

function renderResults(results) {
  const buckets = {
    ready: document.getElementById("bucket-ready"),
    trainable: document.getElementById("bucket-trainable"),
    long: document.getElementById("bucket-long"),
  };

  for (const list of Object.values(buckets)) {
    list.innerHTML = "";
  }

  for (const result of results) {
    const item = document.createElement("li");
    item.innerHTML = `<strong>${result.title}</strong> <span class="badge badge-${result.bucket}">${bucketLabel(result.bucket)}</span><div class="score">${result.field} | match ${result.match}% | gap ${result.gap}%</div>`;
    buckets[result.bucket].appendChild(item);
  }
}

function renderPersonaQaMatrix(results) {
  const persona = catalog.personas[personaSelect.value];
  const chosenFields = new Set(persona.focusFields);
  const rows = results
    .filter((result) => chosenFields.has(result.field))
    .sort((a, b) => b.match - a.match)
    .map((result) => `
      <tr>
        <td>${result.field}</td>
        <td>${result.title}</td>
        <td>${bucketLabel(result.bucket)}</td>
        <td>${result.match}%</td>
        <td>${result.gap}%</td>
      </tr>
    `)
    .join("");

  qaTableBody.innerHTML = rows;
}

function runRecommendations() {
  const userProfile = readSkills();
  const scored = catalog.targets.map((target) => scoreTarget(userProfile, target));
  renderResults(scored);
  renderPersonaQaMatrix(scored);
}

function applySelectedPersona() {
  const persona = catalog.personas[personaSelect.value];
  personaDescription.textContent = `${persona.description} QA focus fields: ${persona.focusFields.join(", ")}.`;
  currentRole.value = persona.role;
  renderSkillInputs(persona.profile);
  runRecommendations();
}

for (const roleName of Object.keys(catalog.roles)) {
  const option = document.createElement("option");
  option.value = roleName;
  option.textContent = roleName;
  currentRole.appendChild(option);
}

for (const personaName of Object.keys(catalog.personas)) {
  const option = document.createElement("option");
  option.value = personaName;
  option.textContent = personaName;
  personaSelect.appendChild(option);
}

currentRole.addEventListener("change", () => {
  renderSkillInputs(catalog.roles[currentRole.value]);
  runRecommendations();
});
personaSelect.addEventListener("change", applySelectedPersona);
runButton.addEventListener("click", runRecommendations);
resetButton.addEventListener("click", applySelectedPersona);

personaSelect.value = Object.keys(catalog.personas)[0];
applySelectedPersona();
