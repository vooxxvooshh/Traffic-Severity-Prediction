const SEVERITY_LABELS = {
  1: { name: "Low", desc: "Minor impact — limited disruption expected." },
  2: { name: "Moderate", desc: "Noticeable impact — some delays likely." },
  3: { name: "High", desc: "Significant impact — substantial delays possible." },
  4: { name: "Severe", desc: "Critical impact — major disruption expected." },
};

async function loadWeatherOptions() {
  const select = document.getElementById("weather");
  try {
    const res = await fetch("/weather-options");
    if (!res.ok) return;
    const data = await res.json();
    const preferred = new Set([
      "Clear", "Cloudy", "Fair", "Mostly Cloudy", "Partly Cloudy", "Overcast",
      "Light Rain", "Rain", "Heavy Rain", "Light Snow", "Snow", "Fog", "Haze",
      "Thunder", "Thunderstorm", "Scattered Clouds",
    ]);
    const options = data.conditions.filter((c) => preferred.has(c));
    const list = options.length ? options : data.conditions.slice(0, 20);
    select.innerHTML = list
      .map((c) => `<option value="${c}">${c}</option>`)
      .join("");
  } catch {
    /* keep static fallback options */
  }
}

loadWeatherOptions();

const form = document.getElementById("predict-form");
const placeholder = document.getElementById("result-placeholder");
const content = document.getElementById("result-content");
const errorEl = document.getElementById("result-error");
const badge = document.getElementById("severity-badge");
const level = document.getElementById("result-level");
const desc = document.getElementById("result-desc");

function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
  placeholder.classList.remove("hidden");
  content.classList.add("hidden");
}

function hideError() {
  errorEl.classList.add("hidden");
}

function showResult(prediction) {
  hideError();
  const info = SEVERITY_LABELS[prediction] || {
    name: `Level ${prediction}`,
    desc: "Prediction returned from the model.",
  };

  badge.textContent = `Severity ${prediction}`;
  badge.className = `severity-badge severity-${prediction}`;
  level.textContent = info.name;
  desc.textContent = info.desc;

  placeholder.classList.add("hidden");
  content.classList.remove("hidden");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();

  const payload = {
    temperature: parseFloat(document.getElementById("temperature").value),
    humidity: parseFloat(document.getElementById("humidity").value),
    visibility: parseFloat(document.getElementById("visibility").value),
    wind_speed: parseFloat(document.getElementById("wind_speed").value),
    weather_condition: document.getElementById("weather").value,
    amenity: document.getElementById("amenity").checked,
    bump: document.getElementById("bump").checked,
    crossing: document.getElementById("crossing").checked,
    junction: document.getElementById("junction").checked,
    traffic_signal: document.getElementById("traffic_signal").checked,
  };

  form.classList.add("loading");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      const detail = data.detail;
      const msg = Array.isArray(detail)
        ? detail.map((d) => d.msg).join(", ")
        : detail || "Prediction failed. Please check your inputs.";
      showError(msg);
      return;
    }

    showResult(data.prediction);
  } catch {
    showError("Could not reach the server. Try again later.");
  } finally {
    form.classList.remove("loading");
  }
});
