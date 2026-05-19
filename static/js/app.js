const SEVERITY_LABELS = {
  1: { name: "Low", desc: "Minor impact — limited disruption expected." },
  2: { name: "Moderate", desc: "Noticeable impact — some delays likely." },
  3: { name: "High", desc: "Significant impact — substantial delays possible." },
  4: { name: "Severe", desc: "Critical impact — major disruption expected." },
};

function authHeaders() {
  const token = localStorage.getItem("access_token");
  const headers = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

async function loadWeatherOptions() {
  const select = document.getElementById("weather");
  if (!select) return;
  try {
    const res = await fetch("/weather-options");
    if (!res.ok) return;
    const data = await res.json();
    select.innerHTML = data.conditions
      .slice(0, 40)
      .map((c) => `<option value="${c}">${c}</option>`)
      .join("");
  } catch {
    /* keep defaults */
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
  if (!errorEl) return;
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
  placeholder?.classList.remove("hidden");
  content?.classList.add("hidden");
}

function hideError() {
  errorEl?.classList.add("hidden");
}

function showResult(prediction) {
  hideError();
  const info = SEVERITY_LABELS[prediction] || {
    name: `Level ${prediction}`,
    desc: "Prediction from model.",
  };
  badge.textContent = `Severity ${prediction}`;
  badge.className = `severity-badge severity-${prediction}`;
  level.textContent = info.name;
  desc.textContent = info.desc;
  placeholder?.classList.add("hidden");
  content?.classList.remove("hidden");
}

if (form) {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError();

    const payload = {
      temperature: parseFloat(document.getElementById("temperature").value),
      humidity: parseFloat(document.getElementById("humidity").value),
      visibility: parseFloat(document.getElementById("visibility").value),
      wind_speed: parseFloat(document.getElementById("wind_speed").value),
      precipitation: parseFloat(document.getElementById("precipitation")?.value || 0),
      pressure: parseFloat(document.getElementById("pressure")?.value || 29.9),
      distance: parseFloat(document.getElementById("distance")?.value || 0.1),
      hour: parseInt(document.getElementById("hour")?.value || 12, 10),
      dayofweek: parseInt(document.getElementById("dayofweek")?.value || 0, 10),
      state: document.getElementById("state")?.value || "CA",
      street: document.getElementById("street")?.value || "",
      weather_condition: document.getElementById("weather").value,
      sunrise_sunset: document.getElementById("sunrise_sunset")?.value || "Day",
      amenity: document.getElementById("amenity")?.checked || false,
      bump: document.getElementById("bump")?.checked || false,
      crossing: document.getElementById("crossing")?.checked || false,
      junction: document.getElementById("junction")?.checked || false,
      traffic_signal: document.getElementById("traffic_signal")?.checked || false,
    };

    form.classList.add("loading");

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: authHeaders(),
        credentials: "include",
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (response.status === 401) {
        window.location.href = "/login";
        return;
      }

      if (!response.ok) {
        const detail = data.detail;
        showError(
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail.map((d) => d.msg).join(", ")
              : "Prediction failed."
        );
        return;
      }

      showResult(data.prediction);
    } catch {
      showError("Could not reach the server.");
    } finally {
      form.classList.remove("loading");
    }
  });
}
