function showAuthError(message, isSuccess = false) {
  const el = document.getElementById("auth-error");
  if (!el) return;
  el.textContent = message;
  el.classList.remove("hidden", "success");
  if (isSuccess) el.classList.add("success");
}

function hideAuthError() {
  const el = document.getElementById("auth-error");
  if (el) el.classList.add("hidden");
}

function storeToken(token) {
  if (token) localStorage.setItem("access_token", token);
}

async function parseError(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) return data.detail.map((d) => d.msg).join(", ");
    return data.message || "Request failed.";
  } catch {
    return "Request failed.";
  }
}

const loginForm = document.getElementById("login-form");
if (loginForm) {
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideAuthError();

    const body = {
      email: document.getElementById("email").value.trim(),
      password: document.getElementById("password").value,
    };

    try {
      const res = await fetch("/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        credentials: "include",
      });
      if (!res.ok) {
        showAuthError(await parseError(res));
        return;
      }
      const data = await res.json();
      storeToken(data.access_token);
      window.location.href = "/dashboard";
    } catch {
      showAuthError("Network error. Please try again.");
    }
  });
}

const signupForm = document.getElementById("signup-form");
if (signupForm) {
  signupForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideAuthError();

    const password = document.getElementById("password").value;
    const confirm = document.getElementById("confirm_password").value;
    if (password !== confirm) {
      showAuthError("Passwords do not match.");
      return;
    }

    const body = {
      name: document.getElementById("name").value.trim(),
      email: document.getElementById("email").value.trim(),
      password,
      confirm_password: confirm,
    };

    try {
      const res = await fetch("/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        credentials: "include",
      });
      if (!res.ok) {
        showAuthError(await parseError(res));
        return;
      }
      const data = await res.json();
      storeToken(data.access_token);
      window.location.href = "/dashboard";
    } catch {
      showAuthError("Network error. Please try again.");
    }
  });
}

const logoutBtn = document.getElementById("logout-btn");
if (logoutBtn) {
  logoutBtn.addEventListener("click", async () => {
    await fetch("/logout", { method: "POST", credentials: "include" });
    localStorage.removeItem("access_token");
    window.location.href = "/login";
  });
}
