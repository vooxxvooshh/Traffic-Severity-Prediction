import os
from pathlib import Path

import joblib
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field

from auth import (
    authenticate_user,
    create_access_token,
    create_user,
    get_current_user,
    setup_auth,
)
from inference import build_feature_vector

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "traffic_model.pkl"
ARTIFACTS_PATH = BASE_DIR / "model_artifacts.pkl"

app = FastAPI(
    title="Traffic Severity Predictor",
    description="Predict US traffic accident severity with authentication.",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

model = None
artifacts = None

PROTECTED_PAGES = {"/dashboard"}


def load_ml_assets() -> None:
    global model, artifacts
    if MODEL_PATH.exists() and ARTIFACTS_PATH.exists():
        model = joblib.load(MODEL_PATH)
        artifacts = joblib.load(ARTIFACTS_PATH)
    elif MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        artifacts = joblib.load(BASE_DIR / "weather_encoder.pkl") if (BASE_DIR / "weather_encoder.pkl").exists() else {}


@app.on_event("startup")
def on_startup() -> None:
    setup_auth()
    load_ml_assets()


@app.middleware("http")
async def protect_dashboard(request: Request, call_next):
    if request.url.path in PROTECTED_PAGES:
        token = request.cookies.get("access_token")
        if not token:
            return RedirectResponse(url="/login", status_code=302)
        try:
            from auth import decode_token

            decode_token(token)
        except HTTPException:
            return RedirectResponse(url="/login", status_code=302)
    return await call_next(request)


class SignupBody(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    email: EmailStr
    password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)


class LoginBody(BaseModel):
    email: EmailStr
    password: str


class InputData(BaseModel):
    temperature: float
    humidity: float = Field(..., ge=0, le=100)
    visibility: float = Field(..., ge=0)
    wind_speed: float = Field(..., ge=0)
    weather_condition: str
    precipitation: float = 0.0
    pressure: float = 29.9
    distance: float = 0.1
    hour: int = Field(12, ge=0, le=23)
    dayofweek: int = Field(0, ge=0, le=6)
    sunrise_sunset: str = "Day"
    state: str = "CA"
    street: str = ""
    amenity: bool = False
    bump: bool = False
    crossing: bool = False
    junction: bool = False
    traffic_signal: bool = False


def _set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=os.getenv("RENDER", "") == "true",
        max_age=60 * 60 * 24,
    )


@app.get("/")
async def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/login")
async def login_page():
    return FileResponse(BASE_DIR / "login.html")


@app.get("/signup")
async def signup_page():
    return FileResponse(BASE_DIR / "signup.html")


@app.get("/dashboard")
async def dashboard_page():
    return FileResponse(BASE_DIR / "dashboard.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "artifacts_loaded": artifacts is not None and "feature_names" in (artifacts or {}),
    }


@app.post("/signup")
async def signup(body: SignupBody, response: Response):
    if body.password != body.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    user = create_user(body.name, body.email, body.password)
    token = create_access_token(user["id"], user["email"])
    _set_auth_cookie(response, token)
    return {"message": "Account created.", "user": user, "access_token": token}


@app.post("/login")
async def login(body: LoginBody, response: Response):
    user = authenticate_user(body.email, body.password)
    token = create_access_token(user["id"], user["email"])
    _set_auth_cookie(response, token)
    return {"message": "Login successful.", "user": user, "access_token": token}


@app.post("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"message": "Logged out."}


@app.get("/me")
async def me(user: dict = Depends(get_current_user)):
    return user


@app.get("/weather-options")
async def weather_options():
    if artifacts and "weather" in artifacts:
        return {"conditions": list(artifacts["weather"].classes_)}
    legacy = BASE_DIR / "weather_encoder.pkl"
    if legacy.exists():
        enc = joblib.load(legacy)
        return {"conditions": list(enc.classes_)}
    return {"conditions": ["Clear", "Cloudy", "Rain", "Fog", "Snow"]}


@app.post("/predict")
async def predict(data: InputData, user: dict = Depends(get_current_user)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run main.py to train.")

    payload = data.model_dump()
    if artifacts and "feature_names_all" in artifacts:
        features = build_feature_vector(payload, artifacts)
    else:
        raise HTTPException(status_code=503, detail="Model artifacts missing. Retrain with main.py.")

    prediction = model.predict(features)
    return {
        "prediction": int(prediction[0]),
        "user": user["email"],
    }
