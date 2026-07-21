import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response
from supabase import create_client

from models import lifetime_models

load_dotenv()

_auth_client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])


def require_auth(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        resp = _auth_client.auth.get_user(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    if not resp or not resp.user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return resp.user


@asynccontextmanager
async def lifespan(app: FastAPI):
    lifetime_models.load()
    yield


app = FastAPI(title="FunnelIQ", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config.js")
def config_js():
    js = (
        f'window.SUPABASE_URL = "{os.environ["SUPABASE_URL"]}";\n'
        f'window.SUPABASE_ANON_KEY = "{os.environ["SUPABASE_ANON_KEY"]}";\n'
    )
    return Response(content=js, media_type="application/javascript")


@app.get("/api/lifetime/comparison", dependencies=[Depends(require_auth)])
def lifetime_comparison():
    return {
        "comparison": lifetime_models.comparison,
        "feature_defaults": lifetime_models.feature_defaults,
    }


@app.post("/api/lifetime/predict", dependencies=[Depends(require_auth)])
def lifetime_predict(features: dict):
    return lifetime_models.predict(features)


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/dashboard")
def dashboard():
    return FileResponse("static/dashboard.html")
