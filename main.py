import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response

load_dotenv()

app = FastAPI(title="FunnelIQ")


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


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/dashboard")
def dashboard():
    return FileResponse("static/dashboard.html")
