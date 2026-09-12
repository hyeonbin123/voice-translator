from fastapi import FastAPI

from app.routers import health

app = FastAPI(title="voice-translator API", version="0.1.0")

# Every route lives under /api, so the built frontend and the API can share one origin
# (the Vite dev server proxies /api to this app).
app.include_router(health.router, prefix="/api")
