# backend_api.py
# ============================================================
# ProjHub AI — FastAPI Wrapper (for backend team)
# ============================================================
# This is a ready-to-run FastAPI server that exposes
# the AI system as REST endpoints.
#
# RUN:
#   pip install fastapi uvicorn
#   uvicorn backend_api:app --reload --port 8000
#
# ENDPOINTS:
#   POST /api/suggest     → Scenario 1
#   POST /api/analyze     → Scenario 2
#   GET  /api/meta        → tracks, levels, domains
#   GET  /health          → health check
# ============================================================

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional
from dotenv import load_dotenv
from ai_core import ProjHubAI, VALID_TRACKS, VALID_LEVELS, VALID_DOMAINS

load_dotenv()

from contextlib import asynccontextmanager

_ai: ProjHubAI = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ai
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set in environment variables")
    _ai = ProjHubAI(groq_api_key=key)
    print("ProjHub AI ready ✓")
    yield

app = FastAPI(title="ProjHub AI API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response schemas ────────────────────────────────────────────────

class Skill(BaseModel):
    track: str
    level: str

    @field_validator("track")
    @classmethod
    def validate_track(cls, v):
        if v not in VALID_TRACKS:
            raise ValueError(f"track must be one of {VALID_TRACKS}")
        return v

    @field_validator("level")
    @classmethod
    def validate_level(cls, v):
        if v not in VALID_LEVELS:
            raise ValueError(f"level must be one of {VALID_LEVELS}")
        return v

class SuggestRequest(BaseModel):
    skills: List[Skill]
    domain: Optional[str] = None

class AnalyzeRequest(BaseModel):
    idea: str


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "ProjHub AI is running ✓", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "ProjHub AI"}


@app.get("/api/meta")
def meta():
    """Returns valid tracks, levels, and domains — useful for Flutter dropdowns."""
    return {
        "tracks":  VALID_TRACKS,
        "levels":  VALID_LEVELS,
        "domains": VALID_DOMAINS,
    }


@app.post("/api/suggest")
def suggest_projects(req: SuggestRequest):
    """
    Scenario 1 — generate project ideas from team skills.

    Request body:
    {
      "skills": [
        {"track": "AI", "level": "Intermediate"},
        {"track": "Flutter", "level": "Beginner"}
      ],
      "domain": "Healthcare & Wellness"   // optional
    }
    """
    try:
        skills = [{"track": s.track, "level": s.level} for s in req.skills]
        result = _ai.scenario_one(skills=skills, domain=req.domain)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
def analyze_idea(req: AnalyzeRequest):
    """
    Scenario 2 — analyze idea and find similar projects.

    Request body:
    {
      "idea": "AI chatbot for university academic advising"
    }
    """
    try:
        result = _ai.scenario_two(idea=req.idea)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))