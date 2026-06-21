# main.py
# ProjHub AI — FastAPI Server
# Deployed on Hugging Face Spaces
#
# v2 additions (existing endpoints UNCHANGED):
#   1. Hybrid Search (BM25 + Semantic) with lru_cache
#   2. Dynamic similarity threshold
#   3. Logging middleware
#   4. GET  /api/ai/stats/skills   - skill distribution for Bar Chart
#   5. GET  /api/ai/skill-gap      - skill gap analysis for Horizontal Bar Chart
#   6. Fallback mechanism if Groq fails

import os
import json
import time
import logging
import numpy as np
import urllib.request
from contextlib import asynccontextmanager
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# ----------------------------------------------
# CONFIG
# ----------------------------------------------

API_BASE_URL      = "https://projecthubb.runasp.net"
PROJECTS_ENDPOINT = f"{API_BASE_URL}/api/Projects"
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")

VALID_TRACKS = ["AI", "Backend", "Flutter", "UI/UX", "Data Science", "Mobile", "Web"]

TAG_MAPPING = {
    "ai": "AI", "backend": "Backend", "flutter": "Flutter",
    "ui/ux": "UI/UX", "uiux": "UI/UX", "ui": "UI/UX",
    "data science": "Data Science", "datascience": "Data Science",
    "mobile": "Mobile", "web": "Web",
}

# ----------------------------------------------
# LOGGING
# ----------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("projhub_ai")

# ----------------------------------------------
# REQUEST / RESPONSE MODELS
# ----------------------------------------------

class SkillItem(BaseModel):
    track: str
    level: str

class Scenario1Request(BaseModel):
    skills: List[SkillItem]
    domain: Optional[str] = None

class Scenario2Request(BaseModel):
    idea: str

# ----------------------------------------------
# HELPERS
# ----------------------------------------------

def normalize_tags(raw_tags) -> list:
    if not raw_tags:
        return []
    if isinstance(raw_tags, str):
        raw_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    if isinstance(raw_tags, list):
        result = []
        for t in raw_tags:
            if isinstance(t, str):
                mapped = TAG_MAPPING.get(t.lower().strip(), t.strip())
                if mapped:
                    result.append(mapped)
        return list(set(result))
    return []

def safe_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    return str(val)

# ----------------------------------------------
# FETCH PROJECTS FROM API
# ----------------------------------------------

def fetch_projects_from_api() -> list:
    req = urllib.request.Request(
        PROJECTS_ENDPOINT,
        headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))

    if isinstance(data, dict):
        data = data.get("projects") or data.get("data") or data.get("items") or []

    projects = []
    for item in data:
        if not isinstance(item, dict):
            continue
        tags = normalize_tags(item.get("tags"))
        desc = safe_str(item.get("description"))
        if not desc.strip():
            desc = safe_str(item.get("category"))

        projects.append({
            "id":          safe_str(item.get("id")),
            "title":       safe_str(item.get("title")) or "Untitled",
            "description": desc,
            "tags":        tags,
            "category":    safe_str(item.get("category")),
            "authorName":  safe_str(item.get("authorName")),
            "githubUrl":   safe_str(item.get("githubUrl")),
        })
    return projects

# ----------------------------------------------
# VECTOR STORE  (+ Hybrid Search: BM25 + Semantic)
# ----------------------------------------------

class VectorStore:
    def __init__(self, projects: list):
        import logging as _logging
        _logging.getLogger("sentence_transformers").setLevel(_logging.ERROR)
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from sentence_transformers import SentenceTransformer
        import faiss
        self._faiss = faiss
        self.model  = SentenceTransformer("all-MiniLM-L6-v2")
        self.projects = projects
        self._build_index()

    def _project_text(self, p: dict) -> str:
        return f"{p['title']}. {p['description']} Tracks: {', '.join(p['tags'])} Category: {p.get('category','')}"

    def _build_index(self):
        texts = [self._project_text(p) for p in self.projects]

        self.embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        dim = self.embeddings.shape[1] if len(self.embeddings) else 384
        self.index = self._faiss.IndexFlatIP(dim)
        if len(self.embeddings) > 0:
            self.index.add(self.embeddings.astype(np.float32))

        from rank_bm25 import BM25Okapi
        tokenized = [t.lower().split() for t in texts] or [[]]
        self.bm25 = BM25Okapi(tokenized) if texts else None

        self._cached_search.cache_clear()

    def _dynamic_threshold(self, scores: np.ndarray) -> float:
        if scores.size == 0:
            return 0.25
        top = float(scores.max())
        if top >= 0.5:
            return 0.30
        if top >= 0.30:
            return 0.20
        return max(0.10, top * 0.5)

    def _semantic_scores(self, query: str) -> np.ndarray:
        if len(self.projects) == 0:
            return np.array([])
        vec = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        k = len(self.projects)
        scores, indices = self.index.search(vec, k)
        full = np.zeros(len(self.projects), dtype=np.float32)
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                full[idx] = score
        return full

    def _bm25_scores(self, query: str) -> np.ndarray:
        if not self.bm25 or len(self.projects) == 0:
            return np.zeros(len(self.projects))
        raw = np.array(self.bm25.get_scores(query.lower().split()), dtype=np.float32)
        if raw.size > 0 and raw.max() > 0:
            raw = raw / raw.max()
        return raw

    @lru_cache(maxsize=256)
    def _cached_search(self, query: str, top_k: int, alpha: float):
        sem  = self._semantic_scores(query)
        bm25 = self._bm25_scores(query)
        if sem.size == 0:
            return []

        hybrid = alpha * sem + (1 - alpha) * bm25
        threshold = self._dynamic_threshold(hybrid)

        ranked_idx = np.argsort(-hybrid)[:top_k]
        results = []
        for idx in ranked_idx:
            if hybrid[idx] >= threshold:
                p = self.projects[idx].copy()
                p["similarity_score"] = round(float(hybrid[idx]), 3)
                results.append(p)
        return results

    def search(self, query: str, top_k=5, threshold=None, alpha=0.65) -> list:
        results = self._cached_search(query, top_k, alpha)
        if threshold is not None:
            results = [r for r in results if r["similarity_score"] >= threshold]
        return results

    def reload(self):
        self.projects = fetch_projects_from_api()
        self._build_index()

# ----------------------------------------------
# GROQ SERVICE  (+ Fallback mechanism)
# ----------------------------------------------

class GroqService:
    def __init__(self):
        from groq import Groq
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model  = "llama-3.3-70b-versatile"

    def _call(self, prompt: str, max_tokens=1000) -> str:
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return res.choices[0].message.content.strip()

    def _parse_json(self, text: str):
        cleaned = text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)

    def _safe_call_json(self, prompt: str, max_tokens: int, fallback):
        try:
            return self._parse_json(self._call(prompt, max_tokens=max_tokens))
        except Exception as e:
            logger.warning(f"Groq call failed, using fallback. Reason: {e}")
            return fallback

    def suggest_projects(self, skills: list, domain: str = None) -> list:
        skills_text = "\n".join(f"  - {s['track']}: {s['level']}" for s in skills)
        domain_line = f"\nThe project MUST be in the domain of: {domain}" if domain else ""
        prompt = f"""You are an expert academic project advisor for university students.
The team has these skills:
{skills_text}
{domain_line}
Generate exactly 3 graduation project ideas that fit these skills.
Rules:
- Match complexity to skill levels (Beginner=simple, Advanced=complex)
- Each project must use at least one of the team's tracks
- Be specific and realistic
- tech_stack must be general tools based on the tracks
- how_it_works must be 3-4 simple steps
- recommended_tracks must be a JSON array of strings ONLY
Respond ONLY with a valid JSON array, no extra text:
[
  {{
    "title": "Project Title",
    "description": "2-3 sentence description.",
    "recommended_tracks": ["Track1", "Track2"],
    "tech_stack": {{
      "Frontend": "e.g. Flutter",
      "Backend": "e.g. FastAPI",
      "Database": "e.g. PostgreSQL",
      "Hosting": "e.g. Firebase Hosting"
    }},
    "how_it_works": ["Step 1: ...", "Step 2: ...", "Step 3: ..."]
  }}
]"""
        primary_track = skills[0]["track"] if skills else "Backend"
        fallback = [{
            "title": f"{primary_track} Capstone Project",
            "description": f"A practical graduation project that applies your {primary_track} skills "
                            f"to solve a real academic or campus problem. "
                            f"(AI suggestion service is temporarily unavailable - this is a generic fallback idea.)",
            "recommended_tracks": [primary_track],
            "tech_stack": {"Frontend": "Flutter", "Backend": "FastAPI", "Database": "PostgreSQL", "Hosting": "Firebase Hosting"},
            "how_it_works": ["Step 1: Define the problem", "Step 2: Build an MVP", "Step 3: Test with real users"],
        }]

        raw = self._safe_call_json(prompt, max_tokens=1000, fallback=fallback)
        for item in raw:
            if isinstance(item.get("recommended_tracks"), str):
                item["recommended_tracks"] = [t.strip() for t in item["recommended_tracks"].split(",")]
        return raw

    def analyze_idea(self, idea: str) -> dict:
        prompt = f"""You are an expert academic project advisor for university students.
A student has this project idea: "{idea}"
Respond ONLY with a valid JSON object:
{{
  "refined_title": "A clear professional title",
  "description": "2-3 sentence description.",
  "detected_tracks": ["Track1", "Track2"],
  "tech_stack": {{
    "Frontend": "e.g. Flutter",
    "Backend": "e.g. FastAPI",
    "Database": "e.g. PostgreSQL",
    "Hosting": "e.g. Firebase Hosting"
  }},
  "how_it_works": ["Step 1: ...", "Step 2: ...", "Step 3: ...", "Step 4: ..."]
}}
Available tracks: AI, Backend, Flutter, UI/UX, Data Science, Mobile, Web
detected_tracks must be a JSON array of strings ONLY."""

        fallback = {
            "refined_title": idea[:60],
            "description": "AI analysis service is temporarily unavailable. "
                            "Showing your original idea with similar projects from the database instead.",
            "detected_tracks": [],
            "tech_stack": {},
            "how_it_works": [],
        }

        raw = self._safe_call_json(prompt, max_tokens=800, fallback=fallback)
        if isinstance(raw.get("detected_tracks"), str):
            raw["detected_tracks"] = [t.strip() for t in raw["detected_tracks"].split(",")]
        return raw

    def explain_similarity(self, idea: str, projects: list) -> list:
        projects_text = "\n".join(
            f'  {i+1}. "{p["title"]}": {p["description"]}'
            for i, p in enumerate(projects)
        )
        prompt = f"""A student has this idea: "{idea}"
Similar projects found:
{projects_text}
Write ONE specific sentence per project explaining what makes it similar.
Respond ONLY with a JSON array of strings:
["explanation 1", "explanation 2", ...]"""

        fallback = ["Similar in topic and tracks to your idea." for _ in projects]
        return self._safe_call_json(prompt, max_tokens=400, fallback=fallback)

    def skill_gap_recommendation(self, project_title: str, missing: list, weak: list) -> str:
        if not missing and not weak:
            return f'Your team is fully equipped for "{project_title}".'
        prompt = f"""A student team wants to build: "{project_title}"
Missing skills entirely: {', '.join(missing) if missing else 'none'}
Weak skills (Beginner level): {', '.join(weak) if weak else 'none'}
Write ONE short, encouraging sentence (max 25 words) recommending what the team should learn or practice first.
Respond with plain text only, no quotes, no markdown."""
        fallback = (
            f"Consider strengthening: {', '.join(missing + weak)} before starting."
            if (missing or weak) else "Your team looks ready for this project."
        )
        try:
            return self._call(prompt, max_tokens=60).strip().strip('"')
        except Exception as e:
            logger.warning(f"Groq skill-gap recommendation failed, using fallback. Reason: {e}")
            return fallback

# ----------------------------------------------
# APP STARTUP
# ----------------------------------------------

store: VectorStore = None
groq:  GroqService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, groq
    logger.info("Loading projects from API...")
    projects = fetch_projects_from_api()
    store = VectorStore(projects)
    groq  = GroqService()
    logger.info(f"Ready - {len(projects)} projects indexed.")
    yield
    logger.info("Shutting down.")

app = FastAPI(title="ProjHub AI", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------
# LOGGING MIDDLEWARE
# ----------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 1)
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)"
    )
    return response

# ----------------------------------------------
# ENDPOINTS - EXISTING (unchanged behavior)
# ----------------------------------------------

def _build_skills_chart() -> dict:
    """نفس منطق /api/ai/stats/skills بس كـ helper داخلي نستخدمه جوه /suggest"""
    counts = {}
    for p in store.projects:
        for tag in p.get("tags", []):
            counts[tag] = counts.get(tag, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return {
        "labels": [k for k, _ in sorted_items],
        "values": [v for _, v in sorted_items],
        "total_projects": len(store.projects),
    }


def _build_skill_gap_chart(skills: List[SkillItem], domain: Optional[str]) -> dict:
    """نفس منطق /api/ai/skill-gap بس كـ helper داخلي نستخدمه جوه /suggest"""
    team_tracks = {s.track: s.level for s in skills}
    query = " ".join(team_tracks.keys()) + (f" {domain}" if domain else "")
    candidate_projects = store.search(query, top_k=6) if query.strip() else store.projects[:6]

    results = []
    for p in candidate_projects:
        project_tracks = p.get("tags", [])
        missing_tracks = [t for t in project_tracks if t not in team_tracks]
        weak_tracks    = [t for t in project_tracks if team_tracks.get(t) == "Beginner"]

        total_required = max(len(project_tracks), 1)
        gap_score = round((len(missing_tracks) + 0.5 * len(weak_tracks)) / total_required, 2)
        gap_score = min(gap_score, 1.0)

        if gap_score == 0:
            status = "ready"
        elif gap_score <= 0.4:
            status = "minor_gap"
        else:
            status = "needs_work"

        recommendation = groq.skill_gap_recommendation(p.get("title", "this project"), missing_tracks, weak_tracks)

        results.append({
            "project_id":      p.get("id"),
            "project_title":   p.get("title"),
            "project_tracks":  project_tracks,
            "missing_tracks":  missing_tracks,
            "weak_tracks":     weak_tracks,
            "gap_score":       gap_score,
            "status":          status,
            "recommendation":  recommendation,
        })

    ready_count      = sum(1 for r in results if r["status"] == "ready")
    needs_work_count = sum(1 for r in results if r["status"] == "needs_work")

    return {
        "projects": results,
        "summary": {
            "total_analyzed": len(results),
            "ready_projects": ready_count,
            "needs_work":     needs_work_count,
            "minor_gap":      len(results) - ready_count - needs_work_count,
        },
    }


@app.post("/api/ai/suggest")
def suggest(req: Scenario1Request):
    if not req.skills:
        raise HTTPException(status_code=400, detail="skills list is empty")

    skills_dicts = [{"track": s.track, "level": s.level} for s in req.skills]
    suggestions  = groq.suggest_projects(skills_dicts, domain=req.domain)

    result = []
    for s in suggestions:
        query   = s["title"] + " " + s["description"]
        similar = store.search(query, top_k=3)

        similar_clean = []
        for p in similar:
            similar_clean.append({
                "id":               safe_str(p.get("id")),
                "title":            safe_str(p.get("title")),
                "description":      safe_str(p.get("description")),
                "tags":             normalize_tags(p.get("tags")),
                "authorName":       safe_str(p.get("authorName")),
                "githubUrl":        safe_str(p.get("githubUrl")),
                "similarity_score": p.get("similarity_score", 0.0),
            })

        result.append({
            "title":              safe_str(s.get("title")),
            "description":        safe_str(s.get("description")),
            "recommended_tracks": s.get("recommended_tracks", []),
            "tech_stack":         s.get("tech_stack", {}),
            "how_it_works":       s.get("how_it_works", []),
            "similar_projects":   similar_clean,
        })

    # الإضافة الجديدة: الـ charts بترجع تلقائي جوه نفس الـ response
    # أي حقل قديم فوق ده متغيرش؛ دول حقلين إضافيين بس على نفس مستوى "suggestions"
    return {
        "suggestions":      result,
        "skills_chart":     _build_skills_chart(),
        "skill_gap_chart":  _build_skill_gap_chart(req.skills, req.domain),
    }


@app.post("/api/ai/analyze")
def analyze(req: Scenario2Request):
    if len(req.idea.strip()) < 10:
        raise HTTPException(status_code=400, detail="idea is too short")

    analysis     = groq.analyze_idea(req.idea)
    search_query = safe_str(analysis.get("refined_title")) + " " + safe_str(analysis.get("description", req.idea))
    similar      = store.search(search_query, top_k=5)
    explanations = groq.explain_similarity(req.idea, similar) if similar else []

    similar_clean = []
    for i, p in enumerate(similar):
        similar_clean.append({
            "id":                     safe_str(p.get("id")),
            "title":                  safe_str(p.get("title")),
            "description":            safe_str(p.get("description")),
            "tags":                   normalize_tags(p.get("tags")),
            "authorName":             safe_str(p.get("authorName")),
            "githubUrl":              safe_str(p.get("githubUrl")),
            "similarity_score":       p.get("similarity_score", 0.0),
            "similarity_explanation": explanations[i] if i < len(explanations) else "",
        })

    return {
        "refined_title":    safe_str(analysis.get("refined_title")),
        "description":      safe_str(analysis.get("description")),
        "detected_tracks":  analysis.get("detected_tracks", []),
        "tech_stack":       analysis.get("tech_stack", {}),
        "how_it_works":     analysis.get("how_it_works", []),
        "similar_projects": similar_clean,
        "skills_chart":     _build_skills_chart(),
    }


@app.post("/api/ai/reload")
def reload_projects():
    store.reload()
    return {"message": f"Reloaded. {len(store.projects)} projects indexed."}


@app.get("/api/ai/health")
def health():
    return {"status": "ok", "projects_indexed": len(store.projects)}


@app.get("/")
def root():
    return {"message": "ProjHub AI is running", "docs": "/docs"}

# ----------------------------------------------
# ENDPOINTS - NEW
# ----------------------------------------------

@app.get("/api/ai/stats/skills")
def skills_distribution():
    """
    Skill/track distribution across all projects currently indexed.
    Ready for a Bar Chart directly in Flutter:
    { "labels": ["AI","Backend",...], "values": [12,8,...] }
    """
    counts = {}
    for p in store.projects:
        for tag in p.get("tags", []):
            counts[tag] = counts.get(tag, 0) + 1

    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    return {
        "labels": labels,
        "values": values,
        "total_projects": len(store.projects),
    }


@app.post("/api/ai/skill-gap")
def skill_gap(req: Scenario1Request):
    """
    Skill gap analysis between the team's skills and a set of candidate
    projects (closest matches to the team's tracks/domain).
    Returns per-project missing_tracks, weak_tracks, gap_score, status,
    recommendation, plus an overall summary - ready for a Horizontal Bar
    Chart in Flutter (use gap_score as the bar value).
    """
    if not req.skills:
        raise HTTPException(status_code=400, detail="skills list is empty")

    team_tracks = {s.track: s.level for s in req.skills}

    query = " ".join(team_tracks.keys()) + (f" {req.domain}" if req.domain else "")
    candidate_projects = store.search(query, top_k=6) if query.strip() else store.projects[:6]

    results = []
    for p in candidate_projects:
        project_tracks = p.get("tags", [])
        missing_tracks = [t for t in project_tracks if t not in team_tracks]
        weak_tracks    = [t for t in project_tracks if team_tracks.get(t) == "Beginner"]

        total_required = max(len(project_tracks), 1)
        gap_score = round((len(missing_tracks) + 0.5 * len(weak_tracks)) / total_required, 2)
        gap_score = min(gap_score, 1.0)

        if gap_score == 0:
            status = "ready"
        elif gap_score <= 0.4:
            status = "minor_gap"
        else:
            status = "needs_work"

        recommendation = groq.skill_gap_recommendation(p.get("title", "this project"), missing_tracks, weak_tracks)

        results.append({
            "project_id":      p.get("id"),
            "project_title":   p.get("title"),
            "project_tracks":  project_tracks,
            "missing_tracks":  missing_tracks,
            "weak_tracks":     weak_tracks,
            "gap_score":       gap_score,
            "status":          status,
            "recommendation":  recommendation,
        })

    ready_count      = sum(1 for r in results if r["status"] == "ready")
    needs_work_count = sum(1 for r in results if r["status"] == "needs_work")

    return {
        "team_skills": {s.track: s.level for s in req.skills},
        "projects": results,
        "summary": {
            "total_analyzed": len(results),
            "ready_projects": ready_count,
            "needs_work":     needs_work_count,
            "minor_gap":      len(results) - ready_count - needs_work_count,
        },
    }