# main.py
# ProjHub AI — FastAPI Server
# Deployed on Hugging Face Spaces

import os
import json
import numpy as np
import urllib.request
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# ──────────────────────────────────────────────
# CONFIG — من الـ Environment Variables
# ──────────────────────────────────────────────

API_BASE_URL      = "https://projecthubb.runasp.net"
PROJECTS_ENDPOINT = f"{API_BASE_URL}/api/Projects"
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")   # ← هيتحط في Hugging Face Secrets

VALID_TRACKS = ["AI", "Backend", "Flutter", "UI/UX", "Data Science", "Mobile", "Web"]


# ──────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ──────────────────────────────────────────────

class SkillItem(BaseModel):
    track: str
    level: str

class Scenario1Request(BaseModel):
    skills: List[SkillItem]
    domain: Optional[str] = None

class Scenario2Request(BaseModel):
    idea: str


# ──────────────────────────────────────────────
# FETCH PROJECTS FROM API
# ──────────────────────────────────────────────

def fetch_projects_from_api() -> list:
    req = urllib.request.Request(
        PROJECTS_ENDPOINT,
        headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))

    mapping = {
        "ai": "AI", "backend": "Backend", "flutter": "Flutter",
        "ui/ux": "UI/UX", "uiux": "UI/UX", "ui": "UI/UX",
        "data science": "Data Science", "datascience": "Data Science",
        "mobile": "Mobile", "web": "Web",
    }
    projects = []
    for item in data:
        raw_tags = item.get("tags") or []
        tags = list({mapping.get(t.lower().strip(), t) for t in raw_tags})
        projects.append({
            "id":          item.get("id"),
            "title":       item.get("title", "Untitled"),
            "description": item.get("description", ""),
            "tags":        tags,
            "category":    item.get("category", ""),
            "authorName":  item.get("authorName", ""),
            "githubUrl":   item.get("githubUrl", ""),
        })
    return projects


# ──────────────────────────────────────────────
# VECTOR STORE
# ──────────────────────────────────────────────

class VectorStore:
    def __init__(self, projects: list):
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from sentence_transformers import SentenceTransformer
        import faiss
        self._faiss   = faiss
        self.model    = SentenceTransformer("all-MiniLM-L6-v2")
        self.projects = projects
        self._build_index()

    def _build_index(self):
        texts = [
            f"{p['title']}. {p['description']} Tracks: {', '.join(p['tags'])} Category: {p.get('category','')}"
            for p in self.projects
        ]
        self.embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        dim = self.embeddings.shape[1]
        self.index = self._faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))

    def search(self, query: str, top_k=5, threshold=0.25) -> list:
        vec = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        k   = min(top_k, len(self.projects))
        scores, indices = self.index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold:
                p = self.projects[idx].copy()
                p["similarity_score"] = round(float(score), 3)
                results.append(p)
        return sorted(results, key=lambda x: x["similarity_score"], reverse=True)

    def reload(self):
        self.projects = fetch_projects_from_api()
        self._build_index()


# ──────────────────────────────────────────────
# GROQ SERVICE
# ──────────────────────────────────────────────

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
        return json.loads(text.replace("```json", "").replace("```", "").strip())

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
        return self._parse_json(self._call(prompt))

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

Available tracks: AI, Backend, Flutter, UI/UX, Data Science, Mobile, Web"""
        return self._parse_json(self._call(prompt, max_tokens=800))

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
        return self._parse_json(self._call(prompt, max_tokens=400))


# ──────────────────────────────────────────────
# APP STARTUP
# ──────────────────────────────────────────────

store: VectorStore = None
groq:  GroqService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, groq
    print("🚀 Loading projects from API...")
    projects = fetch_projects_from_api()
    store = VectorStore(projects)
    groq  = GroqService()
    print(f"✅ Ready — {len(projects)} projects indexed.")
    yield
    print("👋 Shutting down.")

app = FastAPI(title="ProjHub AI", version="1.0.0", lifespan=lifespan)

# CORS — مهم جداً عشان الـ .NET والـ Flutter يقدروا يكلموا الـ API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────

@app.post("/api/ai/suggest")
def suggest(req: Scenario1Request):
    if not req.skills:
        raise HTTPException(status_code=400, detail="skills list is empty")

    skills_dicts = [{"track": s.track, "level": s.level} for s in req.skills]
    suggestions  = groq.suggest_projects(skills_dicts, domain=req.domain)

    result = []
    for s in suggestions:
        query   = s["title"] + " " + s["description"]
        similar = store.search(query, top_k=3, threshold=0.25)
        result.append({
            "title":              s.get("title"),
            "description":        s.get("description"),
            "recommended_tracks": s.get("recommended_tracks", []),
            "tech_stack":         s.get("tech_stack", {}),
            "how_it_works":       s.get("how_it_works", []),
            "similar_projects": [
                {
                    "id":               p.get("id"),
                    "title":            p.get("title"),
                    "description":      p.get("description"),
                    "tags":             p.get("tags"),
                    "authorName":       p.get("authorName"),
                    "githubUrl":        p.get("githubUrl"),
                    "similarity_score": p.get("similarity_score"),
                }
                for p in similar
            ],
        })
    return {"suggestions": result}


@app.post("/api/ai/analyze")
def analyze(req: Scenario2Request):
    if len(req.idea.strip()) < 10:
        raise HTTPException(status_code=400, detail="idea is too short")

    analysis     = groq.analyze_idea(req.idea)
    search_query = analysis.get("refined_title", "") + " " + analysis.get("description", req.idea)
    similar      = store.search(search_query, top_k=5, threshold=0.25)
    explanations = groq.explain_similarity(req.idea, similar) if similar else []

    return {
        "refined_title":   analysis.get("refined_title"),
        "description":     analysis.get("description"),
        "detected_tracks": analysis.get("detected_tracks", []),
        "tech_stack":      analysis.get("tech_stack", {}),
        "how_it_works":    analysis.get("how_it_works", []),
        "similar_projects": [
            {
                "id":                     p.get("id"),
                "title":                  p.get("title"),
                "description":            p.get("description"),
                "tags":                   p.get("tags"),
                "authorName":             p.get("authorName"),
                "githubUrl":              p.get("githubUrl"),
                "similarity_score":       p.get("similarity_score"),
                "similarity_explanation": explanations[i] if i < len(explanations) else "",
            }
            for i, p in enumerate(similar)
        ],
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
    return {"message": "ProjHub AI is running 🚀", "docs": "/docs"}