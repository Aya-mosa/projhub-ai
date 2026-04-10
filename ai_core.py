# ai_core.py
# ============================================================
# ProjHub AI — Core Logic (Backend Integration Ready)
# ============================================================
# This file contains the pure AI logic with NO terminal UI.
# The backend team plugs this into Firebase Functions or FastAPI.
#
# HOW TO USE:
#   from ai_core import ProjHubAI
#   ai = ProjHubAI(groq_api_key="gsk_...")
#   result = ai.scenario_one(skills, domain)
#   result = ai.scenario_two(idea)
# ============================================================

import json
import os
import logging
import numpy as np

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


# ──────────────────────────────────────────────────────────────
# DATABASE
# ──────────────────────────────────────────────────────────────

PROJECTS_DB = [
    {"id": 1,  "title": "Smart Study Planner",              "description": "An AI system that generates optimized study schedules for students based on workload and learning patterns.",                                      "tags": ["AI", "Backend"]},
    {"id": 2,  "title": "University Chatbot Assistant",     "description": "A conversational AI chatbot that helps students navigate university services and provides academic guidance using NLP.",                            "tags": ["AI", "Backend", "Web"]},
    {"id": 3,  "title": "Student Project Marketplace",      "description": "A web platform where students can showcase, sell, and collaborate on academic projects with peer review.",                                         "tags": ["Web", "Backend", "UI/UX"]},
    {"id": 4,  "title": "Campus Food Delivery App",         "description": "A mobile application for ordering food from campus cafeterias with real-time tracking and payment integration.",                                   "tags": ["Flutter", "Backend", "Mobile"]},
    {"id": 5,  "title": "Academic Performance Predictor",   "description": "A machine learning system that predicts student academic performance based on grades, attendance, and behavior.",                                   "tags": ["AI", "Data Science", "Backend"]},
    {"id": 6,  "title": "Online Exam Platform",             "description": "A secure web-based exam system with anti-cheating, timer controls, and automated grading.",                                                        "tags": ["Web", "Backend", "UI/UX"]},
    {"id": 7,  "title": "Student Health Tracker",           "description": "A Flutter app that tracks student health metrics, sleep patterns, and stress with personalized wellness tips.",                                    "tags": ["Flutter", "Mobile", "AI"]},
    {"id": 8,  "title": "Research Paper Summarizer",        "description": "An NLP tool that automatically summarizes academic research papers and extracts key insights using transformers.",                                  "tags": ["AI", "Data Science", "Web"]},
    {"id": 9,  "title": "Campus Events Manager",            "description": "A platform for managing university events, registrations, notifications, and feedback collection.",                                                "tags": ["Flutter", "Web", "Backend"]},
    {"id": 10, "title": "Plagiarism Detection System",      "description": "A system that detects plagiarism in student assignments using cosine similarity and TF-IDF algorithms.",                                          "tags": ["AI", "Backend", "Web"]},
    {"id": 11, "title": "Job Recommendation for Graduates", "description": "A recommendation engine matching graduates with jobs based on skills, GPA, and project portfolio.",                                               "tags": ["AI", "Data Science", "Backend"]},
    {"id": 12, "title": "Smart Attendance System",          "description": "A face recognition attendance tracking system for classrooms using computer vision and deep learning.",                                            "tags": ["AI", "Backend"]},
    {"id": 13, "title": "Collaborative Code Editor",        "description": "A real-time collaborative code editor for students with syntax highlighting and version control.",                                                 "tags": ["Web", "Backend", "UI/UX"]},
    {"id": 14, "title": "Mental Health Support App",        "description": "A mobile app providing mental health resources, mood tracking, and anonymous peer support for students.",                                          "tags": ["Flutter", "Mobile", "Backend"]},
    {"id": 15, "title": "Library Book Recommendation",      "description": "An AI-powered recommendation system for university library books based on reading history and academic interests.",                                "tags": ["AI", "Data Science", "Web"]},
    {"id": 16, "title": "E-Learning Platform",              "description": "A full-featured online learning platform where instructors upload courses and students enroll, watch videos, and take quizzes.",                   "tags": ["Web", "Backend", "UI/UX"]},
    {"id": 17, "title": "Student Carpool Finder",           "description": "A mobile app that connects university students who commute from the same area to share rides and reduce transport costs.",                         "tags": ["Flutter", "Mobile", "Backend"]},
    {"id": 18, "title": "AI Resume Builder",                "description": "An AI-powered tool that helps students create professional resumes by analyzing their skills, projects, and academic history.",                     "tags": ["AI", "Web", "UI/UX"]},
    {"id": 19, "title": "Campus Lost and Found System",     "description": "A web and mobile platform where students report lost or found items on campus with image upload and location tagging.",                            "tags": ["Flutter", "Web", "Backend"]},
    {"id": 20, "title": "Graduation Project Tracker",       "description": "A management system for tracking graduation project progress, milestones, supervisor feedback, and team collaboration.",                          "tags": ["Web", "Backend", "UI/UX"]},
    {"id": 21, "title": "Timetable Generator",              "description": "An AI system that automatically generates conflict-free class timetables for university departments based on constraints and preferences.",         "tags": ["AI", "Backend", "Web"]},
    {"id": 22, "title": "Student Freelance Portal",         "description": "A platform connecting students who offer freelance services (design, coding, writing) with clients inside and outside the university.",           "tags": ["Web", "Backend", "UI/UX"]},
    {"id": 23, "title": "Smart Parking System",             "description": "An IoT-integrated mobile app that shows real-time campus parking availability and lets students reserve spots in advance.",                        "tags": ["Flutter", "Mobile", "Backend"]},
    {"id": 24, "title": "Course Recommendation Engine",     "description": "A data-driven system that recommends elective courses to students based on their major, GPA, interests, and career goals.",                       "tags": ["AI", "Data Science", "Backend"]},
    {"id": 25, "title": "Peer Tutoring Marketplace",        "description": "A platform where advanced students offer paid or volunteer tutoring sessions to peers, with scheduling and rating features.",                     "tags": ["Web", "Backend", "Flutter"]},
    {"id": 26, "title": "Campus Safety Alert App",          "description": "A mobile application that sends real-time safety alerts, emergency notifications, and incident reports to all campus users.",                     "tags": ["Flutter", "Mobile", "Backend"]},
    {"id": 27, "title": "AI Code Review Assistant",         "description": "An AI tool that automatically reviews student code submissions, detects bugs, suggests improvements, and gives quality scores.",                  "tags": ["AI", "Backend", "Web"]},
    {"id": 28, "title": "University News Aggregator",       "description": "A mobile app that aggregates news, announcements, and updates from all university departments into one personalized feed.",                       "tags": ["Flutter", "Mobile", "Backend"]},
    {"id": 29, "title": "Student Budget Tracker",           "description": "A personal finance mobile app tailored for students to track expenses, set budgets, and get saving tips based on spending patterns.",             "tags": ["Flutter", "Mobile", "Data Science"]},
    {"id": 30, "title": "Scholarship Finder",               "description": "A web platform that matches students with available scholarships based on their academic profile, nationality, and field of study.",              "tags": ["Web", "Backend", "AI"]},
    {"id": 31, "title": "ProjHub",                          "description": "A collaborative platform for university students to create, publish, and showcase academic projects. Features an AI-powered recommendation system that suggests project ideas based on team skills and detects similar existing projects using NLP and vector similarity search.", "tags": ["AI", "Web", "Backend", "UI/UX", "Data Science"]},
]

VALID_TRACKS  = ["AI", "Backend", "Flutter", "UI/UX", "Data Science", "Mobile", "Web"]
VALID_LEVELS  = ["Beginner", "Intermediate", "Advanced"]
VALID_DOMAINS = [
    "Artificial Intelligence & Machine Learning",
    "Business & E-Commerce",
    "Education & E-Learning",
    "Healthcare & Wellness",
    "Social & Community",
    "Productivity & Tools",
    "Entertainment & Media",
    "Finance & Banking",
    "Security & Privacy",
]


# ──────────────────────────────────────────────────────────────
# VECTOR STORE  (FAISS + sentence-transformers)
# ──────────────────────────────────────────────────────────────

class _VectorStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def init(self):
        if self._initialized:
            return
        from sentence_transformers import SentenceTransformer
        import faiss
        self._faiss = faiss
        self.model  = SentenceTransformer("all-MiniLM-L6-v2")
        self._build_index()
        self._initialized = True

    def _build_index(self):
        texts = [
            f"{p['title']}. {p['description']} Tracks: {', '.join(p['tags'])}"
            for p in PROJECTS_DB
        ]
        self.embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        dim         = self.embeddings.shape[1]
        self.index  = self._faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 5, threshold: float = 0.25):
        vec = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold:
                p = PROJECTS_DB[idx].copy()
                p["similarity_score"] = round(float(score), 3)
                results.append(p)
        return sorted(results, key=lambda x: x["similarity_score"], reverse=True)


# ──────────────────────────────────────────────────────────────
# MAIN CLASS — used by backend team
# ──────────────────────────────────────────────────────────────

class ProjHubAI:
    """
    Main AI class. Instantiate once and reuse.

    Usage:
        ai = ProjHubAI(groq_api_key="gsk_...")
        result = ai.scenario_one(skills=[...], domain="Healthcare & Wellness")
        result = ai.scenario_two(idea="AI chatbot for advising")
    """

    def __init__(self, groq_api_key: str):
        from groq import Groq
        self._client = Groq(api_key=groq_api_key)
        self._model  = "llama-3.3-70b-versatile"
        self._store  = _VectorStore()
        self._store.init()

    # ── internal helpers ──────────────────────────────────────

    def _call(self, prompt: str, max_tokens: int = 1000) -> str:
        res = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return res.choices[0].message.content.strip()

    def _parse_json(self, text: str):
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)

    # ── PUBLIC: Scenario 1 ────────────────────────────────────

    def scenario_one(self, skills: list, domain: str = None) -> dict:
        """
        Generate project ideas based on team skills.

        Args:
            skills: list of {"track": str, "level": str}
                    track  ∈ VALID_TRACKS
                    level  ∈ VALID_LEVELS
            domain: optional domain string from VALID_DOMAINS

        Returns dict:
        {
          "domain": str,
          "suggestions": [
            {
              "title": str,
              "description": str,
              "recommended_tracks": [str],
              "tech_stack": {"Frontend": str, "Backend": str, ...},
              "how_it_works": [str],
              "similar_projects": [
                {"id": int, "title": str, "similarity_score": float, "tags": [str]}
              ]
            }
          ]
        }
        """
        # validate
        for s in skills:
            if s.get("track") not in VALID_TRACKS:
                raise ValueError(f"Invalid track: {s.get('track')}. Must be one of {VALID_TRACKS}")
            if s.get("level") not in VALID_LEVELS:
                raise ValueError(f"Invalid level: {s.get('level')}. Must be one of {VALID_LEVELS}")

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
- If a domain is specified, all 3 ideas must belong to that domain
- tech_stack must be general tools based on the tracks
- how_it_works must be 3-4 simple steps explaining the user journey

Respond ONLY with a valid JSON array, no extra text:
[
  {{
    "title": "Project Title",
    "description": "2-3 sentence description of the project and its value.",
    "recommended_tracks": ["Track1", "Track2"],
    "tech_stack": {{
      "Frontend": "e.g. Flutter or React",
      "Backend": "e.g. Firebase or FastAPI",
      "Database": "e.g. Firestore or PostgreSQL",
      "Hosting": "e.g. Firebase Hosting"
    }},
    "how_it_works": [
      "Step 1: ...",
      "Step 2: ...",
      "Step 3: ...",
      "Step 4: ..."
    ]
  }}
]"""

        suggestions = self._parse_json(self._call(prompt))

        # attach similar projects to each suggestion
        for s in suggestions:
            query   = s["title"] + " " + s["description"]
            similar = self._store.search(query, top_k=3, threshold=0.25)
            s["similar_projects"] = [
                {
                    "id":               p["id"],
                    "title":            p["title"],
                    "similarity_score": p["similarity_score"],
                    "tags":             p["tags"],
                }
                for p in similar
            ]

        return {"domain": domain or "Open", "suggestions": suggestions}

    # ── PUBLIC: Scenario 2 ────────────────────────────────────

    def scenario_two(self, idea: str) -> dict:
        """
        Analyze an idea and find similar projects.

        Args:
            idea: free-text project idea (min 10 chars)

        Returns dict:
        {
          "refined_title": str,
          "description": str,
          "detected_tracks": [str],
          "tech_stack": {"Frontend": str, ...},
          "how_it_works": [str],
          "similar_projects": [
            {
              "id": int,
              "title": str,
              "similarity_score": float,
              "explanation": str,
              "tags": [str]
            }
          ]
        }
        """
        if len(idea.strip()) < 10:
            raise ValueError("Idea is too short. Please provide more detail.")

        # Step 1: full idea analysis
        analysis_prompt = f"""You are an expert academic project advisor for university students.

A student has this project idea: "{idea}"

Analyze it and provide a full structured breakdown.

Respond ONLY with a valid JSON object, no extra text:
{{
  "refined_title": "A clear, professional project title",
  "description": "2-3 sentence description of what the project does and its value.",
  "detected_tracks": ["Track1", "Track2"],
  "tech_stack": {{
    "Frontend": "e.g. Flutter or React",
    "Backend": "e.g. Firebase or FastAPI",
    "Database": "e.g. Firestore or PostgreSQL",
    "Hosting": "e.g. Firebase Hosting"
  }},
  "how_it_works": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ...",
    "Step 4: ..."
  ]
}}

Available tracks: {", ".join(VALID_TRACKS)}"""

        analysis = self._parse_json(self._call(analysis_prompt, max_tokens=800))

        # Step 2: find similar projects
        search_query = analysis.get("refined_title", idea) + " " + analysis.get("description", idea)
        similar      = self._store.search(search_query, top_k=5, threshold=0.25)

        # Step 3: generate explanations
        explanations = []
        if similar:
            projects_text = "\n".join(
                f'  {i+1}. "{p["title"]}": {p["description"]}'
                for i, p in enumerate(similar)
            )
            exp_prompt = f"""A student has this idea: "{idea}"

Similar projects:
{projects_text}

Write ONE specific sentence per project explaining the concrete shared concept
that makes it similar to the student's idea. Be precise, not generic.

Respond ONLY with a JSON array of strings:
["explanation 1", "explanation 2", ...]"""
            try:
                explanations = self._parse_json(self._call(exp_prompt, max_tokens=400))
            except Exception:
                explanations = ["Similar project found in the database."] * len(similar)

        analysis["similar_projects"] = [
            {
                "id":               p["id"],
                "title":            p["title"],
                "similarity_score": p["similarity_score"],
                "explanation":      explanations[i] if i < len(explanations) else "",
                "tags":             p["tags"],
            }
            for i, p in enumerate(similar)
        ]

        return analysis

    # ── PUBLIC: helpers ───────────────────────────────────────

    @staticmethod
    def get_valid_tracks()  -> list: return VALID_TRACKS
    @staticmethod
    def get_valid_levels()  -> list: return VALID_LEVELS
    @staticmethod
    def get_valid_domains() -> list: return VALID_DOMAINS
    @staticmethod
    def get_projects_db()   -> list: return PROJECTS_DB