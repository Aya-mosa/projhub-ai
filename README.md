# ProjHub AI

AI microservice powering ProjHub  a graduation-project
management platform. This service is the intelligence layer only; the full
platform also includes a Flutter frontend and a .NET backend (not part of
this repo).

## What it does

- 🎯 **Project idea suggestions** — generates 3 tailored graduation project
  ideas based on a team's skills and tracks, using Groq's Llama 3.3 70B
- 🔍 **Idea analysis** — takes a rough project idea and refines it into a
  clear title, tech stack, and step-by-step plan
- 📊 **Skill-gap analysis** — compares a team's skills against real projects
  in the platform and flags what's missing
- 🔎 **Hybrid search** — combines BM25 keyword search with semantic
  embeddings (FAISS + sentence-transformers) to find similar projects
- 📈 **Stats endpoints** — skill/track distribution ready for charts on the
  frontend

## Tech stack

- **FastAPI** — web framework
- **Groq (Llama 3.3 70B)** — LLM for suggestions & analysis
- **sentence-transformers + FAISS** — semantic search
- **rank-bm25** — keyword search
- Deployed on Hugging Face Spaces

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|--------------|
| POST | `/api/ai/suggest` | Suggest project ideas from team skills |
| POST | `/api/ai/analyze` | Analyze and refine a project idea |
| POST | `/api/ai/skill-gap` | Skill gap analysis vs. real projects |
| GET | `/api/ai/stats/skills` | Skill distribution across all projects |
| POST | `/api/ai/reload` | Reload the project index from the API |
| GET | `/api/ai/health` | Health check |

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn main:app --reload
```

## Notes

- Includes a fallback mechanism: if the Groq API fails, the service still
  returns a usable generic response instead of erroring out.
- This repo mirrors the [Hugging Face Space](https://huggingface.co/spaces/aya3330/Projhub_ai) it's deployed on.
