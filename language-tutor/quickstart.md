# Quickstart: Run & Host Language Tutor with Docker

## Overview
This project provides:
- **ML pipeline (optional for demo)**: `python -m src.preprocess`, `python -m src.classifier`, `python -m src.evaluate`
- **Duolingo-like tutor UI (Gradio)**: `python -m app.app`
- **Groq-only backend** (no local transformers required)

This guide focuses on **hosting the UI with Docker**.

---

## 1) Prerequisites
1. Install **Docker Desktop** (Windows).
2. Get your **Groq API key**.

---

## 2) Create a `.env` file (required for Groq)
From the repo root (where `language-tutor/docker-compose.yml` exists):

Create:
- `language-tutor/.env` (recommended) or set env vars in your shell.

Example `language-tutor/.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
PORT=7860
```

> If `GROQ_API_KEY` is missing, the UI will start but tutor responses will fail.

---

## 3) Build and run with Docker Compose
From the directory that contains `docker-compose.yml`:

```bash
cd "e:/SRH University/Artificial Intelligence/language-tutor/language-tutor"
docker compose up --build
```

Docker will build the image using `language-tutor/Dockerfile` and start the Gradio server.

---

## 4) Open the app
After startup, open:
- **http://localhost:7860**

---

## 5) Verify container logs
In another terminal:

```bash
docker compose logs -f
```

---

## 6) Common issues
### A) “tokenizers / transformers build fails”
This project is intended to be **Groq-only** at runtime and uses **scikit-learn** for ML.
If you still see tokenizers/transformers compilation errors, ensure you are:
- running Docker build from `language-tutor/language-tutor`
- using the provided `language-tutor/requirements.txt`

### B) Port already in use
Change `PORT` in `.env` (and optionally `docker-compose.yml`).

---

## 7) (Optional) Run ML scripts locally for report figures
From `language-tutor/language-tutor`:

```bash
python -m src.preprocess
python -m src.classifier
python -m src.evaluate
```

Then use generated files:
- `outputs/figures/*`
- `outputs/results/metrics.json`

---

## Docker file references
- `language-tutor/language-tutor/Dockerfile`
- `language-tutor/language-tutor/docker-compose.yml`

End of quickstart.
