# Language Tutor (Duolingo-like) — ML + Fine-tuned LLM

This project delivers a **language-learning tutor** with:

1) A **Duolingo-like UI** (Gradio) that chats in multiple languages using **Groq** (and optionally a local fine-tuned Llama).
2) A required **machine-learning pipeline** (dataset → preprocessing → feature engineering → train two models → evaluation + tuning → visualizations).

## Key features
- Train **two classic ML models** (from the required list) for a language-learning task.
- Evaluate using:
  - cross-validation
  - confusion matrix
  - precision / recall / F1
  - ROC curve + AUC (when applicable)
  - GridSearchCV tuning
- Provide a tutor chatbot that:
  - gives grammar/vocabulary feedback
  - uses Groq real-time responses
  - falls back gracefully if local model artifacts are missing

## Folder structure
- `src/` : preprocessing, training, evaluation, LLM inference
- `data/raw/` and `data/processed/` : datasets and processed features
- `outputs/figures/` and `outputs/results/` : plots and metrics
- `app/app.py` : Gradio UI

## Setup
### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Environment variables
Create a `.env` file (or set env vars) with:
- `GROQ_API_KEY`
- `HF_API_TOKEN` (optional; only needed if downloading models/tokenizers)

**Do not hardcode keys into code.**

## Run (recommended)
### A) Train classic ML models
```bash
python -m src.preprocess
python -m src.classifier
python -m src.evaluate
```

### B) Start the tutor UI
```bash
python -m app.app
```

## Notes about the assignment rubric
- The ML portion uses a language-related dataset and produces evaluation artifacts in `outputs/`.
- The chatbot portion uses the same theme (grammar/vocabulary coaching) but uses Groq for reliable inference.


