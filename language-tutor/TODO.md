# language-tutor TODO

## ML pipeline (scikit-learn)
- [x] Remove HF/transformers dependencies from `language-tutor/language-tutor/requirements.txt` to avoid `tokenizers` build failures.
- [x] Implement `language-tutor/language-tutor/src/preprocess.py` (synthetic multilingual learner dataset, TF-IDF friendly CSV).
- [x] Implement `language-tutor/language-tutor/src/classifier.py` (train Logistic Regression + SVM pipelines and save to `models/`).
- [x] Implement `language-tutor/language-tutor/src/evaluate.py` (metrics, confusion matrix plots, CV accuracy + small grid search).

## Groq-only chatbot
- [x] Implement `language-tutor/language-tutor/src/finetune.py` as Groq-only `TutorLLM` wrapper (no local transformers).
- [ ] Wire/verify `language-tutor/language-tutor/app/app.py` uses `TutorLLM` correctly.

## Deliverables
- [ ] Generate `outputs/figures/*` and `outputs/results/metrics.json` by running the ML scripts.
- [ ] Create PowerPoint + demo script/steps based on outputs.

