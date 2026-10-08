# Authorship Verification for Academic Integrity

Given two pieces of writing, did the same person write both?

This project tries to answer that question so educators can spot a submitted essay that a student may not have written themselves (for example, ghost-written or heavily AI-assisted). It was built in two stages:

| Version | Approach | Status |
|---|---|---|
| **V1** | Classical ML: handcrafted stylometric features + 5 classifiers, explained with SHAP | Complete |
| **V2** | Multi-agent LLM debate: an **Analyst**, a **Skeptic** and a **Judge** argue over authorship, grounded in V1's features and retrieved writing samples | Complete (research prototype) |

> **Timeline:** This work was done between **May 2025 and March 2026**. It was developed locally and pushed to GitHub in one commit at the end, so the commit history doesn't show the full development timeline.

---

## How it works

### V1: Stylometric machine learning

1. **Features** ([features.py](features.py), [features/handcrafted.py](features/handcrafted.py)): each text is converted into a style vector:
   - CEFR vocabulary-level distribution (A1–C2)
   - Sentence length, variance and structure variation
   - Part-of-speech tag patterns and adverb placement (spaCy, NLTK)
   - Readability scores (textstat, textcomplexity)
   - GPT-2 perplexity (Hugging Face Transformers, PyTorch)
   - TF-IDF token baseline
2. **Pairs**: two texts become one vector `|features_A − features_B|`.
3. **Classifiers** (scikit-learn): Logistic Regression, Random Forest, SVM, KNN and Gradient Boosting, evaluated with 5-fold cross-validation.
4. **Interpretability** ([interpretability/shap_analysis.py](interpretability/shap_analysis.py)): SHAP shows which features drove each prediction.

### V2: Multi-agent LLM debate

```
 Text A, Text B
      │
      ├─► Handcrafted features ──► "evidence packet" (features/evidence_packet.py)
      ├─► FAISS retrieval of similar verified writing (retrieval/)
      ▼
 ┌──────────┐   verdict + reasoning   ┌──────────┐   challenges   ┌──────────┐
 │ Analyst  │ ──────────────────────► │ Skeptic  │ ─────────────► │  Judge   │ ──► SAME / DIFFERENT
 └──────────┘                         └──────────┘                │          │     + confidence
                                                                  └──────────┘     + educator summary
```

- **Analyst** ([agents/analyst_agent.py](agents/analyst_agent.py)): makes an initial call from the text and the evidence packet.
- **Skeptic** ([agents/skeptic_agent.py](agents/skeptic_agent.py)): looks for weaknesses and alternative explanations, such as a topic shift being mistaken for a different author.
- **Judge** ([agents/judge_agent.py](agents/judge_agent.py)): weighs both sides and returns a final verdict, a confidence score and a plain-language summary for educators.
- **Orchestrator** ([debate/orchestrator.py](debate/orchestrator.py)): runs 1–2 debate rounds.
- `*_textonly` variants: close-reading prompts that work only from the raw text (an ICL variant is also included).

The models are configurable: OpenAI (GPT-4o / GPT-5.4 / o4-mini) and Anthropic Claude (used for the Skeptic in the text-only setup).

---

## Results

Reuters-50-50, 100 balanced pairs per run. *Overall* is the mean of the PAN shared-task metrics (AUC, c@1, F0.5u, F1, Brier).

| System | Overall | AUC |
|---|---|---|
| Token-only baseline (Wan, 2024) | ~0.57 | – |
| **V1**: Gradient Boosting, full features | 0.707 | – |
| **V2**: Analyst agent only (seed 123) | **0.782** | **0.812** |
| V2: Analyst agent only (3-seed mean) | 0.743 | 0.757 |
| V2: Full Analyst → Skeptic → Judge debate (seed 123) | 0.723 | 0.747 |

**Key findings**
- V1's handcrafted features beat the token-only baseline by about 25%.
- An LLM agent reasoning over those features beat V1 (up to 0.782 overall).
- The adversarial debate **did not** improve on the Analyst alone. Across 3 seeds the Judge overturned more correct verdicts than incorrect ones. Figures are in [experiments/results/figures/](experiments/results/figures/) and the full breakdown is in [experiments/results/full_comparison.txt](experiments/results/full_comparison.txt).

---

## Repository layout

```
├── agents/            LLM agents (Analyst, Skeptic, Judge) + prompts/
├── debate/            Debate orchestrator
├── features/          V2 handcrafted features + evidence-packet formatting
├── retrieval/         FAISS index builder and retriever
├── evaluation/        PAN metrics, ablation configs, topic-stratified eval
├── interpretability/  SHAP analysis and reasoning-trace parser
├── experiments/       Experiment runners, result inspection, figure generation
├── web/               Flask demo app ("AI INK")
├── data/              Dataset loaders (raw data is NOT included; see below)
├── tests/             Unit tests for the analyst and debate
├── config.py          All settings, read from .env
│
├── main.py            V1 entry point
├── classifier.py, features.py, preprocessing.py,
│   data_processing.py, postprocessing.py,
│   pipeline_model_setup.py, utills.py      V1 pipeline modules
└── AV_V2_Project_Spec.md                    Full V2 design spec
```

---

## Getting started

### 1. Install

Requires Python 3.11.

```bash
git clone https://github.com/jjang-07/academic-authorship-verification.git
cd academic-authorship-verification
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Open `.env` and add your `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. You can also change which model each agent uses. **Never commit `.env`.**

### 3. Get the data

The datasets are not in this repo. Download them and place them under `data/raw/`:

- **Reuters-50-50 (C50)**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/217/reuter+50+50) → `data/raw/reuter+50+50/`
- **CEFR word list**: already included at `data/raw/ENGLISH_CEFR_WORDS.csv`

The student-essay corpus used in some experiments is private and is not distributed.

### 4. Run

```bash
# V1 classical pipeline
python main.py

# V2 main experiment (all configs). Try --dry-run first to avoid API costs.
python experiments/run_experiment.py --dry-run --max-pairs 3
python experiments/run_experiment.py --configs FULL_V2_TEXTONLY --max-pairs 100 --seed 123

# Inspect a run's per-pair debate transcripts
python experiments/inspect_results.py experiments/results/<details>.jsonl --wrong

# Regenerate figures
python experiments/generate_visualizations.py

# Web demo → http://localhost:5050
python web/app.py

# Tests
python -m pytest tests/
```

> ⚠️ A full 100-pair LLM run takes about 5 hours (there is a delay between API calls) and costs real API credits.

---

## Tech stack

Python · scikit-learn · spaCy · NLTK · PyTorch · Hugging Face Transformers (GPT-2) · textstat · FAISS · SHAP · OpenAI API · Anthropic API · Flask · pandas / NumPy · matplotlib / seaborn
