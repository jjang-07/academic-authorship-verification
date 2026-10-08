# Authorship Verification V2 — Technical Project Specification
## For: Cursor AI Development Assistant

---

## Project Background & Goal

This is the second version of a research project on **Authorship Verification (AV) for academic integrity** — detecting whether a student's submitted essay was actually written by them, or generated/heavily assisted by AI.

**V1 Summary (already completed):**
- Built a traditional ML pipeline using hybrid feature vectors: token-level baselines (TF-IDF) + handcrafted stylometric features
- Handcrafted features included: CEFR vocabulary distributions, sentence length & variance, sentence structure variation, perplexity, readability metrics (Flesch-Kincaid etc.), adverbial placement patterns, POS tag patterns
- Trained 5 binary classifiers (Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting) on same-author vs. different-author text pairs
- Best model: Gradient Boosting (Full Feature Set) — Overall score 0.707 on Reuters-50-50
- Outperformed token-only baseline (Wan 2024) by ~25%
- Used Shapley (SHAP) values for interpretability

**V1 Known Limitations:**
- Small labeled dataset (especially the real-world high school essay corpus)
- Topic shifts between documents hurt performance (a student writing about history vs. tech looks like a different author)
- Single-model decision — no cross-checking or adversarial verification
- Handcrafted features alone hit a ceiling

**V2 Goal:**
Build a **Multi-Agent LLM Debate Framework** for authorship verification that:
1. Uses the validated handcrafted features from V1 as structured evidence for LLM reasoning
2. Employs multiple LLM agents that debate a verdict adversarially
3. Uses RAG-style retrieval to ground agents in similar verified writings, solving the small-dataset problem
4. Remains interpretable for educators (SHAP + natural language reasoning traces)
5. Is benchmarked against V1, Huang et al. (2024) LIP, and PAN 2024/2025 baselines

---

## Repository Structure (Recommended)

```
av-v2/
├── data/
│   ├── raw/                  # Raw corpora (Reuters-50-50, PAN, essays)
│   ├── processed/            # Preprocessed text pairs
│   └── vector_store/         # FAISS or Chroma index for RAG retrieval
├── features/
│   ├── extractor.py          # Master feature extraction pipeline
│   ├── handcrafted.py        # All handcrafted stylometric features from V1
│   ├── token_baselines.py    # TF-IDF and token-level features
│   └── evidence_packet.py    # Formats feature vectors into structured prompt evidence
├── retrieval/
│   ├── indexer.py            # Builds and saves the vector store
│   └── retriever.py          # Retrieves top-k similar verified writings at inference
├── agents/
│   ├── base_agent.py         # Base LLM agent class (handles prompt construction, API calls)
│   ├── analyst_agent.py      # Agent A: Stylometric Analyst
│   ├── skeptic_agent.py      # Agent B: Challenger/Skeptic
│   ├── judge_agent.py        # Agent C: Integrator/Judge
│   └── prompts/
│       ├── analyst_prompt.txt
│       ├── skeptic_prompt.txt
│       └── judge_prompt.txt
├── debate/
│   ├── orchestrator.py       # Controls debate rounds, passes messages between agents
│   └── arbitration.py        # Final verdict logic (confidence-weighted or judge LLM)
├── interpretability/
│   ├── shap_analysis.py      # SHAP values on feature vectors (carried over from V1)
│   └── trace_parser.py       # Parses and formats agent reasoning traces for educators
├── evaluation/
│   ├── metrics.py            # AUC, c@1, F0.5, F1, Brier score
│   ├── ablation.py           # Runs ablation study configurations
│   └── topic_stratified.py   # Evaluates performance split by topic overlap
├── experiments/
│   └── run_experiment.py     # Master experiment runner
├── config.py                 # API keys, model names, hyperparameters
└── requirements.txt
```

---

## Component 1: Data & Preprocessing

### Datasets to Support
| Dataset | Purpose | Format |
|---|---|---|
| Reuters-50-50 | V1 benchmark, continuity baseline | 5000 docs, 100 authors |
| PAN 2024 AV Benchmark | SOTA comparison, AI-generated text framing | Text pairs + labels |
| PAN 2023 AV Benchmark | Cross-year robustness check | Text pairs + labels |
| High School Essay Corpus | Real-world unique contribution | ~Small, collected manually |
| Blog Authorship Corpus | Direct comparison to Huang et al. (2024) | Kaggle available |

### Preprocessing Steps (carried over from V1, extended)
- Normalize to lowercase
- Tokenization (NLTK `word_tokenize`)
- POS tagging (spaCy `en_core_web_sm` or `en_core_web_trf`)
- Dependency parsing (spaCy)
- Sentence segmentation
- Remove noise (URLs, special chars, formatting artifacts)
- Generate balanced text pairs:
  - **Positive pairs** (same-author): randomly sample 2 docs from same author
  - **Negative pairs** (different-author): randomly sample 1 doc each from 2 different authors
  - Ensure 50/50 class balance

### Key File: `data/preprocessor.py`
Should output a standardized `TextPair` object:
```python
@dataclass
class TextPair:
    text_a: str           # Verified author text
    text_b: str           # Unknown/test text
    label: int            # 1 = same author, 0 = different author
    author_id: str        # For stratified evaluation
    topic_a: str          # For topic-stratified analysis
    topic_b: str
    source_dataset: str   # Which corpus this came from
```

---

## Component 2: Feature Extraction Module

This is the backbone carried over from V1, now extended and reformatted as structured evidence for LLM prompts.

### 2.1 Handcrafted Stylometric Features (`features/handcrafted.py`)

Implement all of the following as functions that take a raw string and return a float or dict:

**Vocabulary Features:**
- CEFR level distribution (A1/A2/B1/B2/C1/C2 word ratios) — use a CEFR word list lookup
- Type-Token Ratio (TTR) and corrected TTR (CTTR)
- Average word length
- Hapax legomena ratio (words appearing exactly once / total words)
- Lexical density (content words / total words)

**Sentence-Level Features:**
- Mean sentence length (in tokens)
- Sentence length variance / standard deviation
- Sentence length distribution (histogram bins: very short <8, short 8-15, medium 15-25, long >25)
- Sentence structure variation score (ratio of unique syntactic templates)
- Subordinate clause frequency (using dependency parse)

**Syntactic/POS Features:**
- POS tag unigram distribution (ratio of NOUN, VERB, ADJ, ADV, CONJ per doc)
- POS bigram patterns (e.g., ADJ-NOUN frequency)
- Adverbial placement patterns (sentence-initial vs. mid vs. final adverb ratios)
- Passive voice frequency (detect via dependency parse `nsubjpass`)

**Discourse/Readability Features:**
- Flesch-Kincaid Grade Level
- Gunning Fog Index
- Coleman-Liau Index
- Perplexity (computed using a base LM, e.g., GPT-2 via HuggingFace)
- Discourse connective frequency (however, therefore, furthermore, etc.)

**Punctuation/Style Features:**
- Comma frequency per sentence
- Semicolon and colon usage rate
- Exclamation/question mark ratio
- Average paragraph length

### 2.2 Token-Level Baselines (`features/token_baselines.py`)
- TF-IDF (character 3-5 grams, word unigrams — `sklearn TfidfVectorizer`)
- Function word frequency vector (top 150 most common English function words)

### 2.3 Evidence Packet Formatter (`features/evidence_packet.py`)

This is the critical bridge between V1 features and V2 LLM agents. Takes two feature vectors (one per text) and formats them into a human-readable, structured block for injection into agent prompts.

```python
def format_evidence_packet(features_a: dict, features_b: dict) -> str:
    """
    Returns a formatted string like:
    
    === STYLOMETRIC EVIDENCE PACKET ===
    
    VOCABULARY COMPLEXITY:
    - Text A CEFR distribution: A1: 12%, A2: 28%, B1: 35%, B2: 18%, C1: 6%, C2: 1%
    - Text B CEFR distribution: A1: 8%, A2: 22%, B1: 30%, B2: 25%, C1: 12%, C2: 3%
    - Interpretation: Text B uses notably more advanced vocabulary (C1+C2: 15% vs 7%)
    
    SENTENCE STRUCTURE:
    - Text A mean sentence length: 14.2 tokens (std: 4.1)
    - Text B mean sentence length: 22.7 tokens (std: 9.3)
    - Interpretation: Text B uses significantly longer, more variable sentences
    
    [... all features formatted similarly ...]
    
    SIMILARITY DELTAS (|A - B| per feature):
    - sentence_length_mean_delta: 8.5 (HIGH)
    - cefr_c1c2_ratio_delta: 0.08 (MODERATE)
    - passive_voice_freq_delta: 0.02 (LOW)
    ...
    """
```

The "Interpretation" and delta annotations are important — they give LLM agents pre-digested signals rather than raw numbers.

---

## Component 3: RAG-Style Retrieval

### Purpose
At inference time, retrieve the top-k most stylistically similar *verified* writing samples to each input text. These serve as in-context anchors — giving agents concrete comparison examples even when labeled data is sparse.

### 3.1 Vector Store (`retrieval/indexer.py`)
- Use **FAISS** (fast, local, no external service needed) or **ChromaDB** (easier API)
- Index all verified-author texts from your corpus
- Embedding: **do not use raw text embeddings alone** — embed using your feature vectors (or a concatenation of feature vector + lightweight sentence embedding like `sentence-transformers/all-MiniLM-L6-v2`) to ensure retrieval is stylistically driven, not topic-driven
- Each indexed entry stores: `{author_id, raw_text, feature_vector, source_dataset}`

```python
def build_index(verified_texts: List[TextEntry]) -> FAISSIndex:
    # Extract feature vectors for all verified texts
    # Optionally concatenate with sentence embeddings
    # Build and save FAISS index
    pass

def retrieve_similar(query_text: str, k: int = 3) -> List[TextEntry]:
    # Extract features from query_text
    # Query FAISS index
    # Return top-k most stylistically similar verified texts
    pass
```

### 3.2 Usage in Pipeline
Before agents are called, retrieve top-3 similar writings for each input text. Include a brief summary of these retrievals in the evidence packet so agents can reason like: *"Text B resembles writing samples from Author X and Y in vocabulary complexity, but differs in sentence rhythm."*

---

## Component 4: Multi-Agent Debate Framework

This is the core novel contribution of V2.

### 4.1 Agent Architecture Overview

```
TextPair + EvidencePacket + RetrievedSamples
          |
    ┌─────▼──────┐
    │  Agent A   │  ← Stylometric Analyst
    │  (Analyst) │    Initial verdict + reasoning
    └─────┬──────┘
          │ verdict_a + reasoning_a
    ┌─────▼──────┐
    │  Agent B   │  ← Skeptic/Challenger
    │  (Skeptic) │    Challenges verdict_a, finds weaknesses
    └─────┬──────┘
          │ challenge + updated_confidence
    ┌─────▼──────┐
    │  Agent C   │  ← Judge/Integrator
    │   (Judge)  │    Weighs debate, produces final verdict
    └─────┬──────┘
          │
    Final Verdict + Confidence + Explanation
```

### 4.2 Base Agent (`agents/base_agent.py`)

```python
class BaseAgent:
    def __init__(self, model: str, system_prompt: str, temperature: float = 0.2):
        self.model = model          # e.g., "gpt-4o", "claude-3-5-sonnet", "mistral-large"
        self.system_prompt = system_prompt
        self.temperature = temperature  # Low temp for consistency

    def call(self, user_message: str) -> AgentResponse:
        # Calls LLM API (OpenAI, Anthropic, or together.ai for open-source)
        # Returns structured AgentResponse object
        pass

@dataclass
class AgentResponse:
    verdict: str            # "SAME_AUTHOR" or "DIFFERENT_AUTHOR"
    confidence: float       # 0.0 to 1.0
    reasoning: str          # Natural language explanation
    key_features_cited: List[str]   # Which features drove the decision
    raw_response: str       # Full LLM output for logging
```

### 4.3 Agent A — Stylometric Analyst (`agents/analyst_agent.py`)

**Role:** Make the initial AV verdict grounded strictly in the evidence packet and retrieved samples.

**System prompt key instructions:**
- You are an expert forensic linguist and stylometric analyst
- You will receive two texts and a structured evidence packet of their stylometric features
- You will also receive 3 retrieved writing samples that are stylistically similar to each text
- Make a verdict: SAME_AUTHOR or DIFFERENT_AUTHOR
- Justify your verdict by explicitly citing specific features from the evidence packet
- Provide a confidence score (0.0–1.0)
- Do NOT be swayed by topic content — focus only on style

**Output format (enforce via prompt):**
```
VERDICT: [SAME_AUTHOR / DIFFERENT_AUTHOR]
CONFIDENCE: [0.0-1.0]
KEY FEATURES: [list the 3-5 most decisive features]
REASONING: [2-3 paragraph natural language explanation]
```

### 4.4 Agent B — Skeptic/Challenger (`agents/skeptic_agent.py`)

**Role:** Receive Agent A's verdict and actively try to find flaws in it.

**System prompt key instructions:**
- You are a critical peer reviewer of forensic linguistic analyses
- You will receive Agent A's verdict and reasoning, plus the same evidence packet
- Your job is to challenge Agent A's conclusion — find alternative explanations, confounding factors, features that were overlooked or misinterpreted
- Specifically consider: topic overlap effects, genre differences, text length effects, and any features where the delta is actually LOW (suggesting similarity)
- If you agree with Agent A after analysis, state so and explain why the verdict is robust
- Provide your own confidence score

**Output format:**
```
STANCE: [AGREE / DISAGREE / PARTIALLY_DISAGREE]
CONFIDENCE: [0.0-1.0]
CHALLENGES: [list specific challenges to Agent A's reasoning]
OVERLOOKED_EVIDENCE: [features Agent A underweighted]
REVISED_REASONING: [your own 2-3 paragraph analysis]
```

### 4.5 Agent C — Judge (`agents/judge_agent.py`)

**Role:** Receive the full debate transcript and produce a final calibrated verdict.

**System prompt key instructions:**
- You are a senior forensic linguist making a final authorship determination
- You have the original evidence packet, Agent A's analysis, and Agent B's challenge
- Weigh the arguments carefully — do not simply default to Agent A
- Produce a final verdict with a well-calibrated confidence score
- Summarize the key reasons for your decision in plain language suitable for a teacher/educator to read

**Output format:**
```
FINAL_VERDICT: [SAME_AUTHOR / DIFFERENT_AUTHOR]
FINAL_CONFIDENCE: [0.0-1.0]
DECISIVE_FACTORS: [top 3 features/arguments that determined outcome]
EDUCATOR_SUMMARY: [1 paragraph plain-English explanation for teachers]
AGENT_AGREEMENT: [FULL / PARTIAL / NONE]
```

### 4.6 Debate Orchestrator (`debate/orchestrator.py`)

Controls the flow and number of rounds.

```python
class DebateOrchestrator:
    def __init__(self, analyst, skeptic, judge, max_rounds=1):
        self.analyst = analyst
        self.skeptic = skeptic
        self.judge = judge
        self.max_rounds = max_rounds  # Start with 1 round, ablate with 2

    def run(self, text_pair: TextPair, evidence_packet: str, retrieved_samples: List) -> DebateResult:
        # Round 1: Analyst makes initial verdict
        analyst_response = self.analyst.call(
            build_analyst_prompt(text_pair, evidence_packet, retrieved_samples)
        )
        
        # Round 1: Skeptic challenges
        skeptic_response = self.skeptic.call(
            build_skeptic_prompt(text_pair, evidence_packet, analyst_response)
        )
        
        # Optional Round 2: Analyst can respond to challenge
        # (keep this optional — adds cost, test if it improves performance)
        
        # Judge integrates and decides
        judge_response = self.judge.call(
            build_judge_prompt(text_pair, evidence_packet, analyst_response, skeptic_response)
        )
        
        return DebateResult(
            analyst=analyst_response,
            skeptic=skeptic_response,
            judge=judge_response,
            final_verdict=judge_response.verdict,
            final_confidence=judge_response.confidence
        )
```

### 4.7 Model Configuration (Heterogeneous Agents)
Use different models for each agent to prevent echo-chamber effects:
- **Agent A (Analyst):** `gpt-4o` or `claude-sonnet` — strong reasoning, good at structured analysis
- **Agent B (Skeptic):** `mistral-large` or `gpt-4o-mini` — cost-effective, good at critique
- **Agent C (Judge):** `gpt-4o` or `claude-sonnet` — needs strongest calibration

If budget is a concern, all three can use `gpt-4o-mini` for development and switch to stronger models for final evaluation runs.

---

## Component 5: Interpretability Layer

### 5.1 SHAP Analysis (`interpretability/shap_analysis.py`)
- Carry over V1 SHAP pipeline: train a lightweight Gradient Boosting model on feature vectors (same as V1) alongside the LLM pipeline
- Use SHAP to show feature importance at the instance level
- This gives a quantitative, reproducible interpretability layer independent of LLM reasoning

### 5.2 Reasoning Trace Parser (`interpretability/trace_parser.py`)
- Parse `judge_response.educator_summary` and `analyst_response.key_features_cited`
- Format into a clean educator-facing report:

```
═══════════════════════════════════════════
AUTHORSHIP VERIFICATION REPORT
═══════════════════════════════════════════
Verdict: DIFFERENT AUTHOR (Confidence: 82%)

Most Important Signals:
  1. Vocabulary complexity jumped significantly (C1+ words: 7% → 15%)
  2. Sentence length increased by 8.5 tokens on average
  3. Passive voice usage doubled

Plain English Summary:
  "The submitted essay uses substantially more advanced vocabulary
   and longer, more complex sentences than the student's verified
   prior writing. The stylometric patterns are inconsistent with
   the same author..."

Note: This is a flagging tool for educator review, not a final judgment.
═══════════════════════════════════════════
```

---

## Component 6: Evaluation Framework

### 6.1 Metrics (`evaluation/metrics.py`)
Implement all five V1 metrics for direct comparability:
- **AUC** — `sklearn.metrics.roc_auc_score`
- **c@1** — AV-specific metric that rewards abstaining on uncertain cases: `c@1 = (1/n)(nc + nu * nc/n)` where nc = correct, nu = unanswered
- **F0.5** — Precision-weighted F score (`sklearn.metrics.fbeta_score(beta=0.5)`)
- **F1** — Standard F1
- **Brier Score** — Calibration of confidence scores (`sklearn.metrics.brier_score_loss`)
- **Overall** — Mean of all five (same as V1)

### 6.2 Ablation Study (`evaluation/ablation.py`)
Run four configurations to isolate each component's contribution:

| Config | Features | Retrieval | Multi-Agent | Expected |
|---|---|---|---|---|
| V1 Baseline | ✓ | ✗ | ✗ (GB classifier) | ~0.707 |
| LIP-Style | ✓ | ✗ | ✗ (single LLM) | Comparable to Huang et al. |
| Naive MAD | ✗ | ✗ | ✓ (no features) | Tests if debate alone helps |
| Full V2 | ✓ | ✓ | ✓ | Best expected |

### 6.3 Topic-Stratified Evaluation (`evaluation/topic_stratified.py`)
- Tag each text pair by topic (history, science, English, tech, etc.)
- Compute metrics separately for: same-topic pairs, cross-topic pairs
- V1's known weakness was cross-topic performance — this directly measures improvement
- Report delta between V2 and V1 on cross-topic pairs specifically

### 6.4 Baselines to Beat
- V1 Gradient Boosting Full Set: Overall 0.707
- Wan (2024) Logistic Regression: Overall 0.562
- PAN 2024 TF-IDF SVM baseline
- PAN 2024 Binoculars (LLM perplexity-based)
- Huang et al. (2024) LIP — replicate their setup on Blog Authorship Corpus for direct comparison

---

## Implementation Order (Recommended)

1. **Set up data pipeline first** — preprocessor, TextPair dataclass, load all datasets
2. **Port V1 feature extraction** — refactor V1 code into `features/handcrafted.py` cleanly
3. **Build evidence packet formatter** — this is the critical bridge, test it looks readable
4. **Build and test single analyst agent** — get one LLM call working end-to-end before adding debate
5. **Add RAG retrieval** — build FAISS index, verify retrieval is stylistically sensible
6. **Add skeptic and judge agents** — complete the debate loop
7. **Build evaluation framework** — metrics, ablation configs, topic-stratified eval
8. **Run ablation experiments** — confirm each component adds value
9. **Interpretability layer** — SHAP + educator report formatter
10. **Final experiments on all datasets** — PAN, Reuters, Blog, HS essays

---

## Key Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| Feature vectors drive retrieval (not raw embeddings) | Prevents topic-biased retrieval — ensures RAG is stylistically grounded |
| Heterogeneous agent models | Prevents echo chamber, adds diversity of reasoning paths |
| Evidence packet is pre-interpreted (with deltas + "Interpretation:" labels) | LLMs reason better with digested signals than raw numbers |
| Max 1-2 debate rounds | MAD literature shows diminishing returns; NeurIPS 2025 shows gains flatten quickly |
| Keep V1 SHAP alongside LLM pipeline | Provides quantitative, reproducible interpretability independent of LLM outputs |
| c@1 metric | Standard in PAN AV benchmarks — allows direct comparison with published results |
| Topic-stratified evaluation | Directly measures V1's known limitation; core evidence of V2 improvement |

---

## Dependencies

```
# Core NLP
nltk
spacy  # + python -m spacy download en_core_web_trf
transformers
sentence-transformers

# Feature extraction
scikit-learn
textstat          # Readability metrics
cefr-cefrpy       # CEFR vocabulary lookup (or use custom wordlist)

# LLM APIs
openai
anthropic

# Retrieval
faiss-cpu         # or chromadb

# Interpretability
shap

# Evaluation
numpy
pandas
scipy

# Utilities
tqdm
pydantic          # For dataclass validation
python-dotenv     # API key management
```

---

## Notes for Cursor

- All LLM API keys should live in `.env` and be loaded via `python-dotenv` — never hardcoded
- All experiment results should be logged to `experiments/results/` as JSON for reproducibility
- The `TextPair` dataclass is the universal data contract — all components should accept and return it
- When in doubt about feature implementation details, refer to V1 paper methodology: handcrafted features use spaCy for POS/dependency, NLTK for tokenization, textstat for readability
- The `evidence_packet.py` formatter is the most important file to get right — the quality of LLM reasoning is highly dependent on how clearly features are presented
- For development/testing, use `gpt-4o-mini` for all agents to reduce API cost; swap in stronger models for final evaluation runs
