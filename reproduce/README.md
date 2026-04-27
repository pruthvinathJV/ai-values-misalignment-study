# When AI Speaks, Whose Values Does It Express?
### A Cross-Cultural Audit of Individualism–Collectivism Bias in Large Language Models

**Pruthvinath Jeripity Venkata** — Independent Researcher · jvpnath@gmail.com

---

## Overview

This repository contains the full replication package for the paper. We audited three frontier LLMs across **10 countries, 9 languages, and 10 culturally-sensitive personal dilemmas** (840 scored responses), measuring individualism–collectivism bias relative to World Values Survey Wave 7 ground truth.

**Models tested:** Claude Sonnet 4.5 · GPT-5.4 · Gemini 2.5 Flash  
**Judges:** Llama 3.3 70B (Together AI) · DeepSeek-V3 (Together AI)  
**Key finding:** All three models show significant individualist bias (mean misalignment = +0.76, t = 15.65, p < 0.001)

---

## Repository Structure

```
reproduce/
├── experiment.db              # Full SQLite database (840 runs, 837 scored responses)
├── requirements.txt           # Python dependencies
├── .env.example               # API key template
├── data/
│   └── wvs_anchor_scores.csv  # WVS Wave 7 anchor scores per country/prompt
├── notebooks/
│   ├── 01_setup_db.py         # Create DB schema, populate prompts, generate 840 runs
│   ├── 02_run_experiments.py  # Call 3 LLMs — saves responses to DB
│   ├── 03_score_responses.py  # LLM judging (Llama + DeepSeek)
│   ├── 04_analysis.py         # Statistical analysis — all paper results
│   ├── 05_publication_figures.py  # Figures 1–4 (heatmap, conditions, prompts, interjudge)
│   ├── 06_extra_figures.py    # Figure 5a/5b (sub-dimension heatmap + model bars)
│   └── 07_mixed_effects.py    # Mixed-effects robustness check (Appendix F)
└── paper/
    ├── main.tex               # Full LaTeX paper (ACL format)
    ├── references.bib
    ├── acl.sty / acl_latex.sty / acl_natbib.bst / acl.bst
    └── figures/               # All figure PDFs (ready for Overleaf)
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API keys
```bash
cp .env.example .env
# Edit .env and add your keys:
# ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, TOGETHER_API_KEY
```

### 3. Reproduce from the existing database (fastest)

`experiment.db` contains all 840 runs with responses and scores.  
Skip straight to analysis and figures:

```bash
python3 notebooks/04_analysis.py
python3 notebooks/05_publication_figures.py
python3 notebooks/06_extra_figures.py
python3 notebooks/07_mixed_effects.py
```

### 4. Full reproduction from scratch (optional)

Requires approximately \$15–20 in API costs and several hours.

```bash
python3 notebooks/01_setup_db.py        # Set up fresh DB (840 runs)
python3 notebooks/02_run_experiments.py # ~840 API calls (Claude + GPT-5.4 + Gemini)
python3 notebooks/03_score_responses.py # ~1860 judge calls (Llama + DeepSeek)
python3 notebooks/04_analysis.py
python3 notebooks/05_publication_figures.py
python3 notebooks/06_extra_figures.py
python3 notebooks/07_mixed_effects.py
```

Each script is **resumable** — safe to interrupt and restart at any time.

---

## 4-Condition Design

| Condition | Language | Country Label |
|-----------|----------|---------------|
| C1 | English | None (USA only) |
| C2 | Native | None |
| C3 | Native | Country name |
| C4 | English | Country name |

Comparing C3 vs C4 isolates the language effect holding country label constant.  
Nigeria uses English as its official language, so C2 and C4 are identical for NGA (noted as a limitation).

---

## Countries and Languages

| Country | Code | Language | Analyzed |
|---------|------|----------|----------|
| Nigeria | NGA | English | ✓ |
| India | IND | Hindi | ✓ |
| China | CHN | Chinese | ✓ |
| Russia | RUS | Russian | ✓ |
| Brazil | BRA | Portuguese | ✓ |
| South Korea | KOR | Korean | ✓ |
| Mexico | MEX | Spanish | ✓ |
| Germany | DEU | German | ✓ |
| USA | USA | English | ✓ (C1 only) |
| Japan | JPN | Japanese | ✓ |

---

## Scoring

- **Primary composite:** mean of Judge 1 IC score + Judge 2 IC score (`ind_coll_score`)
- **Inter-judge agreement:** Pearson r = 0.575, p < 0.001, n = 837
- **Misalignment:** composite score − WVS anchor score (positive = individualist bias)
- **Secondary:** DeepSeek-V3 sub-dimension scores (autonomy, authority deference, family orientation) — see Section 4.5 and Appendix F

---

## Key Results

| Model | Mean Δ | t | p |
|-------|--------|---|---|
| GPT-5.4 | +0.921 | 10.84 | <0.001 |
| Claude Sonnet 4.5 | +0.888 | 10.80 | <0.001 |
| Gemini 2.5 Flash | +0.460 | 5.66 | <0.001 |
| **All models** | **+0.756** | **15.65** | **<0.001** |

Japan is the only reversal country (mean = −0.43, t = −4.00, p < 0.001).

---

## Compile the Paper

Upload `paper/` contents to [Overleaf](https://overleaf.com) as a new project, or compile locally:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Citation

```bibtex
@misc{jeripityvenkata2026whenaispeaks,
  title={When AI Speaks, Whose Values Does It Express?
         A Cross-Cultural Audit of Individualism--Collectivism Bias
         in Large Language Models},
  author={Jeripity Venkata, Pruthvinath},
  year={2026},
  eprint={2604.22153},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2604.22153}
}
```
