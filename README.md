# When AI Speaks, Whose Values Does It Express?
### A Cross-Cultural Audit of Individualism–Collectivism Bias in Large Language Models

**Pruthvinath Jeripity Venkata** — Independent Researcher · jvpnath@gmail.com

---

## Summary

This project audits cultural value bias in three frontier LLMs by presenting them with 10 culturally loaded personal dilemmas across 10 countries and 9 languages, then measuring how individualist or collectivist their advice is relative to World Values Survey (WVS) Wave 7 ground truth.

**Models:** Claude Sonnet 4.5 · GPT-5.4 · Gemini 2.5 Flash  
**Countries:** Nigeria, India, China, Russia, Brazil, South Korea, Mexico, Germany, USA, Japan  
**Finding:** All three models show significant individualist bias (mean misalignment = +0.76 on a 1–5 scale; t = 15.65, p < 0.001), strongest in Nigeria (+1.85) and India (+0.82). Japan is the only reversal country (−0.43), where models encode traditional stereotypes rather than contemporary WVS-measured values.

---

## Replication Package

Everything needed to reproduce the paper is in the [`reproduce/`](reproduce/) folder:

```
reproduce/
├── experiment.db              # Full SQLite database (840 runs, 837 scored responses)
├── requirements.txt           # pip install -r requirements.txt
├── .env.example               # Copy to .env and add API keys
├── data/wvs_anchor_scores.csv # WVS Wave 7 anchor values
├── notebooks/                 # 7 numbered scripts (run in order)
└── paper/                     # LaTeX source (ACL format)
```

See [`reproduce/README.md`](reproduce/README.md) for full quickstart and reproduction instructions.

---

## Paper

**arXiv:** https://arxiv.org/abs/2604.22153

The paper is in [`paper/main.tex`](paper/main.tex) (ACL two-column format).  
Upload the `paper/` folder to [Overleaf](https://overleaf.com) or compile locally:

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Key Results

| Model | Mean Misalignment | t | p |
|-------|-------------------|---|---|
| GPT-5.4 | +0.921 | 10.84 | <0.001 |
| Claude Sonnet 4.5 | +0.888 | 10.80 | <0.001 |
| Gemini 2.5 Flash | +0.460 | 5.66 | <0.001 |
| **All models combined** | **+0.756** | **15.65** | **<0.001** |

Positive = individualist bias relative to WVS country-level values (1–5 scale).

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
