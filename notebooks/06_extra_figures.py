"""
Notebook 10 — Pipeline overview figure + Sub-dimension secondary analysis
Run: python3 notebooks/10_pipeline_and_subdim_figures.py
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats

DB_PATH = "experiment.db"
FIG_DIR = "paper/figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif"],
    "font.size":   10,
    "figure.dpi":  300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ── Load data ─────────────────────────────────────────────────────────────────
con = sqlite3.connect(DB_PATH)
df = pd.read_sql("""
    SELECT ru.model, ru.condition, ru.country, ru.prompt_id,
           r.is_refusal,
           j2.ind_coll_score, j2.autonomy_score,
           j2.authority_score, j2.family_score,
           pw.anchor_score AS wvs_anchor
    FROM runs ru
    JOIN responses r ON ru.run_id = r.run_id
    JOIN judge2_scores j2 ON ru.run_id = j2.run_id
    LEFT JOIN prompt_wvs_scores pw ON (ru.prompt_id=pw.prompt_id AND ru.country=pw.country)
    WHERE r.is_refusal = 0 AND ru.country != 'EGY'
      AND j2.ind_coll_score IS NOT NULL
""", con)
con.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 0 — Pipeline Overview
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 0 — Pipeline overview...")

fig, ax = plt.subplots(figsize=(15, 6.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 6.5)
ax.axis("off")

# ── Helper functions ──────────────────────────────────────────────────────────
def box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=9, bold=False,
        text_color="black", radius=0.15, valign="center", subtext=None):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.05,rounding_size={radius}",
                           facecolor=facecolor, edgecolor=edgecolor,
                           linewidth=1.2, zorder=3)
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    cy = y + h / 2
    if subtext:
        ax.text(x + w/2, cy + 0.15, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=text_color, zorder=4)
        ax.text(x + w/2, cy - 0.20, subtext, ha="center", va="center",
                fontsize=fontsize - 1.5, color="#555555", style="italic", zorder=4)
    else:
        ax.text(x + w/2, cy, text, ha="center", va=valign,
                fontsize=fontsize, fontweight=weight, color=text_color, zorder=4,
                multialignment="center")

def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14),
                zorder=2)

# ── Column x-positions ────────────────────────────────────────────────────────
C0 = 0.15   # Study design
C1 = 2.85   # Prompts generated
C2 = 5.25   # Three LLMs
C3 = 8.05   # Scoring
C4 = 11.2   # Output / analysis

# ─────────────────────────────── Col 0: Study Design ─────────────────────────
ax.text(C0 + 1.05, 6.25, "Study Design", ha="center", fontsize=10,
        fontweight="bold", color="#222222")

BOX_W0 = 2.1
items_col0 = [
    ("10 Countries\n(5 continents)", "#d4e6f1"),
    ("10 Dilemma\nPrompts (P01–P10)", "#d4e6f1"),
    ("7 Languages", "#d4e6f1"),
    ("4 Conditions\n(C1–C4)", "#d4e6f1"),
]
y_starts = [4.60, 3.50, 2.40, 1.15]
for (txt, col), ys in zip(items_col0, y_starts):
    box(ax, C0, ys, BOX_W0, 0.85, txt, col, "#2980b9", fontsize=8.5)

# Multiplication arrow → 930
ax.text(C0 + BOX_W0/2, 0.72, "3 models × 310 combos", ha="center",
        fontsize=7.5, color="#666666", style="italic")
ax.text(C0 + BOX_W0/2, 0.38, "= 930 API calls", ha="center",
        fontsize=8.5, fontweight="bold", color="#1a5276")

# ─────────────────────────────── Col 1: Prompt Assembly ──────────────────────
ax.text(C1 + 1.05, 6.25, "Prompt Assembly", ha="center", fontsize=10,
        fontweight="bold", color="#222222")

box(ax, C1, 3.5, 2.1, 2.2,
    "Each prompt =\nDilemma text\n+ Language (C2/C3)\n+ Country label (C3/C4)\nor English/no label (C1)",
    "#eaf4fb", "#2980b9", fontsize=8)

box(ax, C1, 1.8, 2.1, 1.4,
    "C1: English, no label\nC2: Native, no label\nC3: Native + label\nC4: English + label",
    "#fdfefe", "#7f8c8d", fontsize=7.5)

box(ax, C1, 0.2, 2.1, 1.3,
    "WVS Wave 7\nAnchor scores\n(ground truth)", "#fef9e7", "#d4ac0d", fontsize=8.5)

# ─────────────────────────────── Col 2: Three LLMs ───────────────────────────
ax.text(C2 + 1.35, 6.25, "Frontier LLMs", ha="center", fontsize=10,
        fontweight="bold", color="#222222")

llms = [
    ("Claude Sonnet 4.5", "claude-sonnet-4-5\n-20250929", "#fdebd0", "#e67e22"),
    ("GPT-5.4",           "gpt-5.4-2026-03-05",           "#fdf2f8", "#8e44ad"),
    ("Gemini 2.5 Flash",  "gemini-2.5-flash",             "#e8f8f5", "#1e8449"),
]
y_llm = [4.55, 2.95, 1.35]
for (name, ver, fc, ec), y in zip(llms, y_llm):
    box(ax, C2, y, 2.7, 1.3, name, fc, ec, fontsize=9, bold=True,
        subtext=ver)

ax.text(C2 + 1.35, 0.65, "temp = 0, no system prompt\nstateless API calls",
        ha="center", fontsize=7.5, color="#666666", style="italic")

# ─────────────────────────────── Col 3: Scoring ──────────────────────────────
ax.text(C3 + 1.45, 6.25, "Scoring", ha="center", fontsize=10,
        fontweight="bold", color="#222222")

# Judge 1
box(ax, C3, 4.55, 2.9, 1.5,
    "Judge 1\nLlama 3.3 70B\n(Together AI)",
    "#eaf7fb", "#1a7aab", fontsize=8, bold=False)
ax.text(C3 + 1.45, 4.35, "IC score only\n(non-IC dims compressed)",
        ha="center", fontsize=7, color="#888888", style="italic")

# Judge 2
box(ax, C3, 2.65, 2.9, 1.5,
    "Judge 2\nDeepSeek-V3\n(Together AI)",
    "#eafaf1", "#1e8449", fontsize=8, bold=False)
ax.text(C3 + 1.45, 2.45, "IC + Autonomy +\nAuthority + Family",
        ha="center", fontsize=7, color="#555555", style="italic")

# Inter-judge
box(ax, C3, 1.2, 2.9, 1.1,
    "Inter-judge: r = 0.575\n(IC scores, n=837)",
    "#f9f9f9", "#aaaaaa", fontsize=7.5)

# Composite
box(ax, C3, 0.1, 2.9, 0.9,
    "Composite = mean(J1_IC, J2_IC)",
    "#fdfefe", "#555555", fontsize=7.5, bold=False)

# ─────────────────────────────── Col 4: Analysis ─────────────────────────────
ax.text(C4 + 1.35, 6.25, "Analysis & Findings", ha="center", fontsize=10,
        fontweight="bold", color="#222222")

findings = [
    ("H1: Individualist bias\n(t=15.65, p<0.001)", "#fdedec", "#c0392b"),
    ("H2: Model comparison\n(Claude≈GPT-5.4 > Gemini)", "#fef9e7", "#b7950b"),
    ("H3: Language vs. label\n(Claude & Gemini differ)", "#eaf4fb", "#2471a3"),
    ("H4: Domain effects\n(Marriage > Authority)", "#f4ecf7", "#7d3c98"),
    ("Secondary: Sub-dims\n(DeepSeek, Autonomy=4.54)", "#e8f8f5", "#1e8449"),
]
y_f = [5.20, 4.15, 3.10, 2.05, 1.00]
for (txt, fc, ec), y in zip(findings, y_f):
    box(ax, C4, y, 2.7, 0.85, txt, fc, ec, fontsize=8)

# WVS comparison bracket at bottom
ax.annotate("", xy=(C4, 0.55), xytext=(C1 + 2.1, 0.55),
            arrowprops=dict(arrowstyle="-|>", color="#d4ac0d", lw=1.5,
                            connectionstyle="arc3,rad=0"), zorder=2)
ax.text((C1 + 2.1 + C4)/2, 0.35, "Misalignment = Composite − WVS anchor",
        ha="center", fontsize=8, color="#b7950b", style="italic")

# ── Arrows between columns ────────────────────────────────────────────────────
# Col0 → Col1
for y in [5.02, 3.92, 2.82, 1.57]:
    arrow(ax, C0 + BOX_W0, y, C1, y, color="#2980b9", lw=1.2)

# Col1 → Col2 (prompts → LLMs)
for y_src, y_dst in [(4.6, 5.2), (4.6, 3.6), (4.6, 2.0)]:
    arrow(ax, C1 + 2.1, y_src, C2, y_dst, color="#555555", lw=1.2)

# Col2 → Col3 (LLMs → judges)
for y_src in [5.2, 3.6, 2.0]:
    arrow(ax, C2 + 2.7, y_src, C3, 5.0, color="#888888", lw=0.8)
    arrow(ax, C2 + 2.7, y_src, C3, 3.0, color="#888888", lw=0.8)

# Col3 → Col4
arrow(ax, C3 + 2.9, 1.55, C4, 3.5, color="#555555", lw=1.5)

# Title
ax.text(7.5, 6.5, "Study Pipeline: When AI Speaks, Whose Values Does It Express?",
        ha="center", va="center", fontsize=11, fontweight="bold", color="#1a1a1a")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig0_pipeline.pdf", format="pdf")
plt.savefig(f"{FIG_DIR}/fig0_pipeline.png", format="png", dpi=300)
plt.close()
print("  Saved fig0_pipeline.pdf / .png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Sub-dimension analysis (DeepSeek-V3 scores)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 5 — Sub-dimension analysis (DeepSeek)...")

DIMS     = ["ind_coll_score", "autonomy_score", "authority_score", "family_score"]
DIM_LBLS = ["Indiv–Coll\n(IC)", "Autonomy", "Authority\nDeference", "Family\nOrientation"]
DIM_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]

PROMPT_ORDER = ["P03","P08","P06","P01","P10","P09","P04","P02","P07","P05"]
PROMPT_LABELS = {
    "P01": "P01\nCareer vs parents",
    "P02": "P02\nWomen's career",
    "P03": "P03\nChallenge manager",
    "P04": "P04\nArranged marriage",
    "P05": "P05\nUnhappy marriage",
    "P06": "P06\nEldest abroad",
    "P07": "P07\nMental health",
    "P08": "P08\nReligion vs career",
    "P09": "P09\nQuestion doctor",
    "P10": "P10\nReport family",
}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# ── Left panel: per-prompt grouped bars ──
ax = axes[0]
rng = np.random.default_rng(42)
prompt_means = df.groupby("prompt_id")[DIMS].mean()
prompt_means = prompt_means.loc[PROMPT_ORDER]

n_prompts = len(PROMPT_ORDER)
n_dims    = len(DIMS)
bar_w     = 0.18
x         = np.arange(n_prompts)

for i, (dim, lbl, col) in enumerate(zip(DIMS, DIM_LBLS, DIM_COLORS)):
    offset = (i - (n_dims-1)/2) * bar_w
    vals   = prompt_means[dim].values
    bars   = ax.bar(x + offset, vals, bar_w, label=lbl.replace("\n", " "),
                    color=col, alpha=0.85, zorder=3)

ax.axhline(y=3.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6,
           label="Neutral (3.0)", zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([PROMPT_LABELS[p].split("\n")[0] for p in PROMPT_ORDER],
                   fontsize=8.5)
ax.set_ylabel("Mean Score (1=collectivist, 5=individualist)", fontsize=9)
ax.set_ylim(1.5, 5.5)
ax.set_title("(a) Sub-dimension scores by dilemma topic\n(DeepSeek-V3 judge)",
             fontsize=9.5, pad=4, loc="left")
ax.legend(fontsize=8, frameon=True, loc="upper left", ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add prompt descriptions below x-axis
for i, pid in enumerate(PROMPT_ORDER):
    short = PROMPT_LABELS[pid].split("\n")[1]
    ax.text(i, 1.3, short, ha="center", va="top", fontsize=6.5,
            color="#555555", rotation=20)

# ── Right panel: per-model radar / bar ──
ax2 = axes[1]

MODEL_ORDER  = ["claude", "gpt4o", "gemini"]
MODEL_LABELS_D = {
    "claude": "Claude\nSonnet 4.5",
    "gpt4o":  "GPT-5.4",
    "gemini": "Gemini\n2.5 Flash",
}
MODEL_COLORS_D = {"claude": "#E69F00", "gpt4o": "#56B4E9", "gemini": "#009E73"}

model_means = df.groupby("model")[DIMS].mean()
bar_w2 = 0.2
x2     = np.arange(len(DIMS))

for i, model in enumerate(MODEL_ORDER):
    offset = (i - 1) * bar_w2
    vals   = [model_means.loc[model, d] for d in DIMS]
    ax2.bar(x2 + offset, vals, bar_w2,
            label=MODEL_LABELS_D[model].replace("\n", " "),
            color=MODEL_COLORS_D[model], alpha=0.85, zorder=3)

ax2.axhline(y=3.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
ax2.set_xticks(x2)
ax2.set_xticklabels(DIM_LBLS, fontsize=9.5)
ax2.set_ylim(1.5, 5.5)
ax2.set_title("(b) Sub-dimension scores by model\n(DeepSeek-V3 judge, $n=837$)",
              fontsize=9.5, pad=4, loc="left")
ax2.legend(fontsize=8.5, frameon=True, loc="upper right")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Annotate means on bars
for i, model in enumerate(MODEL_ORDER):
    offset = (i - 1) * bar_w2
    for j, dim in enumerate(DIMS):
        val = model_means.loc[model, dim]
        ax2.text(j + offset, val + 0.06, f"{val:.2f}", ha="center",
                 va="bottom", fontsize=6.5, fontweight="bold", color="#333333")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_subdimensions.pdf", format="pdf")
plt.savefig(f"{FIG_DIR}/fig5_subdimensions.png", format="png", dpi=300)
plt.close()
print("  Saved fig5_subdimensions.pdf / .png")

print("\nDone.")
