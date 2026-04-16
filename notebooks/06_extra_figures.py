"""
06 — Pipeline overview + Sub-dimension secondary analysis
Run: python3 notebooks/06_extra_figures.py
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import TwoSlopeNorm

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
# FIGURE 0 — Pipeline Overview  (compact, tight layout)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 0 — Pipeline overview (compact)...")

W, H = 14.0, 5.0
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.axis("off")

# ── Helpers ───────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, lines, fc, ec, lw=1.4, fs=8.5, bold_first=False):
    """Rounded box; lines = list of strings stacked top→bottom."""
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3, clip_on=False)
    ax.add_patch(patch)
    n = len(lines)
    step = h / (n + 1)
    for i, txt in enumerate(lines):
        fw = "bold" if (i == 0 and bold_first) else "normal"
        sty = "italic" if txt.startswith("(") else "normal"
        ax.text(x + w/2, y + h - step*(i+1) + step*0.1,
                txt, ha="center", va="center",
                fontsize=fs, fontweight=fw, fontstyle=sty,
                color="#111111", zorder=4)

def harrow(ax, x1, x2, y, color="#333333", lw=2.2):
    """Straight horizontal arrow."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=16), zorder=5)

def darrow(ax, x1, y1, x2, y2, color="#555555", lw=1.8):
    """Diagonal arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=14,
                        connectionstyle="arc3,rad=0.0"), zorder=5)

# ── Layout constants ──────────────────────────────────────────────────────────
#   5 columns: Design | Prompts | LLMs | Scoring | Results
#   Reduced box widths → 0.60 gap between columns → visible arrows
BH    = 0.56   # standard box height
BH_L  = 0.64   # tall LLM box height

# Col x-lefts and widths  (gap between cols = 0.60)
X  = [0.10,  2.70,  5.40,  8.20, 11.50]
BW = [2.00,  2.10,  2.20,  2.70,  2.40]

# Row y-centers (3 rows: top, mid, bot)
Y3 = [3.95, 2.68, 1.42]   # 3 LLMs / judges rows
Y2 = [3.30, 1.90]          # 2-row layout

# ── Column headers ────────────────────────────────────────────────────────────
headers = ["[1] Study Design", "[2] Prompt Assembly",
           "[3] Frontier LLMs", "[4] LLM Scoring", "[5] Analysis"]
hcolors = ["#1a5276","#117a65","#6c3483","#154360","#922b21"]
for i, (hdr, hc) in enumerate(zip(headers, hcolors)):
    ax.text(X[i] + BW[i]/2, H - 0.22, hdr,
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=hc)

# ── Col 0: Study Design ───────────────────────────────────────────────────────
design_items = [
    (["10 Countries", "(5 continents, incl.", "Africa/Asia/Europe)"], "#d6eaf8", "#2980b9"),
    (["10 Dilemmas", "(P01–P10: marriage,", "career, authority…)"], "#d6eaf8", "#2980b9"),
    (["7 Languages +", "4 Conditions (C1–C4)", "= 930 API calls"], "#d6eaf8", "#1a5276"),
]
ys0 = [3.62, 2.48, 1.18]
bh0 = [0.80, 0.80, 0.90]
for (lines, fc, ec), y, bh in zip(design_items, ys0, bh0):
    rbox(ax, X[0], y, BW[0], bh, lines, fc, ec, fs=7.8)

# ── Col 1: Prompt Assembly ────────────────────────────────────────────────────
rbox(ax, X[1], 3.10, BW[1], 1.36,
     ["Dilemma text", "+ Native/English (C2/C3)", "+ Country label (C3/C4)", "→ 1 prompt per run"],
     "#d5f5e3", "#1e8449", fs=7.8)

rbox(ax, X[1], 1.10, BW[1], 1.50,
     ["WVS Wave 7", "Anchor scores", "(ground truth for", "each country/prompt)"],
     "#fef9e7", "#b7950b", fs=7.8)

# ── Col 2: Three LLMs ────────────────────────────────────────────────────────
llm_data = [
    (["Claude Sonnet 4.5", "claude-sonnet-4-5-20250929"], "#fdebd0", "#e67e22"),
    (["GPT-5.4",           "gpt-5.4-2026-03-05"],         "#f5eef8", "#7d3c98"),
    (["Gemini 2.5 Flash",  "gemini-2.5-flash"],            "#e8f8f5", "#1e8449"),
]
for (lines, fc, ec), y in zip(llm_data, Y3):
    rbox(ax, X[2], y - 0.36, BW[2], BH_L, lines, fc, ec,
         fs=8.2, bold_first=True, lw=1.8)

ax.text(X[2]+BW[2]/2, 0.82, "temp=0 · no system prompt · stateless",
        ha="center", fontsize=7.2, color="#555555", style="italic")

# ── Col 3: Scoring ────────────────────────────────────────────────────────────
rbox(ax, X[3], 3.58, BW[3], 0.94,
     ["Judge 1 — Llama 3.3 70B (Together AI)",
      "IC score only  (non-IC dims compressed)"],
     "#eaf7fb", "#1a7aab", fs=8.0, bold_first=True)

rbox(ax, X[3], 2.28, BW[3], 0.94,
     ["Judge 2 — DeepSeek-V3 (Together AI)",
      "IC · Autonomy · Authority · Family"],
     "#eafaf1", "#1e8449", fs=8.0, bold_first=True)

rbox(ax, X[3], 1.08, BW[3], 0.78,
     ["Inter-judge r = 0.575  (p<0.001, n=837)",
      "Composite = mean(J1_IC, J2_IC)"],
     "#f8f9fa", "#888888", fs=7.8)

ax.text(X[3]+BW[3]/2, 0.50,
        "Misalignment = Composite − WVS anchor",
        ha="center", fontsize=8.0, fontweight="bold",
        color="#7d6608", style="italic")

# ── Col 4: Findings ───────────────────────────────────────────────────────────
findings = [
    (["H1: All models individualist-biased",  "mean=+0.76, t=15.65, p<0.001"], "#fdedec","#c0392b"),
    (["H2: Claude ≈ GPT-5.4 > Gemini",        "d=0.024 vs d≈0.32"],            "#fef9e7","#b7950b"),
    (["H3: Language effect varies by model",  "Claude↓ Gemini↑ GPT-5.4 n.s."], "#eaf4fb","#2471a3"),
    (["H4: Marriage > Authority bias",         "P02=+1.85, P03=−0.26"],         "#f4ecf7","#7d3c98"),
    (["Sub-dims: Autonomy=4.54 (ceiling!)",   "Family=2.77 (collectivist)"],    "#e8f8f5","#1e8449"),
]
y_f = [4.05, 3.22, 2.39, 1.56, 0.73]
for (lines, fc, ec), y in zip(findings, y_f):
    rbox(ax, X[4], y, BW[4], 0.70, lines, fc, ec, fs=7.5, bold_first=True)

# ── Arrows ────────────────────────────────────────────────────────────────────
# Col 0 → Col 1: one arrow per design box at its vertical centre
harrow(ax, X[0]+BW[0]+0.02, X[1]-0.02, ys0[0]+bh0[0]/2, "#2980b9", 2.0)
harrow(ax, X[0]+BW[0]+0.02, X[1]-0.02, ys0[1]+bh0[1]/2, "#2980b9", 2.0)
harrow(ax, X[0]+BW[0]+0.02, X[1]-0.02, ys0[2]+bh0[2]/2, "#2980b9", 2.0)

# Col 1 → Col 2: prompts to each LLM
for y_dst in Y3:
    darrow(ax, X[1]+BW[1]+0.02, 2.76, X[2]-0.02, y_dst+BH_L/2, "#333333", 1.8)

# Col 2 → Col 3: each LLM to both judges
for y_src in Y3:
    darrow(ax, X[2]+BW[2]+0.02, y_src+BH_L/2, X[3]-0.02, 4.06, "#1a7aab", 1.6)
    darrow(ax, X[2]+BW[2]+0.02, y_src+BH_L/2, X[3]-0.02, 2.72, "#1e8449", 1.6)

# WVS → scoring composite
harrow(ax, X[1]+BW[1]+0.02, X[3]+BW[3]/2, 1.46, "#b7950b", 1.8)
ax.annotate("", xy=(X[3]+BW[3]/2, 1.40), xytext=(X[3]+BW[3]/2, 0.92),
    arrowprops=dict(arrowstyle="-|>", color="#b7950b", lw=1.6, mutation_scale=12), zorder=5)

# Col 3 → Col 4
harrow(ax, X[3]+BW[3]+0.02, X[4]-0.02, 2.40, "#922b21", 2.2)

plt.tight_layout(pad=0.3)
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

# ── Left panel: heatmap (4 dims × 10 prompts) ────────────────────────────────
ax = axes[0]
prompt_means = df.groupby("prompt_id")[DIMS].mean()
prompt_means = prompt_means.loc[PROMPT_ORDER]

# rows = dimensions, cols = prompts
data = prompt_means[DIMS].T.values   # shape (4, 10)

norm = TwoSlopeNorm(vmin=1.5, vcenter=3.0, vmax=5.5)
im   = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")

ax.set_xticks(range(len(PROMPT_ORDER)))
xlbls = [f"{p}\n{PROMPT_LABELS[p].split(chr(10))[1]}" for p in PROMPT_ORDER]
ax.set_xticklabels(xlbls, fontsize=7.5, rotation=20, ha="right")
ax.set_yticks(range(len(DIMS)))
ax.set_yticklabels(["IC Score", "Autonomy", "Authority\nDeference", "Family\nOrientation"],
                   fontsize=9)

# Annotate each cell
for i in range(len(DIMS)):
    for j in range(len(PROMPT_ORDER)):
        val = data[i, j]
        color = "white" if abs(val - 3.0) > 0.8 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7.5, color=color, fontweight="bold")

# Grid
for j in range(len(PROMPT_ORDER)):
    ax.axvline(j - 0.5, color="white", linewidth=0.5)
for i in range(len(DIMS)):
    ax.axhline(i - 0.5, color="white", linewidth=0.5)

cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Score\n(1=collectivist, 5=individualist)", fontsize=7.5)
cbar.ax.tick_params(labelsize=7)

ax.set_title("(a) Sub-dimension mean scores by dilemma topic\n"
             "(DeepSeek-V3 judge, averaged over all countries and models)",
             fontsize=9, pad=4, loc="left")

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
