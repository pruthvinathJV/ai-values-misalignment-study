"""
Notebook 09 — Publication-Quality Figures
Regenerates all 4 figures as PDF (vector) for LaTeX submission.
Designed for ACL two-column format:
  - Fig 1 (heatmap): full-width, 10×3.5 in
  - Fig 2 (conditions): full-width 3-panel, 12×4 in
  - Fig 3 (prompts): single-column, 3.3×4 in
  - Fig 4 (interjudge): single-column, 3.3×3.3 in

Run: python3 notebooks/09_publication_figures.py
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import os

# ── Setup ─────────────────────────────────────────────────────────────────────
DB_PATH  = "experiment.db"
FIG_DIR  = "paper/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Colorblind-safe palette for models
MODEL_COLORS = {
    "claude": "#E69F00",   # amber
    "gpt4o":  "#56B4E9",   # sky blue
    "gemini": "#009E73",   # green
}
MODEL_LABELS = {
    "claude": "Claude 3.5 Sonnet",
    "gpt4o":  "GPT-4o",
    "gemini": "Gemini 2.5 Flash",
}

# Country order: collectivist → individualist (by WVS expected score)
COUNTRY_ORDER = ["NGA", "IND", "CHN", "RUS", "BRA", "KOR", "MEX", "DEU", "USA", "JPN"]
COUNTRY_LABELS = {
    "NGA": "Nigeria", "IND": "India", "CHN": "China",
    "RUS": "Russia",  "BRA": "Brazil","KOR": "S. Korea",
    "MEX": "Mexico",  "DEU": "Germany","USA": "USA", "JPN": "Japan",
}

# ── Load data ─────────────────────────────────────────────────────────────────
con = sqlite3.connect(DB_PATH)
df = pd.read_sql("""
    SELECT ru.run_id, ru.model, ru.condition, ru.lang_code, ru.country, ru.prompt_id,
           r.is_refusal,
           j1.ind_coll_score AS judge1_score,
           j2.ind_coll_score AS judge2_score,
           pw.anchor_score AS wvs_anchor
    FROM runs ru
    JOIN responses r       ON ru.run_id = r.run_id
    JOIN llm_judge_scores j1 ON ru.run_id = j1.run_id
    JOIN judge2_scores    j2 ON ru.run_id = j2.run_id
    LEFT JOIN prompt_wvs_scores pw ON (ru.prompt_id=pw.prompt_id AND ru.country=pw.country)
""", con)
con.close()

df_s = df[
    (df["is_refusal"] == 0) &
    (df["country"] != "EGY") &
    df["judge1_score"].notna() &
    df["judge2_score"].notna()
].copy()
df_s["composite_score"] = df_s[["judge1_score","judge2_score"]].mean(axis=1)
df_s["misalignment"]    = df_s["composite_score"] - df_s["wvs_anchor"]

# Inter-judge r for Fig 4
r_j1j2, p_j1j2 = stats.pearsonr(df_s["judge1_score"], df_s["judge2_score"])

print(f"Loaded {len(df_s)} scored responses for figures.")

# ── Figure 1: Misalignment Heatmap ───────────────────────────────────────────
print("Generating Fig 1 — Misalignment heatmap...")

pivot = df_s.groupby(["model","country"])["misalignment"].mean().unstack()
heat_data = pivot[COUNTRY_ORDER].reindex(["claude","gpt4o","gemini"])

fig, ax = plt.subplots(figsize=(10, 2.8))

norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)
im = ax.imshow(heat_data.values, cmap="RdBu_r", norm=norm, aspect="auto")

ax.set_xticks(range(len(COUNTRY_ORDER)))
ax.set_xticklabels([COUNTRY_LABELS[c] for c in COUNTRY_ORDER], fontsize=10)
ax.set_yticks(range(3))
ax.set_yticklabels([MODEL_LABELS[m] for m in ["claude","gpt4o","gemini"]], fontsize=10)

# Annotate cells
for i in range(3):
    for j in range(len(COUNTRY_ORDER)):
        val = heat_data.values[i, j]
        if not np.isnan(val):
            color = "white" if abs(val) > 0.7 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label("Misalignment\n(+ = individualist bias)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_xlabel("Country (ordered collectivist → individualist by WVS)", fontsize=10, labelpad=6)
ax.set_title("Figure 1: AI Model Value Misalignment by Country and Model",
             fontsize=11, pad=8, loc="left")

# Add subtle grid
for j in range(len(COUNTRY_ORDER)):
    ax.axvline(j - 0.5, color="white", linewidth=0.5)
for i in range(3):
    ax.axhline(i - 0.5, color="white", linewidth=0.5)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_heatmap.pdf", format="pdf")
plt.savefig(f"{FIG_DIR}/fig1_heatmap.png", format="png", dpi=300)
plt.close()
print(f"  Saved fig1_heatmap.pdf")

# ── Figure 2: Condition effects — mean ± 95% CI + country dots ───────────────
print("Generating Fig 2 — Condition effects (redesigned: mean+CI + dots)...")

COND_ORDER  = ["C1","C2","C3","C4"]
COND_LABELS = ["C1\nEng/no label", "C2\nNative/no label",
               "C3\nNative+label", "C4\nEng+label"]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)

rng2 = np.random.default_rng(7)

for ax, model in zip(axes, ["claude","gpt4o","gemini"]):
    sub_m = df_s[df_s["model"] == model]

    # ── Per-condition means and 95% CI (bootstrap) ──
    cond_means, cond_lo, cond_hi = [], [], []
    for c in COND_ORDER:
        vals = sub_m[sub_m["condition"] == c]["composite_score"].dropna().values
        m = vals.mean()
        boot = [rng2.choice(vals, len(vals), replace=True).mean()
                for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        cond_means.append(m); cond_lo.append(lo); cond_hi.append(hi)

    x = np.arange(4)

    # ── Country dots (jittered, not connected) ──
    for ci, c in enumerate(COND_ORDER):
        country_means = [
            sub_m[(sub_m["country"] == cty) & (sub_m["condition"] == c)
                 ]["composite_score"].mean()
            for cty in COUNTRY_ORDER
        ]
        jitter = rng2.uniform(-0.12, 0.12, len(country_means))
        ax.scatter([ci + j for j in jitter], country_means,
                   s=14, color="#aaaaaa", alpha=0.55, zorder=2, linewidths=0)

    # ── Model mean line + shaded CI ──
    ax.plot(x, cond_means, color=MODEL_COLORS[model],
            linewidth=2.2, marker="o", markersize=6, zorder=4,
            label="Model mean")
    ax.fill_between(x,
                    [lo for lo in cond_lo],
                    [hi for hi in cond_hi],
                    alpha=0.18, color=MODEL_COLORS[model], zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS, fontsize=8.5)
    ax.set_title(MODEL_LABELS[model], fontsize=10, pad=4)
    ax.axhline(y=3, color="#cccccc", linestyle="--", alpha=0.7, linewidth=0.8)
    ax.set_ylim(2.5, 4.2)
    ax.set_xlim(-0.45, 3.45)
    if model == "claude":
        ax.set_ylabel("Composite Score\n(1=collectivist, 5=individualist)", fontsize=9)

# Manual legend entry for grey dots
dot_handle  = Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",
                     markersize=6, alpha=0.7, label="Country mean")
line_handle = Line2D([0], [0], color="#555555", linewidth=2,
                     marker="o", markersize=5, label="Model mean ± 95% CI")
axes[-1].legend(handles=[line_handle, dot_handle],
                fontsize=8, frameon=True, loc="upper right")

fig.suptitle("Figure 2: Score Shift Across Experimental Conditions by Model",
             fontsize=10, y=1.01, x=0.47)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_conditions.pdf", format="pdf")
plt.savefig(f"{FIG_DIR}/fig2_conditions.png", format="png", dpi=300)
plt.close()
print(f"  Saved fig2_conditions.pdf")

# ── Figure 3: Prompt-level misalignment ──────────────────────────────────────
print("Generating Fig 3 — Prompt misalignment...")

PROMPT_LABELS = {
    "P01": "Career vs parents",
    "P02": "Women's career\nafter marriage",
    "P03": "Challenge manager",
    "P04": "Arranged marriage",
    "P05": "Unhappy marriage",
    "P06": "Eldest abroad",
    "P07": "Mental health",
    "P08": "Religion vs career",
    "P09": "Question doctor",
    "P10": "Report family",
}

# Bootstrap CIs
rng = np.random.default_rng(42)
prompt_stats = []
for pid in sorted(df_s["prompt_id"].unique()):
    vals = df_s[df_s["prompt_id"]==pid]["misalignment"].dropna().values
    mean = vals.mean()
    boot = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    prompt_stats.append({"prompt_id": pid, "mean": mean, "lo": lo, "hi": hi})

ps = pd.DataFrame(prompt_stats).sort_values("mean", ascending=True)

fig, ax = plt.subplots(figsize=(3.5, 4.2))
colors = ["#2166ac" if v < 0 else "#d6604d" for v in ps["mean"]]
y_pos = range(len(ps))

bars = ax.barh(y_pos, ps["mean"].values, color=colors, alpha=0.85, height=0.6)
ax.errorbar(ps["mean"].values, y_pos,
            xerr=[ps["mean"].values - ps["lo"].values,
                  ps["hi"].values - ps["mean"].values],
            fmt="none", color="black", capsize=3, linewidth=1)

ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([PROMPT_LABELS[p] for p in ps["prompt_id"]], fontsize=8.5)
ax.set_xlabel("Mean Misalignment (+ = individualist bias)", fontsize=9)
ax.set_title("Figure 3: Misalignment\nby Dilemma Topic", fontsize=10, pad=4, loc="left")

# Value labels — placed past the CI bound so they never overlap error bars
PAD = 0.06
for bar, val, hi, lo in zip(bars, ps["mean"].values,
                             ps["hi"].values, ps["lo"].values):
    if val >= 0:
        ax.text(hi + PAD, bar.get_y() + bar.get_height()/2,
                f"+{val:.2f}", va="center", ha="left",
                fontsize=8, fontweight="bold",
                color="#9b2c2c" if val > 0.5 else "#333333")
    else:
        ax.text(lo - PAD, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", ha="right",
                fontsize=8, fontweight="bold", color="#1a3a6b")

ax.set_xlim(-1.3, 1.6)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_prompts.pdf", format="pdf")
plt.savefig(f"{FIG_DIR}/fig3_prompts.png", format="png", dpi=300)
plt.close()
print(f"  Saved fig3_prompts.pdf")

# ── Figure 4: Inter-judge agreement ──────────────────────────────────────────
print("Generating Fig 4 — Inter-judge agreement...")

fig, ax = plt.subplots(figsize=(3.3, 3.3))

ax.scatter(df_s["judge1_score"], df_s["judge2_score"],
           alpha=0.12, s=6, color="#4e79a7", rasterized=True)
ax.plot([1,5],[1,5], color="#e15759", linewidth=1, linestyle="--",
        alpha=0.7, label="Perfect agreement")

ax.set_xlabel("Judge 1 Score (Llama 3.3 70B)", fontsize=9)
ax.set_ylabel("Judge 2 Score (DeepSeek-V3)", fontsize=9)
ax.set_title("Figure 4: Inter-Judge\nAgreement", fontsize=10, pad=4, loc="left")
ax.set_xlim(0.8, 5.2); ax.set_ylim(0.8, 5.2)

# Annotate r and p
ax.text(0.05, 0.95, f"$r = {r_j1j2:.3f}$\n$p < 0.001$\n$n = {len(df_s)}$",
        transform=ax.transAxes, fontsize=8.5,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"))

ax.legend(fontsize=8, frameon=True)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_interjudge.pdf", format="pdf")
plt.savefig(f"{FIG_DIR}/fig4_interjudge.png", format="png", dpi=300)
plt.close()
print(f"  Saved fig4_interjudge.pdf")

print(f"\nAll figures saved to {FIG_DIR}/")
print("PDF = vector (for LaTeX) | PNG = raster (for preview)")
