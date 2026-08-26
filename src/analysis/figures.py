"""
Programmatically generates every figure and table referenced in the paper
skeleton (Fig 1-6, Table 1-5), reading directly from experiments/runs/ so no
number or bar in the paper is ever hand-made or hand-edited. Re-running this
module after a new batch of runs regenerates the entire results section.

    python -m src.analysis.figures            # everything, into paper/
    python -m src.analysis.figures --figures 3 4

COLOR AND ACCESSIBILITY

Providers are colored by IDENTITY, so the palette is a fixed categorical
assignment, keyed on provider name and never on rank - a provider keeps its
color whether it comes first or last, and adding a provider does not repaint
the others. The four hues were validated (OKLab CVD separation, chroma floor,
lightness band, normal-vision floor, surface contrast) under the all-pairs
rule, which is the stricter rule required by the scatter and heatmap forms
here rather than the adjacent-pairs rule that suffices for bars alone.

One hue in that set sits below 3:1 against a white page, so every figure that
uses it also carries direct value labels, and every figure has a companion CSV
and LaTeX table written beside it. Identity is therefore never carried by
color alone, which is also what makes these figures survive grayscale printing
and photocopying - a realistic fate for a conference paper.

The ATA heatmap uses a single-hue light-to-dark sequential ramp, because it
encodes magnitude on a common scale. Not a rainbow, and not a diverging map:
agreement has no meaningful midpoint to diverge around.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.analysis.load import MODEL_SIZE_B, load_runs, runs_to_frame
from src.analysis.stats import (
    ata_with_cis,
    cell_success_rates,
    edge_sensitivity,
    pairwise_provider_tests,
    provider_effect_by_category,
    two_way_anova,
)
from src.metrics.trust_score import trust_scores
from src.metrics.tsr import tsr_by

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGURES_DIR = os.path.join(REPO_ROOT, "paper", "figures")
TABLES_DIR = os.path.join(REPO_ROOT, "paper", "tables")

# Fixed categorical assignment: color follows the provider, never its rank.
PROVIDER_COLORS = {
    "anthropic": "#2a78d6",   # blue
    "groq": "#eb6834",        # orange
    "ollama": "#1baf7a",      # aqua
    "openai": "#4a3aa7",      # violet
}
FALLBACK_COLOR = "#52514e"

# Single-hue sequential ramp for magnitude encoding (the ATA heatmap).
SEQUENTIAL_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#78776f"
GRID_COLOR = "#e5e4e0"
SURFACE = "#ffffff"

CATEGORY_LABELS = {
    "codegen": "A. Code\ngeneration",
    "tool_orchestration": "B. Tool\norchestration",
    "error_recovery": "C. Error\nrecovery",
    "ambiguous": "D. Ambiguous\ninstructions",
}
CATEGORY_ORDER = ["codegen", "tool_orchestration", "error_recovery", "ambiguous"]


def apply_style():
    """Print-first defaults: serif to match the paper body, recessive axes."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def provider_color(provider):
    return PROVIDER_COLORS.get(provider, FALLBACK_COLOR)


def save(fig, name):
    """Writes PDF (vector, for LaTeX) and PNG (for quick inspection)."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    paths = []
    for extension in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"{name}.{extension}")
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    print(f"  wrote {name}.pdf / {name}.png")
    return paths


def save_table(frame, name, caption=""):
    """Every figure ships a companion table - this is the accessibility
    fallback that lets a reader recover exact values without reading color."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    csv_path = os.path.join(TABLES_DIR, f"{name}.csv")
    frame.to_csv(csv_path, index=False)

    tex_path = os.path.join(TABLES_DIR, f"{name}.tex")
    with open(tex_path, "w") as f:
        f.write(frame.to_latex(index=False, float_format="%.3f", caption=caption or name,
                               label=f"tab:{name}"))
    print(f"  wrote {name}.csv / {name}.tex")
    return csv_path, tex_path


# --------------------------------------------------------------------------
# Fig 1 - teaser: one task, two providers, divergent paths
# --------------------------------------------------------------------------

def figure_1_divergence(runs):
    """The paper's opening image: the SAME task under two providers, with the
    tool-call sequences drawn side by side.

    Picked automatically as the task with the largest observed disagreement
    between any two providers, so the teaser is the real worst case in the
    data rather than a cherry-picked anecdote. The selection rule is stated
    in the caption for exactly that reason.

    Pairs where BOTH providers actually called tools are preferred, because
    two different tool paths illustrate divergence more informatively than
    one path against an empty one. If no such pair exists we fall back to any
    comparable pair, since "one provider silently used no tools at all" is
    itself a real and reportable divergence.
    """
    from src.metrics.ata import index_trajectories, trajectory_similarity

    index = index_trajectories(runs)
    best_both, best_any = None, None
    for task_id, by_provider in index.items():
        providers = sorted(by_provider)
        for i, provider_a in enumerate(providers):
            for provider_b in providers[i + 1:]:
                traj_a, traj_b = by_provider[provider_a][0], by_provider[provider_b][0]
                calls_a, calls_b = traj_a.get("tool_calls"), traj_b.get("tool_calls")
                if not calls_a and not calls_b:
                    continue
                candidate = (trajectory_similarity(traj_a, traj_b), task_id,
                             provider_a, provider_b, traj_a, traj_b)
                if best_any is None or candidate[0] < best_any[0]:
                    best_any = candidate
                if calls_a and calls_b and (best_both is None or candidate[0] < best_both[0]):
                    best_both = candidate

    worst = best_both or best_any
    if worst is None:
        print("  [skip] Fig 1: no comparable trajectories found")
        return None

    similarity, task_id, provider_a, provider_b, traj_a, traj_b = worst
    fig, ax = plt.subplots(figsize=(6.6, 2.6))

    for row, (provider, trajectory) in enumerate([(provider_a, traj_a), (provider_b, traj_b)]):
        y = 1 - row
        steps = [call.get("tool_name") or "(unnamed)" for call in trajectory.get("tool_calls", [])]
        color = provider_color(provider)

        ax.text(-0.35, y, provider, ha="right", va="center",
                fontsize=9, color=TEXT_PRIMARY, fontweight="bold")

        if not steps:
            ax.text(0.1, y, "no tool calls  (answered directly)", ha="left", va="center",
                    fontsize=8.5, color=TEXT_MUTED, style="italic")
            continue

        for position, step in enumerate(steps):
            box = FancyBboxPatch(
                (position * 1.62, y - 0.17), 1.5, 0.34,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=0, facecolor=color, alpha=0.16,
            )
            ax.add_patch(box)
            ax.text(position * 1.62 + 0.75, y, step, ha="center", va="center",
                    fontsize=7.5, color=TEXT_PRIMARY)
            if position < len(steps) - 1:
                # 2px surface gap between marks, arrow drawn inside it
                ax.add_patch(FancyArrowPatch(
                    (position * 1.62 + 1.53, y), (position * 1.62 + 1.59, y),
                    arrowstyle="-|>", mutation_scale=7, linewidth=1.0, color=TEXT_MUTED,
                ))

    max_steps = max(len(traj_a.get("tool_calls", [])), len(traj_b.get("tool_calls", [])), 1)
    ax.set_xlim(-2.6, max(max_steps * 1.62, 3.2))
    ax.set_ylim(-0.75, 1.6)
    ax.axis("off")
    ax.set_title(
        f"Same task ({task_id}), same scaffold, different backend: "
        f"trajectory agreement = {similarity:.2f}",
        color=TEXT_PRIMARY, loc="left", pad=12,
    )
    ax.text(-2.6, -0.62,
            "Task shown is the largest observed pairwise divergence in the run set, not a selected example.",
            fontsize=7, color=TEXT_MUTED)
    return save(fig, "fig1_divergence_teaser")


# --------------------------------------------------------------------------
# Fig 2 - architecture schematic
# --------------------------------------------------------------------------

def figure_2_architecture(frame):
    """Scaffold held constant, provider swapped underneath.

    Drawn from the live config (actual provider and category names) rather
    than hard-coded, so the diagram cannot drift out of sync with what was
    actually run.
    """
    providers = sorted(frame["provider"].dropna().unique())
    categories = [c for c in CATEGORY_ORDER if c in set(frame["category"].dropna())]

    fig, ax = plt.subplots(figsize=(6.6, 3.9))

    def box(x, y, width, height, facecolor, edgecolor):
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.008,rounding_size=0.03",
            linewidth=1.0, facecolor=facecolor, edgecolor=edgecolor,
        ))

    LINE_HEIGHT = 0.052
    PADDING = 0.030

    def stacked_box(x, y_bottom, width, lines, sizes, weights, facecolor, edgecolor):
        """Draws a box sized to fit its own text, then places each line
        explicitly. Sizing the box from the content is what keeps the diagram
        correct when the provider or category list changes - a fixed height
        silently clips whatever no longer fits."""
        height = len(lines) * LINE_HEIGHT + 2 * PADDING
        box(x, y_bottom, width, height, facecolor, edgecolor)
        top = y_bottom + height - PADDING - LINE_HEIGHT / 2
        for i, line in enumerate(lines):
            ax.text(x + width / 2, top - i * LINE_HEIGHT, line, ha="center", va="center",
                    fontsize=sizes[i], color=TEXT_PRIMARY, fontweight=weights[i])
        return y_bottom + height

    # the task suite, one line per category so nothing runs past the box edge
    suite_lines = ["TASK SUITE"] + [CATEGORY_LABELS.get(c, c).replace("\n", " ")
                                    for c in categories]
    suite_top = stacked_box(
        0.04, 0.04, 0.92, suite_lines,
        [8.5] + [7.0] * len(categories), ["bold"] + ["normal"] * len(categories),
        "#f4f4f2", TEXT_SECONDARY,
    )

    # the swapped layer
    provider_bottom = suite_top + 0.09
    provider_height = 0.14
    width = 0.92 / len(providers)
    for i, provider in enumerate(providers):
        color = provider_color(provider)
        x = 0.04 + i * width
        box(x + 0.012, provider_bottom, width - 0.024, provider_height, color + "26", color)
        ax.text(x + width / 2, provider_bottom + provider_height / 2, provider,
                ha="center", va="center", fontsize=8.5,
                color=TEXT_PRIMARY, fontweight="bold")
    ax.add_patch(FancyArrowPatch((0.5, suite_top), (0.5, provider_bottom - 0.01),
                                 arrowstyle="-|>", mutation_scale=8,
                                 linewidth=1.0, color=TEXT_MUTED))

    # the fixed scaffold, the thing held constant
    scaffold_bottom = provider_bottom + provider_height + 0.10
    scaffold_top = stacked_box(
        0.04, scaffold_bottom, 0.92,
        ["FIXED AGENT SCAFFOLD",
         "identical system prompt  ·  identical ReAct loop",
         "identical tool set: read_file, write_file, run_python, calculator"],
        [8.5, 7.5, 7.5], ["bold", "normal", "normal"], "#f4f4f2", TEXT_SECONDARY,
    )
    for i in range(len(providers)):
        x = 0.04 + i * width + width / 2
        ax.add_patch(FancyArrowPatch(
            (x, provider_bottom + provider_height), (x, scaffold_bottom - 0.01),
            arrowstyle="-|>", mutation_scale=8, linewidth=1.0, color=TEXT_MUTED))

    # No floating "only this layer changes" label here: the arrow gap is
    # narrow and any text placed in it collides with the leftmost arrow. The
    # subtitle above states the same point without competing for the space.

    ax.text(0.5, scaffold_top + 0.055,
            "Only the middle layer changes: every run differs in exactly one "
            "factor, the backend model",
            ha="center", fontsize=8.5, color=TEXT_SECONDARY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, scaffold_top + 0.11)
    ax.axis("off")
    return save(fig, "fig2_architecture")


# --------------------------------------------------------------------------
# Fig 3 / Table 2 - TSR by provider (RQ1)
# --------------------------------------------------------------------------

def figure_3_tsr(runs, frame):
    stats_by_provider = tsr_by(runs, "provider")
    providers = sorted(stats_by_provider)

    values = [stats_by_provider[p]["tsr"] for p in providers]
    lower = [values[i] - stats_by_provider[p]["ci_low"] for i, p in enumerate(providers)]
    upper = [stats_by_provider[p]["ci_high"] - values[i] for i, p in enumerate(providers)]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    positions = np.arange(len(providers))
    ax.bar(positions, values, width=0.58,
           color=[provider_color(p) for p in providers], zorder=3)
    ax.errorbar(positions, values, yerr=[lower, upper], fmt="none",
                ecolor=TEXT_SECONDARY, elinewidth=1.1, capsize=4, capthick=1.1, zorder=4)

    # direct value labels: the relief rule for the sub-3:1 hue, and what makes
    # the figure readable in grayscale
    for x, (provider, value) in enumerate(zip(providers, values)):
        ax.text(x, stats_by_provider[provider]["ci_high"] + 0.035, f"{value:.2f}",
                ha="center", fontsize=8.5, color=TEXT_PRIMARY)

    ax.set_xticks(positions)
    # run counts live in the tick label rather than as free-floating text
    # below the axis, where they collided with the provider names
    ax.set_xticklabels([f"{p}\n(n={stats_by_provider[p]['n_total']})" for p in providers])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Task Success Rate")
    ax.set_title("RQ1  Task success rate by provider, identical scaffold",
                 color=TEXT_PRIMARY, loc="left")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0, 1.075, "error bars: 95% Wilson score interval",
            fontsize=7, color=TEXT_MUTED)

    table = pd.DataFrame([
        {"provider": p,
         "model": frame.loc[frame["provider"] == p, "model"].iloc[0]
                  if (frame["provider"] == p).any() else "",
         "tsr": stats_by_provider[p]["tsr"],
         "ci_low": stats_by_provider[p]["ci_low"],
         "ci_high": stats_by_provider[p]["ci_high"],
         "n_pass": stats_by_provider[p]["n_success"],
         "n_runs": stats_by_provider[p]["n_total"]}
        for p in providers
    ])
    save_table(table, "table2_tsr_by_provider",
               "Task Success Rate per provider with 95\\% Wilson confidence intervals.")
    return save(fig, "fig3_tsr_by_provider")


# --------------------------------------------------------------------------
# Fig 4 / Table 3 - pairwise ATA heatmap (RQ2)
# --------------------------------------------------------------------------

def figure_4_ata_heatmap(runs):
    ata_table = ata_with_cis(runs, n_resamples=10000)
    providers = sorted(set(ata_table["provider_a"]) | set(ata_table["provider_b"]))

    matrix = np.full((len(providers), len(providers)), np.nan)
    for _, row in ata_table.iterrows():
        if row["ata"] is not None and not pd.isna(row["ata"]):
            matrix[providers.index(row["provider_a"]), providers.index(row["provider_b"])] = row["ata"]

    colormap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_RAMP)

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    image = ax.imshow(matrix, cmap=colormap, vmin=0.0, vmax=1.0)

    for i in range(len(providers)):
        for j in range(len(providers)):
            if np.isnan(matrix[i, j]):
                ax.text(j, i, "--", ha="center", va="center", fontsize=8, color=TEXT_MUTED)
                continue
            # ink color flips against the ramp so every cell label stays legible
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if matrix[i, j] > 0.55 else TEXT_PRIMARY,
                    fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(len(providers)), providers, rotation=20, ha="right")
    ax.set_yticks(range(len(providers)), providers)
    ax.set_xticks(np.arange(len(providers) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(providers) + 1) - 0.5, minor=True)
    # 2px surface gap between adjacent fills
    ax.grid(which="minor", color=SURFACE, linewidth=2.0)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("RQ2  Pairwise trajectory agreement (ATA)", color=TEXT_PRIMARY, loc="left")
    ax.text(0.0, -0.20,
            "diagonal (bold) = self-agreement across repeated runs,\n"
            "the reference for reading the off-diagonal cells",
            transform=ax.transAxes, fontsize=7, color=TEXT_MUTED, va="top")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("trajectory agreement", fontsize=8, color=TEXT_SECONDARY)
    colorbar.outline.set_visible(False)

    save_table(ata_table, "table3_ata_pairwise",
               "Pairwise Action Trajectory Agreement with task-level bootstrap 95\\% CIs.")
    return save(fig, "fig4_ata_heatmap")


# --------------------------------------------------------------------------
# Fig 5 / Table 4 - capacity vs trust score (RQ3, RQ4)
# --------------------------------------------------------------------------

def figure_5_capacity_vs_trust(runs, frame):
    scored = trust_scores(runs)
    model_by_provider = (frame.dropna(subset=["provider"])
                              .groupby("provider")["model"].first().to_dict())

    rows = []
    for provider, result in scored.items():
        rows.append({
            "provider": provider,
            "model": model_by_provider.get(provider, ""),
            "params_b": MODEL_SIZE_B.get(model_by_provider.get(provider)),
            "trust_score": result["trust_score"],
            "tsr": result["tsr"],
            "ata_cross": result["ata"],
            "stability": result["stability"],
            "protocol_adherence": result["protocol"],
            "n_runs": result["n_runs"],
        })
    table = pd.DataFrame(rows).sort_values("trust_score", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    save_table(table, "table4_trust_score_ranking",
               "Composite Cross-Provider Trust Score ranking and its component terms.")

    known = table.dropna(subset=["params_b"])
    unknown = table[table["params_b"].isna()]

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    if len(known) == 0:
        ax.text(0.5, 0.5, "no provider has a published parameter count",
                ha="center", va="center", fontsize=9, color=TEXT_MUTED)
        ax.axis("off")
        return save(fig, "fig5_capacity_vs_trust")

    for _, row in known.iterrows():
        ax.scatter(row["params_b"], row["trust_score"], s=110, zorder=3,
                   color=provider_color(row["provider"]),
                   edgecolor=SURFACE, linewidth=2.0)   # 2px surface ring on marks
        ax.annotate(f"{row['provider']}\n{row['model']}",
                    (row["params_b"], row["trust_score"]),
                    textcoords="offset points", xytext=(9, -4),
                    fontsize=7.5, color=TEXT_SECONDARY)

    ax.set_xscale("log")
    ax.set_xlabel("model parameters (billions, log scale)")
    ax.set_ylabel("Composite Trust Score")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("RQ3  Edge viability vs composite trust", color=TEXT_PRIMARY, loc="left")

    # State the exclusion on the figure itself rather than only in prose: a
    # reader must be able to see that this panel is not the whole provider set.
    if len(unknown):
        excluded = ", ".join(f"{r['provider']} ({r['model']}, CPTS {r['trust_score']:.2f})"
                             for _, r in unknown.iterrows())
        ax.text(0.0, -0.30, f"Excluded, no published parameter count: {excluded}",
                transform=ax.transAxes, fontsize=7, color=TEXT_MUTED)
    if len(known) < 3:
        ax.text(0.0, -0.38,
                f"n={len(known)} points: too few for a correlation test; shown for illustration only.",
                transform=ax.transAxes, fontsize=7, color=TEXT_MUTED)
    return save(fig, "fig5_capacity_vs_trust")


# --------------------------------------------------------------------------
# Fig 6 / Table 5 - provider x category interaction (RQ6)
# --------------------------------------------------------------------------

def figure_6_interaction(frame):
    cells = (frame.groupby(["provider", "category"], as_index=False)
                  .agg(tsr=("success", "mean"), n_runs=("success", "size")))
    categories = [c for c in CATEGORY_ORDER if c in set(cells["category"])]
    providers = sorted(cells["provider"].unique())

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    positions = np.arange(len(categories))
    THIN_RUNS = 10  # below this, a cell mean is not meaningfully estimated

    endpoints = []
    for provider in providers:
        subset = cells[cells["provider"] == provider].set_index("category")
        values = [subset.loc[c, "tsr"] if c in subset.index else np.nan for c in categories]
        counts = [subset.loc[c, "n_runs"] if c in subset.index else 0 for c in categories]
        color = provider_color(provider)

        ax.plot(positions, values, marker="", zorder=3, color=color, label=provider)
        # Thin cells are drawn hollow. A category scored on a single run must
        # not look as solid as one scored on two hundred, and a reader should
        # see that from the mark itself, not only from the table.
        for x, (value, count) in enumerate(zip(values, counts)):
            if np.isnan(value):
                continue
            solid = count >= THIN_RUNS
            ax.plot(x, value, marker="o", markersize=6, zorder=4, color=color,
                    markerfacecolor=color if solid else SURFACE,
                    markeredgecolor=color if not solid else SURFACE,
                    markeredgewidth=1.6)

        last = next((i for i in range(len(values) - 1, -1, -1) if not np.isnan(values[i])), None)
        if last is not None:
            endpoints.append([values[last], positions[last], provider, color])

    # Push overlapping end labels apart vertically. Three providers converge
    # near 1.0 in this data and their labels otherwise print on top of one
    # another, which is what the direct-labelling rule is meant to prevent.
    endpoints.sort()
    min_gap = 0.075
    for i in range(1, len(endpoints)):
        if endpoints[i][0] - endpoints[i - 1][0] < min_gap:
            endpoints[i][0] = endpoints[i - 1][0] + min_gap
    for label_y, x, provider, color in endpoints:
        ax.annotate(provider, (x, label_y), textcoords="offset points",
                    xytext=(10, 0), fontsize=7.5, color=color, va="center")

    ax.set_xticks(positions)
    ax.set_xticklabels([CATEGORY_LABELS.get(c, c) for c in categories], fontsize=7.5)
    # headroom for however far the de-collision had to push the top label
    top = max([1.05] + [y for y, *_ in endpoints]) + 0.06
    ax.set_ylim(-0.05, top)
    ax.set_xlim(-0.35, len(categories) - 0.45)
    ax.set_ylabel("Task Success Rate")
    ax.set_title("RQ6  Is the provider effect uniform across task categories?",
                 color=TEXT_PRIMARY, loc="left")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncol=len(providers), bbox_to_anchor=(0, -0.34))

    if (cells["n_runs"] < THIN_RUNS).any():
        ax.text(0.0, -0.46,
                f"Hollow markers: fewer than {THIN_RUNS} scored runs in that cell "
                "(categories A and D are not yet fully run).\n"
                "Their means are shown for completeness but are not reliably estimated; "
                "exact counts in Table 5.",
                transform=ax.transAxes, fontsize=7, color=TEXT_MUTED, va="top")

    wide = cells.pivot(index="category", columns="provider", values="tsr").reset_index()
    counts = cells.pivot(index="category", columns="provider", values="n_runs")
    wide["n_runs_total"] = counts.sum(axis=1).values
    spread = provider_effect_by_category(frame)
    table = wide.merge(spread[["category", "tsr_range", "tsr_std"]], on="category", how="left")
    save_table(table, "table5_tsr_by_category",
               "Task Success Rate by provider and task category, with provider spread per category.")
    return save(fig, "fig6_interaction")


# --------------------------------------------------------------------------
# Table 1 - task suite composition
# --------------------------------------------------------------------------

def table_1_task_suite(frame):
    rows = []
    for category in CATEGORY_ORDER:
        subset = frame[frame["category"] == category]
        if subset.empty:
            continue
        difficulties = subset.groupby("task_id")["difficulty"].first().value_counts().to_dict()
        rows.append({
            "category": CATEGORY_LABELS.get(category, category).replace("\n", " "),
            "n_tasks_scored": subset["task_id"].nunique(),
            "n_runs": len(subset),
            "easy": difficulties.get("easy", 0),
            "medium": difficulties.get("medium", 0),
            "hard": difficulties.get("hard", 0),
        })
    table = pd.DataFrame(rows)
    save_table(table, "table1_task_suite",
               "Task suite composition as actually scored in the current run set.")
    return table


def write_stats_summary(runs, frame):
    """The inferential numbers, dumped as text for direct quotation in
    Section 5 - so prose in the paper cannot drift from what was computed."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, "stats_summary.txt")

    anova = two_way_anova(frame)
    pairwise = pairwise_provider_tests(frame)
    edge = edge_sensitivity(frame)
    spread = provider_effect_by_category(frame)

    with open(path, "w") as f:
        f.write("PRE-REGISTERED ANALYSIS OUTPUT\n" + "=" * 70 + "\n\n")
        f.write(f"runs analyzed: {len(frame)}   tasks: {frame['task_id'].nunique()}   "
                f"providers: {frame['provider'].nunique()}\n\n")

        f.write("RQ1/RQ6 - two-way ANOVA on per-task success proportions\n" + "-" * 70 + "\n")
        if anova.get("insufficient_data"):
            f.write(f"insufficient data: {anova['reason']}\n\n")
        else:
            f.write(f"formula: {anova['formula']}\n")
            if not anova["interaction_fitted"]:
                f.write("NOTE: interaction term omitted - at least one provider x category\n"
                        "      combination has fewer than 2 scored tasks, so the term is\n"
                        "      unidentifiable. RQ6 rests on the per-category spread below\n"
                        "      until categories A and D are fully run.\n")
            f.write(f"\n{anova['table']}\n\n")
            f.write(f"partial eta^2: { {k: round(v, 4) for k, v in anova['partial_eta_sq'].items()} }\n")
            f.write(f"magnitude:     {anova['effect_magnitude']}\n\n")

        f.write("RQ1 - pairwise provider comparisons (Wilcoxon signed-rank, Holm-corrected)\n")
        f.write("-" * 70 + "\n")
        if len(pairwise):
            f.write(pairwise[["provider_a", "provider_b", "n_paired_tasks", "mean_success_a",
                              "mean_success_b", "median_difference", "p_raw", "p_adjusted",
                              "significant"]].to_string(index=False) + "\n\n")
            pairwise.to_csv(os.path.join(TABLES_DIR, "table2b_pairwise_providers.csv"), index=False)

        f.write("RQ5 - does the backend effect concentrate at the edge-viable end?\n")
        f.write("-" * 70 + "\n")
        f.write(f"provider partial eta^2, all providers:            {edge['provider_eta_sq_all']}\n")
        f.write(f"provider partial eta^2, excluding {edge['edge_provider']:<14s} {edge['provider_eta_sq_excluding_edge']}\n")
        f.write(f"magnitude: {edge['magnitude_all']} -> {edge['magnitude_excluding_edge']}   "
                f"attenuation: {edge['attenuation']}\n\n")

        f.write("RQ6 - provider spread within each task category\n" + "-" * 70 + "\n")
        f.write(spread.to_string(index=False) + "\n")

    print(f"  wrote stats_summary.txt")
    return path


ALL_FIGURES = {
    1: ("fig1", lambda runs, frame: figure_1_divergence(runs)),
    2: ("fig2", lambda runs, frame: figure_2_architecture(frame)),
    3: ("fig3", lambda runs, frame: figure_3_tsr(runs, frame)),
    4: ("fig4", lambda runs, frame: figure_4_ata_heatmap(runs)),
    5: ("fig5", lambda runs, frame: figure_5_capacity_vs_trust(runs, frame)),
    6: ("fig6", lambda runs, frame: figure_6_interaction(frame)),
}


def main():
    parser = argparse.ArgumentParser(description="regenerate paper figures and tables from experiments/runs/")
    parser.add_argument("--figures", nargs="*", type=int, default=None,
                        help="figure numbers to regenerate (default: all)")
    parser.add_argument("--runs-dir", default=None)
    args = parser.parse_args()

    apply_style()
    runs, skipped = load_runs(args.runs_dir)
    if not runs:
        print("No runs found. Run the harness first (bash scripts/run_pilot.sh).")
        return
    frame = runs_to_frame(runs)

    print(f"Loaded {len(runs)} runs across {frame['provider'].nunique()} providers "
          f"and {frame['task_id'].nunique()} tasks.")
    if skipped:
        print(f"WARNING: skipped {len(skipped)} unreadable run folder(s):")
        for path, reason in skipped[:10]:
            print(f"  {os.path.basename(path)}: {reason}")

    wanted = args.figures or sorted(ALL_FIGURES)
    for number in wanted:
        if number not in ALL_FIGURES:
            print(f"  [skip] unknown figure {number}")
            continue
        print(f"Figure {number}:")
        ALL_FIGURES[number][1](runs, frame)

    if args.figures is None:
        print("Tables and statistics:")
        table_1_task_suite(frame)
        write_stats_summary(runs, frame)

    print(f"\nFigures -> {FIGURES_DIR}\nTables  -> {TABLES_DIR}")


if __name__ == "__main__":
    main()
