"""Render the learner-facing OLM projection as three clean cards (Fig 3).

Replaces the three full-screen platform screenshots (Supporting / Reconnecting /
Advancing) with a single, crisp, vector multi-panel figure that shows exactly the
fields the OLM projects (mode label, why, system behaviour, expected action, exit
condition) for each mode. Output: ../figures/fig3_olm_cards.{pdf,png}.
"""

from __future__ import annotations

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_FIG_DIR = os.path.join(os.path.dirname(ROOT), "figures")
CODE_FIG_DIR = os.path.join(ROOT, "figures")
for d in (PAPER_FIG_DIR, CODE_FIG_DIR):
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

CARDS = [
    {
        "panel": "(a)",
        "mode": "Supporting",
        "sub": "STRUGGLING mode",
        "colour": "#0072B2",
        "fields": [
            ("Why", "You may benefit from extra support in areas where progress has been slower."),
            ("What the system will do", "Offer targeted, bounded support and avoid overloading you."),
            ("What you should do", "Review the suggested support, then try the next problem."),
            ("What will change this", "Once you show progress, support eases off."),
        ],
    },
    {
        "panel": "(b)",
        "mode": "Reconnecting",
        "sub": "LAPSED mode",
        "colour": "#E69F00",
        "fields": [
            ("Why", "It has been a while since your last session; let's pick up where you left off."),
            ("What the system will do", "Send a brief, low-pressure prompt to help you restart."),
            ("What you should do", "Resume with a short activity to rebuild momentum."),
            ("What will change this", "Renewed activity returns the system to a normal stance."),
        ],
    },
    {
        "panel": "(c)",
        "mode": "Advancing",
        "sub": "ACCELERATING mode",
        "colour": "#009E73",
        "fields": [
            ("Why", "You are making strong progress and can take on more."),
            ("What the system will do", "Suggest stretch goals while you keep momentum."),
            ("What you should do", "Take on the suggested stretch goal when ready."),
            ("What will change this", "If your pace slows, the system rebalances."),
        ],
    },
]


def draw_card(ax, card):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    col = card["colour"]
    # Card body.
    ax.add_patch(FancyBboxPatch((0.04, 0.03), 0.92, 0.94,
                                boxstyle="round,pad=0.0,rounding_size=0.03",
                                linewidth=1.1, edgecolor="#B0B7BD", facecolor="white",
                                mutation_aspect=1.0, zorder=1))
    # Header band.
    ax.add_patch(FancyBboxPatch((0.04, 0.83), 0.92, 0.14,
                                boxstyle="round,pad=0.0,rounding_size=0.03",
                                linewidth=0, facecolor=col, zorder=2))
    ax.text(0.08, 0.905, card["mode"], color="white", fontsize=14, fontweight="bold",
            va="center", zorder=3)
    ax.text(0.92, 0.905, card["panel"], color="white", fontsize=11, va="center",
            ha="right", zorder=3)
    ax.text(0.08, 0.862, card["sub"], color="white", fontsize=8, va="center",
            zorder=3, alpha=0.95)
    # Fields.
    y = 0.78
    for name, value in card["fields"]:
        ax.text(0.08, y, name, color=col, fontsize=9.5, fontweight="bold", va="top")
        wrapped = textwrap.fill(value, width=34)
        nlines = wrapped.count("\n") + 1
        ax.text(0.08, y - 0.045, wrapped, color="#222222", fontsize=9, va="top",
                linespacing=1.25)
        y -= 0.045 + nlines * 0.052 + 0.028


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
    for ax, card in zip(axes, CARDS):
        draw_card(ax, card)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.08)
    for ext, d in (("pdf", PAPER_FIG_DIR), ("png", CODE_FIG_DIR)):
        fig.savefig(os.path.join(d, f"fig3_olm_cards.{ext}"), bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print("wrote fig3_olm_cards.pdf/png")


if __name__ == "__main__":
    main()
