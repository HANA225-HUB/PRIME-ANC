"""Rebuild Fig. 2 from bundled numerical results."""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
STYLE = {
    "neural": ("PRIME-ANC", "#0072B2", "o", "-", True),
    "anchor": ("Ratio-base", "#009E73", "s", "-", False),
    "direct_ratio": ("Projected ratio", "#D55E00", "^", "--", False),
    "fomaml": ("FOMAML", "#A43E91", "D", ":", False),
    "fxlms_weighted": ("Weighted FxLMS", "#73797D", "x", "-.", False),
}
TITLE = {
    "tenpath": "(a) Ten-path: 3 support / 7 held",
    "pandar": "(b) PANDAR: 7 support / 16 held",
}
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9.8,
        "axes.labelsize": 9.8,
        "axes.titlesize": 10,
        "axes.linewidth": 0.72,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.025, dpi=320
        )
    plt.close(fig)


def decorate(ax, domain):
    ax.set_title(TITLE[domain], pad=3, fontweight="bold")
    ax.grid(axis="y", color="#D9E0E4", lw=0.55, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def frequency():
    path = ROOT / "results/paper/frequency_random10.json"
    if not path.exists():
        print("Frequency data file is missing.")
        return
    data = json.loads(path.read_text())["datasets"]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.25), sharex=True)
    for ax, domain in zip(axes, ("tenpath", "pandar")):
        for low, high, color, alpha in (
            (20, 50, "#F2F2F2", 0.72),
            (50, 5000, "#DCEBF2", 0.38),
            (5000, 8000, "#F7E6D6", 0.64),
            (8000, 20000, "#F2F2F2", 0.72),
        ):
            ax.axvspan(low, high, color=color, alpha=alpha, lw=0, zorder=0)
        for boundary in (50, 5000, 8000):
            ax.axvline(boundary, color="#A9B0B4", lw=0.48, ls=(0, (3, 2)), zorder=0.5)
        lower, upper = [], []
        for key, (label, color, marker, ls, fill) in STYLE.items():
            if key not in data[domain]:
                continue
            d = data[domain][key]
            x = np.asarray(d["centers_hz"], dtype=float)
            y = np.asarray(d["mean_band_nr_db"], dtype=float)
            sd = np.asarray(d["sd_band_nr_db"], dtype=float)
            show = (x >= 25) & (x <= 20000) & np.isfinite(y) & np.isfinite(sd)
            x, y, sd = x[show], y[show], sd[show]
            lower.append(float((y - sd).min()))
            upper.append(float((y + sd).max()))
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.10, lw=0)
            ax.plot(
                x,
                y,
                color=color,
                lw=1.45,
                ls=ls,
                marker=marker,
                ms=3.4,
                markevery=2,
                mfc=color if fill else "white",
                mec=color,
                mew=0.8,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_xlim(20, 20000)
        ymin = min(-12 if domain == "tenpath" else -8, np.floor(min(lower) - 1))
        ymax = max(36 if domain == "tenpath" else 30, np.ceil(max(upper) + 1))
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(np.arange(np.ceil(ymin / 10) * 10, ymax + 1, 10))
        ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 8000, 20000])
        ax.set_xticklabels(
            ["20", "50", "100", "200", "500", "1k", "2k", "5k", "8k", "20k"]
        )
        ax.tick_params(axis="both", labelsize=9.6)
        decorate(ax, domain)
    axes[0].set_ylabel("Noise reduction (dB)")
    fig.supxlabel("Frequency (Hz)", fontsize=9.8, y=0.018)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.998),
        ncol=5,
        frameon=True,
        fancybox=False,
        edgecolor="#C4CDD2",
        borderpad=0.28,
        columnspacing=0.80,
        handlelength=1.75,
        handletextpad=0.40,
        fontsize=9.2,
    )
    # Relate the background regions to the two frequency-selective loss terms.
    bands = [
        Patch(facecolor=color, edgecolor="#BEC5C9", label=label)
        for color, label in [
            ("#F2F2F2", "20-50 Hz: diagnostic"),
            ("#DCEBF2", "50 Hz-5 kHz: NR + AMP"),
            ("#F7E6D6", "5-8 kHz: AMP only"),
            ("#F2F2F2", "8-20 kHz: diagnostic"),
        ]
    ]
    fig.legend(
        handles=bands,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=4,
        frameon=False,
        handlelength=0.9,
        handleheight=0.65,
        handletextpad=0.35,
        columnspacing=0.75,
        fontsize=9,
    )
    # Reserve one compact row for the key, keeping almost the same axes height.
    fig.subplots_adjust(left=0.073, right=0.995, bottom=0.226, top=0.70, wspace=0.16)
    save(fig, "main_k0_frequency")


if __name__ == "__main__":
    frequency()
