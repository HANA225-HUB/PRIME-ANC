"""Rebuild Fig. 3: common GN, GPU PCG budgets, and the CPU direct solve point."""

import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "figures").mkdir(exist_ok=True)
STYLES = {
    "prism": ("PRIME-ANC", "#0072B2", "o", "-", True),
    "minimum_phase": ("Ratio-base", "#009E73", "s", "-", False),
    "direct_ratio": ("Projected ratio", "#D55E00", "^", "--", False),
    "fomaml": ("FOMAML", "#A43E91", "D", ":", False),
    "wmmse_gpu": ("WMMSE-PCG (GPU)", "#686E75", "x", "--", False),
    "wmmse_cpu": ("WMMSE (CPU)", "#4F555C", "D", "None", False),
}
CPU_MARKER_SIZE = 2.4
CPU_LEGEND_MARKER_SIZE = 2.7
CPU_MARKER_EDGE_WIDTH = 1.05
LEGEND_OPTICAL_LOWERING_PT = 0.15
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10.3,
        "axes.labelsize": 10.3,
        "axes.titlesize": 10.3,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.hashsalt": "PRIME_ANC_refinement",
    }
)


def line(ax, x, y, sd, key):
    label, color, marker, style, filled = STYLES[key]
    ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.13, linewidth=0, zorder=1)
    ax.plot(
        x,
        y,
        color=color,
        marker=marker,
        linestyle=style,
        linewidth=1.15,
        markersize=3.15,
        markeredgewidth=0.75,
        markerfacecolor=color if filled else "white",
        markeredgecolor=color,
        label=label,
        zorder=3,
    )


class CpuLegendHandler(HandlerLine2D):
    """Optically align only the legend glyph; data markers stay centered."""

    def create_artists(self, *args, **kwargs):
        artists = super().create_artists(*args, **kwargs)
        for artist in artists:
            artist.set_ydata(
                np.asarray(artist.get_ydata()) - LEGEND_OPTICAL_LOWERING_PT
            )
        return artists


def main():
    gn = json.loads((ROOT / "results/paper/gn_analysis.json").read_text())["datasets"]
    curve = list(
        csv.DictReader((ROOT / "results/paper/quality_time_isolated.csv").open())
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(7.04, 2.12), gridspec_kw={"width_ratios": [1, 1, 1.15]}
    )
    plotted = {
        "common_gn": [],
        "time_curves": [],
        "direct_point": None,
        "wmmse_gn": "Separate Table 3 control; not part of the PCG curve.",
    }
    for ax, domain, title in zip(
        axes[:2], ["tenpath", "pandar"], ["(a) Ten-path: 3/7", "(b) PANDAR: 7/16"]
    ):
        ax.axvspan(-0.12, 0.12, color="#EDF4F7", alpha=0.9, zorder=0)
        ax.axvline(0, color="#8AA1AC", linewidth=0.5, linestyle=(0, (2, 2)))
        for key in ["prism", "minimum_phase", "direct_ratio", "fomaml"]:
            rows = gn[domain]["methods"][key]
            x = np.arange(4)
            y = np.array([rows[str(k)]["nr_db"]["mean"] for k in range(4)])
            sd = np.array([rows[str(k)]["nr_db"]["sd_across_splits"] for k in range(4)])
            line(ax, x, y, sd, key)
            plotted["common_gn"].append(
                {
                    "domain": domain,
                    "method": key,
                    "K": x.tolist(),
                    "NR": y.tolist(),
                    "SD": sd.tolist(),
                }
            )
        ax.set_xlim(-0.2, 3.18)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_ylim((3, 28) if domain == "tenpath" else (5, 24))
        ax.set_yticks([5, 10, 15, 20, 25] if domain == "tenpath" else [5, 10, 15, 20])
        ax.set_xlabel("GN updates $K$", labelpad=2)
        ax.set_title(title, fontweight="bold", pad=5)
    for method, key in [
        ("PRIME+GN", "prism"),
        ("Ratio-base+GN", "minimum_phase"),
        ("WMMSE-FFT-PCG", "wmmse_gpu"),
    ]:
        rows = sorted(
            (r for r in curve if r["method"] == method), key=lambda r: int(r["budget"])
        )
        assert rows and all(float(r["time_ms"]) > 0 for r in rows)
        x = np.array([float(r["time_ms"]) for r in rows])
        y = np.array([float(r["nr_db"]) for r in rows])
        sd = np.array([float(r["nr_db_split_sd"]) for r in rows])
        line(axes[2], x, y, sd, key)
        plotted["time_curves"].append(
            {
                "method": method,
                "budget": [int(r["budget"]) for r in rows],
                "time_ms": x.tolist(),
                "NR": y.tolist(),
                "SD": sd.tolist(),
            }
        )
    direct = [r for r in curve if r["method"] == "WMMSE-Direct"]
    assert len(direct) == 1 and int(direct[0]["budget"]) == 0
    d = direct[0]
    axes[2].plot(
        float(d["time_ms"]),
        float(d["nr_db"]),
        color=STYLES["wmmse_cpu"][1],
        marker=STYLES["wmmse_cpu"][2],
        markersize=CPU_MARKER_SIZE,
        markerfacecolor="white",
        markeredgewidth=CPU_MARKER_EDGE_WIDTH,
        linestyle="None",
        zorder=5,
    )
    plotted["direct_point"] = {
        k: float(d[k])
        for k in ["time_ms", "nr_db", "nr_db_split_sd", "time_ms_split_sd"]
    }
    axes[2].set_xscale("log")
    axes[2].set_xlim(0.07, 40)
    axes[2].set_xticks([0.1, 1, 10])
    axes[2].set_xticklabels(["0.1", "1", "10"])
    axes[2].set_ylim(8, 24)
    axes[2].set_yticks([10, 15, 20])
    axes[2].set_title("(c) PANDAR: design time", fontweight="bold", pad=5)
    axes[2].set_xlabel("Core design time (ms)", labelpad=2)
    axes[0].set_ylabel("Target-band NR (dB)", labelpad=3)
    for ax in axes:
        ax.grid(axis="y", color="#D9E0E4", linewidth=0.55, alpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10.3, pad=2, length=3)
    handles = [
        Line2D(
            [],
            [],
            label=label,
            color=color,
            marker=marker,
            linestyle=style,
            linewidth=1.2,
            markersize=CPU_LEGEND_MARKER_SIZE if key == "wmmse_cpu" else 3.5,
            markeredgewidth=CPU_MARKER_EDGE_WIDTH if key == "wmmse_cpu" else 0.75,
            markerfacecolor=color if filled else "white",
        )
        for key, (label, color, marker, style, filled) in STYLES.items()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.51, 1.00),
        ncol=6,
        frameon=True,
        fancybox=False,
        edgecolor="#C4CDD2",
        borderpad=0.24,
        handlelength=1.35,
        handletextpad=0.32,
        columnspacing=0.65,
        fontsize=10.3,
        handler_map={handles[-1]: CpuLegendHandler()},
    )
    fig.subplots_adjust(left=0.066, right=0.992, bottom=0.24, top=0.75, wspace=0.30)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(
            ROOT / "figures" / f"refinement_quality_time.{ext}",
            bbox_inches="tight",
            pad_inches=0.025,
            dpi=260,
            metadata={"CreationDate": None, "ModDate": None}
            if ext == "pdf"
            else {"Date": None}
            if ext == "svg"
            else None,
        )
    plt.close(fig)
    (ROOT / "results/FIG3_PLOTTED_DATA.json").write_text(
        json.dumps(plotted, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
