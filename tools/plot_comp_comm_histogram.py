#!/usr/bin/env python3
"""Plot computation/communication timing histograms vs number of GPUs.

This script uses the summarized results table produced from:
run_extract_comp_comm.sh

It plots 4 bar types:
- llama-500M computation
- llama-500M communication
- llama-1.3B computation
- llama-1.3B communication

Color encodes model, hatch pattern encodes computation vs communication.
It also overlays strong-scaling efficiency curves for the two models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def _choose_font_family() -> str:
    preferred = (
        "Times New Roman",
        "Times",
        "Nimbus Roman No9 L",
        "DejaVu Serif",
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred:
        if font_name in available:
            return font_name
    return "serif"


PLOT_FONT_FAMILY = _choose_font_family()
matplotlib.rcParams["font.family"] = PLOT_FONT_FAMILY

axis_font = {"fontname": PLOT_FONT_FAMILY, "fontsize": 24}
legend_font = font_manager.FontProperties(
    family=PLOT_FONT_FAMILY,
    style="normal",
    size=14,
)
ticks_font = {"fontname": PLOT_FONT_FAMILY, "fontsize": 22}


def _apply_ticks_font(ax) -> None:
    ax.tick_params(axis="both", labelsize=ticks_font["fontsize"])
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname(ticks_font["fontname"])


def _strong_scaling_efficiency(total_times: List[float], gpus: List[int]) -> List[float]:
    base_gpu = gpus[0]
    base_time = total_times[0]
    efficiencies: List[float] = []
    for gpu, total_time in zip(gpus, total_times):
        ideal_speedup = gpu / base_gpu
        actual_speedup = base_time / total_time
        efficiencies.append((actual_speedup / ideal_speedup) * 100.0)
    return efficiencies


RESULTS: Dict[str, Dict[str, Dict[str, List[float]]]] = {
    "baseline": {
        "llama-500M": {
            "computation": [7641.176, 3844.888, 2015.251, 969.977],
            "communication": [851.556, 998.918, 1070.430, 1153.109],
        },
        "llama-1.3B": {
            "computation": [1901.497, 970.636, 505.049, 258.183],
            "communication": [2223.410, 2585.913, 2825.554, 2952.411],
        },
    },
    "dist-optm": {
        "llama-500M": {
            "computation": [7752.739, 3864.752, 1997.306, 968.497],
            "communication": [647.234, 761.111, 825.369, 892.914],
        },
        "llama-1.3B": {
            "computation": [1899.251, 949.508, 490.164, 250.203],
            "communication": [1663.144, 1987.380, 2136.466, 2232.581],
        },
    },
}

GPUS = [4, 8, 16, 32]

MODEL_COLORS = {
    "llama-500M": "#BBD5E7",
    "llama-1.3B": "#FCD7AF",
}

METRIC_HATCH = {
    "computation": "",
    "communication": "/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot computation/communication histogram from summarized table results."
    )
    parser.add_argument(
        "--method",
        choices=["baseline", "dist-optm"],
        default="dist-optm",
        help="Which method to plot (default: dist-optm).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Default: ./comp_comm_hist_<method>.png",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI (default: 300).",
    )
    return parser.parse_args()


def build_legend_handles() -> List[Patch]:
    # Matplotlib packs legend items column-first with ncol=2, so this order gives:
    # col1: llama-500M, llama-1.3B
    # col2: Computation, Communication
    return [
        Patch(facecolor=MODEL_COLORS["llama-500M"], edgecolor="black", label="llama-500M"),
        Patch(facecolor=MODEL_COLORS["llama-1.3B"], edgecolor="black", label="llama-1.3B"),
        Patch(facecolor="white", edgecolor="black", hatch=METRIC_HATCH["computation"], label="Computation"),
        Patch(facecolor="white", edgecolor="black", hatch=METRIC_HATCH["communication"], label="Communication"),
    ]


def plot_histogram(method: str, output_path: Path, dpi: int) -> None:
    method_results = RESULTS[method]

    x = np.arange(len(GPUS), dtype=float)
    width = 0.34

    fig = plt.figure(figsize=(10.2, 6.0), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 1, height_ratios=[0.16, 1.0], hspace=0.02)
    legend_ax = fig.add_subplot(grid_spec[0])
    ax = fig.add_subplot(grid_spec[1])

    bar_containers = []
    model_offsets = {
        "llama-500M": -0.20,
        "llama-1.3B": 0.20,
    }
    for model, offset in model_offsets.items():
        comp_values = np.array(method_results[model]["computation"], dtype=float)
        comm_values = np.array(method_results[model]["communication"], dtype=float)

        comp_bars = ax.bar(
            x + offset,
            comp_values,
            width=width,
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.8,
            hatch=METRIC_HATCH["computation"],
        )
        comm_bars = ax.bar(
            x + offset,
            comm_values,
            width=width,
            bottom=comp_values,
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.8,
            hatch=METRIC_HATCH["communication"],
        )
        bar_containers.extend([comp_bars, comm_bars])

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in GPUS])
    ax.set_xlabel("Number of GPUs", **axis_font)
    ax.set_ylabel("Time (ms)", **axis_font)
    _apply_ticks_font(ax)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, color="0.6", alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.75)

    #     all_bar_heights = [bar.get_height() for bars in bar_containers for bar in bars]
    #     label_offset = 0.012 * max(all_bar_heights) if all_bar_heights else 0.0
    # 
    #     for bars in bar_containers:
    #         for bar in bars:
    #             height = bar.get_height()
    #             ax.text(
    #                 bar.get_x() + bar.get_width() / 2.0,
    #                 height + label_offset,
    #                 f"{height:.0f}",
    #                 ha="center",
    #                 va="bottom",
    #                 fontname=ticks_font["fontname"],
    #                 fontsize=ticks_font["fontsize"],
    #                 fontweight="bold",
    #             )
    # 
    #     if all_bar_heights:
    #         ax.set_ylim(top=max(all_bar_heights) + 4.0 * label_offset)

    total_500m = [
        c + m
        for c, m in zip(
            method_results["llama-500M"]["computation"],
            method_results["llama-500M"]["communication"],
        )
    ]
    total_13b = [
        c + m
        for c, m in zip(
            method_results["llama-1.3B"]["computation"],
            method_results["llama-1.3B"]["communication"],
        )
    ]
    eff_500m = _strong_scaling_efficiency(total_500m, GPUS)
    eff_13b = _strong_scaling_efficiency(total_13b, GPUS)

    ax_eff = ax.twinx()
    line_500m, = ax_eff.plot(
        x,
        eff_500m,
        color="#78A9CD",
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.4,
        linewidth=2.0,
        linestyle="-",
        label="llama-500M efficiency",
    )
    line_13b, = ax_eff.plot(
        x,
        eff_13b,
        color="#F8AB61",
        marker="s",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.4,
        linewidth=2.0,
        linestyle="-",
        label="llama-1.3B efficiency",
    )
    ax_eff.axhline(100.0, color="0.35", linestyle=":", linewidth=1.0)
    ax_eff.set_ylabel("Strong Scaling Efficiency (%)", **axis_font)
    _apply_ticks_font(ax_eff)
    ax_eff.grid(False)
    all_eff = eff_500m + eff_13b
    max_eff = max(all_eff)
    min_eff = min(all_eff)
    ax_eff.set_ylim(max(0.0, min_eff - 12.0), max(105.0, max_eff + 8.0))

    red_offsets = [(0, 10), (0, 10), (0, 10), (0, 10)]
    blue_offsets = [(12, -10), (0, 10), (0, 10), (0, 10)]

    for (x_i, y_i), (x_off, y_off) in zip(zip(x, eff_500m), red_offsets):
        ax_eff.annotate(
            f"{y_i:.1f}%",
            (x_i, y_i),
            textcoords="offset points",
            xytext=(x_off, y_off),
            ha="center",
            va="bottom" if y_off >= 0 else "top",
            fontname=ticks_font["fontname"],
            fontsize=max(8, ticks_font["fontsize"] - 1),
            fontweight="bold",
            color="#78A9CD",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#78A9CD",
                "linewidth": 0.8,
                "alpha": 0.95,
            },
            zorder=5,
        )

    for (x_i, y_i), (x_off, y_off) in zip(zip(x, eff_13b), blue_offsets):
        ax_eff.annotate(
            f"{y_i:.1f}%",
            (x_i, y_i),
            textcoords="offset points",
            xytext=(x_off, y_off),
            ha="center",
            va="bottom" if y_off >= 0 else "top",
            fontname=ticks_font["fontname"],
            fontsize=max(8, ticks_font["fontsize"] - 1),
            fontweight="bold",
            color="#F8AB61",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#F8AB61",
                "linewidth": 0.8,
                "alpha": 0.95,
            },
            zorder=5,
        )
    legend_ax.axis("off")
    legend_handles = build_legend_handles()
    legend_labels = [h.get_label() for h in legend_handles]
    legend = legend_ax.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
        prop=legend_font,
        columnspacing=1.8,
        handletextpad=0.8,
        handlelength=2.6,
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_facecolor("white")

    eff_legend = ax_eff.legend(
        handles=[line_500m, line_13b],
        loc="upper right",
        frameon=True,
        prop=legend_font,
    )
    eff_legend.get_frame().set_edgecolor("black")
    eff_legend.get_frame().set_facecolor("white")

    fig.tight_layout(rect=(0, 0.0, 1, 0.86))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        bbox_extra_artists=[legend, eff_legend],
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.output:
        output_path = Path(args.output).expanduser()
    else:
        output_path = Path.cwd() / f"comp_comm_hist_{args.method}.png"

    plot_histogram(method=args.method, output_path=output_path, dpi=args.dpi)
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
