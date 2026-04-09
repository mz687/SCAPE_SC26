"""Plot strong scaling efficiency curves for SCAPE step-time benchmarks."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt


def _choose_font_family():
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

axis_font = {"fontname": PLOT_FONT_FAMILY, "fontsize": 26}
legend_font = font_manager.FontProperties(
    family=PLOT_FONT_FAMILY,
    style="normal",
    size=18.4,
)
ticks_font = {"fontname": PLOT_FONT_FAMILY, "fontsize": 24}


def _apply_ticks_font(ax):
    ax.tick_params(axis="both", labelsize=ticks_font["fontsize"])
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname(ticks_font["fontname"])


GPU_COUNTS = [4, 8, 16, 32]

SERIES = [
    ("AdamS", [4108.618, 3512.179, 3284.186, 3209.108]),
    ("AdamS w/ dist-optm", [3533.861, 2872.987, 2586.799, 2449.124]),
    (r"SCAPE ($d$=0.1)", [2350.108, 1471.044, 1077.386, 939.491]),
    (r"SCAPE ($d$=0.1) w/ dist-optm", [2482.778, 1621.271, 1254.008, 1256.732]),
    (
        r"SCAPE ($d$=0.1) w/ dist-optm & CPUOffload",
        [2509.014, 1638.641, 1304.381, 1158.430],
    ),
    (r"SCAPE ($d$=0.01)", [2131.106, 1242.153, 863.740, 632.725]),
    (r"SCAPE ($d$=0.01) w/ dist-optm", [2066.784, 1192.304, 776.442, 607.993]),
    (
        r"SCAPE ($d$=0.01) w/ dist-optm & CPUOffload",
        [2090.083, 1180.006, 787.959, 575.508],
    ),
]

COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#9C755F",
]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "h"]
LINESTYLES = ["-", "-", "-", "--", "--", "-.", "-.", ":"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot strong scaling efficiency curves from the benchmark table."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/strong_scaling_efficiency.png"),
        help="Path to the output figure. Default: %(default)s",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Raster DPI used when saving bitmap formats. Default: %(default)s",
    )
    parser.add_argument(
        "--legend",
        choices=["on", "off"],
        default="on",
        help="Turn legend rendering on or off. Default: %(default)s",
    )
    return parser.parse_args()


def compute_efficiency(step_times_ms: list[float]) -> list[float]:
    baseline_gpu_count = float(GPU_COUNTS[0])
    baseline_time_ms = float(step_times_ms[0])
    efficiencies = []
    for gpu_count, time_ms in zip(GPU_COUNTS, step_times_ms):
        efficiency = (baseline_time_ms * baseline_gpu_count) / (float(time_ms) * float(gpu_count))
        efficiencies.append(efficiency * 100.0)
    return efficiencies


def plot_strong_scaling_efficiency(output_path: Path, dpi: int, legend_mode: str) -> None:
    base_fig_width = 13
    base_fig_height = 6
    legend_height_ratio = 0.22

    if legend_mode == "on":
        fig = plt.figure(
            figsize=(base_fig_width, base_fig_height * (1.0 + legend_height_ratio)),
            constrained_layout=True,
        )
        grid_spec = fig.add_gridspec(2, 1, height_ratios=[legend_height_ratio, 1.0])
        legend_ax = fig.add_subplot(grid_spec[0])
        ax = fig.add_subplot(grid_spec[1])
    else:
        fig = plt.figure(figsize=(base_fig_width, base_fig_height), constrained_layout=True)
        legend_ax = None
        ax = fig.add_subplot(111)
    handle_map = {}

    for series_idx, (label, values) in enumerate(SERIES):
        efficiencies = compute_efficiency(values)
        (line_handle,) = ax.plot(
            GPU_COUNTS,
            efficiencies,
            color=COLORS[series_idx % len(COLORS)],
            linestyle=LINESTYLES[series_idx % len(LINESTYLES)],
            linewidth=3.0,
            marker=MARKERS[series_idx % len(MARKERS)],
            markersize=9.0,
            markeredgecolor="black",
            markeredgewidth=0.8,
            label=label,
        )
        handle_map[label] = line_handle

    ax.axhline(100.0, color="0.4", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Number of GPUs", **axis_font)
    ax.set_ylabel("Strong Scaling Efficiency (%)", **axis_font)
    ax.set_xscale("log", base=2)
    ax.set_xticks(GPU_COUNTS)
    ax.set_xticklabels([str(count) for count in GPU_COUNTS])
    ax.set_ylim(0.0, 105.0)
    _apply_ticks_font(ax)
    ax.set_axisbelow(True)
    ax.grid(axis="both", linestyle="--", linewidth=0.6, color="0.6", alpha=0.8)

    legend_labels = [label for label, _ in SERIES]
    legend_handles = [handle_map[label] for label in legend_labels]
    legend_ncol = 2

    legend_artists = []
    if legend_mode == "on":
        assert legend_ax is not None
        legend_ax.axis("off")
        legend = legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="center",
            ncol=legend_ncol,
            frameon=True,
            prop=legend_font,
            columnspacing=1.8,
            handletextpad=0.8,
        )
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_facecolor("white")
        legend_artists.append(legend)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": dpi, "bbox_inches": "tight"}
    if legend_artists:
        save_kwargs["bbox_extra_artists"] = legend_artists
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_strong_scaling_efficiency(args.output, args.dpi, args.legend)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
