"""Plot grouped bar charts for SCAPE step-time benchmarks."""

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
    size=18,
)
ticks_font = {"fontname": PLOT_FONT_FAMILY, "fontsize": 24}


def _apply_ticks_font(ax):
    ax.tick_params(axis="both", labelsize=ticks_font["fontsize"])
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname(ticks_font["fontname"])


GPU_COUNTS = [4, 8, 16, 32, 64]

SERIES = [
    ("AdamS", [14026.968, 9110.506, 6582.949, 5447.671, 4987.218]),
    ("AdamS w/ dist-optm", [13372.755, 8277.200, 5706.667, 4424.497, 3890.172]),
    (r"SCAPE ($d$=0.1)", [11788.137, 6348.521, 3706.622, 2432.346, 1983.079]),
    (r"SCAPE ($d$=0.1) w/ dist-optm", [11566.835, 6330.285, 3670.689, 2427.945, 1950.844]),
    (
        r"SCAPE ($d$=0.1) w/ dist-optm & CPUOffload",
        [11601.631, 6316.475, 3720.530, 2449.380, 1970.447],
    ),
    (r"SCAPE ($d$=0.01)", [11275.744, 5879.909, 3201.328, 1909.330, 1473.452]),
    (r"SCAPE ($d$=0.01) w/ dist-optm", [11070.216, 5746.153, 3058.312, 1743.110, 1216.384]),
    (
        r"SCAPE ($d$=0.01) w/ dist-optm & CPUOffload",
        [11271.244, 5727.574, 3084.744, 1797.297, 1279.944],
    ),
]

HATCHES = ["", "//", "xx", "..", "oo", "++", "--", "||"]
FACECOLORS = [
    "#EF9A9A",
    "#FFCDD2",
    "#81C784",
    "#A5D6A7",
    "#C8E6C9",
    "#90CAF9",
    "#BBDEFB",
    "#E3F2FD",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the benchmark table as a grouped bar chart."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("step_time_histogram_bw.png"),
        help="Path to the output figure. Default: %(default)s",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster DPI used when saving bitmap formats. Default: %(default)s",
    )
    parser.add_argument(
        "--legend",
        choices=["on", "off"],
        default="on",
        help="Turn legend rendering on or off. Default: %(default)s",
    )
    return parser.parse_args()


def bar_offsets(num_groups: int, num_series: int, bar_width: float) -> list[list[float]]:
    half_span = (num_series - 1) * bar_width / 2.0
    offsets: list[list[float]] = []
    for series_idx in range(num_series):
        shift = series_idx * bar_width - half_span
        offsets.append([group_idx + shift for group_idx in range(num_groups)])
    return offsets


def plot_grouped_bars(output_path: Path, dpi: int, legend_mode: str) -> None:
    num_groups = len(GPU_COUNTS)
    num_series = len(SERIES)
    bar_width = 0.1
    x_positions = list(range(num_groups))
    offsets = bar_offsets(num_groups, num_series, bar_width)

    base_fig_width = 14
    base_fig_height = 6
    legend_height_ratio = 0.18
    width_ratios = [0.11, 2.96, 0.11]

    if legend_mode == "on":
        fig = plt.figure(
            figsize=(base_fig_width, base_fig_height * (1.0 + legend_height_ratio)),
            constrained_layout=True,
        )
        grid_spec = fig.add_gridspec(
            2,
            3,
            height_ratios=[legend_height_ratio, 1.0],
            width_ratios=width_ratios,
        )
        legend_ax = fig.add_subplot(grid_spec[0, 1])
        ax = fig.add_subplot(grid_spec[1, 1])
    else:
        fig = plt.figure(figsize=(base_fig_width, base_fig_height), constrained_layout=True)
        grid_spec = fig.add_gridspec(1, 3, width_ratios=width_ratios)
        legend_ax = None
        ax = fig.add_subplot(grid_spec[0, 1])
    handle_map = {}

    for series_idx, (label, values) in enumerate(SERIES):
        bars = ax.bar(
            offsets[series_idx],
            values,
            width=bar_width,
            label=label,
            color=FACECOLORS[series_idx % len(FACECOLORS)],
            edgecolor="black",
            linewidth=1.0,
            hatch=HATCHES[series_idx % len(HATCHES)],
        )
        handle_map[label] = bars[0]

    ax.set_xlabel("Number of GPUs", **axis_font)
    ax.set_ylabel("Time per Step (ms)", **axis_font)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(count) for count in GPU_COUNTS])
    _apply_ticks_font(ax)
    ax.set_axisbelow(True)
    ax.grid(axis="both", linestyle="--", linewidth=0.6, color="0.6", alpha=0.8)
    ax.set_ylim(bottom=0)

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
            loc="lower center",
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
    plot_grouped_bars(args.output, args.dpi, args.legend)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
