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
    ("AdamS", [8691.338, 4904.025, 3047.295, 2150.509, 1753.660]),
    ("AdamS w/ dist-optm", [8446.571, 4696.993, 2809.307, 1892.287, 1505.554]),
    (r"SCAPE ($d$=0.1)", [7948.528, 4190.240, 2253.698, 1326.400, 1077.404]),
    (r"SCAPE ($d$=0.1) w/ dist-optm", [7952.240, 4194.192, 2290.764, 1417.444, 1220.763]),
    (
        r"SCAPE ($d$=0.1) w/ dist-optm & CPUOffload",
        [7978.089, 4172.545, 2297.055, 1398.886, 1250.710],
    ),
    (r"SCAPE ($d$=0.01)", [7927.775, 4092.057, 2155.387, 1234.875, 908.190]),
    (r"SCAPE ($d$=0.01) w/ dist-optm", [8072.750, 3986.416, 2235.580, 1246.357, 830.583]),
    (
        r"SCAPE ($d$=0.01) w/ dist-optm & CPUOffload",
        [7900.234, 3978.836, 2125.496, 1201.943, 775.217],
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
        default=Path("step_time_histogram_500M.png"),
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
