"""Plot memory usage bars in separate figures for each model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
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

axis_font = {"fontname": PLOT_FONT_FAMILY, "fontsize": 30}
legend_font = font_manager.FontProperties(
    family=PLOT_FONT_FAMILY,
    style="normal",
    size=24,
)
ticks_font = {"fontname": PLOT_FONT_FAMILY, "fontsize": 24}

METHODS = [
    "AdamS",
    "AdamS+DO",
    "SCAPE",
    "SCAPE+res.",
    "SCAPE+DO",
    "SCAPE+DO+res.",
]

HAS_RES_OFFLOAD = [False, False, False, True, False, True]
REFERENCE_INDICES = {0, 1}
DIST_OPTM_INDICES = {1, 4, 5}

MODEL_VALUES = {
    "llama-500M": [71812.00, 70914.00, 76344.00, 74760.00, 77722.00, 76620.00],
    "llama-1.3B": [49460.00, 40922.00, 54026.00, 49392.00, 60486.00, 55852.00],
    "llama-1.8B": [90366.00, 81176.00, None, 93256.00, None, 93354.00],
}

REF_COLOR = "#FFCDD2"
NO_RES_COLOR = "#C8E6C9"
RES_COLOR = "#E3F2FD"
OOM_COLOR = "#C62828"
BAR_LABEL_FONTSIZE = 20


def _apply_ticks_font(ax) -> None:
    ax.tick_params(axis="both", labelsize=ticks_font["fontsize"])
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname(ticks_font["fontname"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot memory usage as separate figures for each model. "
            "The --output path is used as the filename prefix."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("memory_usage_grouped_bar.png"),
        help=(
            "Base output path. Per-model files are saved as "
            "<stem>_<model><suffix> for each entry in MODEL_VALUES. "
            "Default: %(default)s"
        ),
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


def _plot_model_panel(ax, values: list[float | None]) -> None:
    x_positions = list(range(len(METHODS)))
    max_valid = max(float(v) for v in values if v is not None)
    oom_stub_height = max_valid * 0.018

    heights = [oom_stub_height if value is None else float(value) for value in values]
    bar_colors: list[str] = []
    bar_hatches: list[str] = []
    for idx in range(len(values)):
        if idx in REFERENCE_INDICES:
            bar_colors.append(REF_COLOR)
        elif HAS_RES_OFFLOAD[idx]:
            bar_colors.append(RES_COLOR)
        else:
            bar_colors.append(NO_RES_COLOR)
        bar_hatches.append("//" if idx in DIST_OPTM_INDICES else "")

    bars = ax.bar(
        x_positions,
        heights,
        width=0.72,
        color=bar_colors,
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )

    for idx, bar in enumerate(bars):
        bar.set_hatch(bar_hatches[idx])

    max_annotation_y = 0.0
    for idx, value in enumerate(values):
        bar = bars[idx]
        x_center = bar.get_x() + bar.get_width() / 2.0
        y_top = bar.get_height() + max_valid * 0.010

        if value is None:
            bar.set_facecolor("white")
            bar.set_edgecolor(OOM_COLOR)
            bar.set_hatch("xx")
            bar.set_linewidth(1.2)
            ax.text(
                x_center,
                y_top,
                "OOM",
                color=OOM_COLOR,
                ha="center",
                va="bottom",
                fontsize=BAR_LABEL_FONTSIZE,
                fontname=PLOT_FONT_FAMILY,
                fontweight="bold",
                zorder=4,
            )
        else:
            value_label = f"{int(round(float(value))):,}"
            ax.text(
                x_center,
                y_top,
                value_label,
                color="black",
                ha="center",
                va="bottom",
                fontsize=BAR_LABEL_FONTSIZE,
                fontname=PLOT_FONT_FAMILY,
                fontweight="bold",
                zorder=4,
            )

        max_annotation_y = max(max_annotation_y, y_top)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(METHODS, rotation=45, ha="right")
    _apply_ticks_font(ax)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, color="0.6", alpha=0.8)
    ax.set_ylim(bottom=0.0, top=max(max_valid * 1.14, max_annotation_y + max_valid * 0.05))


def _legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=REF_COLOR, edgecolor="black", linewidth=1.0, label="Reference (AdamS)"),
        Patch(facecolor=NO_RES_COLOR, edgecolor="black", linewidth=1.0, label="SCAPE (no res. offload)"),
        Patch(facecolor=RES_COLOR, edgecolor="black", linewidth=1.0, label="SCAPE (res. offload)"),
        Patch(facecolor="white", edgecolor="black", linewidth=1.0, hatch="//", label="Dist-optm enabled"),
    ]


def _plot_single_figure(output_path: Path, values: list[float | None], dpi: int, legend_mode: str) -> None:
    base_fig_width = 11.5
    base_fig_height = 8
    legend_height_ratio = 0.18

    if legend_mode == "on":
        fig = plt.figure(
            figsize=(base_fig_width, base_fig_height * (1.0 + legend_height_ratio)),
            constrained_layout=True,
        )
        grid_spec = fig.add_gridspec(2, 1, height_ratios=[legend_height_ratio, 1.0])
        legend_ax = fig.add_subplot(grid_spec[0, 0])
        ax = fig.add_subplot(grid_spec[1, 0])
    else:
        fig, ax = plt.subplots(1, 1, figsize=(base_fig_width, base_fig_height), constrained_layout=True)
        legend_ax = None

    _plot_model_panel(ax, values)
    ax.set_ylabel("Memory Usage (MB)", **axis_font)

    legend_artists = []
    if legend_mode == "on":
        assert legend_ax is not None
        legend_ax.axis("off")
        handles = _legend_handles()
        legend = legend_ax.legend(
            handles,
            [h.get_label() for h in handles],
            loc="lower center",
            ncol=2,
            frameon=True,
            prop=legend_font,
            columnspacing=2.0,
            handletextpad=0.8,
        )
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_facecolor("white")
        legend_artists.append(legend)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {"dpi": dpi, "bbox_inches": "tight"}
    if legend_artists:
        save_kwargs["bbox_extra_artists"] = legend_artists
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def plot_memory_bars(output_path: Path, dpi: int, legend_mode: str) -> list[Path]:
    suffix = output_path.suffix if output_path.suffix else ".png"
    stem = output_path.stem if output_path.stem else "memory_usage_grouped_bar"
    out_dir = output_path.parent

    outputs: list[Path] = []
    for model_label, values in MODEL_VALUES.items():
        model_out = out_dir / f"{stem}_{model_label}{suffix}"
        _plot_single_figure(model_out, values, dpi, legend_mode)
        outputs.append(model_out)
    return outputs


def main() -> None:
    args = parse_args()
    output_files = plot_memory_bars(args.output, args.dpi, args.legend)
    for path in output_files:
        print(f"Saved figure to {path}")


if __name__ == "__main__":
    main()
