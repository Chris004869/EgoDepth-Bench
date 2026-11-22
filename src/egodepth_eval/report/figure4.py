"""Fig. 4: five interaction-diagnostic panels, drawn only from measured statistics.

    (a) whole-image / hand / object / background MAE_Z on the same annotated frames
    (b) A_tau curves for hand and object, 0-10 cm, with prediction coverage
    (c) Z x angle error heatmaps with GT and prediction support
    (d) active minus low-motion difference on retained site-Z-angle strata, with CI
    (e) shared qualitative examples

NO PANEL IS EVER SIMULATED. A panel with no data draws an explicit "no data"
notice naming what is missing. There is no code path that invents a curve, and
`--allow_synthetic` does not exist.

MISSING REFERENCE AND MISSING PREDICTION ARE DRAWN DIFFERENTLY. In the heatmaps
and the qualitative columns, a cell or pixel with no reference is hatched grey and
one with reference but no prediction is cross-hatched, so "we could not measure
here" never reads as "the error is low here".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Shared colours. Kept in one place so every panel and the qualitative columns
# agree; a per-panel colour scale would make the panels non-comparable.
ROI_COLOURS = {
    "full": "#4C6EF5",
    "hand": "#E8590C",
    "object": "#2F9E44",
    "background": "#868E96",
    "unknown": "#CED4DA",
    "active_object": "#1971C2",
}
NO_REFERENCE_COLOUR = "#DEE2E6"
NO_PREDICTION_COLOUR = "#F8F0FC"
ERROR_CMAP = "viridis"


def _colormap(name: str):
    """Fetch a colormap copy across matplotlib versions.

    `matplotlib.cm.get_cmap` was removed in 3.9; `matplotlib.colormaps` is the
    replacement. Both are tried so this module works on either.
    """
    import matplotlib

    try:
        return matplotlib.colormaps[name].copy()
    except (AttributeError, KeyError):
        return matplotlib.cm.get_cmap(name).copy()


def _figure_text(axis, message: str, title: str) -> None:
    """Draw an explicit missing-data notice instead of an empty or fake panel."""
    axis.text(
        0.5, 0.5, message, ha="center", va="center", fontsize=8, wrap=True,
        transform=axis.transAxes, color="#C92A2A",
    )
    axis.set_title(title, fontsize=9, loc="left")
    axis.set_xticks([])
    axis.set_yticks([])


def _metric(result: dict, stratum: str, name: str):
    return (
        result.get("strata", {}).get(stratum, {}).get("metrics", {}).get(name)
        if result else None
    )


def _support(result: dict, stratum: str) -> dict:
    return result.get("strata", {}).get(stratum, {}).get("support", {}) if result else {}


# ------------------------------------------------------------------- panel (a)

def panel_region_errors(axis, results: dict, methods: Sequence[str]) -> dict:
    """Region-minus-whole MAE on each ROI's own reference-bearing frame subset."""
    rois = ['hand', 'object', 'background']
    available = [m for m in methods if m in results]
    populations = {m: results[m].get('population', {}).get('token_set_id') for m in available}
    metadata = {'population_token_set_ids': populations,
                'single_population': len(set(populations.values())) <= 1}
    if not available or not all('paired_regions' in results[m] for m in available):
        _figure_text(axis, 'missing per-region matched-frame comparisons', '(a) Region versus whole-image error')
        return {'status': 'missing', **metadata}
    if not metadata['single_population']:
        _figure_text(axis, 'methods have different annotation populations', '(a) Region versus whole-image error')
        return {'status': 'population_mismatch', **metadata}
    positions = np.arange(3); width = .8 / len(available); drawn = {}; any_value = False
    for index, method in enumerate(available):
        drawn[method] = {}
        for position, roi in enumerate(rois):
            block = results[method]['paired_regions'].get(roi, {})
            pair = block.get('paired_difference', {}); value = pair.get('difference')
            x = positions[position] + index * width
            if value is None:
                axis.text(x, 0, 'n/a', ha='center', fontsize=5); continue
            any_value = True
            axis.bar(x, value, width, label=method if position == 0 else None,
                     color=ROI_COLOURS[roi], edgecolor='black', linewidth=.4,
                     alpha=.55+.45*index/max(1,len(available)-1))
            ci = pair.get('ci95')
            if ci is not None:
                axis.errorbar(x, value, yerr=[[max(0,value-ci[0])],[max(0,ci[1]-value)]],
                              fmt='none', color='black', capsize=2, linewidth=.6)
            axis.annotate(f"{block.get('n_frames',0)}f/{pair.get('n_sites_compared',0)}s",
                          (x,value), xytext=(0,3 if value >= 0 else -9),
                          textcoords='offset points', ha='center', fontsize=5)
            drawn[method][roi] = block
    if not any_value:
        _figure_text(axis, 'no supported regional comparisons', '(a) Region versus whole-image error')
        return {'status': 'no_values', **metadata}
    axis.axhline(0, color='black', linewidth=.5)
    axis.set_xticks(positions + width*(len(available)-1)/2)
    axis.set_xticklabels(rois, fontsize=7)
    axis.set_ylabel('Region minus whole-image MAE$_Z$ (cm)', fontsize=8)
    axis.set_title('(a) Region versus whole-image error', fontsize=9, loc='left')
    axis.tick_params(labelsize=7)
    handles, labels = axis.get_legend_handles_labels()
    if handles: axis.legend(fontsize=5)
    return {'status':'ok','drawn':drawn,**metadata,
            'support_label':'frames/sites; each region has its own matched whole-image baseline'}


# ------------------------------------------------------------------- panel (b)

def panel_tolerance_curves(axis, results: dict, methods: Sequence[str]) -> dict:
    """A_tau over 0-10 cm for hand and object, with coverage in the legend."""
    drawn: dict = {}
    plotted = 0
    for method in methods:
        result = results.get(method)
        if not result:
            continue
        for roi, style in (("hand", "-"), ("object", "--")):
            block = result.get("strata", {}).get(f"roi::{roi}", {})
            curve = block.get("metrics", {}).get("a_tau") or {}
            if not curve:
                continue
            taus = sorted(float(k) for k in curve)
            values = [curve[f"{t:.3f}"] for t in taus]
            if all(v is None for v in values):
                continue
            coverage = block.get("metrics", {}).get("coverage")
            axis.plot(
                [100 * t for t in taus],
                [np.nan if v is None else 100 * v for v in values],
                style, color=ROI_COLOURS[roi], linewidth=1.2,
                label=f"{method} {roi} (cov {100 * coverage:.0f}%)"
                if coverage is not None else f"{method} {roi}",
            )
            drawn[f"{method}::{roi}"] = {"taus_cm": [100 * t for t in taus],
                                         "a_tau": values, "coverage": coverage}
            plotted += 1

    if not plotted:
        _figure_text(
            axis,
            "no A_tau curves: the annotated subset has no scored hand/object pixels",
            "(b) Hand and object error tolerances",
        )
        return {"status": "missing", "reason": "no a_tau data"}

    axis.set_xlabel(r"Z error tolerance $\tau$ (cm)", fontsize=8)
    axis.set_ylabel(r"$A_\tau$ (%)", fontsize=8)
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 100)
    # A curve that is genuinely ~0 across the whole range is a real measurement,
    # not a missing one, and at this axis scale it is indistinguishable from an
    # unplotted line. Say so on the panel.
    peak = max(
        (max((v or 0.0) for v in series["a_tau"]) for series in drawn.values()),
        default=0.0,
    )
    if peak < 0.02:
        axis.text(
            5.0, 50.0,
            f"all curves below {100 * peak:.2f}%: predictions rarely land within\n"
            "10 cm of the reference in this population",
            ha="center", va="center", fontsize=6, color="#C92A2A",
        )
    axis.set_title("(b) Hand and object error tolerances", fontsize=9, loc="left")
    axis.tick_params(labelsize=7)
    axis.grid(alpha=0.25, linewidth=0.4)
    axis.legend(fontsize=5, loc="upper left")
    return {
        "status": "ok", "drawn": drawn, "max_a_tau": peak,
        "note": (
            "missing predictions stay in the denominator; a curve at ~0 is a "
            "measured result, not absent data"
        ),
    }


# ------------------------------------------------------------------- panel (c)

def panel_z_angle_heatmap(
    axis, results: dict, method: str, z_labels: Sequence[str], angle_labels: Sequence[str],
    vmin: Optional[float] = None, vmax: Optional[float] = None,
) -> dict:
    """MAE_Z per Z x angle cell, with unmeasurable cells visually distinct."""
    result = results.get(method)
    if not result:
        _figure_text(axis, f"no result for {method}", "(c) Optical depth and viewing angle")
        return {"status": "missing", "reason": f"no result for {method}"}

    grid = np.full((len(z_labels), len(angle_labels)), np.nan)
    reference = np.zeros_like(grid)
    scored = np.zeros_like(grid)
    for z_index, z_label in enumerate(z_labels):
        for angle_index, angle_label in enumerate(angle_labels):
            key = f"cell::full::z={z_label}::angle={angle_label}"
            block = result.get("strata", {}).get(key, {})
            metrics = block.get("metrics", {})
            reference[z_index, angle_index] = metrics.get("n_reference") or 0
            scored[z_index, angle_index] = metrics.get("n_scored") or 0
            value = metrics.get("mae_z_cm")
            if value is not None:
                grid[z_index, angle_index] = float(value)

    if not np.isfinite(grid).any():
        _figure_text(
            axis,
            "no Z x angle cell has both reference and predictions",
            "(c) Optical depth and viewing angle",
        )
        return {"status": "missing", "reason": "no populated cells"}

    # Distinguish "no reference here" from "reference but no prediction".
    masked = np.ma.masked_invalid(grid)
    colormap = _colormap(ERROR_CMAP)
    colormap.set_bad(NO_REFERENCE_COLOUR)
    image = axis.imshow(
        masked, aspect="auto", cmap=colormap, origin="lower", vmin=vmin, vmax=vmax
    )
    for z_index in range(len(z_labels)):
        for angle_index in range(len(angle_labels)):
            if reference[z_index, angle_index] == 0:
                marker, colour = "no GT", "#495057"
            elif scored[z_index, angle_index] == 0:
                marker, colour = "no pred", "#C92A2A"
                from matplotlib.patches import Rectangle

                axis.add_patch(Rectangle(
                    (angle_index - 0.5, z_index - 0.5), 1, 1,
                    facecolor=NO_PREDICTION_COLOUR, hatch="xx", edgecolor="#C92A2A",
                    linewidth=0.4,
                ))
            else:
                marker, colour = f"{grid[z_index, angle_index]:.0f}", "white"
            axis.text(angle_index, z_index, marker, ha="center", va="center",
                      fontsize=5, color=colour)

    axis.set_xticks(range(len(angle_labels)))
    axis.set_xticklabels(angle_labels, fontsize=5, rotation=30)
    axis.set_yticks(range(len(z_labels)))
    axis.set_yticklabels(z_labels, fontsize=5)
    axis.set_xlabel(r"field angle $\theta$ (deg)", fontsize=8)
    axis.set_ylabel("reference Z (m)", fontsize=8)
    axis.set_title(f"(c) Optical depth and viewing angle -- {method}", fontsize=9, loc="left")
    bar = axis.figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    bar.set_label(r"MAE$_Z$ (cm)", fontsize=7)
    bar.ax.tick_params(labelsize=6)
    return {
        "status": "ok",
        "grid_mae_z_cm": grid.tolist(),
        "n_reference": reference.tolist(),
        "n_scored": scored.tolist(),
    }


# ------------------------------------------------------------------- panel (d)

def panel_motion_contrast(axis, motion_payload: Optional[dict]) -> dict:
    """Active minus low-motion difference, unweighted and standardized, with CI."""
    if not motion_payload or not motion_payload.get("comparisons"):
        _figure_text(
            axis,
            "no matched motion contrast: this needs accepted active-manipulation "
            "AND low-motion frames sharing site-Z-angle strata",
            "(d) Matched motion conditions",
        )
        return {"status": "missing", "reason": "no motion comparisons"}

    comparisons = motion_payload["comparisons"]
    labels = [c["label"] for c in comparisons]
    positions = np.arange(len(labels))
    for offset, key, colour in ((-0.16, "unweighted", "#4C6EF5"),
                                (0.16, "standardized", "#E8590C")):
        supported = [
            (index, comparison[key])
            for index, comparison in enumerate(comparisons)
            if comparison.get(key) is not None
        ]
        # The point estimate and its interval have independent availability.
        # Keep the legend entry even if every estimate of this kind is undefined.
        axis.plot(
            [index + offset for index, _ in supported],
            [value for _, value in supported],
            "o", color=colour, label=key, markersize=3,
        )
        for index, comparison in enumerate(comparisons):
            if comparison.get(key) is None:
                label = "raw n/a" if key == "unweighted" else "std n/a"
                axis.text(
                    index + offset, 0.97, label,
                    transform=axis.get_xaxis_transform(),
                    ha="center", va="top", fontsize=4, color=colour,
                    rotation=90,
                )
                continue
            interval = comparison.get(f"{key}_ci95")
            if interval is None:
                continue
            # Draw the actual endpoints, including percentile intervals that do
            # not contain the point estimate; the marker is drawn separately.
            lower, upper = interval
            midpoint = (lower + upper) / 2
            axis.errorbar(
                index + offset, midpoint,
                yerr=[[midpoint - lower], [upper - midpoint]],
                fmt="none", color=colour, capsize=2, linewidth=1.0,
            )
    axis.axhline(0, color="black", linewidth=0.6, linestyle=":")
    # Reserve both positions even when a condition has no numeric marker.
    axis.set_xlim(-0.5, len(labels) - 0.5)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=6, rotation=20)
    axis.set_ylabel(r"active $-$ low-motion MAE$_Z$ (cm)", fontsize=8)
    axis.set_title("(d) Matched motion conditions", fontsize=9, loc="left")
    axis.tick_params(labelsize=7)
    axis.legend(fontsize=5)
    for index, comparison in enumerate(comparisons):
        axis.text(
            index, axis.get_ylim()[0],
            f"{comparison.get('n_strata_retained', 0)} strata\n"
            f"{comparison.get('n_strata_discarded', 0)} dropped",
            fontsize=4, ha="center", va="bottom",
        )
    return {"status": "ok", "comparisons": comparisons}


# ------------------------------------------------------------------- panel (e)

def panel_qualitative(
    axes, examples: Sequence[dict], depth_vmin: float = 0.25, depth_vmax: float = 2.0
) -> dict:
    """RGB+ROI, reference Z+mask, a monocular representative, MapAnything before/after."""
    if not examples:
        for axis in np.atleast_1d(axes).ravel():
            _figure_text(axis, "no shared qualitative examples available",
                         "(e) Shared qualitative examples")
        return {"status": "missing", "reason": "no examples"}

    drawn: list = []
    axes = np.atleast_2d(axes)
    for row, example in enumerate(examples[: axes.shape[0]]):
        columns = example.get("columns", [])
        for column in range(axes.shape[1]):
            axis = axes[row, column]
            axis.set_xticks([])
            axis.set_yticks([])
            if column >= len(columns):
                axis.axis("off")
                continue
            payload = columns[column]
            array = payload.get("array")
            if array is None:
                _figure_text(axis, payload.get("missing", "missing"), "")
                continue
            array = np.asarray(array)
            if payload.get("kind") == "rgb":
                axis.imshow(array.astype(np.uint8))
            else:
                masked = np.ma.masked_invalid(array.astype(np.float32))
                colormap = _colormap(payload.get("cmap", ERROR_CMAP))
                # Same convention as panel (c): unmeasurable is grey, never a colour
                # that reads as a low error.
                colormap.set_bad(NO_REFERENCE_COLOUR)
                axis.imshow(
                    masked, cmap=colormap,
                    vmin=payload.get("vmin", depth_vmin),
                    vmax=payload.get("vmax", depth_vmax),
                )
            if row == 0:
                axis.set_title(payload.get("title", ""), fontsize=6)
            if column == 0:
                axis.set_ylabel(example.get("sample_token", "")[:10], fontsize=5)
        drawn.append(example.get("sample_token"))
    return {"status": "ok", "sample_tokens": drawn}


# ---------------------------------------------------------------- composition

def build_figure4(
    results: dict,
    output_path: Path,
    methods: Sequence[str],
    z_labels: Sequence[str],
    angle_labels: Sequence[str],
    heatmap_method: Optional[str] = None,
    motion_payload: Optional[dict] = None,
    examples: Optional[Sequence[dict]] = None,
    status: str = "development",
) -> dict:
    """Draw all five panels; return a per-panel record of what was and was not drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(11.0, 12.0))
    outer = gridspec.GridSpec(3, 2, figure=figure, height_ratios=[1.0, 1.0, 1.25],
                              hspace=0.38, wspace=0.28)

    record: dict = {"status": status, "panels": {}}
    record["panels"]["a"] = panel_region_errors(figure.add_subplot(outer[0, 0]), results, methods)
    record["panels"]["b"] = panel_tolerance_curves(figure.add_subplot(outer[0, 1]), results, methods)
    record["panels"]["c"] = panel_z_angle_heatmap(
        figure.add_subplot(outer[1, 0]), results,
        heatmap_method or (methods[0] if methods else ""), z_labels, angle_labels,
    )
    record["panels"]["d"] = panel_motion_contrast(
        figure.add_subplot(outer[1, 1]), motion_payload
    )

    examples = list(examples or [])
    n_rows = max(1, min(2, len(examples)))
    inner = gridspec.GridSpecFromSubplotSpec(
        n_rows, 5, subplot_spec=outer[2, :], hspace=0.12, wspace=0.06
    )
    qualitative_axes = np.array(
        [[figure.add_subplot(inner[r, c]) for c in range(5)] for r in range(n_rows)]
    )
    record["panels"]["e"] = panel_qualitative(qualitative_axes, examples)

    suffix = "" if status == "formal" else f"  [{status}]"
    figure.suptitle(
        f"Interaction diagnostics on held-out sites{suffix}", fontsize=11, y=0.995
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(figure)

    record["output_pdf"] = str(output_path)
    record["output_png"] = str(output_path.with_suffix(".png"))
    record["missing_panels"] = [
        key for key, value in record["panels"].items() if value.get("status") != "ok"
    ]
    record["note"] = (
        "Every panel is drawn from measured statistics. A panel with no data shows "
        "an explicit notice; no curve, heatmap or example is simulated."
    )
    (output_path.parent / f"{output_path.stem}_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    return record
