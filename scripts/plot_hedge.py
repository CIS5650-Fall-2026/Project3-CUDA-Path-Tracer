#!/usr/bin/env python3
"""Render dependency-free SVG charts from measured hedge evidence summaries."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import tempfile
from typing import Any, Iterable


SCHEMA_VERSION = 1
COLORS = {
    "counter": "#2463a6",
    "halton": "#d06531",
    "gpu": "#4477aa",
    "loop": "#22856b",
    "process": "#9a5aa6",
    "midpoint": "#6b75b8",
    "full": "#c2574b",
}


class EvidenceError(ValueError):
    """Raised when an evidence summary is missing or internally malformed."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: Any, context: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{context} must be finite")
    if nonnegative and result < 0:
        raise EvidenceError(f"{context} must be nonnegative")
    return result


def require_finite_json_numbers(value: Any, context: str = "summary") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceError(f"{context} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            require_finite_json_numbers(child, f"{context}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite_json_numbers(child, f"{context}.{key}")
        return
    raise EvidenceError(f"{context} contains unsupported JSON data")


def positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvidenceError(f"{context} must be a positive integer")
    return value


def text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def load_summary(directory: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = directory.resolve()
    path = directory if directory.is_file() else directory / "summary.json"
    if not path.is_file():
        raise EvidenceError(f"{label} summary does not exist: {path}")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read {label} summary {path}: {error}") from error
    if not isinstance(parsed, dict):
        raise EvidenceError(f"{label} summary root must be an object: {path}")
    require_finite_json_numbers(parsed, label + " summary")
    source = {
        "path": str(path),
        "sha256": sha256_path(path),
        "bytes": path.stat().st_size,
        "declared_schema_version": parsed.get("schema_version"),
        "declared_status": parsed.get("status"),
    }
    return parsed, source


def phase(summary: dict[str, Any], name: str) -> dict[str, Any] | None:
    phases = summary.get("phases")
    if isinstance(phases, dict) and isinstance(phases.get(name), dict):
        return phases[name]
    candidate = summary.get(name)
    if isinstance(candidate, dict):
        return candidate
    # measure_hedge.py also writes each phase object directly to
    # quality/summary.json and timing/summary.json.  Accept those natural inputs
    # without mistaking an unrelated summary for a measured phase.
    signatures = {
        "quality": ("reference_precision", "convergence", "equal_time"),
        "timing": ("groups", "motion_cost"),
    }
    required = signatures.get(name, ())
    if required and all(key in summary for key in required):
        return summary
    return None


def sample_stats(values: Iterable[float], context: str) -> dict[str, Any]:
    checked = [finite_number(item, context) for item in values]
    if not checked:
        raise EvidenceError(f"{context} contains no measurements")
    mean = math.fsum(checked) / len(checked)
    return {
        "values": checked,
        "count": len(checked),
        "mean": mean,
        "sample_stddev": statistics.stdev(checked) if len(checked) > 1 else 0.0,
        "min": min(checked),
        "max": max(checked),
    }


def convergence_data(quality: dict[str, Any] | None) -> list[dict[str, Any]]:
    if quality is None:
        return []
    records = quality.get("convergence", [])
    if not isinstance(records, list):
        raise EvidenceError("quality.convergence must be an array")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, record in enumerate(records):
        context = f"quality.convergence[{index}]"
        if not isinstance(record, dict):
            raise EvidenceError(f"{context} must be an object")
        scene = record.get("scene")
        if not isinstance(scene, str) or not scene:
            raise EvidenceError(f"{context}.scene must be a nonempty string")
        samples = positive_integer(record.get("samples"), f"{context}.samples")
        identity = (scene, samples)
        if identity in seen:
            raise EvidenceError(f"duplicate convergence entry for {scene}/{samples}")
        seen.add(identity)
        per_seed = record.get("per_seed")
        if not isinstance(per_seed, list) or not per_seed:
            raise EvidenceError(f"{context}.per_seed must be a nonempty array")
        for sampler in ("counter", "halton"):
            values = []
            completed = []
            seeds = []
            for seed_index, paired in enumerate(per_seed):
                seed_context = f"{context}.per_seed[{seed_index}]"
                if not isinstance(paired, dict):
                    raise EvidenceError(f"{seed_context} must be an object")
                seed = positive_integer(paired.get("seed"), f"{seed_context}.seed")
                entry = paired.get(sampler)
                if not isinstance(entry, dict):
                    raise EvidenceError(f"{seed_context}.{sampler} must be an object")
                values.append(finite_number(entry.get("mse"), f"{seed_context}.{sampler}.mse"))
                completed.append(positive_integer(
                    entry.get("completed_samples"),
                    f"{seed_context}.{sampler}.completed_samples"))
                seeds.append(seed)
            stats = sample_stats(values, f"{context}.{sampler}.mse")
            declared = record.get(sampler)
            if isinstance(declared, dict) and "mean_mse" in declared:
                declared_mean = finite_number(
                    declared["mean_mse"], f"{context}.{sampler}.mean_mse")
                if not math.isclose(declared_mean, stats["mean"], rel_tol=1e-12,
                                    abs_tol=1e-15):
                    raise EvidenceError(
                        f"{context}.{sampler}.mean_mse disagrees with per-seed values")
            result.append({
                "scene": scene,
                "samples": samples,
                "sampler": sampler,
                "mse": stats,
                "fixed_seeds": seeds,
                "actual_completed_samples": completed,
                "declared_relative_mse_reduction": finite_number(
                    record.get("relative_mse_reduction"),
                    f"{context}.relative_mse_reduction",
                    nonnegative=False,
                ),
                "strict_improvement": record.get("strict_improvement") is True,
                "intensity_criterion_passed": (
                    isinstance(record.get("intensity_paired"), dict)
                    and record["intensity_paired"].get("passed") is True
                ),
            })
    return result


def metric_object(record: dict[str, Any], *names: str) -> dict[str, Any] | None:
    metrics = record.get("metrics")
    containers = [metrics, record] if isinstance(metrics, dict) else [record]
    for container in containers:
        for name in names:
            candidate = container.get(name)
            if isinstance(candidate, dict) and "values" in candidate:
                values = candidate["values"]
                if not isinstance(values, list) or not values:
                    raise EvidenceError(f"metric {name}.values must be a nonempty array")
                return sample_stats(values, f"metric {name}")
            if isinstance(candidate, list) and candidate:
                return sample_stats(candidate, f"metric {name}")
        for name in names:
            median_key = name + "_median"
            if median_key in container:
                median = finite_number(container[median_key], median_key)
                minimum_key = name + "_min"
                maximum_key = name + "_max"
                if minimum_key not in container or maximum_key not in container:
                    raise EvidenceError(
                        f"metric {name} has a median but no measured min–max range")
                minimum = finite_number(container[minimum_key], minimum_key)
                maximum = finite_number(container[maximum_key], maximum_key)
                if not minimum <= median <= maximum:
                    raise EvidenceError(
                        f"metric {name} median lies outside its measured min–max range")
                return {
                    "values": None,
                    "count": container.get("run_count"),
                    "mean": None,
                    "sample_stddev": None,
                    "median": median,
                    "min": minimum,
                    "max": maximum,
                }
    return None


def metric_median(metric: dict[str, Any]) -> float:
    if "median" in metric:
        return finite_number(metric["median"], "metric.median")
    values = metric.get("values")
    if not isinstance(values, list) or not values:
        raise EvidenceError("metric lacks measured values")
    return statistics.median(values)


def normalize_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "values": metric.get("values"),
        "count": metric.get("count"),
        "median": metric_median(metric),
        "min": finite_number(metric["min"], "metric.min"),
        "max": finite_number(metric["max"], "metric.max"),
    }


def equal_time_data(quality: dict[str, Any] | None) -> list[dict[str, Any]]:
    if quality is None:
        return []
    block = quality.get("equal_time")
    if not isinstance(block, dict):
        return []
    summaries = block.get("summaries", [])
    if not isinstance(summaries, list):
        raise EvidenceError("quality.equal_time.summaries must be an array")
    result = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(summaries):
        context = f"quality.equal_time.summaries[{index}]"
        if not isinstance(record, dict):
            raise EvidenceError(f"{context} must be an object")
        scene = record.get("scene")
        sampler = record.get("sampler")
        if not isinstance(scene, str) or not scene:
            raise EvidenceError(f"{context}.scene must be a nonempty string")
        if sampler not in ("counter", "halton"):
            raise EvidenceError(f"{context}.sampler must be counter or halton")
        identity = (scene, sampler)
        if identity in seen:
            raise EvidenceError(f"duplicate equal-time summary for {scene}/{sampler}")
        seen.add(identity)
        completed = metric_object(record, "completed_samples")
        mse = metric_object(record, "mse")
        gpu = metric_object(record, "renderer_gpu_ms", "gpu_ms")
        loop = metric_object(record, "render_loop_wall_ms", "render_loop_ms")
        process = metric_object(record, "process_elapsed_ms", "process_ms")
        if completed is None or mse is None:
            raise EvidenceError(f"{context} lacks completed-sample or MSE measurements")
        item = {
            "scene": scene,
            "sampler": sampler,
            "budget_ms": finite_number(block.get("budget_ms"),
                                       "quality.equal_time.budget_ms"),
            "completed_samples": normalize_metric(completed),
            "mse": normalize_metric(mse),
        }
        if gpu is not None:
            item["renderer_gpu_ms"] = normalize_metric(gpu)
        if loop is not None:
            item["render_loop_wall_ms"] = normalize_metric(loop)
        if process is not None:
            item["process_elapsed_ms"] = normalize_metric(process)
        seeds = record.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise EvidenceError(f"{context}.seeds must be a nonempty array")
        checked_seeds = [positive_integer(seed, f"{context}.seeds") for seed in seeds]
        metric_names = (
            "completed_samples", "mse", "renderer_gpu_ms",
            "render_loop_wall_ms", "process_elapsed_ms",
        )
        per_seed = []
        for seed_index, seed in enumerate(checked_seeds):
            seed_record: dict[str, Any] = {"seed": seed}
            for name in metric_names:
                if name not in item:
                    continue
                values = item[name].get("values")
                if not isinstance(values, list) or len(values) != len(checked_seeds):
                    raise EvidenceError(
                        f"{context}.{name} values do not align with fixed seeds")
                seed_record[name] = values[seed_index]
            per_seed.append(seed_record)
        adequate = record.get("reference_adequate")
        if not isinstance(adequate, bool):
            raise EvidenceError(f"{context}.reference_adequate must be boolean")
        item["fixed_seeds"] = checked_seeds
        item["reference_adequate"] = adequate
        item["per_seed"] = per_seed
        result.append(item)
    return result


def timing_data(timing: dict[str, Any] | None) -> list[dict[str, Any]]:
    if timing is None:
        return []
    groups = timing.get("groups", [])
    if not isinstance(groups, list):
        raise EvidenceError("timing.groups must be an array")
    result = []
    seen: set[tuple[str, str, str]] = set()
    for index, record in enumerate(groups):
        context = f"timing.groups[{index}]"
        if not isinstance(record, dict):
            raise EvidenceError(f"{context} must be an object")
        kind = record.get("kind")
        scene = record.get("scene")
        sampler = record.get("sampler")
        if not isinstance(kind, str) or not kind:
            raise EvidenceError(f"{context}.kind must be a nonempty string")
        if not isinstance(scene, str) or not scene:
            raise EvidenceError(f"{context}.scene must be a nonempty string")
        if sampler not in ("counter", "halton"):
            raise EvidenceError(f"{context}.sampler must be counter or halton")
        identity = (kind, scene, sampler)
        if identity in seen:
            raise EvidenceError(f"duplicate timing group for {kind}/{scene}/{sampler}")
        seen.add(identity)
        gpu_cost = metric_object(
            record, "renderer_gpu_ms_per_completed_sample", "gpu_ms_per_sample",
            "renderer_gpu_ms_per_sample")
        loop_cost = metric_object(
            record, "render_loop_wall_ms_per_completed_sample",
            "render_loop_ms_per_sample")
        process_cost = metric_object(
            record, "process_elapsed_ms_per_completed_sample",
            "process_ms_per_sample")
        completed = metric_object(record, "completed_samples")
        if gpu_cost is None:
            gpu_total = metric_object(record, "renderer_gpu_ms", "gpu_ms")
            if gpu_total is None or completed is None:
                raise EvidenceError(f"{context} lacks actual GPU cost evidence")
            totals = gpu_total.get("values")
            counts = completed.get("values")
            if not isinstance(totals, list) or not isinstance(counts, list) \
                    or len(totals) != len(counts):
                raise EvidenceError(f"{context} cannot pair GPU time with sample count")
            gpu_cost = sample_stats(
                [total / count for total, count in zip(totals, counts)],
                f"{context}.derived GPU ms/completed sample")
        item = {
            "kind": kind,
            "scene": scene,
            "sampler": sampler,
            "comparison_role": record.get("comparison_role"),
            "warmup_run_ids": record.get("warmup_run_ids"),
            "measured_run_ids": record.get("measured_run_ids"),
            "warmups_excluded_from_summary": record.get(
                "warmups_excluded_from_summary"),
            "renderer_gpu_ms_per_completed_sample": normalize_metric(gpu_cost),
        }
        if loop_cost is not None:
            item["render_loop_wall_ms_per_completed_sample"] = normalize_metric(loop_cost)
        if process_cost is not None:
            item["process_elapsed_ms_per_completed_sample"] = normalize_metric(process_cost)
        if completed is not None:
            item["completed_samples"] = normalize_metric(completed)
        result.append(item)
    return result


def svg_header(width: int, height: int, title: str, subtitle: str,
               evidence_status: str | None = None) -> list[str]:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{text(title)}</title>',
        f'<desc id="desc">{text(subtitle)}</desc>',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        '<g font-family="system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
        'fill="#182433">',
        f'<text x="34" y="39" font-size="24" font-weight="650">{text(title)}</text>',
        f'<text x="34" y="66" font-size="13">{text(subtitle)}</text>',
    ]
    if evidence_status is not None:
        lines.append(
            f'<text x="{width - 34}" y="39" text-anchor="end" font-size="13" '
            f'font-weight="700" fill="#9a3412">{text(evidence_status)}</text>'
        )
    return lines


def write_svg(path: Path, lines: list[str]) -> None:
    lines.extend(["</g>", "</svg>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convergence_svg(path: Path, records: list[dict[str, Any]],
                    qualification: str | None,
                    adequate_references: bool | None,
                    per_scene_reference_adequacy: dict[str, bool] | None = None) -> None:
    scenes = sorted({item["scene"] for item in records})
    width = 1120
    panel_height = 360
    height = 104 + panel_height * len(scenes)
    subtitle = (
        "Linear-RGB MSE versus retained reference; mean points and fixed-seed "
        "min–max whiskers. Axes use measured values."
    )
    disclosures = []
    if qualification is not None:
        disclosures.append(f"qualification: {qualification}")
    if adequate_references is not None:
        disclosures.append(
            "references adequate" if adequate_references else "references inadequate"
        )
    if disclosures:
        subtitle += " Evidence status — " + "; ".join(disclosures) + "."
    status_label = quality_status_label(qualification, adequate_references)
    out = svg_header(
        width, height, "Sampler convergence", subtitle,
        evidence_status=status_label,
    )
    left, right, panel_top = 112, 60, 92
    plot_width = width - left - right
    for panel_index, scene in enumerate(scenes):
        subset = [item for item in records if item["scene"] == scene]
        samples = sorted({item["samples"] for item in subset})
        y_values = [value for item in subset for value in item["mse"]["values"]]
        positive_log = min(y_values) > 0
        if positive_log:
            y_min = 10 ** math.floor(math.log10(min(y_values)))
            y_max = 10 ** math.ceil(math.log10(max(y_values)))
            if y_min == y_max:
                y_max *= 10
        else:
            y_min = 0.0
            y_max = max(y_values) or 1.0
        x_min = math.log2(min(samples))
        x_max = math.log2(max(samples))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        top = panel_top + panel_index * panel_height
        bottom = top + 230

        def x_position(sample: int) -> float:
            return left + (math.log2(sample) - x_min) / (x_max - x_min) * plot_width

        def y_position(value: float) -> float:
            if positive_log:
                fraction = ((math.log10(value) - math.log10(y_min)) /
                            (math.log10(y_max) - math.log10(y_min)))
            else:
                fraction = (value - y_min) / (y_max - y_min)
            return bottom - fraction * (bottom - top)

        scene_label = scene
        if per_scene_reference_adequacy is not None:
            scene_label += (
                " · reference adequate"
                if per_scene_reference_adequacy[scene]
                else " · REFERENCE INADEQUATE"
            )
        out.append(f'<text x="34" y="{top - 13}" font-size="17" font-weight="600">'
                   f'{text(scene_label)}</text>')
        out.append(f'<path d="M{left} {top} V{bottom} H{width-right}" '
                   'fill="none" stroke="#3a4655" stroke-width="1.3"/>')
        for tick_index in range(5):
            if positive_log:
                exponent = (math.log10(y_min) + tick_index / 4 *
                            (math.log10(y_max) - math.log10(y_min)))
                tick_value = 10 ** exponent
            else:
                tick_value = y_max * tick_index / 4
            y = y_position(tick_value) if tick_value > 0 or not positive_log else bottom
            out.append(f'<path d="M{left} {y:.2f} H{width-right}" stroke="#dde3ea"/>')
            out.append(f'<text x="{left-9}" y="{y+4:.2f}" text-anchor="end" '
                       f'font-size="11">{tick_value:.3g}</text>')
        for sample in samples:
            x = x_position(sample)
            out.append(f'<path d="M{x:.2f} {top} V{bottom}" stroke="#edf0f4"/>')
            out.append(f'<text x="{x:.2f}" y="{bottom+22}" text-anchor="middle" '
                       f'font-size="12">{sample}</text>')
        out.append(f'<text x="{(left+width-right)/2:.2f}" y="{bottom+45}" '
                   'text-anchor="middle" font-size="12">samples per pixel (log₂)</text>')
        out.append(f'<text x="24" y="{(top+bottom)/2:.2f}" text-anchor="middle" '
                   'font-size="12" transform="rotate(-90 24 '
                   f'{(top+bottom)/2:.2f})">mean squared error'
                   f'{" (log₁₀)" if positive_log else ""}</text>')
        for sampler in ("counter", "halton"):
            entries = sorted((item for item in subset if item["sampler"] == sampler),
                             key=lambda item: item["samples"])
            points = " ".join(
                f'{x_position(item["samples"]):.2f},{y_position(item["mse"]["mean"]):.2f}'
                for item in entries)
            out.append(f'<polyline points="{points}" fill="none" '
                       f'stroke="{COLORS[sampler]}" stroke-width="2.5"/>')
            for item in entries:
                x = x_position(item["samples"])
                y_low = y_position(item["mse"]["min"])
                y_high = y_position(item["mse"]["max"])
                y = y_position(item["mse"]["mean"])
                out.append(f'<path d="M{x:.2f} {y_low:.2f} V{y_high:.2f} '
                           f'M{x-5:.2f} {y_low:.2f} H{x+5:.2f} '
                           f'M{x-5:.2f} {y_high:.2f} H{x+5:.2f}" '
                           f'stroke="{COLORS[sampler]}" stroke-width="1.4"/>')
                out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" '
                           f'fill="{COLORS[sampler]}"><title>{text(sampler)}; '
                           f'{item["samples"]} spp; mean MSE {item["mse"]["mean"]:.8g}; '
                           f'range {item["mse"]["min"]:.8g}–{item["mse"]["max"]:.8g}; '
                           f'n={item["mse"]["count"]}; seeds '
                           f'{text(item["fixed_seeds"])}; actual completed samples '
                           f'{text(item["actual_completed_samples"])}</title></circle>')
        legend_y = bottom + 68
        for offset, sampler in enumerate(("counter", "halton")):
            legend_x = left + offset * 150
            out.append(f'<path d="M{legend_x} {legend_y} h28" '
                       f'stroke="{COLORS[sampler]}" stroke-width="3"/>')
            out.append(f'<text x="{legend_x+36}" y="{legend_y+4}" font-size="12">'
                       f'{sampler}</text>')
        comparison_labels = []
        for sample in samples:
            paired = {
                item["sampler"]: item for item in subset
                if item["samples"] == sample
            }
            if set(paired) != {"counter", "halton"}:
                continue
            counter = paired["counter"]["mse"]["mean"]
            halton = paired["halton"]["mse"]["mean"]
            if counter <= 0:
                continue
            reduction = (counter - halton) / counter
            label = f"{sample}: {reduction:+.1%}"
            if reduction < 0:
                label += " regression"
            comparison_labels.append(label)
        if comparison_labels:
            out.append(f'<text x="{width-right}" y="{legend_y+4}" '
                       f'text-anchor="end" font-size="12">'
                       f'{text("Halton MSE reductions — " + "; ".join(comparison_labels))}'
                       '</text>')
    write_svg(path, out)


def horizontal_metric_svg(path: Path, title: str, subtitle: str,
                          records: list[dict[str, Any]], metric_key: str,
                          unit: str, labeler,
                          evidence_status: str | None = None) -> None:
    base_width, left, right, top, row_height = 1180, 365, 110, 102, 51
    height = top + row_height * len(records) + 70
    maximum = max(item[metric_key]["max"] for item in records)
    maximum = maximum * 1.12 if maximum > 0 else 1.0
    plot_width = base_width - left - right
    plot_right = left + plot_width
    label_font_size = 12
    label_offset = 8
    label_right_margin = 34
    conservative_character_width = label_font_size * 0.75
    value_labels = [
        (f'{item[metric_key]["median"]:.5g} '
         f'[{item[metric_key]["min"]:.5g}, {item[metric_key]["max"]:.5g}]')
        for item in records
    ]
    label_right_edges = [
        (left + plot_width * item[metric_key]["max"] / maximum + label_offset
         + len(value_label) * conservative_character_width)
        for item, value_label in zip(records, value_labels)
    ]
    width = max(
        base_width,
        math.ceil(max(label_right_edges) + label_right_margin),
    )
    out = svg_header(width, height, title, subtitle, evidence_status)
    for tick in range(6):
        value = maximum * tick / 5
        x = left + plot_width * tick / 5
        out.append(f'<path d="M{x:.2f} {top-12} V{height-48}" stroke="#dde3ea"/>')
        out.append(f'<text x="{x:.2f}" y="{height-25}" text-anchor="middle" '
                   f'font-size="12">{value:.4g}</text>')
    out.append(f'<text x="{(left+plot_right)/2:.2f}" y="{height-5}" '
               f'text-anchor="middle" font-size="12">{text(unit)}</text>')
    for index, item in enumerate(records):
        metric = item[metric_key]
        median = metric["median"]
        y = top + index * row_height
        color = COLORS.get(item.get("sampler"), COLORS.get(item.get("kind"), "#667788"))
        bar_width = plot_width * median / maximum
        min_x = left + plot_width * metric["min"] / maximum
        max_x = left + plot_width * metric["max"] / maximum
        out.append(f'<text x="{left-14}" y="{y+20}" text-anchor="end" '
                   f'font-size="12">{text(labeler(item))}</text>')
        tooltip = ""
        per_seed = item.get("per_seed")
        if isinstance(per_seed, list) and per_seed:
            pairs = [
                f'{entry.get("seed")}:{entry.get(metric_key)}'
                for entry in per_seed
                if isinstance(entry, dict) and metric_key in entry
            ]
            if pairs:
                tooltip = (
                    f'<title>{text(metric_key)} by fixed seed — '
                    f'{text("; ".join(pairs))}</title>'
                )
        out.append(f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="28" '
                   f'rx="3" fill="{color}" opacity="0.9">{tooltip}</rect>')
        out.append(f'<path d="M{min_x:.2f} {y+14} H{max_x:.2f} '
                   f'M{min_x:.2f} {y+8} V{y+20} M{max_x:.2f} {y+8} V{y+20}" '
                   'stroke="#172331" stroke-width="1.5"/>')
        out.append(f'<text x="{max_x+label_offset:.2f}" y="{y+20}" '
                   f'font-size="{label_font_size}">{value_labels[index]}</text>')
    write_svg(path, out)


def comparison_rows(records: list[dict[str, Any]], value_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for item in records:
        grouped.setdefault(item["scene"], {})[item["sampler"]] = item
    result = []
    for scene in sorted(grouped):
        pair = grouped[scene]
        if set(pair) != {"counter", "halton"}:
            continue
        counter = pair["counter"][value_key]["median"]
        halton = pair["halton"][value_key]["median"]
        signed_reduction = ((counter - halton) / counter
                            if counter > 0 else None)
        result.append({
            "scene": scene,
            "counter": counter,
            "halton": halton,
            "signed_halton_reduction": signed_reduction,
            "regression": signed_reduction < 0 if signed_reduction is not None else None,
        })
    return result


def quality_status_label(qualification: str | None,
                         adequate_references: bool | None) -> str | None:
    """Return the short, visible status used on reference-dependent SVGs."""
    if qualification is None and adequate_references is None:
        return None
    status = qualification.upper() if qualification is not None else "STATUS UNKNOWN"
    if adequate_references is None:
        reference = "REFERENCE STATUS UNKNOWN"
    else:
        reference = ("REFERENCES ADEQUATE" if adequate_references
                     else "REFERENCES INADEQUATE")
    return f"QUALITY: {status} · {reference}"


def convergence_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for item in records:
        grouped.setdefault((item["scene"], item["samples"]), {})[
            item["sampler"]
        ] = item
    result = []
    for (scene, samples), pair in sorted(grouped.items()):
        if set(pair) != {"counter", "halton"}:
            raise EvidenceError(f"convergence pair is incomplete for {scene}/{samples}")
        counter = pair["counter"]["mse"]["mean"]
        halton = pair["halton"]["mse"]["mean"]
        if counter <= 0:
            raise EvidenceError(f"counter MSE is not positive for {scene}/{samples}")
        reduction = (counter - halton) / counter
        declared = pair["counter"]["declared_relative_mse_reduction"]
        if not math.isclose(reduction, declared, rel_tol=1e-12, abs_tol=1e-15):
            raise EvidenceError(
                f"declared MSE reduction disagrees for {scene}/{samples}")
        result.append({
            "scene": scene,
            "samples": samples,
            "counter_mean_mse": counter,
            "halton_mean_mse": halton,
            "signed_halton_mse_reduction": reduction,
            "regression": reduction < 0,
            "strict_improvement": pair["counter"]["strict_improvement"],
            "intensity_criterion_passed": pair["counter"][
                "intensity_criterion_passed"
            ],
            "fixed_seeds": pair["counter"]["fixed_seeds"],
            "counter_actual_completed_samples": pair["counter"][
                "actual_completed_samples"
            ],
            "halton_actual_completed_samples": pair["halton"][
                "actual_completed_samples"
            ],
        })
    return result


def quality_disclosure(quality: dict[str, Any] | None) -> dict[str, Any] | None:
    if quality is None:
        return None
    qualification = quality.get("qualification")
    precision = quality.get("reference_precision")
    if not isinstance(qualification, dict):
        raise EvidenceError("quality.qualification must be an object")
    status = qualification.get("status")
    if status not in ("qualified", "not-qualified", "inconclusive"):
        raise EvidenceError("quality.qualification.status is invalid")
    if not isinstance(precision, list) or not precision:
        raise EvidenceError("quality.reference_precision must be a nonempty array")
    references = []
    per_scene_adequacy: dict[str, bool] = {}
    for index, record in enumerate(precision):
        if not isinstance(record, dict) or not isinstance(record.get("adequate"), bool):
            raise EvidenceError(
                f"quality.reference_precision[{index}] lacks boolean adequate status")
        scene = record.get("scene")
        if not isinstance(scene, str) or not scene:
            raise EvidenceError(
                f"quality.reference_precision[{index}].scene must be a nonempty string")
        if scene in per_scene_adequacy:
            raise EvidenceError(f"duplicate reference-precision scene: {scene}")
        per_scene_adequacy[scene] = record["adequate"]
        references.append(record)
    adequate = all(record["adequate"] for record in references)
    if (status == "inconclusive") == adequate:
        raise EvidenceError(
            "quality qualification/reference adequacy statuses are inconsistent")
    return {
        "status": status,
        "adequate_references": adequate,
        "per_scene_reference_adequacy": per_scene_adequacy,
        "qualification": qualification,
        "reference_precision": references,
        "pooled_mse": quality.get("pooled_mse"),
        "comparisons": quality.get("comparisons"),
        "intensity_criteria": quality.get("intensity_criteria"),
    }


def motion_disclosure(timing: dict[str, Any] | None,
                      motion_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not motion_records:
        return None
    if timing is None or not isinstance(timing.get("motion_cost"), dict):
        raise EvidenceError("motion timing records lack timing.motion_cost evidence")
    cost = timing["motion_cost"]
    swept = cost.get("swept_geometry")
    if not isinstance(swept, dict) or swept.get("equal") is not True:
        raise EvidenceError("motion swept-geometry equality is not established")
    normalized_hash = swept.get("normalized_scene_sha256")
    if not isinstance(normalized_hash, str) or len(normalized_hash) != 64:
        raise EvidenceError("motion swept-geometry design hash is missing")
    roles = {item.get("comparison_role"): item for item in motion_records}
    if set(roles) != {"midpoint", "full-shutter"}:
        raise EvidenceError("motion timing comparison roles are incomplete")
    midpoint = roles["midpoint"]["renderer_gpu_ms_per_completed_sample"]["median"]
    full = roles["full-shutter"]["renderer_gpu_ms_per_completed_sample"]["median"]
    ratio = full / midpoint if midpoint > 0 else None
    return {
        "swept_geometry": swept,
        "comparison_roles": ["midpoint", "full-shutter"],
        "renderer_gpu_ms_per_completed_sample_full_over_midpoint": ratio,
        "source_median_ratios": cost.get("median_ratios"),
    }


def create_plots(quality_summary: dict[str, Any] | None,
                 timing_summary: dict[str, Any] | None,
                 output: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    quality = phase(quality_summary, "quality") if quality_summary else None
    timing = phase(timing_summary, "timing") if timing_summary else None
    convergence = convergence_data(quality)
    equal_time = equal_time_data(quality)
    timings = timing_data(timing)
    quality_status = quality_disclosure(quality) if quality is not None else None
    derived: dict[str, Any] = {}
    missing = []
    generated = []

    def retain(filename: str, evidence: str) -> None:
        target = output / filename
        generated.append({
            "filename": filename,
            "sha256": sha256_path(target),
            "bytes": target.stat().st_size,
            "evidence": evidence,
        })

    if convergence:
        if quality_status is None:
            raise EvidenceError("convergence evidence lacks qualification disclosure")
        reference_scenes = quality_status["per_scene_reference_adequacy"]
        missing_reference_scenes = sorted(
            {item["scene"] for item in convergence} - set(reference_scenes)
        )
        if missing_reference_scenes:
            raise EvidenceError(
                "convergence scenes lack reference-precision evidence: "
                + ", ".join(missing_reference_scenes))
        convergence_svg(
            output / "sampler-convergence.svg",
            convergence,
            quality_status["status"],
            quality_status["adequate_references"],
            reference_scenes,
        )
        retain("sampler-convergence.svg", "quality.convergence per-seed measured MSE")
        derived["quality_evidence_status"] = quality_status
        derived["convergence"] = {
            "records": convergence,
            "all_scene_budget_comparisons": convergence_comparisons(convergence),
        }
    else:
        missing.append("quality.convergence")

    if equal_time:
        ordered = sorted(equal_time, key=lambda item: (item["scene"], item["sampler"]))
        if quality_status is None:
            raise EvidenceError("equal-time quality evidence lacks qualification disclosure")
        expected_adequacy = quality_status["per_scene_reference_adequacy"]
        for item in ordered:
            if item["scene"] not in expected_adequacy:
                raise EvidenceError(
                    f'equal-time scene {item["scene"]} lacks reference-precision evidence')
            if item["reference_adequate"] != expected_adequacy[item["scene"]]:
                raise EvidenceError(
                    f'equal-time reference adequacy disagrees for {item["scene"]}')
        status_label = quality_status_label(
            quality_status["status"], quality_status["adequate_references"])
        subtitle = (
            f'Each run requested {ordered[0]["budget_ms"]:.0f} ms with a fixed safety ceiling; '
            "bars show medians and whiskers the full retained fixed-seed range."
        )
        if not quality_status["adequate_references"]:
            quality_subtitle = (
                "INCONCLUSIVE: references are inadequate; displayed MSE is descriptive "
                "against the retained reference, not qualified ground truth."
            )
        else:
            quality_subtitle = subtitle + " MSE uses the retained adequate reference."
        horizontal_metric_svg(
            output / "equal-time-quality.svg", "Equal-time image error", quality_subtitle,
            ordered, "mse", "linear-RGB MSE (lower is better)",
            lambda item: (f'{item["scene"]} · {item["sampler"]} '
                          f'· reference {"adequate" if item["reference_adequate"] else "INADEQUATE"} '
                          f'(n={len(item["fixed_seeds"])})'),
            evidence_status=status_label)
        retain("equal-time-quality.svg", "quality.equal_time measured MSE")
        horizontal_metric_svg(
            output / "equal-time-samples.svg", "Equal-time completed samples", subtitle,
            ordered, "completed_samples", "actual whole completed samples",
            lambda item: (f'{item["scene"]} · {item["sampler"]} '
                          f'(n={len(item["fixed_seeds"])})'))
        retain("equal-time-samples.svg", "quality.equal_time actual completed sample counts")
        gpu_records = [item for item in ordered if "renderer_gpu_ms" in item]
        if gpu_records:
            horizontal_metric_svg(
                output / "equal-time-gpu.svg", "Equal-time measured renderer GPU time",
                "GPU measurements are renderer benchmark values, not the process wall clock.",
                gpu_records, "renderer_gpu_ms", "actual renderer GPU milliseconds",
                lambda item: (f'{item["scene"]} · {item["sampler"]} '
                              f'(n={len(item["fixed_seeds"])})'))
            retain("equal-time-gpu.svg", "quality.equal_time renderer_gpu_ms")
        derived["equal_time"] = {
            "records": ordered,
            "quality_evidence_status": quality_status,
            "mse_comparisons": comparison_rows(ordered, "mse"),
            "completed_sample_comparisons": comparison_rows(ordered, "completed_samples"),
        }
    else:
        missing.append("quality.equal_time")

    if timings:
        static = [item for item in timings if item["kind"] == "static"]
        motion = [item for item in timings if item["kind"] != "static"]
        if static:
            ordered = sorted(static, key=lambda item: (item["scene"], item["sampler"]))
            horizontal_metric_svg(
                output / "fixed-spp-timing.svg", "Fixed-256-spp renderer GPU cost",
                "Bars summarize retained measured repeats only; warmups are excluded. "
                "Whiskers show the measured min–max range.",
                ordered, "renderer_gpu_ms_per_completed_sample",
                "renderer GPU milliseconds per completed sample",
                lambda item: (
                    f'{item["scene"]} · {item["sampler"]} '
                    f'(n={item["renderer_gpu_ms_per_completed_sample"]["count"]})'
                ))
            retain("fixed-spp-timing.svg", "timing.groups static measured GPU cost")
            derived["static_timing"] = {
                "records": ordered,
                "sampler_comparisons": comparison_rows(
                    ordered, "renderer_gpu_ms_per_completed_sample"),
            }
        else:
            missing.append("timing.groups static")
        if motion:
            ordered = sorted(motion, key=lambda item: (item["kind"], item["scene"]))
            motion_status = motion_disclosure(timing, ordered)
            horizontal_metric_svg(
                output / "motion-timing.svg", "Motion shutter renderer GPU cost",
                "Frozen midpoint and full-shutter runs have the same normalized authored "
                "swept-geometry design; bars summarize retained measured repeats only.",
                ordered, "renderer_gpu_ms_per_completed_sample",
                "renderer GPU milliseconds per completed sample",
                lambda item: (
                    f'{item.get("comparison_role") or item["kind"].replace("_", " ")} '
                    f'· {item["scene"]} '
                    f'(n={item["renderer_gpu_ms_per_completed_sample"]["count"]})'
                ))
            retain("motion-timing.svg", "timing.groups motion measured GPU cost")
            derived["motion_timing"] = {
                "records": ordered,
                "motion_cost_and_swept_geometry_evidence": motion_status,
            }
        else:
            missing.append("timing.groups motion")
    else:
        missing.append("timing.groups")
    return generated, derived, missing


def self_test() -> dict[str, Any]:
    seeds = [947, 104729, 130363]
    convergence = []
    for scene_index, scene in enumerate(("diffuse", "glass")):
        for samples in (16, 64, 256):
            per_seed = []
            for seed_index, seed in enumerate(seeds):
                base = (scene_index + 1) * (seed_index + 2) / samples
                per_seed.append({
                    "seed": seed,
                    "counter": {"mse": base, "completed_samples": samples},
                    "halton": {"mse": base * 0.8, "completed_samples": samples},
                })
            convergence.append({
                "scene": scene,
                "samples": samples,
                "counter": {"mean_mse": math.fsum(
                    item["counter"]["mse"] for item in per_seed) / len(per_seed)},
                "halton": {"mean_mse": math.fsum(
                    item["halton"]["mse"] for item in per_seed) / len(per_seed)},
                "relative_mse_reduction": 0.2,
                "strict_improvement": True,
                "intensity_paired": {"passed": True},
                "per_seed": per_seed,
            })
    equal = []
    for scene in ("diffuse", "glass"):
        for sampler, scale in (("counter", 1.0), ("halton", 0.8)):
            equal.append({
                "scene": scene,
                "sampler": sampler,
                "seeds": seeds,
                "reference_adequate": True,
                "metrics": {
                    "completed_samples": {"values": [101, 103, 102]},
                    "mse": {"values": [0.04 * scale, 0.05 * scale, 0.045 * scale]},
                    "renderer_gpu_ms": {"values": [3900, 3920, 3910]},
                },
            })
    groups = []
    for scene in ("diffuse", "glass"):
        for sampler, scale in (("counter", 1.0), ("halton", 1.1)):
            groups.append({
                "kind": "static", "scene": scene, "sampler": sampler,
                "metrics": {
                    "renderer_gpu_ms_per_completed_sample": {
                        "values": [2.0 * scale, 2.1 * scale, 1.9 * scale]},
                    "completed_samples": {"values": [256, 256, 256]},
                },
            })
    for kind, scale in (("motion_midpoint", 1.0), ("motion_full_shutter", 1.2)):
        groups.append({
            "kind": kind, "scene": kind, "sampler": "halton",
            "comparison_role": (
                "midpoint" if kind == "motion_midpoint" else "full-shutter"),
            "warmup_run_ids": ["w1", "w2", "w3"],
            "measured_run_ids": ["m1", "m2", "m3"],
            "warmups_excluded_from_summary": True,
            "metrics": {
                "renderer_gpu_ms_per_completed_sample": {
                    "values": [3.0 * scale, 3.1 * scale, 2.9 * scale]},
                "completed_samples": {"values": [256, 256, 256]},
            },
        })
    summary = {
        "schema_version": 1,
        "status": "passed",
        "phases": {
            "quality": {
                "reference_precision": [{
                    "scene": scene,
                    "adequate": True,
                    "retry_performed": False,
                    "selected_samples_per_seed": 8192,
                    "attempts": [],
                } for scene in ("diffuse", "glass")],
                "convergence": convergence,
                "equal_time": {"budget_ms": 4000, "summaries": equal},
                "pooled_mse": {"counter": 1.0, "halton": 0.8,
                               "relative_reduction": 0.2},
                "comparisons": {"strict_halton_mse_improvements": 6, "total": 6},
                "intensity_criteria": {"passed": 6, "total": 6},
                "qualification": {
                    "status": "qualified",
                    "criteria": {"adequate_references": True},
                    "failed_criteria": [],
                },
            },
            "timing": {
                "groups": groups,
                "motion_cost": {
                    "swept_geometry": {
                        "equal": True,
                        "normalized_scene_sha256": "a" * 64,
                        "excluded_variant_fields": ["Camera.FILE", "Camera.SHUTTER"],
                    },
                    "median_ratios": {
                        "renderer_gpu_ms_per_completed_sample_full_over_midpoint": 1.2,
                    },
                },
            },
        },
    }
    with tempfile.TemporaryDirectory(prefix="plot-hedge-self-test-") as temporary:
        root = Path(temporary)
        generated, derived, missing = create_plots(summary, summary, root)
        expected = {
            "sampler-convergence.svg", "equal-time-quality.svg",
            "equal-time-samples.svg", "equal-time-gpu.svg",
            "fixed-spp-timing.svg", "motion-timing.svg",
        }
        actual = {item["filename"] for item in generated}
        if actual != expected or missing:
            raise AssertionError(f"unexpected self-test plot set: {actual}, missing={missing}")
        for item in generated:
            svg = (root / item["filename"]).read_text(encoding="utf-8")
            if not svg.startswith("<svg") or "</svg>" not in svg:
                raise AssertionError(f"invalid SVG envelope: {item['filename']}")
        if derived["convergence"]["records"][0]["mse"]["count"] != len(seeds):
            raise AssertionError("per-seed variability was not retained")
        bare = root / "bare-phase-inputs"
        bare.mkdir()
        bare_generated, _, bare_missing = create_plots(
            summary["phases"]["quality"], summary["phases"]["timing"], bare)
        if {item["filename"] for item in bare_generated} != expected or bare_missing:
            raise AssertionError("bare phase summary inputs were not recognized")

        regression = comparison_rows([
            {"scene": "synthetic", "sampler": "counter",
             "mse": {"median": 1.0}},
            {"scene": "synthetic", "sampler": "halton",
             "mse": {"median": 1.25}},
        ], "mse")[0]
        if regression["signed_halton_reduction"] != -0.25 \
                or regression["regression"] is not True:
            raise AssertionError("signed regression was not retained")

        inconclusive = json.loads(json.dumps(summary))
        inconclusive_quality = inconclusive["phases"]["quality"]
        inconclusive_quality["qualification"]["status"] = "inconclusive"
        inconclusive_quality["qualification"]["criteria"][
            "adequate_references"
        ] = False
        inconclusive_quality["qualification"]["failed_criteria"] = [
            "adequate_references"
        ]
        for reference in inconclusive_quality["reference_precision"]:
            reference["adequate"] = False
        for equal_time_summary in inconclusive_quality["equal_time"]["summaries"]:
            equal_time_summary["reference_adequate"] = False
        inconclusive_root = root / "inconclusive"
        inconclusive_root.mkdir()
        _, inconclusive_derived, inconclusive_missing = create_plots(
            inconclusive, inconclusive, inconclusive_root)
        if inconclusive_missing:
            raise AssertionError(
                f"inconclusive fixture lost measured series: {inconclusive_missing}")
        visible_status = "QUALITY: INCONCLUSIVE · REFERENCES INADEQUATE"
        for filename in ("sampler-convergence.svg", "equal-time-quality.svg"):
            svg = (inconclusive_root / filename).read_text(encoding="utf-8")
            if visible_status not in svg:
                raise AssertionError(
                    f"inadequate-reference status is not visible in {filename}")
        if inconclusive_derived["equal_time"]["quality_evidence_status"][
                "status"] != "inconclusive":
            raise AssertionError("equal-time derived data lost inconclusive status")
        try:
            metric_object({"mse_median": 0.5}, "mse")
        except EvidenceError:
            pass
        else:
            raise AssertionError("missing measured ranges were silently imputed")
    return {
        "passed": True,
        "synthetic_only": True,
        "generated_svg_count": len(expected),
        "checks": [
            "convergence statistics from per-seed values",
            "signed sampler comparisons and explicit regressions",
            "equal-time actual sample/GPU metrics",
            "visible inconclusive/inadequate-reference disclosure",
            "missing measured ranges rejected without imputation",
            "static and motion timing charts",
            "SVG envelope and hashes",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path,
                        help="quality evidence directory (or its summary.json)")
    parser.add_argument("--timing", type=Path,
                        help="timing evidence directory (or its summary.json)")
    parser.add_argument("--output", type=Path,
                        help="required brand-new output directory")
    parser.add_argument("--self-test", action="store_true",
                        help="exercise synthetic measured-series parsing and SVG generation")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required")
    if args.quality is None and args.timing is None:
        parser.error("at least one of --quality or --timing is required")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output directory already exists: {output}")

    quality_summary = None
    timing_summary = None
    inputs: dict[str, Any] = {}
    try:
        if args.quality is not None:
            quality_summary, inputs["quality"] = load_summary(args.quality, "quality")
        if args.timing is not None:
            timing_summary, inputs["timing"] = load_summary(args.timing, "timing")
        # Parse all numeric evidence before creating the destination. This avoids
        # leaving an apparently usable plot directory after malformed input.
        quality_phase = phase(quality_summary, "quality") if quality_summary else None
        timing_phase = phase(timing_summary, "timing") if timing_summary else None
        convergence = convergence_data(quality_phase)
        equal_time = equal_time_data(quality_phase)
        timings = timing_data(timing_phase)
        disclosure = quality_disclosure(quality_phase) if quality_phase else None
        if not convergence and not equal_time and not timings:
            raise EvidenceError("inputs contain no measured quality or timing series")
        motion_disclosure(
            timing_phase, [item for item in timings if item["kind"] != "static"])
        output.mkdir(parents=True, exist_ok=False)
        generated, derived, missing = create_plots(
            quality_summary, timing_summary, output)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "complete",
            "quality_qualification_status": (
                disclosure["status"] if disclosure is not None else None),
            "quality_references_adequate": (
                disclosure["adequate_references"] if disclosure is not None else None),
            "inputs": inputs,
            "generated_svgs": generated,
            "missing_measured_sections": missing,
            "derived_numeric_data": derived,
            "policy": {
                "external_plotting_packages": False,
                "synthetic_or_imputed_production_values": False,
                "whiskers": "full measured min-to-max range",
                "timing_warmups": "excluded upstream by measure_hedge.py",
            },
        }
        (output / "plot-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (EvidenceError, OSError) as error:
        parser.error(str(error))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
