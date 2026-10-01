#!/usr/bin/env python3
"""Create original SVG benchmark charts from measured CSV/JSON evidence."""
import argparse
import csv
import html
import json
from pathlib import Path
import statistics


BOUNCE_TIME_FIELDS = (
    ('intersection_ms', 'intersection'),
    ('sort_ms', 'sort'),
    ('shade_ms', 'shade'),
    ('compaction_ms', 'compaction'),
    ('count_ms', 'count'),
    ('queue_ms', 'queue'),
    ('queue_host_ms', 'queue_host'),
    ('diffuse_ms', 'diffuse'),
    ('mirror_ms', 'mirror'),
    ('dielectric_ms', 'dielectric'),
    ('emission_ms', 'emission'),
    ('terminal_ms', 'terminal'),
)
BOUNCE_COUNT_FIELDS = (
    'queued_diffuse',
    'queued_mirror',
    'queued_dielectric',
    'queued_emission',
    'queued_terminal',
    'kernel_dispatches',
    'rr_tested',
    'rr_terminated',
)
SPECIALIZED_SHADE_FIELDS = (
    'diffuse', 'mirror', 'dielectric', 'emission', 'terminal')
WAVE_PHASE_LABELS = {
    'intersection': 'intersection',
    'sort': 'legacy sort',
    'queue': 'queue construction',
    'shade': 'unified shade',
    'diffuse': 'diffuse shade',
    'mirror': 'mirror shade',
    'dielectric': 'dielectric shade',
    'emission': 'emission shade',
    'terminal': 'terminal shade',
    'compaction': 'compaction',
    'count': 'active count',
}
WAVE_PHASE_COLORS = {
    'intersection': '#355f8a',
    'sort': '#7b5aa6',
    'queue': '#c05a32',
    'shade': '#3f8f73',
    'diffuse': '#4ca66b',
    'mirror': '#76b7b2',
    'dielectric': '#4f86c6',
    'emission': '#d6a33c',
    'terminal': '#9b8066',
    'compaction': '#d16ba5',
    'count': '#7b8794',
}
LEGACY_BENCHMARK_FIELDS = (
    'record_type', 'sample', 'bounce', 'active_before', 'active_after',
    'intersection_ms', 'sort_ms', 'shade_ms', 'compaction_ms', 'count_ms',
    'host_ms', 'gpu_ms', 'measured_samples', 'render_loop_wall_ms',
)


def chart(path, title, rows, value_key, unit, *, min_key=None, max_key=None,
          subtitle=None):
    width, left, top, row_height, right = 1000, 310, 100, 46, 100
    height = top + len(rows) * row_height + 70
    if min_key is None or max_key is None:
        bounds = None
    else:
        bounds = (min_key, max_key)
    maximum = max((
        max(row[value_key], row.get(max_key, row[value_key]))
        if max_key else row[value_key]
        for row in rows
    ), default=1) * 1.12 or 1
    plot_width = width - left - right
    caption = subtitle or (
        f'Three measured repeats; bars show median {unit}, whiskers min–max. '
        'Lower is faster.')
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f6f8fc"/>',
        '<g font-family="system-ui, sans-serif" fill="#1a2538">',
        f'<text x="28" y="38" font-size="24" font-weight="600">'
        f'{html.escape(title)}</text>',
        f'<text x="28" y="67" font-size="14">'
        f'{html.escape(caption)}</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        x = left + plot_width * tick / 5
        out.append(
            f'<path d="M{x:.2f} {top - 8} V{height - 45}" '
            'stroke="#dce2ec"/>')
        out.append(
            f'<text x="{x:.2f}" y="{height - 24}" text-anchor="middle" '
            f'font-size="13">{value:.2f}</text>')
    colors = ('#2166ac', '#ca6e22', '#318f68')
    for index, row in enumerate(rows):
        y = top + index * row_height
        value = row[value_key]
        out.append(
            f'<text x="{left - 14}" y="{y + 21}" text-anchor="end" '
            f'font-size="14">{html.escape(row["label"])}</text>')
        out.append(
            f'<rect x="{left}" y="{y}" '
            f'width="{plot_width * value / maximum:.2f}" height="29" rx="3" '
            f'fill="{colors[index % len(colors)]}"/>')
        if bounds and all(key in row for key in bounds):
            xmin = left + plot_width * row[min_key] / maximum
            xmax = left + plot_width * row[max_key] / maximum
            out.append(
                f'<path d="M{xmin:.2f} {y + 14} H{xmax:.2f} '
                f'M{xmin:.2f} {y + 8} V{y + 20} '
                f'M{xmax:.2f} {y + 8} V{y + 20}" '
                'stroke="#172235" stroke-width="2"/>')
        out.append(
            f'<text x="{left + plot_width * value / maximum + 8:.2f}" '
            f'y="{y + 21}" font-size="13">{value:.3f}</text>')
    out.extend(('</g>', '</svg>'))
    path.write_text('\n'.join(out) + '\n')


def stacked_wave_phase_chart(path, rows):
    width, left, top, row_height, right = 1180, 330, 154, 42, 120
    height = top + len(rows) * row_height + 72
    totals = [row['wavefront_phase_stack_total_ms_per_sample'] for row in rows]
    maximum = max(totals, default=1) * 1.12 or 1
    plot_width = width - left - right
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f6f8fc"/>',
        '<g font-family="system-ui, sans-serif" fill="#1a2538">',
        '<text x="28" y="38" font-size="24" font-weight="600">'
        'Wavefront scheduling GPU phase medians</text>',
        '<text x="28" y="66" font-size="14">'
        'Stacks show nonoverlapping phase medians in ms/sample; '
        'overall GPU elapsed is in performance-wavefront.svg.</text>',
        '<text x="28" y="87" font-size="14">'
        'Wavefront bars use the five specialized shader timings instead of '
        'aggregate shade_ms; queue_host is excluded.</text>',
    ]
    legend_x = 28
    legend_y = 109
    for phase, label in WAVE_PHASE_LABELS.items():
        estimated_width = 34 + len(label) * 7
        if legend_x + estimated_width > width - 28:
            legend_x = 28
            legend_y += 23
        out.append(
            f'<rect x="{legend_x}" y="{legend_y - 11}" width="13" '
            f'height="13" rx="2" fill="{WAVE_PHASE_COLORS[phase]}"/>')
        out.append(
            f'<text x="{legend_x + 19}" y="{legend_y}" font-size="12">'
            f'{html.escape(label)}</text>')
        legend_x += estimated_width
    for tick in range(6):
        value = maximum * tick / 5
        x = left + plot_width * tick / 5
        out.append(
            f'<path d="M{x:.2f} {top - 8} V{height - 45}" '
            'stroke="#dce2ec"/>')
        out.append(
            f'<text x="{x:.2f}" y="{height - 23}" text-anchor="middle" '
            f'font-size="13">{value:.2f}</text>')
    for index, row in enumerate(rows):
        y = top + index * row_height
        out.append(
            f'<text x="{left - 14}" y="{y + 20}" text-anchor="end" '
            f'font-size="13">{html.escape(row["label"])}</text>')
        cursor = left
        for phase, value in row['wavefront_phase_stack_ms_per_sample'].items():
            segment_width = plot_width * value / maximum
            out.append(
                f'<rect x="{cursor:.2f}" y="{y}" width="{segment_width:.2f}" '
                f'height="27" fill="{WAVE_PHASE_COLORS[phase]}">'
                f'<title>{html.escape(WAVE_PHASE_LABELS[phase])}: '
                f'{value:.6f} ms/sample</title></rect>')
            cursor += segment_width
        out.append(
            f'<text x="{cursor + 7:.2f}" y="{y + 19}" font-size="12">'
            f'{row["wavefront_phase_stack_total_ms_per_sample"]:.3f}</text>')
    out.extend(('</g>', '</svg>'))
    path.write_text('\n'.join(out) + '\n')


def numeric_sum(rows, field):
    values = [row.get(field, '') for row in rows]
    if not values or any(value in (None, '') for value in values):
        return None
    return sum(float(value) for value in values)


def summarize_benchmark(path, samples):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = set(LEGACY_BENCHMARK_FIELDS)
        required.update(field for field, _ in BOUNCE_TIME_FIELDS)
        required.update(BOUNCE_COUNT_FIELDS)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f'{path}: missing benchmark columns {sorted(missing)}')
        records = list(reader)
    if any(None in row or any(value is None for value in row.values())
           for row in records):
        raise ValueError(f'{path}: benchmark row is not aligned with its header')
    overall_rows = [row for row in records if row['record_type'] == 'overall']
    if len(overall_rows) != 1:
        raise ValueError(f'{path}: expected exactly one overall row')
    bounce_rows = [row for row in records if row['record_type'] == 'bounce']
    if not bounce_rows:
        raise ValueError(f'{path}: no bounce rows')
    overall = overall_rows[0]
    phases = {}
    for csv_name, summary_name in BOUNCE_TIME_FIELDS:
        total = numeric_sum(bounce_rows, csv_name)
        if total is not None:
            phases[summary_name] = total / samples
    counters = {}
    for field in BOUNCE_COUNT_FIELDS:
        total = numeric_sum(bounce_rows, field)
        if total is not None:
            counters[field] = total / samples

    rr_rate = None
    tested = numeric_sum(bounce_rows, 'rr_tested')
    terminated = numeric_sum(bounce_rows, 'rr_terminated')
    if tested is not None and terminated is not None and tested > 0:
        rr_rate = 100.0 * terminated / tested

    return {
        'gpu': float(overall['gpu_ms']) / samples,
        'loop': float(overall['render_loop_wall_ms']) / samples,
        'phases': phases,
        'counters': counters,
        'rr_termination_percent': rr_rate,
    }


def shading_value(phases):
    """Return one shading total without double-counting aggregate timings."""
    if 'shade' in phases:
        return phases['shade'], 'aggregate shade timing'
    if all(field in phases for field in SPECIALIZED_SHADE_FIELDS):
        return (sum(phases[field] for field in SPECIALIZED_SHADE_FIELDS),
                'specialized shading fallback (aggregate unavailable)')
    return None, None


def selected_wave_phases(variant, phases):
    """Choose a nonoverlapping phase stack for a wave-scheduling variant."""
    selected = {'intersection': phases['intersection']}
    if variant == 'legacy-sorted':
        selected['sort'] = phases['sort']
    elif variant == 'wavefront':
        selected['queue'] = phases['queue']
    elif variant != 'legacy-unsorted':
        raise ValueError(f'unknown wave scheduling variant: {variant}')

    if variant == 'wavefront':
        for field in SPECIALIZED_SHADE_FIELDS:
            selected[field] = phases[field]
    else:
        selected['shade'] = phases['shade']
    selected['compaction'] = phases['compaction']
    selected['count'] = phases['count']
    return selected


def summarize_runs(directory, measured):
    summaries = []
    identities = dict.fromkeys(
        (row['experiment'], row['variant']) for row in measured)
    for experiment, variant in identities:
        matching = [
            row for row in measured
            if row['experiment'] == experiment and row['variant'] == variant
        ]
        benchmark_runs = []
        for run in matching:
            benchmark_runs.append(summarize_benchmark(
                directory / (run['prefix'] + '.benchmark.csv'),
                float(run['samples'])))
        gpu = [item['gpu'] for item in benchmark_runs]
        loop = [item['loop'] for item in benchmark_runs]
        process = [float(run['process_seconds']) for run in matching]

        phase_keys = set(benchmark_runs[0]['phases'])
        counter_keys = set(benchmark_runs[0]['counters'])
        for item in benchmark_runs[1:]:
            if set(item['phases']) != phase_keys:
                raise ValueError(
                    f'{experiment}/{variant}: inconsistent timing columns')
            if set(item['counters']) != counter_keys:
                raise ValueError(
                    f'{experiment}/{variant}: inconsistent counter columns')
        phases = {
            key: statistics.median(item['phases'][key]
                                   for item in benchmark_runs)
            for key in sorted(phase_keys)
        }
        phase_min = {
            key: min(item['phases'][key] for item in benchmark_runs)
            for key in sorted(phase_keys)
        }
        phase_max = {
            key: max(item['phases'][key] for item in benchmark_runs)
            for key in sorted(phase_keys)
        }
        counters = {
            key: statistics.median(item['counters'][key]
                                   for item in benchmark_runs)
            for key in sorted(counter_keys)
        }
        summary = dict(
            experiment=experiment,
            variant=variant,
            label=experiment + ' / ' + variant,
            gpu_ms_per_sample=statistics.median(gpu),
            gpu_min=min(gpu),
            gpu_max=max(gpu),
            loop_ms_per_sample=statistics.median(loop),
            loop_min=min(loop),
            loop_max=max(loop),
            phases=phases,
            repeats=len(matching),
            process_median_seconds=statistics.median(process),
            process_min_seconds=min(process),
            process_max_seconds=max(process),
            phase_min=phase_min,
            phase_max=phase_max,
            counters_per_sample=counters,
            phase_relationships={
                'shade': (
                    'aggregate shader time in both scheduling modes; the five '
                    'specialized wavefront fields are an alternative breakdown'),
                'queue': 'separate GPU queue-construction timing',
                'queue_host': 'separate host bookkeeping/submission timing',
            },
        )

        shading_runs = [shading_value(item['phases'])
                        for item in benchmark_runs]
        if all(value is not None for value, _ in shading_runs):
            sources = {source for _, source in shading_runs}
            if len(sources) != 1:
                raise ValueError(
                    f'{experiment}/{variant}: inconsistent shading metrics')
            values = [value for value, _ in shading_runs]
            summary.update(
                gpu_shading_ms_per_sample=statistics.median(values),
                gpu_shading_min=min(values),
                gpu_shading_max=max(values),
                gpu_shading_source=sources.pop(),
            )
        rates = [item['rr_termination_percent'] for item in benchmark_runs]
        available_rates = [value for value in rates if value is not None]
        if available_rates:
            if len(available_rates) != len(rates):
                raise ValueError(
                    f'{experiment}/{variant}: inconsistent roulette counters')
            summary.update(
                rr_termination_percent=statistics.median(available_rates),
                rr_termination_min=min(available_rates),
                rr_termination_max=max(available_rates),
            )
        if experiment.startswith('wave-'):
            selected = selected_wave_phases(variant, phases)
            summary.update(
                wavefront_phase_stack_ms_per_sample=selected,
                wavefront_phase_stack_total_ms_per_sample=sum(
                    selected.values()),
                wavefront_phase_stack_note=(
                    'Nonoverlapping measured phase medians. Legacy modes use '
                    'aggregate shade; wavefront mode uses the five specialized '
                    'shader timings instead. queue_host is excluded.'),
            )
        summaries.append(summary)
    return summaries


def comparison(kind, workload, baseline, candidate, metric):
    baseline_value = baseline[metric]
    candidate_value = candidate[metric]
    difference = candidate_value - baseline_value
    if difference < 0:
        direction = 'faster'
    elif difference > 0:
        direction = 'slower'
    else:
        direction = 'unchanged'
    percent = (100.0 * difference / baseline_value
               if baseline_value != 0 else None)
    return dict(
        comparison=kind,
        workload=workload,
        baseline=baseline['variant'],
        candidate=candidate['variant'],
        metric=metric,
        unit={
            'gpu_ms_per_sample': 'milliseconds per sample',
            'loop_ms_per_sample': 'milliseconds per sample',
            'process_median_seconds': 'seconds',
        }[metric],
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        difference=candidate_value - baseline_value,
        relative_change_percent=percent,
        direction=direction,
    )


def performance_comparisons(summaries):
    by_identity = {
        (item['experiment'], item['variant']): item for item in summaries
    }
    comparisons = []
    metrics = (
        'gpu_ms_per_sample',
        'loop_ms_per_sample',
        'process_median_seconds',
    )
    for workload in ('uniform', 'mixed', 'closed', 'mesh-heavy'):
        experiment = 'wave-' + workload
        unsorted = by_identity[(experiment, 'legacy-unsorted')]
        sorted_run = by_identity[(experiment, 'legacy-sorted')]
        wavefront = by_identity[(experiment, 'wavefront')]
        for metric in metrics:
            comparisons.extend((
                comparison('wave-scheduling', workload, unsorted, sorted_run,
                           metric),
                comparison('wave-scheduling', workload, unsorted, wavefront,
                           metric),
                comparison('wave-scheduling', workload, sorted_run, wavefront,
                           metric),
            ))
    for workload in ('open', 'closed'):
        experiment = 'rr-' + workload
        for metric in metrics:
            comparisons.append(comparison(
                'russian-roulette', workload,
                by_identity[(experiment, 'off')],
                by_identity[(experiment, 'on')], metric))
    return comparisons


def discover_cpu_profiles(experiments, explicit):
    if explicit:
        candidates = [explicit]
    else:
        candidates = sorted(experiments.glob('**/summary.json'))
    profiles = []
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            if explicit:
                raise
            continue
        records = data.get('records') if isinstance(data, dict) else None
        if not isinstance(records, list) or not records:
            continue
        required = {
            'scene', 'load_including_bvh_ms', 'bvh_rebuild_ms',
            'load_including_bvh_median_ms', 'bvh_rebuild_median_ms',
        }
        if all(isinstance(record, dict) and required <= set(record)
               for record in records):
            profiles.append((path, data))
    if explicit and not profiles:
        raise ValueError(
            f'{explicit}: not a procedural CPU profile summary')
    if len(profiles) > 1:
        names = ', '.join(str(path) for path, _ in profiles)
        raise ValueError(
            'multiple procedural CPU profiles found; select one with '
            f'--procedural-cpu-profile: {names}')
    return profiles


def plot_cpu_profiles(output, profiles):
    if not profiles:
        return
    rows = []
    for path, profile in profiles:
        for record in profile['records']:
            label = Path(record['scene']).stem
            if len(profiles) > 1:
                label = path.parent.name + ' / ' + label
            load_values = [float(value)
                           for value in record['load_including_bvh_ms']]
            rebuild_values = [float(value)
                              for value in record['bvh_rebuild_ms']]
            rows.extend((
                dict(label=label + ' / load incl. BVH',
                     cpu_ms=float(record['load_including_bvh_median_ms']),
                     cpu_min=min(load_values), cpu_max=max(load_values)),
                dict(label=label + ' / BVH rebuild',
                     cpu_ms=float(record['bvh_rebuild_median_ms']),
                     cpu_min=min(rebuild_values), cpu_max=max(rebuild_values)),
            ))
    chart(
        output / 'performance-procedural-cpu.svg',
        'Procedural mesh CPU loading costs', rows, 'cpu_ms', 'CPU ms',
        min_key='cpu_min', max_key='cpu_max',
        subtitle=(
            'Median CPU milliseconds; whiskers min–max; load includes the '
            'initial BVH. Rebuild is shown separately.'),
    )


def write_single_iteration(directory, output, measured):
    iteration_rows = []
    for scene in ('open', 'closed'):
        run = next(
            row for row in measured
            if row['experiment'] == 'compact-' + scene
            and row['variant'] == 'on')
        with (directory / (run['prefix'] + '.benchmark.csv')).open(
                newline='') as stream:
            records = [row for row in csv.DictReader(stream)
                       if row['record_type'] == 'bounce']
        first_sample = min(int(row['sample']) for row in records)
        for record in records:
            if int(record['sample']) != first_sample:
                continue
            row = dict(
                scene=scene,
                sample=first_sample,
                bounce=int(record['bounce']),
                active_before=int(record['active_before']),
                active_after=int(record['active_after']),
            )
            for field, _ in BOUNCE_TIME_FIELDS:
                if record.get(field, '') != '':
                    row[field] = float(record[field])
            for field in BOUNCE_COUNT_FIELDS:
                if record.get(field, '') != '':
                    row[field] = int(record[field])
            iteration_rows.append(row)

    old_fields = [
        'scene', 'sample', 'bounce', 'active_before', 'active_after',
        'intersection_ms', 'sort_ms', 'shade_ms', 'compaction_ms', 'count_ms',
    ]
    new_fields = [field for field, _ in BOUNCE_TIME_FIELDS
                  if field not in old_fields]
    new_fields.extend(BOUNCE_COUNT_FIELDS)
    fieldnames = old_fields + [
        field for field in new_fields
        if any(field in row for row in iteration_rows)
    ]
    with (directory / 'single-iteration.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(iteration_rows)

    iteration_plot = [
        dict(label=row['scene'].title() + ' / bounce ' + str(row['bounce']),
             active=row['active_after'])
        for row in iteration_rows
    ]
    chart(
        output / 'active-paths-iteration.svg',
        'Single-iteration path survival: open and closed',
        iteration_plot, 'active', 'surviving paths',
        subtitle=(
            'Surviving paths after each bounce in the first measured sample; '
            'seed 947, 160 × 106 pixels.'),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiments', type=Path)
    parser.add_argument(
        '--output', type=Path,
        default=Path(__file__).resolve().parents[1] / 'img')
    parser.add_argument(
        '--procedural-cpu-profile', type=Path,
        help='actual summary.json emitted by profile_mesh_load.py')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.experiments / 'runs.csv').open(newline='') as stream:
        runs = list(csv.DictReader(stream))
    measured = [row for row in runs if row['phase'] == 'measured']
    summaries = summarize_runs(args.experiments, measured)
    (args.experiments / 'timing-summary.json').write_text(
        json.dumps(summaries, indent=2) + '\n')

    chart(
        args.output / 'performance-bvh.svg',
        'BVH traversal across mesh complexity',
        [row for row in summaries if row['experiment'].startswith('bvh-')],
        'gpu_ms_per_sample', 'GPU ms per sample',
        min_key='gpu_min', max_key='gpu_max')
    chart(
        args.output / 'performance-scheduling.svg',
        'Compaction and material sorting',
        [row for row in summaries
         if row['experiment'].startswith(('compact-', 'sort-'))],
        'gpu_ms_per_sample', 'GPU ms per sample',
        min_key='gpu_min', max_key='gpu_max')
    chart(
        args.output / 'performance-features.svg',
        'Surface and camera feature costs',
        [row for row in summaries
         if row['experiment'] in ('texture', 'bump', 'refraction', 'dof', 'nee')],
        'gpu_ms_per_sample', 'GPU ms per sample',
        min_key='gpu_min', max_key='gpu_max')
    chart(
        args.output / 'performance-checkpoint.svg',
        'Checkpointing during rendering',
        [row for row in summaries if row['experiment'] == 'checkpoint'],
        'loop_ms_per_sample', 'render-loop ms per sample',
        min_key='loop_min', max_key='loop_max')

    wave_rows = [row for row in summaries
                 if row['experiment'].startswith('wave-')]
    if wave_rows:
        chart(
            args.output / 'performance-wavefront.svg',
            'Unified and material-queue scheduling', wave_rows,
            'gpu_ms_per_sample', 'GPU ms per sample',
            min_key='gpu_min', max_key='gpu_max',
            subtitle=(
                'Median GPU ms/sample; whiskers min–max; all three scheduling '
                'modes shown. Lower is faster.'))
        if len(wave_rows) != 12:
            raise ValueError(
                f'wavefront phase chart needs 12 variants, found {len(wave_rows)}')
        stacked_wave_phase_chart(
            args.output / 'performance-wavefront-phases.svg', wave_rows)
        wavefront_rows = [row for row in wave_rows
                          if row['variant'] == 'wavefront']
        if (wavefront_rows
                and all('queue' in row['phases']
                        and 'queue_host' in row['phases']
                        for row in wavefront_rows)):
            overhead_rows = []
            for row in wavefront_rows:
                overhead_rows.extend((
                    dict(
                        label=row['experiment'] + ' / queue GPU',
                        overhead=row['phases']['queue'],
                        overhead_min=row['phase_min']['queue'],
                        overhead_max=row['phase_max']['queue'],
                    ),
                    dict(
                        label=row['experiment'] + ' / queue host',
                        overhead=row['phases']['queue_host'],
                        overhead_min=row['phase_min']['queue_host'],
                        overhead_max=row['phase_max']['queue_host'],
                    ),
                ))
            chart(
                args.output / 'performance-wavefront-overhead.svg',
                'Material-queue scheduling overhead', overhead_rows,
                'overhead', 'ms per sample',
                min_key='overhead_min', max_key='overhead_max',
                subtitle=(
                    'Median ms/sample; GPU queue and host submission shown '
                    'independently, without summing them.'))
    rr_rows = [row for row in summaries if row['experiment'].startswith('rr-')]
    if rr_rows:
        chart(
            args.output / 'performance-russian-roulette.svg',
            'Russian roulette: open and closed scenes', rr_rows,
            'gpu_ms_per_sample', 'GPU ms per sample',
            min_key='gpu_min', max_key='gpu_max',
            subtitle=(
                'Median GPU ms/sample; whiskers min–max; paired off/on '
                'timings shown for both workloads.'))
        rr_on_rows = [row for row in rr_rows if row['variant'] == 'on']
        if (rr_on_rows
                and all('rr_termination_percent' in row
                        for row in rr_on_rows)):
            termination_rows = [
                dict(
                    label=row['label'],
                    termination=row['rr_termination_percent'],
                    termination_min=row['rr_termination_min'],
                    termination_max=row['rr_termination_max'],
                )
                for row in rr_on_rows
            ]
            chart(
                args.output / 'russian-roulette-termination.svg',
                'Russian roulette termination rate', termination_rows,
                'termination', 'percent of tested paths',
                min_key='termination_min', max_key='termination_max',
                subtitle=(
                    'Percent of tested paths terminated; median across runs '
                    'with nonzero tests; whiskers min–max.'))

    comparisons = performance_comparisons(summaries)
    (args.experiments / 'performance-comparisons.json').write_text(
        json.dumps(dict(comparisons=comparisons), indent=2) + '\n')

    procedural_rows = [
        row for row in summaries
        if row['experiment'].startswith('procedural-')
    ]
    if (procedural_rows
            and all('gpu_shading_ms_per_sample' in row
                    for row in procedural_rows)):
        chart(
            args.output / 'performance-procedural-shading.svg',
            'Procedural scene GPU shading cost', procedural_rows,
            'gpu_shading_ms_per_sample', 'GPU shading ms per sample',
            min_key='gpu_shading_min', max_key='gpu_shading_max',
            subtitle=(
                'Median GPU shading ms/sample from aggregate shade_ms; '
                'per-type timings are separate breakdowns.'))

    profiles = discover_cpu_profiles(
        args.experiments, args.procedural_cpu_profile)
    plot_cpu_profiles(args.output, profiles)

    # Queue survival at each bounce from the first closed, compacted measured run.
    selected = next(
        row for row in measured
        if row['experiment'] == 'compact-closed' and row['variant'] == 'on')
    with (args.experiments / (selected['prefix'] + '.benchmark.csv')).open(
            newline='') as stream:
        bounce_rows = [row for row in csv.DictReader(stream)
                       if row['record_type'] == 'bounce']
    bounces = sorted(set(int(row['bounce']) for row in bounce_rows))
    survival = [
        dict(
            label='Bounce ' + str(bounce),
            active=statistics.mean(
                float(row['active_after']) for row in bounce_rows
                if int(row['bounce']) == bounce),
        )
        for bounce in bounces
    ]
    chart(
        args.output / 'active-paths.svg',
        'Surviving paths in the closed scene', survival,
        'active', 'surviving paths',
        subtitle=(
            'Mean surviving paths after each bounce across measured '
            'image samples.'))

    write_single_iteration(
        args.experiments, args.output, measured)
    print(json.dumps(dict(
        summaries=summaries,
        comparisons=comparisons,
        procedural_cpu_profiles=[str(path) for path, _ in profiles],
    ), indent=2))


if __name__ == '__main__':
    main()
