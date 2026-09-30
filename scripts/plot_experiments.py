#!/usr/bin/env python3
"""Create original SVG benchmark charts from measured CSV/JSON evidence."""
import argparse
import csv
import html
import json
from pathlib import Path
import statistics


def chart(path, title, rows, value_key, unit):
    width, left, top, row_height, right = 1000, 240, 100, 46, 90
    height = top + len(rows) * row_height + 70
    upper_key = 'gpu_max' if value_key == 'gpu_ms_per_sample' else 'loop_max'
    maximum = max((max(r[value_key], r.get(upper_key, r[value_key])) for r in rows), default=1) * 1.12 or 1
    plot_width = width - left - right
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           f'<rect width="{width}" height="{height}" fill="#f6f8fc"/>',
           '<g font-family="system-ui, sans-serif" fill="#1a2538">',
           f'<text x="28" y="38" font-size="24" font-weight="600">{html.escape(title)}</text>',
           f'<text x="28" y="67" font-size="14">Three measured repeats; bars show median {html.escape(unit)}, whiskers min–max. Lower is faster.</text>']
    for tick in range(6):
        value = maximum * tick / 5
        x = left + plot_width * tick / 5
        out.append(f'<path d="M{x:.2f} {top-8} V{height-45}" stroke="#dce2ec"/>')
        out.append(f'<text x="{x:.2f}" y="{height-24}" text-anchor="middle" font-size="13">{value:.2f}</text>')
    colors = ('#2166ac', '#ca6e22')
    for i, row in enumerate(rows):
        y = top + i * row_height
        value = row[value_key]
        out.append(f'<text x="{left-14}" y="{y+21}" text-anchor="end" font-size="14">{html.escape(row["label"])}</text>')
        out.append(f'<rect x="{left}" y="{y}" width="{plot_width*value/maximum:.2f}" height="29" rx="3" fill="{colors[i%2]}"/>')
        bounds = ('gpu_min', 'gpu_max') if value_key == 'gpu_ms_per_sample' else ('loop_min', 'loop_max')
        if all(key in row for key in bounds):
            xmin, xmax = [left + plot_width * row[key] / maximum for key in bounds]
            out.append(f'<path d="M{xmin:.2f} {y+14} H{xmax:.2f} M{xmin:.2f} {y+8} V{y+20} M{xmax:.2f} {y+8} V{y+20}" stroke="#172235" stroke-width="2"/>')
        out.append(f'<text x="{left+plot_width*value/maximum+8:.2f}" y="{y+21}" font-size="13">{value:.3f}</text>')
    out.extend(('</g>', '</svg>'))
    path.write_text('\n'.join(out) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiments', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'img')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.experiments / 'runs.csv').open(newline='') as stream:
        runs = list(csv.DictReader(stream))
    measured = [r for r in runs if r['phase'] == 'measured']
    summaries = []
    for label, variant in dict.fromkeys((r['experiment'], r['variant']) for r in measured):
        matching = [r for r in measured if r['experiment'] == label and r['variant'] == variant]
        gpu, host, phases = [], [], []
        for run in matching:
            with (args.experiments / (run['prefix'] + '.benchmark.csv')).open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            overall = next(r for r in rows if r['record_type'] == 'overall')
            gpu.append(float(overall['gpu_ms']) / float(run['samples']))
            host.append(float(overall['render_loop_wall_ms']) / float(run['samples']))
            phase = {key: sum(float(r[key]) for r in rows if r['record_type'] == 'bounce') / float(run['samples'])
                     for key in ('intersection_ms', 'sort_ms', 'shade_ms', 'compaction_ms', 'count_ms')}
            phases.append(phase)
        summaries.append(dict(experiment=label, variant=variant, label=label+' / '+variant,
                              gpu_ms_per_sample=statistics.median(gpu), gpu_min=min(gpu), gpu_max=max(gpu),
                              loop_ms_per_sample=statistics.median(host), loop_min=min(host), loop_max=max(host),
                              phases={key:statistics.median(r[key] for r in phases) for key in phases[0]},
                              repeats=len(matching)))
    (args.experiments / 'timing-summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    chart(args.output / 'performance-bvh.svg', 'BVH traversal across mesh complexity',
          [r for r in summaries if r['experiment'].startswith('bvh-')], 'gpu_ms_per_sample', 'GPU ms per sample')
    chart(args.output / 'performance-scheduling.svg', 'Compaction and material sorting',
          [r for r in summaries if r['experiment'].startswith(('compact-', 'sort-'))], 'gpu_ms_per_sample', 'GPU ms per sample')
    chart(args.output / 'performance-features.svg', 'Surface and camera feature costs',
          [r for r in summaries if r['experiment'] in ('texture','bump','refraction','dof','nee')], 'gpu_ms_per_sample', 'GPU ms per sample')
    chart(args.output / 'performance-checkpoint.svg', 'Checkpointing during rendering',
          [r for r in summaries if r['experiment']=='checkpoint'], 'loop_ms_per_sample', 'render-loop ms per sample')
    # Queue survival at each bounce from the first closed, compacted measured run.
    selected = next(r for r in measured if r['experiment']=='compact-closed' and r['variant']=='on')
    with (args.experiments / (selected['prefix'] + '.benchmark.csv')).open(newline='') as stream:
        bounce_rows = [r for r in csv.DictReader(stream) if r['record_type']=='bounce']
    bounces = sorted(set(int(r['bounce']) for r in bounce_rows))
    survival = [dict(label='Bounce '+str(b), active=statistics.mean(float(r['active_after']) for r in bounce_rows if int(r['bounce'])==b)) for b in bounces]
    chart(args.output / 'active-paths.svg', 'Surviving paths in the closed scene', survival, 'active', 'surviving paths')
    # Adjust the generic timing caption for a count chart.
    path=args.output / 'active-paths.svg'
    text=path.read_text().replace('Three measured repeats; bars show median surviving paths, whiskers min–max. Lower is faster.', 'Mean live queue length after each bounce across the measured image samples.')
    path.write_text(text)
    # Preserve a single iteration's queue and phase data for open/closed comparison.
    iteration_rows = []
    for scene in ('open', 'closed'):
        run = next(r for r in measured if r['experiment'] == 'compact-' + scene and r['variant'] == 'on')
        with (args.experiments / (run['prefix'] + '.benchmark.csv')).open(newline='') as stream:
            records = [r for r in csv.DictReader(stream) if r['record_type'] == 'bounce']
        first_sample = min(int(r['sample']) for r in records)
        for record in records:
            if int(record['sample']) == first_sample:
                iteration_rows.append(dict(scene=scene, sample=first_sample,
                    bounce=int(record['bounce']), active_before=int(record['active_before']),
                    active_after=int(record['active_after']),
                    **{key:float(record[key]) for key in ('intersection_ms', 'sort_ms', 'shade_ms', 'compaction_ms', 'count_ms')}))
    with (args.experiments / 'single-iteration.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(iteration_rows[0]))
        writer.writeheader()
        writer.writerows(iteration_rows)
    iteration_plot = [dict(label=r['scene'].title() + ' / bounce ' + str(r['bounce']), active=r['active_after']) for r in iteration_rows]
    path = args.output / 'active-paths-iteration.svg'
    chart(path, 'Single-iteration path survival: open and closed', iteration_plot, 'active', 'surviving paths')
    text = path.read_text().replace('Three measured repeats; bars show median surviving paths, whiskers min–max. Lower is faster.',
        'Live paths after each bounce in the first measured sample; seed 947, 160 × 106 pixels.')
    path.write_text(text)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
