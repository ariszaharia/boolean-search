"""
The final analysis: every confirmatory test of hypotheses.md from the committed run files, the Holm correction over
them, and the exploratory results kept apart (never in the Holm correction).

    venv\\Scripts\\python.exe -m harness.final_analysis

Writes results/analysis/confirmatory.json (every number) and results/analysis/confirmatory.md (the tables).
Nothing is computed by hand: rerunning gives the same numbers (fixed bootstrap seed, harness.stats).

Rules followed (hypotheses.md):
  - Primary analysis: requests where system A returned something; second analysis: empty counts as 0 (Setting).
  - N = 100 is the pool for every test (clarification of H1, "one fixed N"); H3 and H7 after MedCPT at N = 100.
  - Holm over the confirmatory tests that are decided by a p-value (H1, H2, H3 x 2, H4 x 2, H9). H7 (an equivalence
    interval) and H8 (a ratio threshold) are decided by their own pre-registered rules, without a p-value; they are
    reported next to the family. H5 and H6 are not run yet; when they are, they join the family and this is rerun.
"""
import json
import os
import random

from .compare import compare_by_group, compare_runs
from .data import ROOT, load_groups, load_qrels
from .metrics import evaluate_run, means, restrict
from .run_io import read_run, read_set_sizes
from .stats import SEED, equivalent, holm

RUNS = os.path.join(ROOT, 'results', 'runs')
EXPL = os.path.join(ROOT, 'results', 'exploratory')
OUT = os.path.join(ROOT, 'results', 'analysis')
MARGIN = 0.03     # H7
H8_THRESHOLD = 0.80
MODEL = 'gpt-6-luna'


def run(name):
    return read_run(os.path.join(RUNS, name + '.tsv'))


def sizes(name):
    return read_set_sizes(os.path.join(RUNS, name + '_sizes.tsv'))


def pick(comparison, measure, analysis='nonempty'):
    x = comparison['measures'][measure][analysis]
    return {k: x[k] for k in ('n', 'mean_a', 'mean_b', 'mean_diff', 'ci_low', 'ci_high', 'p_value')}


def group_test(split, a, b, group, measure, sizes_a=None):
    q, groups = load_qrels(split), load_groups(split)
    out = compare_by_group(run(a), run(b), q, {group: groups[group]}, sizes_a=sizes_a)[group]
    return {'primary': pick(out, measure), 'second': pick(out, measure, 'empty_as_zero')}


def whole_test(split, a, b, measure, sizes_a=None):
    out = compare_runs(run(a), run(b), load_qrels(split), sizes_a=sizes_a)
    return {'primary': pick(out, measure), 'second': pick(out, measure, 'empty_as_zero'),
            'zero_result_rate_a': out['zero_result_rate_a']}


def ratio_with_interval(num, den, qrels, n=10_000):
    a = evaluate_run(run(num), qrels)['nDCG@10']
    b = evaluate_run(run(den), qrels)['nDCG@10']
    qs = sorted(a)
    rng = random.Random(SEED)
    ratios = []
    for _ in range(n):
        s = [rng.choice(qs) for _ in qs]
        ratios.append(sum(a[x] for x in s) / sum(b[x] for x in s))
    ratios.sort()
    return {'ratio': sum(a.values()) / sum(b.values()), 'ci_low': ratios[int(0.025 * n) - 1],
            'ci_high': ratios[int(0.975 * n) - 1], 'share_below_threshold': sum(r < H8_THRESHOLD for r in ratios) / n}


def main():
    t = {}   # confirmatory tests decided by a p-value
    t['H1 precision, constraint, E1 - B1'] = group_test('targeted', 'E1_targeted', 'B1_targeted', 'constraint',
                                                        'P@min(|S|,100)', sizes('E1_targeted'))
    t['H2 recall@100, paraphrase, E1 - B1'] = group_test('targeted', 'E1_targeted', 'B1_targeted', 'paraphrase',
                                                         'R@100', sizes('E1_targeted'))
    for g in ('exclusion', 'identifier'):
        t[f'H3 nDCG@10, {g}, E2 - B1 (reranked)'] = group_test('targeted', 'E2-rr100-medcpt_targeted',
                                                              'B1-rr100-medcpt_targeted', g, 'nDCG@10')
    for split in ('test', 'targeted'):
        t[f'H4 recall@100, {split}, E2 - E1'] = whole_test(split, f'E2_{split}', f'E1_{split}', 'R@100',
                                                           sizes(f'E2_{split}'))
    t['H9 nDCG@10, test, E3 - E1'] = whole_test('test', 'E3_test', 'E1_test', 'nDCG@10', sizes('E3_test'))
    adjusted = holm({name: v['primary']['p_value'] for name, v in t.items()})
    for name in t:
        t[name]['holm'] = adjusted[name]

    def sig(name):
        return t[name]['holm']['reject']

    def d(name):
        return t[name]['primary']['mean_diff']

    verdicts = {
        'H1': 'holds' if sig('H1 precision, constraint, E1 - B1') and d('H1 precision, constraint, E1 - B1') > 0
        else 'falsified',
        'H2': 'holds' if sig('H2 recall@100, paraphrase, E1 - B1') and d('H2 recall@100, paraphrase, E1 - B1') < 0
        else 'falsified',
        'H3': 'holds' if any(sig(n) and d(n) > 0 for n in t if n.startswith('H3')) else 'falsified',
        'H4': 'holds' if all(sig(n) and d(n) > 0 for n in t if n.startswith('H4')) else 'falsified',
        'H9': 'falsified' if sig('H9 nDCG@10, test, E3 - E1') else 'holds',
    }

    # H7: equivalence after reranking, test, N = 100.
    h7 = {}
    for b in ('B1', 'B3'):
        c = whole_test('test', 'E2-rr100-medcpt_test', f'{b}-rr100-medcpt_test', 'nDCG@10')
        h7[f'E2 - {b} (reranked)'] = {**c, 'equivalent': equivalent(c['primary'], MARGIN),
                                      'above_margin': c['primary']['ci_low'] > MARGIN}
    verdicts['H7'] = ('falsified' if h7['E2 - B1 (reranked)']['above_margin'] else 'not falsified') + \
                     ('; equivalence shown' if all(v['equivalent'] for v in h7.values()) else
                      '; equivalence not shown')

    # H8: E2 / E4 nDCG@10 on test, not reranked; primary budget 10.
    qt = load_qrels('test')
    h8 = {b: ratio_with_interval('E2_test', f'E4-b{b}-{MODEL}_test', qt) for b in (3, 5, 10)}
    verdicts['H8'] = 'holds' if h8[10]['ratio'] >= H8_THRESHOLD else 'falsified'

    # Exploratory (never in Holm).
    expl = {}
    for b in ('B1', 'B3', 'E2'):
        expl[f'E4 (10 calls) - {b}, test, not reranked'] = whole_test('test', f'E4-b10-{MODEL}_test', f'{b}_test',
                                                                      'nDCG@10')
        expl[f'E4 (10 calls) - {b}, test, reranked'] = whole_test('test', f'E4-b10-{MODEL}-rr100-medcpt_test',
                                                                  f'{b}-rr100-medcpt_test', 'nDCG@10')
    qg, groups = load_qrels('targeted'), load_groups('targeted')
    for tag, path in (('not reranked', f'E4-not-b10-{MODEL}_targeted'),
                      ('reranked', f'E4-not-b10-{MODEL}-rr100-medcpt_targeted')):
        a = read_run(os.path.join(EXPL, path + '.tsv'))
        for b, bname in (('B1', 'B1-rr100-medcpt_targeted' if tag == 'reranked' else 'B1_targeted'),
                         ('B3', 'B3-rr100-medcpt_targeted' if tag == 'reranked' else 'B3_targeted')):
            out = compare_by_group(a, run(bname), qg, {'exclusion': groups['exclusion']})['exclusion']
            expl[f'E4+NOT - {b}, exclusion, {tag}'] = {'primary': pick(out, 'nDCG@10', 'empty_as_zero')}

    # Means for the summary tables.
    tables = {}
    for split, systems in (('test', ['B1', 'B2', 'B3', 'E0', 'E1', 'E2', 'E3', f'E4-b10-{MODEL}']),):
        q = load_qrels(split)
        tables[f'{split}, not reranked'] = {s: means(evaluate_run(run(f'{s}_{split}'), q)) for s in systems}
        tables[f'{split}, reranked (N = 100)'] = {s: means(evaluate_run(run(f'{s}-rr100-medcpt_{split}'), q))
                                                   for s in ['B1', 'B3', 'E2', f'E4-b10-{MODEL}']}
    for tag, fmt in (('not reranked', '{}_targeted'), ('reranked (N = 100)', '{}-rr100-medcpt_targeted')):
        per = {s: evaluate_run(run(fmt.format(s)), qg) for s in ['B1', 'B2', 'B3', 'E2', f'E4-b10-{MODEL}']}
        tables[f'targeted by group, {tag}'] = {
            g: {s: means(restrict(per[s], groups[g]))['nDCG@10'] for s in per}
            for g in ('constraint', 'exclusion', 'phrase', 'identifier', 'paraphrase')}

    result = {'confirmatory': t, 'H7': h7, 'H8': h8, 'verdicts': verdicts, 'exploratory': expl, 'tables': tables,
              'holm_family': list(t), 'margin_H7': MARGIN, 'threshold_H8': H8_THRESHOLD, 'seed': SEED}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'confirmatory.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1)
    write_markdown(result)
    for h, v in verdicts.items():
        print(f'{h}: {v}')
    print(f'-> {os.path.relpath(OUT, ROOT)}/confirmatory.md and .json')


def fmt_ci(x):
    return f'{x["mean_diff"]:+.3f} [{x["ci_low"]:+.3f}, {x["ci_high"]:+.3f}]'


def write_markdown(r):
    lines = ['# Final analysis', '', 'Generated by `python -m harness.final_analysis` from the committed run files. '
             'Primary analysis = requests where system A returned something; N = 100; bootstrap 10,000 resamples, '
             f'seed {r["seed"]}.', '', '## Confirmatory tests (Holm over the p-value tests)', '',
             '| test | n | A | B | A - B [95% CI] | p | Holm p | significant |', '|---|---|---|---|---|---|---|---|']
    for name, v in r['confirmatory'].items():
        x = v['primary']
        lines.append(f'| {name} | {x["n"]} | {x["mean_a"]:.3f} | {x["mean_b"]:.3f} | {fmt_ci(x)} | '
                     f'{x["p_value"]:.4f} | {v["holm"]["p_adjusted"]:.4f} | {"yes" if v["holm"]["reject"] else "no"} |')
    lines += ['', '## H7 (equivalence, margin +-0.03, test, reranked)', '',
              '| comparison | n | A - B [95% CI] | inside +-0.03 | above +0.03 |', '|---|---|---|---|---|']
    for name, v in r['H7'].items():
        lines.append(f'| {name} | {v["primary"]["n"]} | {fmt_ci(v["primary"])} | {v["equivalent"]} | '
                     f'{v["above_margin"]} |')
    lines += ['', '## H8 (E2 / E4 nDCG@10, test, not reranked; threshold 0.80, primary budget 10)', '',
              '| E4 budget | ratio | 95% CI | share of resamples below 0.80 |', '|---|---|---|---|']
    for b, v in r['H8'].items():
        lines.append(f'| {b} | {v["ratio"]:.3f} | [{v["ci_low"]:.3f}, {v["ci_high"]:.3f}] | '
                     f'{v["share_below_threshold"]:.2f} |')
    lines += ['', '## Verdicts', ''] + [f'- **{h}**: {v}' for h, v in r['verdicts'].items()]
    lines += ['', '## Exploratory (not in Holm, not pre-registered tests)', '', '| comparison | n | A - B [95% CI] | p |',
              '|---|---|---|---|']
    for name, v in r['exploratory'].items():
        x = v['primary']
        lines.append(f'| {name} | {x["n"]} | {fmt_ci(x)} | {x["p_value"]:.4f} |')
    lines += ['', '## Tables (nDCG@10, all requests, empty = 0)', '']
    for name, tab in r['tables'].items():
        if name.startswith('targeted'):
            systems = list(next(iter(tab.values())))
            lines += [f'### {name}', '', '| group | ' + ' | '.join(systems) + ' |', '|---' * (len(systems) + 1) + '|']
            lines += [f'| {g} | ' + ' | '.join(f'{row[s]:.3f}' for s in systems) + ' |' for g, row in tab.items()]
        else:
            lines += [f'### {name}', '', '| system | nDCG@10 | R@100 |', '|---|---|---|']
            lines += [f'| {s} | {m["nDCG@10"]:.3f} | {m["R@100"]:.3f} |' for s, m in tab.items()]
        lines.append('')
    with open(os.path.join(OUT, 'confirmatory.md'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
