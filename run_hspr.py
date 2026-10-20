"""Replay HSPR with frozen paper configurations and externally supplied data."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from hspr.groups import signature_bank
from hspr.inference import replay
from hspr.metrics import pooled_summary

DATA_NAMES = {'lite': 'swe_lite_summary', 'verified': 'swe_verified_summary',
              'full': 'swe_test_summary', 'pro': 'swe_pro_summary'}
VARIANTS = ('hspr', 'mean', 'no_hierarchy', 'score', 'latent',
            'fixed_score_no_hierarchy', 'uniform', 'midrange')


def load_data(root, benchmark):
    directory = Path(root) / DATA_NAMES[benchmark]
    matrix = np.loadtxt(directory / 'matrix.csv', delimiter=',', ndmin=2)
    metadata = json.loads((directory / 'metadata.json').read_text())
    agents = [a['folder'] for a in metadata['agents']]
    if len(agents) != len(matrix) or len(set(agents)) != len(agents):
        raise ValueError('Response rows and unique agent identifiers differ')
    if not np.isfinite(matrix).all() or not np.isin(matrix, [0, 1]).all():
        raise ValueError('Expected a complete binary response matrix')
    return matrix, {agent: i for i, agent in enumerate(agents)}


def evaluate_fold(job):
    root, choice, fractions, variant, seeds, limit = job
    matrix, index = load_data(root, choice['benchmark'])
    source_ids, target_ids = choice['source_agents'], choice['target_agents']
    if len(set(source_ids)) != len(source_ids) or len(set(target_ids)) != len(target_ids):
        raise ValueError('Duplicate cohort identifiers')
    if set(source_ids) & set(target_ids):
        raise ValueError('Source and target cohorts overlap')
    source = matrix[[index[a] for a in source_ids]]
    if round(.1 * matrix.shape[1]) != choice['budget']:
        raise ValueError('Task bank differs from the paper configuration')
    bank = signature_bank(source)
    budgets = [round(matrix.shape[1] * f) for f in fractions]
    targets = target_ids[:limit] if limit is not None else target_ids
    records = []
    for agent in targets:
        outcomes = matrix[index[agent]]
        for seed in seeds:
            for budget, value, selected in replay(
                    bank, source, outcomes, budgets, choice['global_config'],
                    choice['group_precision'], choice['policy'], seed,
                    variant=variant, fold=choice['fold']):
                estimator = 'mean' if variant == 'mean' else 'median'
                calibration = choice[estimator + '_calibration']
                record = dict(fold=choice['fold'], agent=agent, seed=seed,
                              method=variant, budget=budget,
                              prediction=value['scores'][estimator][calibration],
                              selected_items=selected, query_count=len(selected),
                              median_mc_ci95=value['median_mc_ci95'])
                # Full outcomes supply the reference score after the estimate.
                record['truth'] = float(outcomes.mean())
                records.append(record)
    print(choice['benchmark'], 'fold', choice['fold'], 'completed', flush=True)
    return choice['benchmark'], records


def json_finite(value):
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_finite(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('paper_configs.json'))
    parser.add_argument('--benchmark', choices=(*DATA_NAMES, 'all'), default='lite')
    parser.add_argument('--fractions', type=float, nargs='+', choices=(.05, .1, .15, .2), default=[.1])
    parser.add_argument('--variant', choices=VARIANTS, default='hspr')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--folds', type=int, nargs='+', choices=range(4), default=list(range(4)))
    parser.add_argument('--limit-targets', type=int, help='Per-fold limit for checking execution only')
    parser.add_argument('--output', type=Path, default=Path('outputs/hspr.json'))
    args = parser.parse_args()
    if args.workers < 1 or (args.limit_targets is not None and args.limit_targets < 1):
        parser.error('Workers and target limits must be positive')
    seeds = args.seeds
    if seeds is None:
        seeds = [11, 23, 37, 53, 71] if args.variant == 'uniform' else [0] if args.variant == 'midrange' else list(range(5))
    choices = json.loads(args.config.read_text())['folds']
    jobs = [(str(args.data_root), c, args.fractions, args.variant, seeds, args.limit_targets)
            for c in choices if c['fold'] in args.folds
            and (args.benchmark == 'all' or c['benchmark'] == args.benchmark)]
    if not jobs:
        parser.error('No matching configurations')
    groups = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for benchmark, records in pool.map(evaluate_fold, jobs):
            groups.setdefault(benchmark, []).extend(records)
    result = {'partial_cohort': args.limit_targets is not None or len(args.folds) != 4,
              'variant': args.variant, 'benchmarks': {b: {'records': r, 'summary': pooled_summary(r)}
                                                   for b, r in groups.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_finite(result), indent=2, allow_nan=False) + '\n')
    print('Saved', args.output)


if __name__ == '__main__':
    main()
