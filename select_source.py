"""Three-stage leave-one-source-agent-out configuration selection."""
import argparse
import json
from pathlib import Path

import numpy as np

from hspr.bank import fit_bank
from hspr.decisions import decisions
from hspr.groups import signature_bank
from hspr.observations import SealedResponses, candidates, predict_sealed
from hspr.posterior import ExchangeableLatentPosterior
from run_hspr import DATA_NAMES, load_data


def walk_source(bank, row, budget, config, kappa, policy, seed):
    state = ExchangeableLatentPosterior(bank, config, kappa, 13)
    query = SealedResponses(row)
    remaining = [np.flatnonzero(bank['group_ids'] == g).tolist()
                 for g in range(len(bank['group_sizes']))]
    rng = np.random.default_rng(seed)
    for _ in range(budget):
        gain = state.state.moments()['score_gain'] if policy == 'posterior_score' else state.latent_gain()
        group = int(np.argmax(gain))
        item = remaining[group].pop(int(rng.integers(len(remaining[group]))))
        state.update(group, query.reveal(item))
    return state


def select(source, budget):
    truth = source.mean(1)
    grid = candidates()
    errors = np.empty((len(source), len(grid)))
    for held in range(len(source)):
        bank = fit_bank(np.delete(source, held, axis=0), 'zero_inflated')
        for j, config in enumerate(grid):
            prediction, _ = predict_sealed(bank, SealedResponses(source[held]), budget, config)
            errors[held, j] = abs(prediction - truth[held])
    winner = min(range(len(grid)), key=lambda j: (errors[:, j].mean(), errors[:, j].max(), j))
    config = {k: grid[winner][k] for k in ('rank', 'precision', 'acquisition')}
    print('Global rank and precision selected', flush=True)
    choices = [(kappa, policy, calibration) for kappa in (.25, 1., 4., 16.)
               for policy in ('posterior_score', 'posterior_latent') for calibration in (0., .5)]
    errors = np.empty((3, len(source), len(choices)))
    banks = [signature_bank(np.delete(source, held, axis=0)) for held in range(len(source))]
    for held, bank in enumerate(banks):
        for seed in range(3):
            for j in range(0, len(choices), 2):
                kappa, policy, _ = choices[j]
                state = walk_source(bank, source[held], budget, config, kappa, policy, seed)
                scores = state.estimate()
                for offset, calibration in enumerate((0., .5)):
                    errors[seed, held, j + offset] = abs(scores[calibration] - truth[held])
        print('Hierarchy source row', held + 1, '/', len(source), flush=True)
    winner = min(range(len(choices)), key=lambda j: (errors[:, :, j].mean(),
                                                    errors[:, :, j].mean(0).max(), j))
    kappa, policy, _ = choices[winner]
    errors = np.empty((3, len(source), 2, 2))
    for held, bank in enumerate(banks):
        for seed in range(3):
            state = walk_source(bank, source[held], budget, config, kappa, policy, seed)
            value = decisions(state)
            lower, upper = value['median_mc_ci95']
            if (abs(value['sample_mean'] - value['raw_mean']) > 6 * value['mean_mc_se'] + 1e-12
                    or upper - lower > 1 / source.shape[1] + 1e-12):
                raise ValueError('Source predictive numerical check failed')
            for estimator, name in enumerate(('mean', 'median')):
                for j, calibration in enumerate((0., .5)):
                    errors[seed, held, estimator, j] = abs(value['scores'][name][calibration] - truth[held])
    calibrations = {}
    for estimator, name in enumerate(('mean', 'median')):
        winner = min(range(2), key=lambda j: (errors[:, :, estimator, j].mean(),
                                             errors[:, :, estimator, j].mean(0).max(), j))
        calibrations[name + '_calibration'] = (0., .5)[winner]
    return dict(global_config=config, group_precision=kappa, policy=policy, **calibrations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--benchmark', choices=DATA_NAMES, required=True)
    parser.add_argument('--fold', type=int, choices=range(4), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(Path(__file__).with_name('paper_configs.json').read_text())
    row = next(c for c in reference['folds'] if c['benchmark'] == args.benchmark and c['fold'] == args.fold)
    matrix, index = load_data(args.data_root, args.benchmark)
    source = matrix[[index[a] for a in row['source_agents']]]
    settings = select(source, row['budget'])
    chosen = {**row, **settings}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'folds': [chosen]}, indent=2) + '\n')
    print('Saved source-only configuration:', args.output)


if __name__ == '__main__':
    main()
