"""Small checks of the exported inference code; no benchmark data required."""
import numpy as np
from scipy.stats import betabinom

from hspr.beta import BetaGroupState
from hspr.decisions import predictive_scores
from hspr.inference import replay


def main():
    state = BetaGroupState([[.1]], [0.], [1.], [8], 'fixed', 1.)
    state.update(0, 1)
    state.update(0, 0)
    draws = predictive_scores(state, 65536)
    counts = np.rint(draws * 8 - 1).astype(int)
    np.testing.assert_allclose(np.bincount(counts, minlength=7) / len(draws),
                               betabinom.pmf(np.arange(7), 6, 1.1, 1.9), atol=.007, rtol=0)
    bank = dict(group_ids=np.array([0, 0, 1, 2, 2, 2]), group_sizes=np.array([2, 1, 3]),
                group_intercept=np.array([-1.2, .2, 1.]), group_slope=np.array([.7, 1.1, .8]),
                group_residual_basis=np.array([[.2], [-.3], [.1]]))
    source = np.array([[0, 0, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]], dtype=float)
    target = np.array([0., 1., 1., 0., 1., 1.])
    config = dict(rank=1, precision=4., acquisition='score')
    for policy in ('posterior_score', 'posterior_latent'):
        first = list(replay(bank, source, target, [3], config, 1., policy))[0]
        changed = target.copy()
        unpaid = np.ones(len(target), dtype=bool)
        unpaid[first[2]] = False
        changed[unpaid] = 1 - changed[unpaid]
        again = list(replay(bank, source, changed, [3], config, 1., policy))[0]
        assert first == again, 'Unqueried outcomes changed inference'
        full = list(replay(bank, source, target, [3, 6], config, 1., policy))
        assert full[0][2] == full[1][2][:3]
        assert len(set(full[1][2])) == 6
        for scores in full[1][1]['scores'].values():
            for score in scores.values():
                assert score == target.mean()
    print('Passed: predictive distribution, sealed observations, nested paths and full budget.')


if __name__ == '__main__':
    main()
