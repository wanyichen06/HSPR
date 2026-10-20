"""Adaptive and fixed-query replay using the deployed HSPR posterior."""
import numpy as np

from .decisions import decisions
from .observations import SealedResponses
from .posterior import ExchangeableLatentPosterior


def qualified_decisions(posterior):
    samples = 131072
    while samples <= 2097152:
        value = decisions(posterior, samples=samples)
        mean_ok = abs(value['sample_mean'] - value['raw_mean']) <= 6 * value['mean_mc_se'] + 1e-12
        lower, upper = value['median_mc_ci95']
        if mean_ok and upper - lower <= 1 / posterior.state.N + 1e-12:
            return value
        samples *= 2
    raise ValueError('Predictive precision limit reached')


def replay(bank, source, target, budgets, config, group_precision, policy,
           seed=0, variant='hspr', fold=0, power=13):
    budgets = sorted(set(budgets))
    if not budgets or budgets[0] < 1 or budgets[-1] > len(target):
        raise ValueError('Invalid distinct-task budget')
    if policy not in ('posterior_score', 'posterior_latent'):
        raise ValueError('Unknown acquisition objective')
    if variant in ('no_hierarchy', 'fixed_score_no_hierarchy'):
        group_precision = None
    if variant in ('score', 'fixed_score_no_hierarchy'):
        policy = 'posterior_score'
    elif variant == 'latent':
        policy = 'posterior_latent'
    posterior = ExchangeableLatentPosterior(bank, config, group_precision, power)
    query = SealedResponses(target)
    remaining = [np.flatnonzero(bank['group_ids'] == g).tolist()
                 for g in range(len(bank['group_sizes']))]
    rng = np.random.default_rng(seed)
    fixed = None
    if variant == 'uniform':
        stream = np.random.SeedSequence([20260930, fold, seed]).spawn(2)[0]
        fixed = np.random.default_rng(stream).permutation(len(target)).tolist()
    elif variant == 'midrange':
        fixed = np.argsort(abs(source.mean(0) - .5), kind='stable').tolist()
    for step in range(1, budgets[-1] + 1):
        if fixed is not None:
            item = fixed[step - 1]
            group = int(bank['group_ids'][item])
        else:
            gain = (posterior.state.moments()['score_gain'] if policy == 'posterior_score'
                    else posterior.latent_gain())
            group = int(np.argmax(gain))
            if not np.isfinite(gain[group]) or not remaining[group]:
                raise ValueError('Acquisition selected an exhausted group')
            item = remaining[group].pop(int(rng.integers(len(remaining[group]))))
        posterior.update(group, query.reveal(item))
        if step in budgets:
            value = qualified_decisions(posterior)
            yield step, value, query.selected.copy()
