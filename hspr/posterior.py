"""Hierarchical posterior with Beta-integrated group likelihoods."""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp, ndtri
from scipy.stats import qmc
from .beta import BetaGroupState
from .global_posterior import LatentPosterior
from .groups import GroupState


class ExchangeableLatentPosterior(LatentPosterior):
    def __init__(self, bank, config, group_precision, power=13):
        self.bank = bank
        self.group_precision = group_precision
        self.map = GroupState(bank, config, group_precision)
        self.sizes = bank['group_sizes']
        self.standard = ndtri(qmc.Sobol(len(self.map.z), scramble=True, seed=20261005).random_base2(power))
        self.refresh()

    def refresh(self):
        covariance = self.map.covariance()[0]
        self.nodes = self.map.z[None, :] + self.standard @ np.linalg.cholesky(covariance).T
        # Group MAP delta is not the conditional Beta mean.
        eta = self.map.offset[:, None] + self.map.loadings @ self.nodes.T
        probability = np.clip(expit(eta), 1e-12, 1-1e-12)
        form = 'none' if self.group_precision is None else 'logit_matched'
        if self.group_precision is None:
            likelihood = self.map.success @ eta - self.map.count @ np.logaddexp(0, eta)
        else:
            concentration = self.group_precision / (probability * (1-probability))
            likelihood = ordered_beta_loglik(probability, concentration, self.map.count, self.map.success)
        log_prior = -.5 * ((self.nodes**2) * self.map.precision).sum(1)
        log_proposal = -.5 * (self.standard**2).sum(1)
        log_weight = likelihood + log_prior - log_proposal
        weight = np.exp(log_weight - logsumexp(log_weight))
        state = BetaGroupState(probability, self.nodes[:, 0], weight, self.sizes, form, self.group_precision)
        state.count = self.map.count.astype(int).copy()
        state.success = self.map.success.astype(int).copy()
        state.unpaid = self.sizes - state.count
        if state.c is not None:
            state.m = (state.c * state.p + state.success[:, None]) / (state.c + state.count[:, None])
            state.square = state.m * state.m
            state.qvar = state.m * (1-state.m) / (state.c + state.count[:, None] + 1)
        state.score_by_node = state.unpaid @ state.m / state.N
        self.state = state


def ordered_beta_loglik(probability, concentration, count, success):
    """Stable Beta-integrated likelihood of observed ordered Bernoulli cells.

    Rising-factorial log1p form avoids subtracting huge beta/gamma functions
    when the logit-matched concentration is large near probability zero/one.
    """
    p = np.asarray(probability)
    count = np.asarray(count, dtype=int)
    success = np.asarray(success, dtype=int)
    failure = count - success
    if np.any(success < 0) or np.any(failure < 0):
        raise ValueError('invalid counts')
    value = success @ np.log(p) + failure @ np.log1p(-p)
    for counts, scale, sign in ((success, concentration * p, 1),
                                (failure, concentration * (1-p), 1),
                                (count, concentration, -1)):
        for j in range(1, int(counts.max(initial=0))):
            included = counts > j
            value += sign * np.log1p(j / scale[included]).sum(0)
    return value


def transfer_gain(state):
    """Binary variance reduction for other unpaid tasks, excluding candidate Y.

    Cov(S_full - Y/N, Y) = Cov(S_full,Y) - Var(Y)/N. This is a
    cross-task information objective, not an MAE/global policy optimum claim.
    """
    moments = state.moments()
    variance = moments['outcome_variance']
    cross = moments['score_outcome_covariance'] - variance/state.N
    gain = cross*cross/variance
    gain[state.unpaid == 0] = -np.inf
    return gain
