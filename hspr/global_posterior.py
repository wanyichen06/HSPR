"""Latent-state quadrature, posterior updates and calibration."""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp, ndtri
from scipy.stats import qmc
from .beta import BetaGroupState
from .groups import GroupState


class LatentPosterior:
    def __init__(self, bank, config, power=11):
        self.bank = bank
        self.map = GroupState(bank, config, None)
        self.sizes = bank['group_sizes']
        dimension = len(self.map.z)
        self.standard = ndtri(qmc.Sobol(dimension, scramble=True, seed=20261005).random_base2(power))
        self.refresh()

    def refresh(self):
        covariance = self.map.covariance()[0]
        chol = np.linalg.cholesky(covariance)
        self.nodes = self.map.z[None, :] + self.standard @ chol.T
        eta = self.map.offset[:, None] + self.map.loadings @ self.nodes.T
        log_likelihood = self.map.success @ eta - self.map.count @ np.logaddexp(0, eta)
        log_prior = -.5 * ((self.nodes**2) * self.map.precision).sum(1)
        # Proposal normal constants are common across nodes and cancel.
        log_proposal = -.5 * (self.standard**2).sum(1)
        log_weight = log_likelihood + log_prior - log_proposal
        weight = np.exp(log_weight - logsumexp(log_weight))
        state = BetaGroupState(expit(eta), self.nodes[:, 0], weight, self.sizes, 'none', None)
        state.count = self.map.count.copy()
        state.success = self.map.success.copy()
        state.unpaid = self.sizes - state.count
        state.score_by_node = state.unpaid @ state.m / state.N
        self.state = state

    def update(self, group, outcome):
        self.map.update(group, outcome)
        self.state.update(group, outcome)
        if self.map.count.sum() % 8 == 0:
            self.refresh()

    def latent_gain(self):
        mean = self.state.m @ self.state.weight
        cross = self.state.m @ (self.state.weight[:, None] * self.nodes)
        cross -= mean[:, None] * (self.state.weight @ self.nodes)[None, :]
        gain = (cross**2).sum(1) / (mean * (1 - mean))
        gain[self.state.unpaid == 0] = -np.inf
        return gain

    def estimate(self):
        moment = self.state.moments()
        residual = (self.map.success.sum() - self.map.count @ moment['mean']) / self.map.count.sum()
        fraction = self.map.count.sum() / self.state.N
        return {c: float(np.clip(moment['score'] + c * (1 - fraction) * residual, 0, 1)) for c in (0., .5)}
