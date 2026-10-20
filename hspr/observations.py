"""Query interface and global source-validation evaluator."""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from .bank import logistic_coordinates


class SealedResponses:
    def __init__(self, responses):
        self.__responses = np.asarray(responses).copy()
        self.selected = []

    def reveal(self, item):
        if item in self.selected:
            raise ValueError('duplicate query')
        self.selected.append(int(item))
        return float(self.__responses[item])


def predict_sealed(bank, query, budget, config):
    loadings = bank['slope'][:, None]
    if config['rank']:
        loadings = np.column_stack([loadings, bank['residual_basis'][:, :config['rank']]])
    precision = np.full(loadings.shape[1], config['precision'])
    if config['rank']:
        precision[0] = .25
    z = np.zeros(loadings.shape[1])
    selected = []
    paid = []
    available = np.ones(len(loadings), dtype=bool)
    for _ in range(budget):
        probabilities = expit(bank['intercept'] + loadings @ z)
        variance = probabilities * (1 - probabilities)
        hessian = np.diag(precision)
        if selected:
            design = loadings[selected]
            hessian = hessian + design.T @ (variance[selected, None] * design)
        covariance = np.linalg.inv(hessian)
        projection = loadings @ covariance
        denominator = 1 + variance * (projection * loadings).sum(1)
        if config['acquisition'] == 'score':
            score_gradient = (variance[:, None] * loadings).mean(0)
            gain = variance * (projection @ score_gradient) ** 2 / denominator
        else:
            gain = variance * (projection * projection).sum(1) / denominator
        gain[~available] = -np.inf
        item = int(np.argmax(gain))
        paid.append(query.reveal(item))
        selected.append(item)
        available[item] = False
        z = logistic_coordinates(bank['intercept'][selected], loadings[selected],
                                 np.asarray(paid), precision, initial=z)
    probabilities = expit(bank['intercept'] + loadings @ z)
    residual = float((np.asarray(paid) - probabilities[selected]).mean())
    probabilities[selected] = paid
    score = float(np.clip(probabilities.mean() + config['calibration'] *
                          (1 - budget / len(loadings)) * residual, 0, 1))
    return score, selected


def candidates():
    models = [{'rank': 0, 'precision': p} for p in (.25, 1.)]
    models += [{'rank': r, 'precision': p} for r in (1, 2, 4) for p in (1., 4., 8.)]
    return [{**m, 'acquisition': acquisition, 'calibration': calibration}
            for m in models for acquisition in ('score', 'information')
            for calibration in (0., .5)]
