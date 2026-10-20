"""Source-only response factors and item initialization."""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit


def fit_bank(source, item_prior='jeffreys'):
    source = np.asarray(source, dtype=float)
    if not np.isfinite(source).all() or np.any((source < 0) | (source > 1)):
        raise ValueError('invalid source responses')
    n, tasks = source.shape
    mean = source.mean(0)
    _, singular, vt = np.linalg.svd(source - mean, full_matrices=False)
    rank = min(8, n - 1, tasks)
    basis = vt[:rank].T
    covariance_loadings = basis * singular[:rank] / np.sqrt(max(1, n - 1))
    if item_prior == 'jeffreys':
        prior_alpha, prior_beta = .5, .5
        smooth = np.clip((source.sum(0) + .5) / (n + 1), .01, .99)
    elif item_prior in ('empirical', 'zero_inflated'):
        if not np.isin(source, [0, 1]).all():
            raise ValueError('beta-binomial item priors require binary source responses')
        from scipy.optimize import minimize
        from scipy.special import betaln, digamma
        counts = source.sum(0)

        def marginal(parameters):
            alpha, beta = np.exp(parameters[:2])
            component = betaln(counts + alpha, n - counts + beta) - betaln(alpha, beta)
            common = digamma(n + alpha + beta) - digamma(alpha + beta)
            ga = digamma(counts + alpha) - digamma(alpha) - common
            gb = digamma(n - counts + beta) - digamma(beta) - common
            if item_prior == 'empirical':
                return -float(component.sum()), -np.array([alpha * ga.sum(), beta * gb.sum()])
            zero_mass = expit(parameters[2])
            log_mixture = np.log1p(-zero_mass) + component
            zeros = counts == 0
            log_mixture[zeros] = np.logaddexp(np.log(zero_mass), log_mixture[zeros])
            responsibility = np.zeros(tasks)
            responsibility[zeros] = np.exp(np.log(zero_mass) - log_mixture[zeros])
            continuous = 1 - responsibility
            return -float(log_mixture.sum()), -np.array([
                alpha * np.sum(continuous * ga), beta * np.sum(continuous * gb),
                responsibility.sum() - tasks * zero_mass])

        center = np.clip(mean.mean(), .001, .999)
        initial = np.log([center, 1 - center]).tolist()
        bounds = [(-6, 6), (-6, 6)]
        if item_prior == 'zero_inflated':
            initial.append(logit(.1))
            bounds.append((-8, 8))
        fitted = minimize(marginal, initial, jac=True, bounds=bounds,
                          method='L-BFGS-B', options={'maxiter': 200})
        if not fitted.success or not np.isfinite(fitted.fun):
            raise ValueError('source empirical item prior did not converge')
        prior_alpha, prior_beta = np.exp(fitted.x[:2])
        smooth = (counts + prior_alpha) / (n + prior_alpha + prior_beta)
        if item_prior == 'zero_inflated':
            zero_mass = expit(fitted.x[2])
            log_continuous_zero = (np.log1p(-zero_mass)
                                   + betaln(prior_alpha, n + prior_beta)
                                   - betaln(prior_alpha, prior_beta))
            continuous_zero = np.exp(log_continuous_zero
                                     - np.logaddexp(np.log(zero_mass), log_continuous_zero))
            smooth[counts == 0] *= continuous_zero
        smooth = np.clip(smooth, .0001, .9999)
    else:
        raise ValueError('unknown item prior')
    score_logits = logit(np.clip((source.sum(1) + .5) / (tasks + 1), .001, .999))
    location = float(score_logits.mean())
    scale = max(float(score_logits.std()), .25)
    ability = (score_logits - location) / scale
    intercept = logit(smooth)
    intercept_prior = intercept.copy()
    slope = np.ones(tasks)
    # Regularized supervised 2PL item calibration on source agents only.
    for _ in range(16):
        prob = expit(intercept + ability[:, None] * slope)
        residual = prob - source
        weight = prob * (1 - prob)
        gb = residual.sum(0) + (intercept - intercept_prior)
        ga = (residual * ability[:, None]).sum(0) + 2 * (slope - 1)
        hbb = weight.sum(0) + 1
        haa = (weight * ability[:, None] ** 2).sum(0) + 2
        hba = (weight * ability[:, None]).sum(0)
        det = hbb * haa - hba ** 2
        intercept -= np.clip((haa * gb - hba * ga) / det, -.5, .5)
        slope -= np.clip((hbb * ga - hba * gb) / det, -.5, .5)
        intercept = np.clip(intercept, -10, 10)
        slope = np.clip(slope, .05, 5)
    prob = expit(intercept + ability[:, None] * slope)
    residual = source - prob
    residual -= residual.mean(0)
    _, rs, rvt = np.linalg.svd(residual, full_matrices=False)
    derivative = np.maximum((prob * (1 - prob)).mean(0), .08)
    residual_basis = (rvt[:rank].T * rs[:rank] / np.sqrt(max(1, n - 1))) / derivative[:, None]
    logistic_basis = covariance_loadings / np.maximum(smooth * (1 - smooth), .08)[:, None]
    return {'source': source, 'mean': mean, 'basis': basis,
            'loadings': covariance_loadings, 'smooth': smooth,
            'intercept': intercept, 'slope': slope,
            'residual_basis': residual_basis, 'logistic_basis': logistic_basis,
            'item_prior': {'name': item_prior, 'alpha': float(prior_alpha),
                           'beta': float(prior_beta),
                           'zero_mass': float(zero_mass) if item_prior == 'zero_inflated' else 0.}}


def logistic_coordinates(offset, design, outcomes, precision, initial=None):
    z = np.zeros(design.shape[1]) if initial is None else np.asarray(initial).copy()
    precision = np.broadcast_to(precision, z.shape)

    def loss(v):
        logits = offset + design @ v
        return float(np.logaddexp(0, logits).sum() - outcomes @ logits
                     + .5 * np.sum(precision * v ** 2))

    current = loss(z)
    for _ in range(25):
        probability = expit(offset + design @ z)
        gradient = design.T @ (probability - outcomes) + precision * z
        hessian = design.T @ ((probability * (1 - probability))[:, None] * design)
        hessian += np.diag(precision)
        step = np.linalg.solve(hessian, gradient)
        if np.max(abs(step)) < 1e-7:
            break
        fraction = 1.
        for _ in range(12):
            candidate = z - fraction * step
            candidate_loss = loss(candidate)
            if candidate_loss <= current:
                z = candidate
                current = candidate_loss
                break
            fraction *= .5
        else:
            break
    return z
