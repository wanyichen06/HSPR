"""Pooled cohort metrics and paired statistical comparisons."""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np

def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | bool]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    tau, rho = rank_statistics(truth, prediction)
    return {
        "score_mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "kendall_tau_b": float(tau) if np.isfinite(tau) else float("nan"),
        "spearman_rho": float(rho) if np.isfinite(rho) else float("nan"),
        "max_absolute_error": float(np.max(np.abs(error))),
        "finite_rank": bool(np.isfinite(tau)),
    }

def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks

def rank_statistics(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    """Return Kendall tau-b and Spearman rho without requiring SciPy.

    The remote execution environment uses SciPy for parity with the earlier
    analyses.  The pure-NumPy implementation keeps local schema/unit tests
    usable when the workstation has an incompatible SciPy/NumPy pair.
    """
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    try:
        from scipy.stats import kendalltau, spearmanr

        tau = float(kendalltau(truth, prediction).statistic)
        rho = float(spearmanr(truth, prediction).statistic)
        return tau, rho
    except (ImportError, ModuleNotFoundError, ValueError):
        n = len(truth)
        concordant = discordant = ties_truth = ties_prediction = 0
        for left in range(n):
            for right in range(left + 1, n):
                dx = truth[left] - truth[right]
                dy = prediction[left] - prediction[right]
                if dx == 0.0 and dy == 0.0:
                    continue
                if dx == 0.0:
                    ties_truth += 1
                elif dy == 0.0:
                    ties_prediction += 1
                elif dx * dy > 0.0:
                    concordant += 1
                else:
                    discordant += 1
        denominator = math.sqrt(
            (concordant + discordant + ties_truth)
            * (concordant + discordant + ties_prediction)
        )
        tau = (concordant - discordant) / denominator if denominator else float("nan")
        truth_rank = _average_ranks(truth)
        prediction_rank = _average_ranks(prediction)
        rho = float(np.corrcoef(truth_rank, prediction_rank)[0, 1])
        return tau, rho

def pooled_summary(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in records:
        groups[(row['method'], int(row['budget']), int(row.get('seed', 0)))].append(row)
    per_seed = defaultdict(list)
    cohorts = {}
    for (method, budget, seed), rows in sorted(groups.items()):
        keys = [(int(r['fold']), r['agent']) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f'duplicate target in {method}/{budget}/{seed}')
        cohort = set(keys)
        if budget in cohorts and cohorts[budget] != cohort:
            raise ValueError(f'unmatched target cohort for {method}/{budget}')
        cohorts[budget] = cohort
        result = metrics(np.array([r['truth'] for r in rows]),
                         np.array([r['prediction'] for r in rows]))
        per_seed[(method, budget)].append({'seed': seed, **result})
    output = {}
    for (method, budget), rows in sorted(per_seed.items()):
        output.setdefault(method, {})[str(budget)] = {
            'aggregation': 'pool_targets_then_average_query_seeds',
            'n_targets': len(cohorts[budget]), 'n_seeds': len(rows),
            'metrics': {
                key: {'mean': float(np.mean([r[key] for r in rows])),
                      'std': float(np.std([r[key] for r in rows], ddof=1)) if len(rows)>1 else 0.0}
                for key in ('score_mae','rmse','bias','kendall_tau_b','spearman_rho','max_absolute_error')
            }, 'per_seed': rows,
        }
    return output

def paired_mae(records: list[dict], ours: str, comparator: str, budget: int,
               seed: int = 20261005, replicates: int = 10000) -> dict:
    from scipy.stats import wilcoxon
    errors = {}
    for method in (ours, comparator):
        by_target = defaultdict(list)
        for r in records:
            if r['method'] == method and r['budget'] == budget:
                by_target[(int(r['fold']),r['agent'])].append(abs(r['prediction']-r['truth']))
        errors[method] = {k: float(np.mean(v)) for k,v in by_target.items()}
    if not errors[ours] or set(errors[ours]) != set(errors[comparator]):
        raise ValueError('paired cohorts differ')
    keys = sorted(errors[ours])
    delta = np.array([errors[ours][k]-errors[comparator][k] for k in keys])
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0,len(delta),size=(replicates,len(delta)))].mean(axis=1)
    p = 1.0 if np.all(delta == 0) else float(wilcoxon(delta, alternative='two-sided').pvalue)
    return {'ours':ours,'comparator':comparator,'budget':budget,'n_targets':len(keys),
            'mae_delta':float(delta.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),
            'wilcoxon_p':p,'target_deltas':[{'fold':k[0],'agent':k[1],'delta':float(v)} for k,v in zip(keys,delta)],
            'bootstrap_replicates':replicates,'bootstrap_seed':seed,
            'scope':'paired configuration bootstrap conditional on archived task bank; query-seed errors averaged before resampling'}

def holm_adjust(comparisons: list[dict]) -> None:
    running = 0.0
    order = sorted(range(len(comparisons)), key=lambda i:comparisons[i]['wilcoxon_p'])
    for rank,i in enumerate(order):
        running = max(running, min(1.0,(len(order)-rank)*comparisons[i]['wilcoxon_p']))
        comparisons[i]['holm_p'] = running
