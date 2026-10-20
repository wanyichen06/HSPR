"""Finite-score predictive sampling and calibrated point decisions."""
from __future__ import annotations

import numpy as np
from scipy.stats import binom
from .posterior import ExchangeableLatentPosterior, transfer_gain


def predictive_scores(state,samples=131072,seed=20261005,chunk=2048):
    if samples<1 or chunk<1:
        raise ValueError('positive Monte Carlo sizes required')
    rng=np.random.default_rng(seed)
    active=np.flatnonzero(state.unpaid>0)
    scores=np.empty(samples)
    paid=state.success.sum()
    for start in range(0,samples,chunk):
        size=min(chunk,samples-start)
        node=rng.choice(len(state.weight),size=size,p=state.weight)
        probability=state.p[active[:,None],node[None,:]].copy()
        if hasattr(state,'component_weights'):
            weights=state.component_weights[:,active[:,None],node[None,:]]
            component=(rng.random((len(active),size))[None,:,:]>
                       np.cumsum(weights,axis=0)).sum(0)
            component=np.minimum(component,len(state.components)-1)
            for j,c in enumerate(state.concentrations):
                if c is None:continue
                g,k=np.nonzero(component==j)
                original=active[g]
                concentration=c[original,node[k]]
                probability[g,k]=rng.beta(concentration*probability[g,k]+state.success[original],
                    concentration*(1-probability[g,k])+state.count[original]-state.success[original])
        elif state.c is not None:
            concentration=state.c[active[:,None],node[None,:]]
            probability=rng.beta(concentration*probability+state.success[active,None],
                concentration*(1-probability)+(state.count-state.success)[active,None])
        count=rng.binomial(state.unpaid[active,None],probability).sum(0)
        scores[start:start+size]=(paid+count)/state.N
    return scores


def decisions(posterior,samples=131072,seed=20261005):
    scores=predictive_scores(posterior.state,samples,seed)
    mean=posterior.state.moments()['score']
    # Order-statistic interval measures Monte Carlo uncertainty, not dataset uncertainty.
    lower,upper=binom.ppf([.025,.975],samples,.5).astype(int)
    lower=max(0,lower-1)
    upper=min(samples-1,upper)
    sorted_scores=np.sort(scores)
    median=float(np.median(scores))
    residual=(posterior.map.success.sum()-posterior.map.count@posterior.state.moments()['mean'])/posterior.map.count.sum()
    correction=(1-posterior.map.count.sum()/posterior.state.N)*residual
    return {'scores':{'mean':posterior.estimate(),
        'median':{c:float(np.clip(median+c*correction,0,1)) for c in (0.,.5)}},
        'raw_median':median,'median_mc_ci95':[float(sorted_scores[lower]),float(sorted_scores[upper])],
        'raw_mean':mean,'sample_mean':float(scores.mean()),
        'mean_mc_se':float(scores.std(ddof=1)/np.sqrt(samples)),
        'samples':samples,'simulation_seed':seed}


def predict_decisions(bank,query,budget,config,precision,policy,seed):
    """Keep the registered exchangeable query path, change its final decision."""
    if policy not in ('posterior_score','posterior_latent','posterior_transfer'):
        raise ValueError('invalid decision experiment query policy')
    posterior=ExchangeableLatentPosterior(bank,config,precision,13)
    remaining=[np.flatnonzero(bank['group_ids']==g).tolist() for g in range(len(bank['group_sizes']))]
    rng=np.random.default_rng(seed)
    selected=[]
    for _ in range(budget):
        if policy=='posterior_score':gain=posterior.state.moments()['score_gain']
        elif policy=='posterior_transfer':gain=transfer_gain(posterior.state)
        else:gain=posterior.latent_gain()
        group=int(np.argmax(gain))
        item=remaining[group].pop(int(rng.integers(len(remaining[group]))))
        selected.append(item)
        posterior.update(group,query.reveal(item))
    return decisions(posterior)|{'selected_items':selected,
        'effective_nodes':float(1/(posterior.state.weight@posterior.state.weight))}
