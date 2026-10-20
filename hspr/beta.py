"""Conditional group-rate posterior and finite-score moments."""
from __future__ import annotations

import numpy as np


class BetaGroupState:
    def __init__(self,probability,ability,prior_weight,sizes,form,concentration):
        self.p=np.clip(np.asarray(probability,dtype=float),1e-12,1-1e-12)
        self.ability=np.asarray(ability,dtype=float)
        self.weight=np.asarray(prior_weight,dtype=float).copy();self.weight/=self.weight.sum()
        self.sizes=np.asarray(sizes,dtype=int);self.unpaid=self.sizes.copy();self.N=self.sizes.sum()
        self.count=np.zeros(len(sizes),dtype=int);self.success=np.zeros(len(sizes),dtype=int)
        if self.p.shape!=(len(sizes),len(self.weight)) or len(self.ability)!=len(self.weight):
            raise ValueError('posterior shapes differ')
        if form=='none':self.c=None
        elif form=='fixed':self.c=np.full_like(self.p,float(concentration))
        elif form=='logit_matched':self.c=float(concentration)/(self.p*(1-self.p))
        else:raise ValueError('invalid concentration form')
        self.m=self.p.copy();self.square=self.m*self.m
        self.qvar=np.zeros_like(self.p) if self.c is None else self.m*(1-self.m)/(self.c+1)
        self.score_by_node=self.unpaid@self.m/self.N

    def moments(self):
        mean=self.m@self.weight;square=self.square@self.weight;rate_var=self.qvar@self.weight
        bernoulli=mean-square
        mean_score=float(self.weight@self.score_by_node)
        conditional_cross=(bernoulli+(self.unpaid-1)*rate_var)/self.N
        between=self.m@(self.weight*self.score_by_node)-mean*mean_score
        cross=conditional_cross+between
        outcome_var=mean*(1-mean)
        gain=cross*cross/outcome_var
        gain[self.unpaid==0]=-np.inf
        variance=float((self.unpaid*bernoulli+self.unpaid*(self.unpaid-1)*rate_var).sum()/self.N**2+
                       self.weight@((self.score_by_node-mean_score)**2))
        score=self.success.sum()/self.N+mean_score
        return {'mean':mean,'score':float(score),'score_variance':variance,
                'score_outcome_covariance':cross,'outcome_variance':outcome_var,'score_gain':gain}

    def ability_gain(self):
        mean=self.m@self.weight;theta_mean=self.weight@self.ability
        cross=self.m@(self.weight*self.ability)-mean*theta_mean
        gain=cross*cross/(mean*(1-mean));gain[self.unpaid==0]=-np.inf
        return gain

    def update(self,group,outcome):
        if outcome not in (0,1) or self.unpaid[group]<=0:raise ValueError('invalid paid observation')
        before=self.unpaid[group]*self.m[group]
        likelihood=self.m[group] if outcome==1 else 1-self.m[group]
        self.weight*=likelihood
        normalizer=self.weight.sum()
        if normalizer<=0 or not np.isfinite(normalizer):raise ValueError('posterior support vanished')
        self.weight/=normalizer
        self.count[group]+=1;self.success[group]+=outcome;self.unpaid[group]-=1
        if self.c is not None:
            self.m[group]=(self.c[group]*self.p[group]+self.success[group])/(self.c[group]+self.count[group])
            self.square[group]=self.m[group]*self.m[group]
            self.qvar[group]=self.m[group]*(1-self.m[group])/(self.c[group]+self.count[group]+1)
        self.score_by_node+=(self.unpaid[group]*self.m[group]-before)/self.N
