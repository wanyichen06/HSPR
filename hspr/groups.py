"""Exact response-signature groups and Gaussian importance proposal."""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from .bank import fit_bank


def signature_bank(source):
    bank=fit_bank(source,'zero_inflated')
    _,groups=np.unique(source.T,axis=0,return_inverse=True)
    sizes=np.bincount(groups)
    # Averaging within exact source patterns removes meaningless SVD row jitter.
    def grouped(array):
        a=np.asarray(array)
        if a.ndim==1:
            return np.bincount(groups,weights=a)/sizes
        return np.column_stack([np.bincount(groups,weights=a[:,j])/sizes for j in range(a.shape[1])])
    bank['group_ids']=groups
    bank['group_sizes']=sizes
    bank['group_intercept']=grouped(bank['intercept'])
    bank['group_slope']=grouped(bank['slope'])
    bank['group_residual_basis']=grouped(bank['residual_basis'])
    return bank


class GroupState:
    def __init__(self,bank,config,group_precision):
        self.offset=bank['group_intercept']
        self.loadings=bank['group_slope'][:,None]
        if config['rank']:
            self.loadings=np.column_stack([self.loadings,bank['group_residual_basis'][:,:config['rank']]])
        self.precision=np.full(self.loadings.shape[1],config['precision'])
        if config['rank']:
            self.precision[0]=.25
        self.group_precision=group_precision
        self.z=np.zeros(self.loadings.shape[1])
        self.delta=np.zeros(len(self.offset))
        self.count=np.zeros(len(self.offset))
        self.success=np.zeros(len(self.offset))

    def probability(self):
        return expit(self.offset+self.loadings@self.z+self.delta)

    def loss(self,z,delta):
        eta=self.offset+self.loadings@z+delta
        value=np.sum(self.count*np.logaddexp(0,eta)-self.success*eta)+.5*np.sum(self.precision*z*z)
        if self.group_precision is not None:
            value+=.5*self.group_precision*(delta@delta)
        return float(value)

    def covariance(self):
        p=self.probability()
        weight=self.count*p*(1-p)
        if self.group_precision is None:
            adjusted=self.loadings
            diagonal=np.zeros(len(p))
            information=weight
        else:
            denominator=self.group_precision+weight
            shrink=self.group_precision/denominator
            adjusted=self.loadings*shrink[:,None]
            diagonal=1/denominator
            information=weight*shrink
        schur=np.diag(self.precision)+self.loadings.T@(information[:,None]*self.loadings)
        return np.linalg.inv(schur),adjusted,diagonal

    def update(self,group,outcome):
        self.count[group]+=1
        self.success[group]+=outcome
        current=self.loss(self.z,self.delta)
        for _ in range(25):
            p=self.probability()
            residual=self.count*p-self.success
            weight=self.count*p*(1-p)
            gradient=self.loadings.T@residual+self.precision*self.z
            if self.group_precision is None:
                hessian=np.diag(self.precision)+self.loadings.T@(weight[:,None]*self.loadings)
                step_z=np.linalg.solve(hessian,gradient)
                step_delta=np.zeros(len(p))
            else:
                denominator=self.group_precision+weight
                gradient_delta=residual+self.group_precision*self.delta
                information=weight*self.group_precision/denominator
                schur=np.diag(self.precision)+self.loadings.T@(information[:,None]*self.loadings)
                rhs=gradient-self.loadings.T@(weight*gradient_delta/denominator)
                step_z=np.linalg.solve(schur,rhs)
                step_delta=(gradient_delta-weight*(self.loadings@step_z))/denominator
            if max(np.max(abs(step_z)),np.max(abs(step_delta)))<1e-7:
                break
            fraction=1.
            for _ in range(12):
                z=self.z-fraction*step_z
                delta=self.delta-fraction*step_delta
                value=self.loss(z,delta)
                if value<=current:
                    self.z,self.delta,current=z,delta,value
                    break
                fraction*=.5
            else:
                break
