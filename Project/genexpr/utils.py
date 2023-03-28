import numpy as np
from scipy.special import rel_entr, loggamma

def KL_div(P, Q):
    """
    Kullback-Leibler divergence D_KL(P || Q) between discrete distributions P and Q. The lengths of P and Q must match.

    Inputs:
        - P [ndarray, shape=(N,)]: array containing p.m.f of P
        - Q [ndarray, shape=(N,)]: array containing p.m.f of Q
            
    Outputs: 
        - D_KL [float]: Kullback-Leibler divergence D_KL(P || Q)
    """

    if len(P) != len(Q):
        return np.inf
    if not(isinstance(P, np.ndarray) and isinstance(Q, np.ndarray)):
        P, Q = np.asarray(P), np.asarray(Q)

    return np.sum(rel_entr(P,Q))



def NBinom(k, mu, eta):
    r = mu/(mu*eta**2 - 1)
    p = 1/(mu*eta**2)

    if p<0 or p>1 or r<0:
        return -1
    else:
        return np.exp(loggamma(k+r)-loggamma(k+1)-loggamma(r) + k*np.log(1-p) + r*np.log(p))