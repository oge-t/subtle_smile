import scipy.optimize as opt
import numpy as np
from scipy.stats import norm


def Call(S, K, T, r, q, sig):
    d1 = (np.log(S/K) + (r-q+sig**2/2)*T)/(sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def Put(S, K, T, r, q, sig):
    d1 = (np.log(S/K) + (r-q+sig**2/2)*T)/(sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def CallDelta(S, K, T, r, q, sig):
    d1 = (np.log(S/K) + (r-q+sig**2/2)*T)/(sig*np.sqrt(T))
    return np.exp(-q*T)*norm.cdf(d1)

def Gamma(S, K, T, r, q, sig):
    d1 = (np.log(S/K) + (r-q+sig**2/2)*T)/(sig*np.sqrt(T))
    return np.exp(-q*T)*np.exp(-0.5*d1*d1)/(S*sig*np.sqrt(T)*np.sqrt(2.0*math.pi))

def impliedvol_call(S,K,T,r,q,price,xtol=1e-10, guess=0.2):
    callfun=lambda iv:Call(S,K,T,r,q,iv)-price
    return float(opt.fsolve(callfun,guess))

def impliedvol_put(S,K,T,r,q,price,guess=0.2):
    putfun=lambda iv:Put(S,K,T,r,q,iv)-price
    return float(opt.fsolve(putfun,guess))
