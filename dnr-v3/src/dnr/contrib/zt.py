#!/usr/bin/env python

## Zero-truncated Poisson and negative binomial distributions for scipy
##  R. Malouf, 4-May-2014
## https://github.com/rmalouf/learning/blob/master/zt.py

from numpy import floor
from scipy.special import expm1
from scipy.stats import nbinom, poisson, rv_discrete


class ztpoisson_gen(rv_discrete):
    def _rvs(self, mu, loc=0, size=1, random_state=None):
        return poisson.ppf(
            random_state.uniform(low=poisson.pmf(0, mu), high=1.0, size=size), mu
        )

    def _pmf(self, k, mu):
        return -poisson.pmf(k, mu) / expm1(-mu)

    def _cdf(self, x, mu):
        k = floor(x)
        if k == 0:
            return 0.0
        else:
            return (poisson.cdf(k, mu) - poisson.pmf(0, mu)) / poisson.sf(0, mu)

    def _ppf(self, q, mu):
        return poisson.ppf(poisson.sf(0, mu) * q + poisson.pmf(0, mu), mu)

    def _stats(self, mu):
        mean = mu * exp(mu) / expm1(mu)
        var = mean * (1.0 - mu / expm1(mu))
        g1 = None
        g2 = None
        return mean, var


ztpoisson = ztpoisson_gen(name="ztpoisson", longname="Zero-truncated Poisson")


class ztnbinom_gen(rv_discrete):
    def _rvs(self, n, p):
        return nbinom.ppf(uniform(low=nbinom.pmf(0, n, p)), n, p)

    def _argcheck(self, n, p):
        return (n >= 0) & (p >= 0) & (p <= 1)

    def _pmf(self, x, n, p):
        if x == 0:
            return 0.0
        else:
            return nbinom.pmf(x, n, p) / nbinom.sf(0, n, p)

    def _cdf(self, x, n, p):
        k = floor(x)
        if k == 0:
            return 0.0
        else:
            return (nbinom.cdf(x, n, p) - nbinom.pmf(0, n, p)) / nbinom.sf(0, n, p)

    def _ppf(self, q, n, p):
        return nbinom.ppf(nbinom.sf(0, n, p) * q + nbinom.pmf(0, n, p), n, p)


#     def _stats(self, n, p):
#         Q = 1.0 / p
#         P = Q - 1.0
#         mu = n*P
#         var = n*P*Q
#         g1 = (Q+P)/sqrt(n*P*Q)
#         g2 = (1.0 + 6*P*Q) / (n*P*Q)
#         return mu, var, g1, g2
ztnbinom = ztnbinom_gen(name="ztnbinom", longname="Zero-truncated negative binomial")
