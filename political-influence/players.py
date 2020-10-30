import numpy as np
import scipy.stats as stats

#MDP class
class players:

    def __init__(self, group=1, shared=False, article=0, clickprob=0.5, clickprobparams=[0.5, 0.1, 0, 1]):
        self.group = group
        self.shared = shared
        self.article = article
        self.clickprob = calcclickprob(*clickprobparams)

def calcclickprob(mu, sigma, upper, lower):
    return stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)