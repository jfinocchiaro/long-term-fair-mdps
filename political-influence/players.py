import numpy as np


#MDP class
class players:

    def __init__(self, group=1, shared=False, articleshown='n', clickprob=0.5, clickprobparams=[0.5, 0.1, 0, 1]):
        self.group = group
        self.shared = shared
        self.articleshown = articleshown
        self.clickprob = calcclickprob(*clickprobparams)

def calcclickprob(mu, sigma, upper, lower)
    return stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)