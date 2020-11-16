import numpy as np
import scipy.stats as stats

#MDP class
class Player:

    def __init__(self, group=1, shared=False, article=0, clicked = 0): #, clickprob=0.5 , clickprobparams=[0.5, 0.1, 0, 1]):
        self.group = group
        self.shared = shared
        self.article = article
        self.clicked = clicked
        #self.clickprob = calcclickprob(*clickprobparams)

def calcclick(player, t, P, q, thetag=0.5, c = 1, v=1):
    g = player.group
    s = player.article
    if t > 1:
        return (q[g] * calcclick(player, t-1, P, q, thetag, c, v) * P[(g,s)]) + ((1 - q[ -1 * g]) * P[(g,s)] * calcclick(player, t-1, P, q, thetag, c, v))
    if t == 1:
        p =  P[(g,1)] * thetag + P[(g,-1)] * (1-thetag)
        #print(p)
        if p >= float(c/v):
            return 1
        else:
            return 0
'''
def calcclickprob(mu, sigma, upper, lower):
    return stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
'''

'''
def calcclick(pga, pgb, thetag, v=1.0, c=1.0):
	if pga * thetag + pgb * (1-thetag) > float(c/v):
		return 1
	else:
		return -1 
'''
    

    
def coin_toss(p):
    '''Mechanism to decide between A (1) or B (-1).'''
    return 2 * np.random.binomial(1, p) - 1

