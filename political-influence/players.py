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

def calcclick(player, t, P, q, theta, c = 1, v=1):
    g = player.group
    s = player.article
    #print(theta)
    thetag = theta[g]
    if t > 1:
        return (q[g] * calcclick(player, t-1, P, q, theta, c, v) * P[(s,g)]) + ((1 - q[ -1 * g]) * P[(s,g)] * calcclick(player, t-1, P, q, theta, c, v))
    if t == 1:
        p =  P[(1,g)] * thetag + P[(-1,g)] * (1-thetag)
        #print('p: ' + str(p) + ' c/v: ' + str(float(c/v)))
        if p >= float(c/v):
            return 1
        else:
            return 0
        
def calcclickdict(player, t, P, q, theta, c, v):
    g = player.group
    s = player.article
    #print(theta)
    thetag = theta[g]
    if t > 1:
        return (q[g] * calcclick(player, t-1, P, q, theta, c, v) * P[(s,g)]) + ((1 - q[ -1 * g]) * P[(s,g)] * calcclick(player, t-1, P, q, theta, c, v))
    if t == 1:
        p =  P[(1,g)] * thetag * (v[(1,g)] - c[(1,g)]) + P[(-1,g)] * (1-thetag) * (v[(-1,g)] - c[(1,g)])
        #print('p: ' + str(p) + ' c/v: ' + str(float(c/v)))
        if p >= 0.:
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

