import numpy as np
import scipy.stats as stats

#player class
class Player:

    def __init__(self, group=1, shared=False, article=0, clicked = 0): 
        self.group = group
        self.shared = shared
        self.article = article
        self.clicked = clicked
        #self.clickprob = calcclickprob(*clickprobparams)

def calcclick(player, t, P, q, theta, c = 1, v=1):
    '''
    params:
    player        (python object): has group membership (A= 1, B = -1), the article they are shown (a = 1, b =-1), and booleans for whether or not they clicked and shared the article 
    t             (int): current timestep.  we have to make sure we actually got to this person
    P             (dict): dictionary indexed by (user group, article shown) of the probability distribution of utility.
    q             (dict): indexed by user group: homophily parameter
    theta         (dict): indexed by group.  Probability of shown a user in group g article A
    c             (int): cost for clicking
    v             (int): value for liking
    '''

    g = player.group
    s = player.article
    #print(theta)
    thetag = theta[g]
    if t > 1:
        return (q[g] * calcclick(player, t-1, P, q, theta, c, v) * P[(g,s)]) + ((1 - q[ -1 * g]) * P[(g,s)] * calcclick(player, t-1, P, q, theta, c, v))
    if t == 1:
        p =  P[(g,1)] * thetag + P[(g,-1)] * (1-thetag)
        if p >= float(c/v):
            return 1
        else:
            return 0
        
def calcclickdict(player, t, P, q, theta, c, v):
    
    '''
    params:
    player        (python object): has group membership (A= 1, B = -1), the article they are shown (a = 1, b =-1), and booleans for whether or not they clicked and shared the article 
    t             (int): current timestep.  we have to make sure we actually got to this person
    P             (dict): dictionary indexed by (user group, article shown) of the probability distribution of utility.
    q             (dict): indexed by user group: homophily parameter
    theta         (dict): indexed by group.  Probability of shown a user in group g article A
    c             (dict): cost for clicking indexed by (user group, article shown)
    v             (dict): value for liking indexed by (user group, article shown)
    '''

    g = player.group
    s = player.article
    thetag = theta[g]
    if t > 1:
        return (q[g] * calcclick(player, t-1, P, q, theta, c, v) * P[(g,s)]) + ((1 - q[ -1 * g]) * P[(g,s)] * calcclick(player, t-1, P, q, theta, c, v))
    if t == 1:
        p =  P[(g,1)] * thetag * (v[(g,1)] - c[(g,1)]) + P[(g,-1)] * (1-thetag) * (v[(g,-1)] - c[(g,1)])
        if p >= 0.:
            return 1
        else:
            return 0
   

    
def coin_toss(p):
    '''Mechanism to decide between A (1) or B (-1).'''
    return (2 * np.random.binomial(1, p) - 1)


