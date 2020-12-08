import numpy as np
import cvxpy as cp
from scipy.special import betainc





def exposure_constraint(g, s, T, pi_g, theta_g, l_gs, l_g_s, q_g):
	"""
	g = group {A, B}
	s = article source {a, b}
	T = time horizon
	pi_g = fraction of users from group g
	theta_g = fraction of users shown article s
	l_gs =  number of users from g that click and like or share s 
	l_g_s = number of users from g' that click and like or share s 
	q_g = probability user in g is replaced by another user from same group

	"""
	l = 0
	for t in T:
		l += (l_gs[t]*q_g) + (l_g_s[t]*(1-q_g))
	
	return ((theta_g*pi_g) + l)


def platform_optimize(articles, T, time_A, time_B, theta_a_t, theta_b_t):

	"""
	Inputs
	groups = {A, B}
	articles = {a, b}
	At time t, {A_t, B_t}
	Decide fraction of users in A_t and B_t to show the article {a, b}
	Proportion of users in A_t shown a = theta_a
	Proportion of users in A_t shown b = 1 - theta_a
	Proportion of users in B_t shown a = theta_b
	Proportion of users in B_t shown b = 1 - theta_b

	Problem
	Maximize l_gs under exposure constraint
		g = {A, B}, s = {a,b} l_gs = number of users that click, like/share the article
		exposure constraint = e_gs


	"""

	for t in T:
		A_t = time_A[t]
		B_t = time_B[t]
		theta_a = theta_a_t[t]
		theta_b = theta_b_t[t]

		# we can complete this after the discussion today


def calcthreshold(P, c, v):
    
    ret_dict = {}
    for g in [-1,1]:
            ret_dict[g] = min(1, max(0, ((c[(1,g)] / v[(1,g)]) - P[(-1, g)]) / (P[(1,g)] - P[(-1,g)])))
    
    return ret_dict

def psi(c, v, F):
    ps = {}
    for g in [-1,1]:
        for s in [-1,1]:
            alpha, beta = F[(g,s)]
            c_ind = c[(g,s)]
            v_ind = v[(g,s)]
            #int from c_gs/v_gs to 1 of p dF_g,s(p), i.e. Probability of click
            ps[(g,s)] =  1 - betainc(alpha, beta, float(c_ind) / v_ind)
    
    return ps

def zeta(g,s,q, c,v,F):
    ps = psi(c,v,F)
    return ps[(g,s)] * q[g] + ps[(-g, s)] * q[-g]

def phi(g,s,q, c,v,F):
    ps = psi(c,v,F)
    return ps[(g,s)] * q[g] + ps[(g,s)] / q[g] - ps[(-g,s)] - (ps[(-g,s)] * q[-g]) / q[g] + ps[(-g,s)] * q[-g]


def l(g,s,t, pi, theta, q, c,v,F):
    ps = psi(c,v,F)
    ti = (g+1) / 2
    if t >= 2:
        return ((q[g] * l(g,s,t-1, pi, theta, q,c,v,F)) + (1 - q[-g]) * l(-g,s,t-1, pi, theta, q,c,v,F)) * ps[(g,s)]
        #return ps[(g,s)] * (q[g] * (phi(g,s,q,c,v,F) ** (t-2) * l(g,s,1, pi, theta,q,c,v,F) + (1 - q[-g]) * zeta(g,s,q,c,v,F) ** (t-2)* l(-g, s, 1, pi, theta, q,c,v,F) )) 
    else:
        if s == 1:
            return pi[g] * theta[ti] * ps[(g,1)]
        if s == -1:
            return pi[g] * (1 - theta[ti]) * ps[(g,-1)]

# utility the platform has for user g clicking on article s        
# indexed by (g,s)        
unit_util = {(1,1)   : 1.,
             (1,-1)  : 1.,
             (-1,1)  : 1.,
             (-1,-1) : 1.}
#throwing this one in there 25 Nov... let's go through and delete the other ones that aren't being used.
def opt(pi, q, T, epsilon,c,v,F, u=unit_util):
    
    '''
    params:
    pi      : dictionary: proportion of users in each group
    q       : dictionary: homophily variable
    T       : int: number of total time steps (max)
    epsilon : double: fairness violation allowed
    c       : dictionary indexed (g,s): cost for clicking by group and article
    v       : dictionary indexed (g,s): value for sharing by group and article
    F       : dictionary indexed (g,s): alpha and beta parameters for beta distribution
    '''

    
    #varaible theta_A, theta_B
    theta = cp.Variable(2)
    objective = cp.Maximize(cp.sum([u[(1,1)] * l(1,1,t, pi, theta, q, c,v,F) + u[(-1,1)] * l(-1,1,t,pi, theta, q, c,v,F) + u[(1,-1)] * l(1,-1,t, pi, theta, q, c,v,F) + u[(-1,-1)] *  l(-1,-1,t, pi, theta, q, c,v,F) for t in range(T)]))
    constraints_theta = [0 <= theta[0], theta[0] <= 1, 0 <= theta[1], theta[1] <= 1]
    
    #generate eta, used as constraints.
    eta = {}
    for s in [-1,1]:
        for g in [-1,1]:
            ti = (g+1) / 2 #theta index.... moving from -1,1 to 0,1
            if s == 1:
                eta[(s,g)] = pi[g] * theta[ti] + sum([ l(g,s,t, pi, theta,q,c,v,F) * q[g] + l(-g,s,t, pi, theta,q,c,v,F) * (1 - q[-g]) for t in range(1,T-1)])
            else:
                eta[(s,g)] = pi[g] * (1-theta[ti]) + sum([ l(g,s,t, pi, theta,q,c,v,F) * q[g] + l(-g,s,t, pi, theta,q,c,v,F) * (1 - q[-g]) for t in range(1,T-1)])
                
    
    
    constraints_eta = [eta[(1,1)] - eta[(-1,-1)] <= epsilon, eta[(-1,-1)] - eta[(1,1)] <= epsilon]
    
    
    prob = cp.Problem(objective, constraints_theta+constraints_eta)
    prob.solve()
    th = {}
    
    #TODO: double check this is correct indexing
    th[1] = max(min(theta.value[1], 1.), 0.)
    th[-1] = max(min(theta.value[0], 1.), 0.)
    return th


#throwing this one in there 25 Nov... let's go through and delete the other ones that aren't being used.
def opt_unconstrained(pi, q, T, epsilon,c,v,F, u=unit_util):
    
    '''
    params:
    pi      : dictionary: proportion of users in each group
    q       : dictionary: homophily variable
    T       : int: number of total time steps (max)
    epsilon : double: fairness violation allowed
    c       : dictionary indexed (g,s): cost for clicking by group and article
    v       : dictionary indexed (g,s): value for sharing by group and article
    F       : dictionary indexed (g,s): alpha and beta parameters for beta distribution
    '''

    
    #varaible theta_A, theta_B
    theta = cp.Variable(2)
    objective = cp.Maximize(cp.sum([u[(1,1)] * l(1,1,t, pi, theta, q, c,v,F) + u[(-1,1)] * l(-1,1,t,pi, theta, q, c,v,F) + u[(1,-1)] * l(1,-1,t, pi, theta, q, c,v,F) + u[(-1,-1)] *  l(-1,-1,t, pi, theta, q, c,v,F) for t in range(T)]))
    constraints_theta = [0 <= theta[0], theta[0] <= 1, 0 <= theta[1], theta[1] <= 1]
    
    prob = cp.Problem(objective, constraints_theta)
    prob.solve()
    th = {}
    th[1] = max(min(theta.value[1], 1.), 0.)
    th[-1] = max(min(theta.value[0], 1.), 0.)
    return th
