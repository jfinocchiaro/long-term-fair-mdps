import numpy as np
#import cvxpy as cp

def calcPCt(PC, timestep, PLC1, PLC0):
	if timestep == 1:
		return PC
	else: 
		p = calcPCt(PC, timestep-1, PLC1, PLC0)
		return PC * ( (PLC1 * p) + (PLC0 * (1-p)) )

'''
def calcPC(group, timestep, PAa, PBa, PLA, PLB, muA=0.5, muB=0.5):
	if group == 1:
		g = 1
		PLC1 = PAa #probability of liking given click
		PLC0 = PLA #probability of liking given no click
		PC = muA #probability of click given group A

	else:
		g = -1
		PLC1 = PBa #probability of like given click
		PLC0 = PLB #probability of live given no click
		PC = muB #probability of click given group B

	PCt = calcPCt(PC, timestep, PLC1, PLC0)

	#return P(C | group) * np.prod([P(L | C=1, group) * P(C=1|group, t) + P(L | C=0, group) * P(C=0|group, t) for t in range timestep])
	pc = PC * np.prod([PLC1 * PCt + PLC0 * (1-PCt) for t in range (1,timestep+1)])
	return pc
'''

#def optimize(epsilon, mA, M, T, PAa= 0.8, PBa=0.2, PLA=0.1, PLB=0.05, muA=0.3, muB=0.3):
def optimize(epsilon, mA, M, T, PCA, PCB, PLC, PLnC, group=1):
	pi = float(mA) / M
	theta = cp.Variable(1)

	#obj_vec = [theta * pi * calcPCt(1, t, PAa, PBa, PLA, PLB, muA, muB) + (1 - theta) * (1 - pi) * calcPCt(-1,t, PAa, PBa, PLA, PLB, muA, muB) for t in range(1, T+1)]
	obj_vec = [theta * pi * calcPCt(PCA, t, PLC[(str(group), '1')], PLnC[(str(group), '1')]) + (1 - theta) * (1 - pi) * calcPCt(PCB, t, PLC[(str(-1 * group), '1')], PLnC[(str(-1 * group), '1')]) for t in range(1, T+1)]
	


	objective = cp.Maximize(cp.sum(obj_vec))
	constraints = [0.5 - epsilon <= theta, theta <= 0.5 - epsilon]
	prob = cp.Problem(objective, constraints)

	# The optimal objective value is returned by `prob.solve()`.
	result = prob.solve()
	# The optimal value for x is stored in `x.value`.
	# print(theta.value)
	return theta.value[0]



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
		g = {A, B}, s = {a,b} l_gs = number of users that click, like and share the article
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
















