import numpy as np
import cvxpy as cp

def calcPCt(PC, timestep, PLC1, PLC0):
	if timestep == 1:
		return PC
	else: 
		p = calcPCt(PC, timestep-1, PLC1, PLC0)
		return PC * ( (PLC1 * p) + (PLC0 * (1-p)) )

def calcPC(group, timestep, PAa, PBa, PLA, PLB, muA, muB):
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


def optimize(epsilon, mA, M, T, PAa= 0.8, PBa=0.2, PLA=0.1, PLB=0.05, muA=0.3, muB=0.3):
	pi = float(mA) / M
	theta = cp.Variable(1)

	obj_vec = [theta * pi * calcPC(1, t, PAa, PBa, PLA, PLB, muA, muB) + (1 - theta) * (1 - pi) * calcPC(-1,t, PAa, PBa, PLA, PLB, muA, muB) for t in range(1, T+1)]
	


	objective = cp.Maximize(cp.sum(obj_vec))
	constraints = [0.5 - epsilon <= theta, theta <= 0.5 - epsilon]
	prob = cp.Problem(objective, constraints)

	# The optimal objective value is returned by `prob.solve()`.
	result = prob.solve()
	# The optimal value for x is stored in `x.value`.
	# print(theta.value)
	return theta.value[0]

def main():
	optimize(0.2, 15, 30, 10)

if __name__ == "__main__":
	main()