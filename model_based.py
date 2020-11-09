import numpy as np
import mdptoolbox.example
import cvxpy as cp
import mdp

#Algorithm 1 from Wen et al. 2019 "Fairness with Dynamics"
def modelbased(M):
    pi = np.zeros((M.A, M.S))

    lam = cp.Variable((M.A,M.S)) #variables for the dual program
    c = cp.Variable(1) #variables for the dual program

    objective = cp.Maximize((1. /(1-M.gamma)) * cp.sum(cp.multiply(M.R, lam)) )
    constraintsindiv = [cp.sum(lam[sp,])== (1 - M.gamma) * M.D[sp] + M.gamma * cp.sum(cp.multiply(lam, M.P[:,:,sp])) for sp in range(M.S)]
    constraintsgroup = [M.calcgroupprob(g) * cp.sum(cp.multiply(cp.multiply(M.grabgroupind(g), lam), cp.multiply(M.grabgroupind(g), M.R))) == c for g in range(M.num_groups)] #TODO: finish adding group constraints -- what is rho?  should R be something else?
    constraints = constraintsindiv + constraintsgroup
    prob = cp.Problem(objective,constraints)
    results = prob.solve(verbose=True)
    print prob.status #says if problem is infeasible or unbounded

    if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
        print("Optimal value: %s" % prob.value)
    for variable in prob.variables():
        print("Variable %s: value %s" % (variable.name(), variable.value))

    #calculate pi from lam
    for s in range(M.S):
        for a in range(M.A):
            pi[a,s] = float(lam[a,s]) / np.sum(lam, axis=0)

    return pi


if __name__ == '__main__':
  M = mdp.mdp()
  modelbased(M)
