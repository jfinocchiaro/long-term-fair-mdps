import numpy as np
import mdptoolbox.example
import cvxpy as cp

#extra functions for the model free algorithm
#for now, take Gamma to be the identity
#TODO: figure this out; seems like we can't take Gamma to be the identity.

#MDP class
class mdp:

    def __init__(self, S=2,A=3,D=np.full(2, 1./2),gamma=0.5, num_groups=2):
        self.S = S
        self.A = A
        self.D =  D
        self.gamma = gamma
        self.P, R = mdptoolbox.example.rand(self.S,self.A) #might need some modifying later so that P and R can be initialized on their own
        self.R = R[:,:,0]
        self.num_groups = num_groups
        self.groups = np.random.randint(0,num_groups,size=S)

    def calcgroupprob(self, g):
        return [sum([self.D[s] for s in range(self.S) if self.groups[s] == g]) for g in range(self.num_groups)][g]

    def grabgroupind(self, g):
        ind = np.ones((self.A, self.S))
        for s in range(self.S):
            if self.groups[s] != g:
                for a in range(self.A):
                    ind[a,s] = 0
        return ind


    #Algorithm 1 from Wen et al. 2019 "Fairness with Dynamics"
    def modelbased(self):
        pi = np.zeros((self.A, self.S))

        lam = cp.Variable((self.A,self.S)) #variables for the dual program
        c = cp.Variable(1) #variables for the dual program

        objective = cp.Maximize((1. /(1-self.gamma)) * cp.sum(cp.multiply(self.R, lam)) )
        constraintsindiv = [cp.sum(lam[sp,])== (1 - self.gamma) * self.D[sp] + self.gamma * cp.sum(cp.multiply(lam, self.P[:,:,sp])) for sp in range(self.S)]
        constraintsgroup = [self.calcgroupprob(g) * cp.sum(cp.multiply(cp.multiply(self.grabgroupind(g), lam), cp.multiply(self.grabgroupind(g), self.R))) == c for g in range(self.num_groups)] #TODO: finish adding group constraints -- what is rho?  should R be something else?
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
        for s in range(self.S):
            for a in range(self.A):
                pi[a,s] = float(lam[a,s]) / np.sum(lam, axis=0)

        return pi

    def genTsteps(self,theta, T=5):
        #BUG: this currently generates the same action at every time step given the use of theta in determining pi_theta
        s = []
        s_curr = np.random.choice(np.arange(0,self.S), p=self.D)
        s.append(s_curr)
        a = []
        #pi^theta is a draw according to a uniform binning of action space.
        action_bins = [(i / float(self.A), (i+1) / float(self.A)) for i in range(self.A)]
        print action_bins
        for act, (min, max) in enumerate(action_bins):
            if theta >= min and theta <= max:
                a_curr = act

        print a_curr
        a.append(a_curr)
        s_past = s_curr

        for t in range(1,T):
            #take a step towards the current state
            s_curr = np.random.choice(np.arange(0,self.S), p=self.P[s_past,a_curr,:])
            s.append(s_curr)

            #determine which action to take by
            for act, (min, max) in enumerate(action_bins):
                if theta >= min and theta <= max:
                    a_curr = act

            print a_curr
            a.append(a_curr)
            s_past = s_curr

        return s,a

    #Algorithm 2 from Wen wet al. 2019 "Fairness with Dynamics"
    def modelfree(self, r=500, n=30, n_prime=10, m=15, alpha=0.5, sigma=0.5,T=5):
        #r = iterations
        #n = parameter samples
        #nprime = top n' samples considered, <= n
        #m = rollout samples
        #alpha = smoothing parameter, in (0,1)
        #sigma = tolerance

        eta_hat = 0. #line 2
        for k in range(r): #line 3
            samples_theta_unbounded = np.random.exponential((1./m) * eta_hat, size=n) #line 4: sample n samples from f_{m^-1}(eta_hat) #TODO: double check this is correct
            samples_theta = np.minimum(np.maximum(samples_theta_unbounded, 0), 1)
            print samples_theta

            for i in range(n): #line 5
                #s[t] and a[t] are determined by a sample of the MDP governed by policy pi_theta.  it's unclear what pi_theta is to me.
                s,a = self.genTsteps(samples_theta[0], T=T)
                R_hat_samples = [np.sum([self.gamma ** t * self.R[s[t]][a[t]] for t in range(T)]) for x in range(m)]
                R_hat_theta_i = np.average(R_hat_samples) #line 6: update R_hat

                rho_maj =1#TODO
                rho_min =1#TODO
                eps_hat_samples = np.abs(np.add(rho_maj, -1* rho_min) )
                eps_hat_theta_i = np.average(eps_hat_samples)#line 7: update eps_hat
            #line 9: sort theta_i in increasing eps_hat
            #line 10: i_prime is the largest i so that the tolerance is attained
            if n_prime <= i_prime: #line 11
                pass#line 12: sort to i_prime theta^i's in decreasing R_hat
            #line 14: update eta_hat
        return np.random.exponential(eta_hat) #line 16






M = mdp()
#print M.modelbased()
print M.modelfree()
