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
