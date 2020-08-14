#-*-coding:utf-8-*-

import numpy as np
import pprint

class Env:
    def __init__(self, state_dim, action_dim, h):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.h = h

    def set_init_state(self):
        """
        Return an initial state
        :return: An initial state of dimension (1,self.state_dim)
        """
        return np.zeros((1, self.state_dim))
    
    def step(self, s, a):
        """
        Draw a random sample with the transition distribution by taking a at state s
        :param s: The current state
        :param a: The current action
        :return: A random draw according to T(s' | s, a)
        """
        pass
        return s
    
    def objective_G(self, trajs):
        """
        Given a finite-horizon sample trajectory, return its objective value
        :param trajs: a set of h-step simulation trajectories
        :return: G(traj) value
        """
        return 0
    
    def objective_H(self, trajs):
        """
        Given a finite-horizon sample trajectory, return its constraint value
        :param trajs: a set of h-step simulation trajectories
        :return: H(trajs) value
        """
        return 0
    
    
""" Environment design
        S^{tilde}: R^2 x {0,1},         state space
        A = {0,1},                      action space
        T: S^{tilde} x A -> S^{tilde},  transition function
            T([alpha+1, beta, z] | [alpha, beta, z], 1) = p_{alpha, beta} = alpha/(alpha+beta)
            T([alpha, beta+1, z] | [alpha, beta, z], 1) = 1 - p_{alpha, beta} = beta/(alpha+beta)
            T([alpha, beta + epsilon, z] | [alpha, beta, z], 0) = 1
        R: S^{tilde} x A -> R,          reward function for the bank
            R([alpha, beta, z], 0) = 0
            R([alpha, beta, z], 1) = (I+P)*alpha/(alpha+beta) - P + lamda*(I+P)^2*alpha*beta/(alpha+beta)/(alpha+beta+1)
        d:                              initial distribution
            d([alpha1, beta1, 1]) = pZ
            d([alpha0+alpha1, beta0+beta1, 0]) = 1-pZ
            
    Fairness constraint
    R_{applicant}(s,a) = a
    E[R_{applicant} | z = 1] = E[R_{applicant} | z = 0]
"""


class EnvLoan(Env):
    def __init__(self, state_dim, action_dim, h, epsilon, alpha0, alpha1, beta0, beta1, pZ, I, P, lamda, gamm, h1, h0):
        self.epsilon = epsilon
        self.pZ = pZ
        self.alpha0, self.alpha1, self.beta0, self.beta1 = alpha0, alpha1, beta0, beta1
        self.I, self.P, self.lamda = I, P, lamda
        
        self.h0, self.h1 = h0, h1
        
        super().__init__(state_dim, action_dim, h)
        "discounting factor"
        self.gamm = gamm
        # "Error bound"
        # self.err = 0.001
        # "Required length of the simulated trajectories"
        # self.h = np.log(self.err * (1 - self.gamm) /
        #                 np.amax([self.I + self.P, self.lamda * (self.I + self.P) ** 2])) / \
        #          np.log(self.gamm)
        # self.h = np.ceil(self.h)
        
    def set_init_state(self, z=None):
        """
        Ramdomly select an initial state and return it
        :param z: binary variable. If none, randomly select one.
        :return: state
        """
        if z is None:
            p = np.random.rand(1)
            if p <= self.pZ:
                state = np.array([[self.alpha1, self.beta1, 1]])
            else:
                state = np.array([[self.alpha0, self.beta0, 0]])
        elif z == 1:
            state = np.array([[self.alpha1, self.beta1, 1]])
        else:
            state = np.array([[self.alpha0, self.beta0, 0]])
        # state = state.astype(np.float32)
        return state
    
    def set_init_state_abz(self, alpha, beta, z):
        """
        Specify the initial state using the alpha, beta and z values
        :param alpha: alpha > 0
        :param beta: beta > 0
        :param z: z = 0, 1
        :return: the state with alpha, beta, z
        """
        # return np.array([[alpha, beta, z]], dtype=np.float32)
        return np.array([[alpha, beta, z]])
    
    def set_init_distribution(self, z):
        """
        Run the transition system for hz steps to generate the initial distribution for group z.
        There will be (hz + 1) possible initial values. The ith value is encoded by (alpha_z + hz-i, beta_z + i)
        :param z: z = 0 or 1.
        :return: an numpy array of (hz+1,), corresponding to the probability to take
                 [(a+hz,b), (a+hz-1,b+1), ..., (a+1,b+hz-1), (a,b+hz)]
        """
        if z == 0:
            a, b = self.alpha0, self.beta0
            hz = self.h0
        elif z == 1:
            a, b = self.alpha1, self.beta1
            hz = self.h1
        else:
            raise ValueError("invalid parameter z. z should be 0 or 1")

        dist = np.zeros(hz + 1)
        dist[0] = 1
            
        for ii in range(hz):
            dist_copy = np.zeros(hz+1)
            for jj in range(ii+1):
                dist_copy[jj] += dist[jj] * (a+ii-jj) / (a+b+ii)
                dist_copy[jj+1] += dist[jj] * (b+jj) / (a+b+ii)
            dist = dist_copy
        init_dist = np.zeros((3, hz+1))
        init_dist[0,:] = np.linspace(hz, 0, hz+1) + a
        init_dist[1,:] = np.linspace(0, hz, hz+1) + b
        init_dist[2,:] = dist
        return init_dist
        
    def step(self, s, a):
        """
        s = [[alpha, beta, z]], a \in {0, 1}
        :param s: (a batch of) states
        :param a: (a batch of) action
        :return: the next state (randomly drawn wrt the given transition)
        """
        if isinstance(s, list):
            s = np.array(s)
        if len(s.shape) == 1:
            s = np.array([s])
        # s = s.astype(np.float32)
        
        s_prime = s.copy()
        p = np.random.rand(s.shape[0])
        
        if not isinstance(a, list) and not isinstance(a, np.ndarray):
            a = [a]
        
        for ii in range(s.shape[0]):
            if a[ii] == 0:
                "a = 0"
                # print("case0")
                s_prime[ii][1] = s_prime[ii][1] + self.epsilon

            elif p[ii] <= s[ii][0] / (s[ii][0] + s[ii][1]):
                "a = 1 and repay the loan"
                # print("case1")
                s_prime[ii][0] += 1
                
            else:
                "a = 1 and decline the loan"
                # print("case2")
                s_prime[ii][1] += 1
        
        # print("s id = "+str(id(s))+", s_prime id = "+str(id(s_prime)))
        # print("s = "+str(s)+", a = "+str(a)+", s_prime = "+str(s_prime))
        return s_prime
    
    def reward(self, s, a):
        """
        State-based reward function for G
        :param s: (a batch of) states [[alpha, beta, z]]
        :param a: (a batch of) actions
        :return: the corresponding reward values
        """
        if len(s.shape) ==1:
            s = np.array([s])
            
        # if not isinstance(a, list):
        #     a = [a]
          
        # print("[reward] s = "+str(s))
        r = np.ones(s.shape[0]) * 2
        
        for ii in range(s.shape[0]):
            if a[ii] == 1:
                tmp = s[ii][0]+s[ii][1]
                r[ii] += (self.I+self.P)*s[ii][0]/tmp - self.P - \
                    self.lamda*(self.I+self.P)*np.sqrt(s[ii][0]*s[ii][1]/tmp/(tmp+1))
                # print("[reward] r[ii] = "+str(r[ii])+", s[ii] = "+str(s[ii]))
        if s.shape[0] == 1:
            r = r[0]
            
        if any(r < 0):
            print(r)
            
        return r
        
    def objective_G(self, trajs):
        """
        Given a list of trajectories, return a list of G values corresponding to each trajectory.
        The G value of each h-step trajectory is the discounted value
        :param trajs: a list of trajectories, each as a list of [[s_0, a_0], [s_1, a_1], ..., [s_{h-1}, a_{h-1}]]
        :return: a list of the length len(trajs)
        """
        # print("len(trajs) = "+str(len(trajs)))
        
        g = [0 for _ in range(len(trajs))]
        for ii in range(len(trajs)):
            
            # print("Length of the " + str(ii) + " trajectory: " + str(len(trajs[ii])))
            # pprint.pprint(trajs[ii])
            
            for jj in range(len(trajs[ii])):
                g[ii] += self.gamm**jj * self.reward(trajs[ii][jj][0], trajs[ii][jj][1])
            # print("g[ii] = "+str(g[ii]))
            # print("-- traj --")
            
        # print("g = "+str(g))
        # print("---- objective_G ----")
        
        return g

    def objective_G_tricky(self, trajs, z):
        """
        Tricky objective G, just for debugging.
        """
        # print("len(trajs) = "+str(len(trajs)))
    
        g = [0 for _ in range(len(trajs))]
        for ii in range(len(trajs)):
            if all([s[1]==1 for s in trajs[ii]]):
                if z == 1:
                    g[ii] = 1.8886
                else:
                    g[ii] = -0.4098
        return g
        
    def objective_H(self, trajs):
        """
        Given a list of trajectories, return a list of H values corresponding to each trajectory
        The H value of each trajectory is the average value of the fairness reward
        :param trajs: a list of trajectories, each as a list [s_0, a_0, s_1, a_1, ..., s_{h-1}, a_{h-1}]
        :return: a list of the length len(trajs)
        """
        h = [0 for _ in range(len(trajs))]
        for ii in range(len(trajs)):
            for jj in range(len(trajs[ii])):
                # print(trajs[ii][jj])
                h[ii] += trajs[ii][jj][1]
            h[ii] = h[ii] / len(trajs[ii])
        return h