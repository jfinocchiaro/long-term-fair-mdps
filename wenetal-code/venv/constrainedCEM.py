#-*-coding:utf-8-*-

import logging
import numpy as np
from operator import itemgetter
import pickle
import csv
import sys
import cv2
import time
import itertools

logger = logging.getLogger("CEM_loan")
effective_digits = 8

pixel_lim = [440, 440]
x_lim = [0, 22]
y_lim = [0, 22]

def map_xy_to_pixel(x, y):
    return (int(round((x - x_lim[0]) / (x_lim[1] - x_lim[0]) * pixel_lim[0])),
            int(round((y_lim[1] - y) / (y_lim[1] - y_lim[0]) * pixel_lim[1])))

class ConstrainedCEM:
    def __init__(self, env, policy, n_samples, h, rho, nl, sess, equal_opp_thres, log_folder):
        """
        Initialize all parameters used by the constrained cross-entropy method
        :param env:         An environment object: given policy pi, return G(pi) and H(pi)
        :param policy:      A neural network S x A -> D(A)
        :param n_samples:   The number of sample polices used in each iteration
        :param h:           The length of each sample trajectory
        :param rho:         The proportion of sample policies as elite samples
        :param nl:          The number of sample trajectories for each sample policy
        :param sess:        The tensorflow session
        :param self.equal_opp_thres: The minimum value for qualified states
        :param log_folder:  The folder to save log files
        """
        self.env = env
        self.policy = policy
        self.n_samples = n_samples
        self.h = h
        self.rho = rho
        self.nl = nl
        self.sess = sess
        
        self.tolerance = 0.1
        self.equal_opp_thres = equal_opp_thres
        
        self.log_folder = log_folder
        self.csv_file_performance = open(self.log_folder + "/performance.csv", 'w', newline='\n')
        self.csv_writer_performance = csv.writer(self.csv_file_performance, dialect='excel')
        
        "Set the initial distribution"
        self.init0 = self.env.set_init_distribution(0)
        self.init1 = self.env.set_init_distribution(1)
        
        self.initialize_v()
        self._initialize_DP()

        self.cond_init0 = self.set_conditional_init(self.init0)
        self.cond_init1 = self.set_conditional_init(self.init1)
        
    def initialize_v(self):
        """
        Initialize the parameter of policy distribution
        :return: Nothing
        """
        weight = self.policy.get_nn_weights()
        self.eta1 = []
        self.eta2 = []
        self.std = []
        self.param_cnt = 0
        
        for w in weight:
            if len(w) == 0:
                self.eta1.append([])
                self.eta2.append([])
                self.std.append([])
            else:
                l1 = []
                l2 = []
                for tmp in w:
                    l1.append(np.ones_like(tmp) * 1)
                    l2.append(np.zeros_like(tmp))
                    self.param_cnt += np.prod(tmp.shape)
                if len(self.eta1) == 0:
                    self.eta1 = [l1]
                    self.eta2 = [l2]
                    self.std = [l1]
                else:
                    self.eta1.append(l1)
                    self.eta2.append(l2)
                    self.std.append(l1)
        self.ave_std = np.sqrt(1)
        self.max_std = np.sqrt(1)
        
    def set_conditional_init(self, init):
        """
        Given a threshold, replace the original initial distribution by the conditional distribution
        over the 'qualified' values
        :param init: the original initial distribution, returned by env.set_init_distribution
        :param threshold: the lower bound of alpha/(alpha+beta)
        :return: a conditional distribution over states where alpha/(alpha+beta) > threshold
        """
        cond_init = init.copy()
        cond_init[2] = np.array([tmp2 if tmp0 / (tmp0 + tmp1) >= self.equal_opp_thres else 0
                                 for tmp0, tmp1, tmp2 in zip(init[0], init[1], init[2])])
        if np.sum(cond_init[2]) > 0:
            cond_init[2] = cond_init[2] / np.sum(cond_init[2])
        else:
            logger.error("init = "+str(init))
            logger.error("threshold = "+str(self.equal_opp_thres))
            raise ValueError("For the given initial distribution and threshold, the conditional distribution is all 0.")
        return cond_init
                
    def get_empty_v(self):
        """
        Initialize the parameter of policy distribution
        :return: A list of parameters with the same structure as self.policy
        """
        weight = self.policy.get_nn_weights()
        res = []
        for w in weight:
            if len(w) == 0:
                res.append([])
            else:
                l1 = []
                for tmp in w:
                    # l1.append(np.zeros_like(tmp, dtype=np.float32))
                    l1.append(np.zeros_like(tmp))
                if len(res) == 0:
                    res = [l1]
                else:
                    res.append(l1)
        
        return res
        
    def sample_random_policy(self):
        """
        Randomly sample a set of weights for the policy network,
        and set the parameters to self.policy.policy_model
        :return: the weights that can be used as the input of self.policy.set_nn_weights(weights)
        """
        weights = []
        summ = 0
        self.max_std = 0
        "self.eta1: E[X^2]"
        "self.eta2: E[X]"
        for e1, e2 in zip(self.eta1, self.eta2):
            if len(e1) == 0 and len(e2) == 0:
                weights.append([])
                continue
            elif (len(e1) == 0 and len(e2) > 0) or (len(e1) > 0 and len(e2) == 0):
                raise ValueError("One of mean and variance is empty while the other is not.")
            
            tmp = []
            for p1, p2 in zip(e1, e2):
                try:
                    tmp.append(np.random.normal(p2, np.sqrt(p1 - np.power(p2,2))))
                    summ += np.sum(np.sum(np.sqrt(p1 - np.power(p2,2))))
                    self.max_std = np.maximum(self.max_std, np.amax(np.sqrt(p1-np.power(p2,2))))
                except ValueError:
                    tmp.append(np.random.normal(p2, np.sqrt(np.maximum(p1 - np.power(p2, 2), 0))))
                    summ += np.sum(np.sum(np.sqrt(np.maximum(p1 - np.power(p2, 2), 1e-5))))
                    self.max_std = np.maximum(self.max_std, np.amax(np.sqrt(np.maximum(p1 - np.power(p2, 2), 1e-5))))
            if len(weights) == 0:
                weights = [tmp]
            else:
                weights.append(tmp)
        
        self.policy.set_nn_weights(weights)
        self.ave_std = summ / self.param_cnt
        
        return weights
    
    def _initialize_DP(self):
        """
        This function computes all possible states that may be visited, which is saved in self.key_states.
        The index of state [ind_step, na, nb] is saved in self.key_index[ind_step, na, nb]
        :return:
        """
        m = self.h + np.maximum(self.env.h0, self.env.h1)
        num_of_key_states = int((m + 1) * (m + 2) * (m + 3) / 6)
        
        print("m = "+str(m)+", num_of_key_states = "+str(num_of_key_states))
        
        self.key_states0 = np.zeros((num_of_key_states, 3))
        self.key_states1 = np.zeros((num_of_key_states, 3))
        
        self.key_reward = np.zeros((num_of_key_states, 4))
        self.key_index = np.zeros((m+1, m+1, m+1))
        self.key_trans0 = np.zeros(num_of_key_states)
        self.key_trans1 = np.zeros(num_of_key_states)
        
        cnt = 0
        for ind_step in range(m, -1, -1):
            for na in range(ind_step+1):
                for nb in range(ind_step-na+1):
                    self.key_index[ind_step, na, nb] = int(cnt)
                    at = na
                    bt = nb + (ind_step - na - nb) * self.env.epsilon
                    a0 = at + self.env.alpha0
                    b0 = bt + self.env.beta0
                    self.key_states0[cnt, :] = np.array([a0 / (a0 + b0), a0 - b0, 0])
                    a1 = at + self.env.alpha1
                    b1 = bt + self.env.beta1
                    self.key_states1[cnt, :] = np.array([a1 / (a1 + b1), a1 - b1, 1])
                    
                    tmp = self.env.reward(np.array([[a0, b0, 0], [a0, b0, 0], [a1, b1, 1], [a1, b1, 1]]),
                                                   [0, 1, 0, 1])
                    self.key_reward[cnt, :] = tmp
                    
                    self.key_trans0[cnt] = a0 / (a0 + b0)
                    self.key_trans1[cnt] = a1 / (a1 + b1)
                    
                    cnt += 1
                    
        self.key_states1_blind = self.key_states1.copy()
        self.key_states1_blind[:,2] = 0
        
    def evaluate_policy(self, fairness_constraint=None):
        """
        Evaluate self.policy for self.h steps, by dynamic programming. Get g_objective and h_objective for the initial
        distributions in both groups. The initial distributions are stored in self.init0 and self.init1.
        :param fairness_constraint: None for demographic parity;
                                    1 for equality of opportunities (should be used together with parameter 'self.equal_opp_thres'.
        :return: [g_objective, h_objective] for self.policy
        """
        
        t_s = time.time()
        max_h = np.maximum(self.env.h0, self.env.h1)
        g0 = np.zeros((self.h+2+self.env.h0, self.h+2+self.env.h0, self.h+2+self.env.h0))
        h0 = np.zeros((self.h+2+self.env.h0, self.h+2+self.env.h0, self.h+2+self.env.h0))
        g1 = np.zeros((self.h+2+self.env.h1, self.h+2+self.env.h1, self.h+2+self.env.h1))
        h1 = np.zeros((self.h+2+self.env.h1, self.h+2+self.env.h1, self.h+2+self.env.h1))
        
        score0 = self.policy.policy_model.predict(self.key_states0)
        score1 = self.policy.policy_model.predict(self.key_states1)
        
        for ind_step in range(self.h+max_h, -1, -1):
            for na in range(ind_step+1):
                for nb in range(ind_step-na+1):
                    "Then n0 = step_ind+ - na - nb"
                    
                    index = int(self.key_index[ind_step, na, nb])
                    
                    if self.env.h0 <= ind_step <= self.h+self.env.h0:
                        g0[ind_step, na, nb] = score0[index][0] * (self.key_reward[index][0] + self.env.gamm * g0[ind_step+1, na, nb]) + \
                                               score0[index][1] * (self.key_reward[index][1] + self.env.gamm * (
                                self.key_trans0[index] * g0[ind_step+1, na+1, nb] +
                                (1-self.key_trans0[index]) * g0[ind_step+1, na, nb+1]))
                        h0[ind_step, na, nb] = score0[index][1] * (1 + self.key_trans0[index] * h0[ind_step + 1, na + 1, nb] +
                                                                   (1-self.key_trans0[index]) * h0[ind_step + 1, na, nb + 1]) + \
                                               score0[index][0] * h0[ind_step+1, na, nb]
                    elif ind_step < self.env.h0:
                        g0[ind_step, na, nb] = self.key_trans0[index] * g0[ind_step + 1, na + 1, nb] + \
                                               (1-self.key_trans0[index]) * g0[ind_step + 1, na, nb + 1]
                        h0[ind_step, na, nb] = self.key_trans0[index] * h0[ind_step + 1, na + 1, nb] + \
                                               (1-self.key_trans0[index]) * h0[ind_step + 1, na, nb + 1]
                    
                    if self.env.h1 <= ind_step <= self.h+self.env.h1:
                        g1[ind_step, na, nb] = score1[index][0] * (self.key_reward[index][2] + self.env.gamm * g1[ind_step+1, na, nb]) + \
                                               score1[index][1] * (self.key_reward[index][3] + self.env.gamm * (
                                self.key_trans1[index] * g1[ind_step+1, na+1, nb] +
                                (1-self.key_trans1[index]) * g1[ind_step+1, na, nb+1]))
                        h1[ind_step, na, nb] = score1[index][1] * (1 + self.key_trans1[index] * h1[ind_step + 1, na + 1, nb] +
                                                                   (1-self.key_trans1[index]) * h1[ind_step + 1, na, nb + 1]) + \
                                               score1[index][0] * h1[ind_step + 1, na, nb]
                    elif ind_step < self.env.h1:
                        g1[ind_step, na, nb] = self.key_trans1[index] * g1[ind_step + 1, na + 1, nb] + \
                                               (1-self.key_trans1[index]) * g1[ind_step + 1, na, nb + 1]
                        h1[ind_step, na, nb] = self.key_trans1[index] * h1[ind_step + 1, na + 1, nb] + \
                                               (1-self.key_trans1[index]) * h1[ind_step + 1, na, nb + 1]
                        
        # print(time.time() - t_s)
        if fairness_constraint is None:
            "demographic parity"
            return self.env.pZ * g1[0,0,0] + (1-self.env.pZ) * g0[0,0,0], \
                   -np.abs(h0[0,0,0] - h1[0,0,0]) / self.h, \
                   h0[0,0,0] / self.h, \
                   h1[0,0,0] / self.h
        
        elif fairness_constraint == 1 and self.equal_opp_thres is not None:
            "equality of opportunity"
            constraint = np.matmul(self.cond_init0[2],
                                   np.array([h0[self.env.h0, ii, self.env.h0-ii] for ii in range(self.env.h0,-1,-1)])) - \
                         np.matmul(self.cond_init1[2],
                                   np.array([h1[self.env.h1, ii, self.env.h1-ii] for ii in range(self.env.h1,-1,-1)]))
            constraint = -np.abs(constraint) / self.h
            
            return self.env.pZ * g1[0, 0, 0] + (1 - self.env.pZ) * g0[0, 0, 0], \
                   constraint, \
                   h0[0, 0, 0] / self.h, \
                   h1[0, 0, 0] / self.h

    def evaluate_race_blind_policy(self):
        """
        Evaluate self.policy for self.h steps, by dynamic programming. Get g_objective and h_objective for the initial
        distributions in both groups. The initial distributions are stored in self.init0 and self.init1.
        :return: [g_objective, h_objective] for self.policy
        """
        t_s = time.time()
        max_h = np.maximum(self.env.h0, self.env.h1)
        g0 = np.zeros((self.h + 2 + self.env.h0, self.h + 2 + self.env.h0, self.h + 2 + self.env.h0))
        h0 = np.zeros((self.h + 2 + self.env.h0, self.h + 2 + self.env.h0, self.h + 2 + self.env.h0))
        g1 = np.zeros((self.h + 2 + self.env.h1, self.h + 2 + self.env.h1, self.h + 2 + self.env.h1))
        h1 = np.zeros((self.h + 2 + self.env.h1, self.h + 2 + self.env.h1, self.h + 2 + self.env.h1))
    
        score0 = self.policy.policy_model.predict(self.key_states0)
        score1 = self.policy.policy_model.predict(self.key_states1_blind)
    
        for ind_step in range(self.h + max_h, -1, -1):
            for na in range(ind_step + 1):
                for nb in range(ind_step - na + 1):
                    "Then n0 = step_ind+ - na - nb"
                
                    index = int(self.key_index[ind_step, na, nb])
                
                    if self.env.h0 <= ind_step <= self.h + self.env.h0:
                        g0[ind_step, na, nb] = score0[index][0] * (
                                self.key_reward[index][0] + self.env.gamm * g0[ind_step + 1, na, nb]) + \
                                               score0[index][1] * (self.key_reward[index][1] + self.env.gamm * (
                                self.key_trans0[index] * g0[ind_step + 1, na + 1, nb] +
                                (1 - self.key_trans0[index]) * g0[ind_step + 1, na, nb + 1]))
                        h0[ind_step, na, nb] = score0[index][1] * (
                                1 + self.key_trans0[index] * h0[ind_step + 1, na + 1, nb] +
                                (1 - self.key_trans0[index]) * h0[ind_step + 1, na, nb + 1]) + \
                                               score0[index][0] * h0[ind_step + 1, na, nb]
                    elif ind_step < self.env.h0:
                        g0[ind_step, na, nb] = self.key_trans0[index] * g0[ind_step + 1, na + 1, nb] + \
                                               (1 - self.key_trans0[index]) * g0[ind_step + 1, na, nb + 1]
                        h0[ind_step, na, nb] = self.key_trans0[index] * h0[ind_step + 1, na + 1, nb] + \
                                               (1 - self.key_trans0[index]) * h0[ind_step + 1, na, nb + 1]
                
                    if self.env.h1 <= ind_step <= self.h + self.env.h1:
                        g1[ind_step, na, nb] = score1[index][0] * (
                                self.key_reward[index][2] + self.env.gamm * g1[ind_step + 1, na, nb]) + \
                                               score1[index][1] * (self.key_reward[index][3] + self.env.gamm * (
                                self.key_trans1[index] * g1[ind_step + 1, na + 1, nb] +
                                (1 - self.key_trans1[index]) * g1[ind_step + 1, na, nb + 1]))
                        h1[ind_step, na, nb] = score1[index][1] * (
                                1 + self.key_trans1[index] * h1[ind_step + 1, na + 1, nb] +
                                (1 - self.key_trans1[index]) * h1[ind_step + 1, na, nb + 1]) + \
                                               score1[index][0] * h1[ind_step + 1, na, nb]
                    elif ind_step < self.env.h1:
                        g1[ind_step, na, nb] = self.key_trans1[index] * g1[ind_step + 1, na + 1, nb] + \
                                               (1 - self.key_trans1[index]) * g1[ind_step + 1, na, nb + 1]
                        h1[ind_step, na, nb] = self.key_trans1[index] * h1[ind_step + 1, na + 1, nb] + \
                                               (1 - self.key_trans1[index]) * h1[ind_step + 1, na, nb + 1]

        constraint = np.matmul(self.cond_init0[2],
                               np.array([h0[self.env.h0, ii, self.env.h0 - ii] for ii in range(self.env.h0, -1,
                                                                                               -1)])) - \
                     np.matmul(self.cond_init1[2],
                               np.array([h1[self.env.h1, ii, self.env.h1 - ii] for ii in range(self.env.h1, -1, -1)]))
        constraint = -np.abs(constraint) / self.h
                         
        return self.env.pZ * g1[0, 0, 0] + (1 - self.env.pZ) * g0[0, 0, 0], \
               -np.abs(h0[0, 0, 0] - h1[0, 0, 0]) / self.h, \
               h0[0, 0, 0] / self.h, \
               h1[0, 0, 0] / self.h, \
               constraint

    def evaluate_conservative_policy(self, pi1):
        """
        Evaluate self.policy for self.h steps, by dynamic programming. Get g_objective and h_objective for the initial
        distributions in both groups. The initial distributions are stored in self.init0 and self.init1.
        The policy at any state shares the same action distribution
        :return: [g_objective, h_objective] for self.policy
        """
        t_s = time.time()
        max_h = np.maximum(self.env.h0, self.env.h1)
        g0 = np.zeros((self.h + 2 + self.env.h0, self.h + 2 + self.env.h0, self.h + 2 + self.env.h0))
        h0 = np.zeros((self.h + 2 + self.env.h0, self.h + 2 + self.env.h0, self.h + 2 + self.env.h0))
        g1 = np.zeros((self.h + 2 + self.env.h1, self.h + 2 + self.env.h1, self.h + 2 + self.env.h1))
        h1 = np.zeros((self.h + 2 + self.env.h1, self.h + 2 + self.env.h1, self.h + 2 + self.env.h1))
    
        score = [1 - pi1, pi1]
    
        for ind_step in range(self.h + max_h, -1, -1):
            for na in range(ind_step + 1):
                for nb in range(ind_step - na + 1):
                    "Then n0 = step_ind+ - na - nb"
                    index = int(self.key_index[ind_step, na, nb])
                
                    if self.env.h0 <= ind_step <= self.h + self.env.h0:
                        g0[ind_step, na, nb] = score[0] * (
                                self.key_reward[index][0] + self.env.gamm * g0[ind_step + 1, na, nb]) + \
                                               score[1] * (self.key_reward[index][1] + self.env.gamm * (
                                self.key_trans0[index] * g0[ind_step + 1, na + 1, nb] +
                                (1 - self.key_trans0[index]) * g0[ind_step + 1, na, nb + 1]))
                        h0[ind_step, na, nb] = score[1] * (
                                1 + self.key_trans0[index] * h0[ind_step + 1, na + 1, nb] +
                                (1 - self.key_trans0[index]) * h0[ind_step + 1, na, nb + 1]) + \
                                               score[0] * h0[ind_step + 1, na, nb]
                    elif ind_step < self.env.h0:
                        g0[ind_step, na, nb] = self.key_trans0[index] * g0[ind_step + 1, na + 1, nb] + \
                                               (1 - self.key_trans0[index]) * g0[ind_step + 1, na, nb + 1]
                        h0[ind_step, na, nb] = self.key_trans0[index] * h0[ind_step + 1, na + 1, nb] + \
                                               (1 - self.key_trans0[index]) * h0[ind_step + 1, na, nb + 1]
                
                    if self.env.h1 <= ind_step <= self.h + self.env.h1:
                        g1[ind_step, na, nb] = score[0] * (
                                self.key_reward[index][2] + self.env.gamm * g1[ind_step + 1, na, nb]) + \
                                               score[1] * (self.key_reward[index][3] + self.env.gamm * (
                                self.key_trans1[index] * g1[ind_step + 1, na + 1, nb] +
                                (1 - self.key_trans1[index]) * g1[ind_step + 1, na, nb + 1]))
                        h1[ind_step, na, nb] = score[1] * (
                                1 + self.key_trans1[index] * h1[ind_step + 1, na + 1, nb] +
                                (1 - self.key_trans1[index]) * h1[ind_step + 1, na, nb + 1]) + \
                                               score[0] * h1[ind_step + 1, na, nb]
                    elif ind_step < self.env.h1:
                        g1[ind_step, na, nb] = self.key_trans1[index] * g1[ind_step + 1, na + 1, nb] + \
                                               (1 - self.key_trans1[index]) * g1[ind_step + 1, na, nb + 1]
                        h1[ind_step, na, nb] = self.key_trans1[index] * h1[ind_step + 1, na + 1, nb] + \
                                               (1 - self.key_trans1[index]) * h1[ind_step + 1, na, nb + 1]

        constraint = np.matmul(self.cond_init0[2],
                               np.array([h0[self.env.h0, ii, self.env.h0 - ii] for ii in range(self.env.h0, -1,
                                                                                               -1)])) - \
                     np.matmul(self.cond_init1[2],
                               np.array([h1[self.env.h1, ii, self.env.h1 - ii] for ii in range(self.env.h1, -1, -1)]))
        constraint = -np.abs(constraint) / self.h
        return self.env.pZ * g1[0, 0, 0] + (1 - self.env.pZ) * g0[0, 0, 0], \
               -np.abs(h0[0, 0, 0] - h1[0, 0, 0]) / self.h, \
                1-pi1, pi1, constraint
        
    def train_demographic_parity(self):
        """
        Train the policy distribution using the constrained cross entropy method, with demographic parity constraint.
        epsilon is set to be self.tolerance
        :return:
        """
        logger.info("----- Demographic parity -----")
        
        samples_G = np.zeros(self.n_samples)
        samples_H = np.zeros(self.n_samples)
        z0_pos_rate = np.zeros(self.n_samples)
        z1_pos_rate = np.zeros(self.n_samples)
        sample_nn_weight_l1 = np.zeros(self.n_samples)
        
        n_best = max(1, int(self.n_samples * self.rho))
        itr_cnt = 0
        nl = self.nl
        
        cnt_good_average = 0
        last_mean_H = 0
        cnt_converged = 0
        
        logger.info("The number of parameters in the policy network = "+str(self.param_cnt))
        logger.info("The number of sample policies (n_samples) = "+str(self.n_samples))
        logger.info("The number of elite samples (n_best) = "+str(n_best))
        
        # while itr_cnt < 100 or (cnt_converged < 5 or cnt_good_average < 5) and itr_cnt < 200:
        while itr_cnt < 80:
            nn_weights = []
            
            # n_samples = int(self.n_samples * (itr_cnt+1)**.1)
            n_samples = self.n_samples
            
            logger.info("[DP Iteration "+str(itr_cnt)+" ] n_samples = "+str(n_samples))
            logger.info("average std value = "+str(self.ave_std)+", maximum std value = "+str(self.max_std))
            logger.info("cnt_good_average = " + str(cnt_good_average))
            logger.info("cnt_converged = " + str(cnt_converged))
                
            "Try different sample policies"
            for ii_policy in range(n_samples):
                policy_sample_weight = self.sample_random_policy()

                nn_weights.append(policy_sample_weight)
                self.policy.set_nn_weights(policy_sample_weight)
                
                "Record the total absolute value of all parameter weights"
                sample_nn_weight_l1[ii_policy] = self.policy.get_nn_weight_sum(policy_sample_weight)
                
                "Evaluate the given policy in self.policy"
                samples_G[ii_policy], samples_H[ii_policy], z0_pos_rate[ii_policy], z1_pos_rate[ii_policy] = self.evaluate_policy()
            
            "Update the policy distribution: first select a subset of elite sample policies"
            data_theta = sorted(list(zip(samples_G,
                                         samples_H,
                                         [ii for ii in range(len(samples_H))],
                                         sample_nn_weight_l1)),
                                key=itemgetter(0))
            data_theta = sorted(data_theta, key=itemgetter(1))
                
            ind_sort = [c for a, b, c, d in data_theta]
            best_inds = ind_sort[-n_best:]
    
            "Note that samples_H is non-positive here"
            if np.amin([samples_H[ii] for ii in best_inds]) > -self.tolerance:
                min_G = np.amin([samples_G[ii] for ii in best_inds])
                ind_sort = [c for a, b, c, d in data_theta if b >= -self.tolerance and a >= min_G]
                data_theta3 = sorted([(ii, samples_G[ii]) for ii in ind_sort], key=itemgetter(1))
                
                ind_sort = [a for a, c in data_theta3]
                best_inds = ind_sort[-n_best:]

            logger.info("ind_sort = "+str(ind_sort))
            # logger.info("(G, H, z0_pos_rate, z1_pos_rate) = " +
            #             str([(a, b, c, d) for a, b, c, d in zip(samples_G, samples_H, z0_pos_rate, z1_pos_rate)]))
            
            logger.info("G_theta for elite samples: "+str([samples_G[ii] for ii in best_inds]))
            logger.info("H_theta for elite samples: "+str([samples_H[ii] for ii in best_inds]))
            logger.info("Average G_theta for elite samples vs all samples: " +
                        str(np.mean([samples_G[ii] for ii in best_inds])) + ", " +
                        str(np.mean(samples_G)))
            logger.info("Average H_theta for elite samples vs all samples: " +
                        str(np.mean([samples_H[ii] for ii in best_inds])) + ", " +
                        str(np.mean(samples_H)))
            logger.info("Group 0 positive rate for elite samples vs all samples: " +
                        str(np.mean([z0_pos_rate[ii] for ii in best_inds])) + ", " +
                        str(np.mean(z0_pos_rate)))
            logger.info("Group 1 positive rate for elite samples vs all samples: " +
                        str(np.mean([z1_pos_rate[ii] for ii in best_inds])) + ", " +
                        str(np.mean(z1_pos_rate)))
            logger.info("Percentage of feasible samples = "+str(np.sum(samples_H > -self.tolerance) / len(samples_H)))
                
            if np.mean(samples_H) > -self.tolerance:
                cnt_good_average += 1
            else:
                cnt_good_average = 0
                
            if np.abs(np.mean(samples_H) - last_mean_H) < 1e-3:
                cnt_converged += 1
            else:
                cnt_converged = 0
            
            "Update the distribution"
            elist_g_list = [samples_G[ii] for ii in best_inds]
            elist_g_min = min(elist_g_list)
            
            norm_const = (max(elist_g_list) - min(elist_g_list)) * 0.2
            if norm_const < 1e-5:
                norm_const = 1
            # norm_const = .5
            
            logger.info("norm_const = "+str(norm_const))

            self.csv_writer_performance.writerow([np.mean(samples_G), np.mean(samples_H),
                                                  np.mean([samples_G[ii] for ii in best_inds]),
                                                  np.mean([samples_H[ii] for ii in best_inds]),
                                                  np.std(samples_G)/np.sqrt(self.n_samples-1),
                                                  np.std(samples_H)/np.sqrt(self.n_samples-1),
                                                  np.mean(z0_pos_rate), np.mean(z1_pos_rate),
                                                  norm_const, np.mean(np.abs(z0_pos_rate - z1_pos_rate))])
            self.csv_file_performance.flush()

            alp = .5
            lam = 0
            # alp = self.alp / (itr_cnt/100 + 1)**0.501
            # lam = self.lam /(itr_cnt+1)**.5

            w_tmp = np.zeros(len(best_inds))
            for ind_best_inds in range(len(best_inds)):
                w_tmp[ind_best_inds] = np.exp((samples_G[best_inds[ind_best_inds]] - elist_g_min) / norm_const)
            print("w_tmp = " + str(w_tmp))
            
            weight_struct = nn_weights[0]
            for ind_weight in range(len(weight_struct)):
                if len(weight_struct[ind_weight]) == 0:
                    print("empty")
                    pass
                else:
                    for ind_w in range(len(weight_struct[ind_weight])):
                        theta_tmp = np.array([np.zeros_like(weight_struct[ind_weight][ind_w])
                                              for _ in range(len(best_inds))])
                        theta_min = np.inf
                        theta_max = -np.inf
                        for ind_best_inds in range(len(best_inds)):
                            theta_tmp[ind_best_inds] = nn_weights[best_inds[ind_best_inds]][ind_weight][ind_w]
                            theta_min = np.minimum(theta_min, np.amin(theta_tmp[ind_best_inds]))
                            theta_max = np.maximum(theta_max, np.amax(theta_tmp[ind_best_inds]))
                            
                        if len(self.eta1[ind_weight][ind_w].shape) == 2:
                            self.eta1[ind_weight][ind_w] = \
                                alp * np.sum(np.square(theta_tmp) * np.reshape(w_tmp, (len(w_tmp), 1, 1)), axis=0) / np.sum(w_tmp) + \
                                (1 - alp) * self.eta1[ind_weight][ind_w]
                            self.eta2[ind_weight][ind_w] = \
                                alp * np.sum(theta_tmp * np.reshape(w_tmp, (len(w_tmp), 1, 1)), axis=0) / np.sum(w_tmp) + \
                                (1 - alp) * self.eta2[ind_weight][ind_w]
                            
                            tmp = self.eta1[ind_weight][ind_w] - np.square(self.eta2[ind_weight][ind_w])
                            empty_tmp = tmp[tmp < 0]
                            if len(empty_tmp):
                                self.eta1[ind_weight][ind_w][tmp<0] += empty_tmp + 1e-5
                            
                        elif len(self.eta1[ind_weight][ind_w].shape) == 1:
                            self.eta1[ind_weight][ind_w] = \
                                alp * np.sum(np.square(theta_tmp) * np.reshape(w_tmp, (len(w_tmp), 1)), axis=0) / np.sum(w_tmp) + \
                                (1 - alp) * self.eta1[ind_weight][ind_w]
                            self.eta2[ind_weight][ind_w] = \
                                alp * np.sum(theta_tmp * np.reshape(w_tmp, (len(w_tmp), 1)), axis=0) / np.sum(w_tmp) + \
                                (1 - alp) * self.eta2[ind_weight][ind_w]
                            
                            tmp = self.eta1[ind_weight][ind_w] - np.square(self.eta2[ind_weight][ind_w])
                            empty_tmp = tmp[tmp < 0]
                            if len(empty_tmp):
                                self.eta1[ind_weight][ind_w][tmp < 0] += empty_tmp + 1e-5
                            
                        self.std[ind_weight][ind_w] = np.maximum(self.std[ind_weight][ind_w], 1e-5)
                        
            print("Checking done. ")
            
            itr_cnt += 1
            print("-----")

        self.policy.policy_model.save(self.log_folder+"/policy_model.h5")

    def train_equality_of_opportunity(self):
        """
        Train the policy distribution using the constrained cross entropy method, with equality of opportunity
        constraints. epsilon is set to be self.tolerance
        :return:
        """
        logger.info("----- Equal of opportunity -----")
        
        samples_G = np.zeros(self.n_samples)
        samples_H = np.zeros(self.n_samples)
        z0_pos_rate = np.zeros(self.n_samples)
        z1_pos_rate = np.zeros(self.n_samples)
        sample_nn_weight_l1 = np.zeros(self.n_samples)
    
        n_best = max(1, int(self.n_samples * self.rho))
        itr_cnt = 0
    
        "The following variables are for termination conditions"
        cnt_good_average = 0
        last_mean_H = 0
        cnt_converged = 0
    
        logger.info("The number of parameters in the policy network = " + str(self.param_cnt))
        logger.info("The number of sample policies (n_samples) = " + str(self.n_samples))
        logger.info("The number of elite samples (n_best) = " + str(n_best))
    
        # while self.ave_std > 0.01 and itr_cnt < 100:
        # while (itr_cnt < 100) or (itr_cnt < 500 and (cnt_good_average < 20 or cnt_converged < 20)):
        while itr_cnt < 80:
        # while itr_cnt < 100 or (cnt_converged < 5 or cnt_good_average < 5):
            nn_weights = []
        
            # n_samples = int(self.n_samples * (itr_cnt+1)**.1)
            n_samples = self.n_samples
        
            logger.info("[Eoo Iteration " + str(itr_cnt) + " ] n_samples = " + str(n_samples))
            logger.info("average std value = " + str(self.ave_std) + ", maximum std value = " + str(self.max_std))
            logger.info("cnt_good_average = " + str(cnt_good_average))
            logger.info("cnt_converged = " + str(cnt_converged))
        
            "Try different sample policies"
            for ii_policy in range(n_samples):
                policy_sample_weight = self.sample_random_policy()
                nn_weights.append(policy_sample_weight)
                self.policy.set_nn_weights(policy_sample_weight)
            
                "Record the total absolute value of all parameter weights"
                sample_nn_weight_l1[ii_policy] = self.policy.get_nn_weight_sum(policy_sample_weight)

                "Evaluate the given policy in self.policy"
                samples_G[ii_policy], samples_H[ii_policy], \
                z0_pos_rate[ii_policy], z1_pos_rate[
                    ii_policy] = self.evaluate_policy(1)
                
            "Update the policy distribution: first select a subset of elite sample policies"
            data_theta = sorted(list(zip(samples_G,
                                         samples_H,
                                         [ii for ii in range(len(samples_H))],
                                         sample_nn_weight_l1)),
                                key=itemgetter(0))
            data_theta = sorted(data_theta, key=itemgetter(1))
        
            ind_sort = [c for a, b, c, d in data_theta]
            best_inds = ind_sort[-n_best:]
        
            "Note that samples_H is non-positive here"
            if np.amin([samples_H[ii] for ii in best_inds]) > -self.tolerance:
                min_G = np.amin([samples_G[ii] for ii in best_inds])
                ind_sort = [c for a, b, c, d in data_theta if b >= -self.tolerance and a >= min_G]
                data_theta3 = sorted([(ii, samples_G[ii]) for ii in ind_sort], key=itemgetter(1))
            
                ind_sort = [a for a, c in data_theta3]
                best_inds = ind_sort[-n_best:]
        
            logger.info("ind_sort = " + str(ind_sort))
            # logger.info("(G, H, z0_pos_rate, z1_pos_rate) = " +
            #             str([(a, b, c, d) for a, b, c, d in zip(samples_G, samples_H, z0_pos_rate, z1_pos_rate)]))
        
            logger.info("G_theta for elite samples: " + str([samples_G[ii] for ii in best_inds]))
            logger.info("H_theta for elite samples: " + str([samples_H[ii] for ii in best_inds]))
            logger.info("Average G_theta for elite samples vs all samples: " +
                        str(np.mean([samples_G[ii] for ii in best_inds])) + ", " +
                        str(np.mean(samples_G)))
            logger.info("Average H_theta for elite samples vs all samples: " +
                        str(np.mean([samples_H[ii] for ii in best_inds])) + ", " +
                        str(np.mean(samples_H)))
            logger.info("Group 0 positive rate for elite samples vs all samples: " +
                    str(np.mean([z0_pos_rate[ii] for ii in best_inds])) + ", " +
                    str(np.mean(z0_pos_rate)))
            logger.info("Group 1 positive rate for elite samples vs all samples: " +
                    str(np.mean([z1_pos_rate[ii] for ii in best_inds])) + ", " +
                    str(np.mean(z1_pos_rate)))
            logger.info("Percentage of feasible samples = " + str(np.sum(samples_H > -self.tolerance) / len(samples_H)))
        
            if np.mean(samples_H) > -self.tolerance:
                cnt_good_average += 1
            else:
                cnt_good_average = 0
        
            if np.abs(np.mean(samples_H) - last_mean_H) < 1e-3:
                cnt_converged += 1
            else:
                cnt_converged = 0
        
            "Update the distribution"
            elist_g_list = [samples_G[ii] for ii in best_inds]
            elist_g_min = min(elist_g_list)
        
            norm_const = (max(elist_g_list) - min(elist_g_list)) * 0.2
            if norm_const < 1e-5:
                norm_const = 1
            # norm_const = .5
        
            logger.info("norm_const = " + str(norm_const))
        
            self.csv_writer_performance.writerow([np.mean(samples_G), np.mean(samples_H),
                                                  np.mean([samples_G[ii] for ii in best_inds]),
                                                  np.mean([samples_H[ii] for ii in best_inds]),
                                                  np.std(samples_G) / np.sqrt(self.n_samples - 1),
                                                  np.std(samples_H) / np.sqrt(self.n_samples - 1),
                                                  np.mean(z0_pos_rate), np.mean(z1_pos_rate),
                                                  norm_const, np.mean(np.abs(z0_pos_rate - z1_pos_rate))])
            self.csv_file_performance.flush()
        
            alp = .5
            lam = 0
            # alp = self.alp / (itr_cnt/100 + 1)**0.501
            # lam = self.lam /(itr_cnt+1)**.5
        
            w_tmp = np.zeros(len(best_inds))
            for ind_best_inds in range(len(best_inds)):
                w_tmp[ind_best_inds] = np.exp((samples_G[best_inds[ind_best_inds]] - elist_g_min) / norm_const)
            print("w_tmp = " + str(w_tmp))
        
            weight_struct = nn_weights[0]
            for ind_weight in range(len(weight_struct)):
                if len(weight_struct[ind_weight]) == 0:
                    print("empty")
                    pass
                else:
                    for ind_w in range(len(weight_struct[ind_weight])):
                        theta_tmp = np.array([np.zeros_like(weight_struct[ind_weight][ind_w])
                                              for _ in range(len(best_inds))])
                        theta_min = np.inf
                        theta_max = -np.inf
                        for ind_best_inds in range(len(best_inds)):
                            theta_tmp[ind_best_inds] = nn_weights[best_inds[ind_best_inds]][ind_weight][ind_w]
                            theta_min = np.minimum(theta_min, np.amin(theta_tmp[ind_best_inds]))
                            theta_max = np.maximum(theta_max, np.amax(theta_tmp[ind_best_inds]))
                    
                        if len(self.eta1[ind_weight][ind_w].shape) == 2:
                            self.eta1[ind_weight][ind_w] = \
                                alp * np.sum(np.square(theta_tmp) * np.reshape(w_tmp, (len(w_tmp), 1, 1)),
                                             axis=0) / np.sum(w_tmp) + \
                                (1 - alp) * self.eta1[ind_weight][ind_w]
                            self.eta2[ind_weight][ind_w] = \
                                alp * np.sum(theta_tmp * np.reshape(w_tmp, (len(w_tmp), 1, 1)), axis=0) / np.sum(
                                    w_tmp) + \
                                (1 - alp) * self.eta2[ind_weight][ind_w]
                        
                            tmp = self.eta1[ind_weight][ind_w] - np.square(self.eta2[ind_weight][ind_w])
                            empty_tmp = tmp[tmp < 0]
                            if len(empty_tmp):
                                self.eta1[ind_weight][ind_w][tmp < 0] += empty_tmp + 1e-5
                    
                        elif len(self.eta1[ind_weight][ind_w].shape) == 1:
                            self.eta1[ind_weight][ind_w] = \
                                alp * np.sum(np.square(theta_tmp) * np.reshape(w_tmp, (len(w_tmp), 1)),
                                             axis=0) / np.sum(w_tmp) + \
                                (1 - alp) * self.eta1[ind_weight][ind_w]
                            self.eta2[ind_weight][ind_w] = \
                                alp * np.sum(theta_tmp * np.reshape(w_tmp, (len(w_tmp), 1)), axis=0) / np.sum(w_tmp) + \
                                (1 - alp) * self.eta2[ind_weight][ind_w]
                        
                            tmp = self.eta1[ind_weight][ind_w] - np.square(self.eta2[ind_weight][ind_w])
                            empty_tmp = tmp[tmp < 0]
                            if len(empty_tmp):
                                self.eta1[ind_weight][ind_w][tmp < 0] += empty_tmp + 1e-5
                    
                        self.std[ind_weight][ind_w] = np.maximum(self.std[ind_weight][ind_w], 1e-5)
        
            print("Checking done. ")
        
            itr_cnt += 1
            print("-----")
    
        self.policy.policy_model.save(self.log_folder + "/policy_model.h5")

    def train_optimistic_DP(self):
        """
        Train the policy distribution using the constrained cross entropy method, with optimistic assumption.
        :return:
        """
        logger.info("----- Optimistic case, DP -----")
    
        samples_G = np.zeros(self.n_samples)
        samples_H = np.zeros(self.n_samples)
        z0_pos_rate = np.zeros(self.n_samples)
        z1_pos_rate = np.zeros(self.n_samples)
        sample_nn_weight_l1 = np.zeros(self.n_samples)
    
        n_best = max(1, int(self.n_samples * self.rho))
        itr_cnt = 0
        nl = self.nl
    
        "The following variables are for termination conditions"
        cnt_good_average = 0
        last_mean_H = 0
        cnt_converged = 0
    
        logger.info("The number of parameters in the policy network = " + str(self.param_cnt))
        logger.info("The number of sample policies (n_samples) = " + str(self.n_samples))
        logger.info("The number of elite samples (n_best) = " + str(n_best))
    
        # while self.ave_std > 0.01 and itr_cnt < 100:
        # while (itr_cnt < 100) or (itr_cnt < 500 and (cnt_good_average < 20 or cnt_converged < 20)):
        while itr_cnt < 80:
        # while itr_cnt < 100 or (cnt_converged < 5 or cnt_good_average < 5):
            nn_weights = []

            n_samples = self.n_samples
        
            logger.info("[Opt-DP Iteration " + str(itr_cnt) + " ] n_samples = " + str(n_samples))
            logger.info("average std value = " + str(self.ave_std) + ", maximum std value = " + str(self.max_std))
            logger.info("cnt_good_average = " + str(cnt_good_average))
            logger.info("cnt_converged = " + str(cnt_converged))
        
            "Try different sample policies"
            for ii_policy in range(n_samples):
                policy_sample_weight = self.sample_random_policy()
                nn_weights.append(policy_sample_weight)
                self.policy.set_nn_weights(policy_sample_weight)
            
                "Record the total absolute value of all parameter weights"
                sample_nn_weight_l1[ii_policy] = self.policy.get_nn_weight_sum(policy_sample_weight)

                "Evaluate the given policy in self.policy"
                samples_G[ii_policy], _, z0_pos_rate[ii_policy], z1_pos_rate[ii_policy] = self.evaluate_policy()
            
                "g_objective and h_objective saves the objective values for each simulation trajectory "
                pos_cnt0_tmp, pos_cnt1_tmp = 0, 0
                
                for ind_x in range(len(self.init0[0])):
                    s_input_0 = np.array([[self.init0[0][ind_x] / (self.init0[0][ind_x] + self.init0[1][ind_x]),
                                           self.init0[0][ind_x] - self.init0[1][ind_x], 0]])
                    score = self.policy.policy_model.predict(s_input_0)
                    pos_cnt0_tmp += score[0][1] * self.init0[2][ind_x]
                    
                for ind_x in range(len(self.init1[0])):
                    s_input_1 = np.array([[self.init1[0][ind_x] / (self.init1[0][ind_x] + self.init1[1][ind_x]),
                                           self.init1[0][ind_x] - self.init1[1][ind_x], 1]])
                    score = self.policy.policy_model.predict(s_input_1)
                    pos_cnt1_tmp += score[0][1] * self.init1[2][ind_x]

                samples_H[ii_policy] = -np.abs(pos_cnt0_tmp - pos_cnt1_tmp)
        
            "Update the policy distribution: first select a subset of elite sample policies"
            data_theta = sorted(list(zip(samples_G,
                                         samples_H,
                                         [ii for ii in range(len(samples_H))],
                                         sample_nn_weight_l1)),
                                key=itemgetter(0))
            data_theta = sorted(data_theta, key=itemgetter(1))
        
            ind_sort = [c for a, b, c, d in data_theta]
            best_inds = ind_sort[-n_best:]
        
            "Note that samples_H is non-positive here"
            if np.amin([samples_H[ii] for ii in best_inds]) > -self.tolerance:
                min_G = np.amin([samples_G[ii] for ii in best_inds])
                ind_sort = [c for a, b, c, d in data_theta if b >= -self.tolerance and a >= min_G]
                data_theta3 = sorted([(ii, samples_G[ii]) for ii in ind_sort], key=itemgetter(1))
            
                ind_sort = [a for a, c in data_theta3]
                best_inds = ind_sort[-n_best:]
        
            logger.info("ind_sort = " + str(ind_sort))
            # logger.info("(G, H, z0_pos_rate, z1_pos_rate) = " +
            #             str([(a, b, c, d) for a, b, c, d in zip(samples_G, samples_H, z0_pos_rate, z1_pos_rate)]))
        
            logger.info("G_theta for elite samples: " + str([samples_G[ii] for ii in best_inds]))
            logger.info("H_theta for elite samples: " + str([samples_H[ii] for ii in best_inds]))
            logger.info("Average G_theta for elite samples vs all samples: " +
                        str(np.mean([samples_G[ii] for ii in best_inds])) + ", " +
                        str(np.mean(samples_G)))
            logger.info("Average H_theta for elite samples vs all samples: " +
                        str(np.mean([samples_H[ii] for ii in best_inds])) + ", " +
                        str(np.mean(samples_H)))
            logger.info("Percentage of feasible samples = " + str(np.sum(samples_H > -self.tolerance) / len(samples_H)))
        
            if np.mean(samples_H) > -self.tolerance:
                cnt_good_average += 1
            else:
                cnt_good_average = 0
        
            if np.abs(np.mean(samples_H) - last_mean_H) < 1e-3:
                cnt_converged += 1
            else:
                cnt_converged = 0
        
            "Update the distribution"
        
            elist_g_list = [samples_G[ii] for ii in best_inds]
            elist_g_min = min(elist_g_list)
        
            norm_const = (max(elist_g_list) - min(elist_g_list)) * 0.2
            if norm_const < 1e-5:
                norm_const = 1
            # norm_const = .5
        
            logger.info("norm_const = " + str(norm_const))
        
            self.csv_writer_performance.writerow([np.mean(samples_G), np.mean(samples_H),
                                                  np.mean([samples_G[ii] for ii in best_inds]),
                                                  np.mean([samples_H[ii] for ii in best_inds]),
                                                  np.std(samples_G) / np.sqrt(self.n_samples - 1),
                                                  np.std(samples_H) / np.sqrt(self.n_samples - 1),
                                                  np.mean(z0_pos_rate), np.mean(z1_pos_rate),
                                                  norm_const, np.mean(np.abs(z0_pos_rate - z1_pos_rate))])
            self.csv_file_performance.flush()
        
            alp = .5
            lam = 0
            # alp = self.alp / (itr_cnt/100 + 1)**0.501
            # lam = self.lam /(itr_cnt+1)**.5
        
            w_tmp = np.zeros(len(best_inds))
            for ind_best_inds in range(len(best_inds)):
                w_tmp[ind_best_inds] = np.exp((samples_G[best_inds[ind_best_inds]] - elist_g_min) / norm_const)
            print("w_tmp = " + str(w_tmp))
        
            weight_struct = nn_weights[0]
            for ind_weight in range(len(weight_struct)):
                if len(weight_struct[ind_weight]) == 0:
                    print("empty")
                    pass
                else:
                    for ind_w in range(len(weight_struct[ind_weight])):
                        theta_tmp = np.array([np.zeros_like(weight_struct[ind_weight][ind_w])
                                              for _ in range(len(best_inds))])
                        theta_min = np.inf
                        theta_max = -np.inf
                        for ind_best_inds in range(len(best_inds)):
                            theta_tmp[ind_best_inds] = nn_weights[best_inds[ind_best_inds]][ind_weight][ind_w]
                            theta_min = np.minimum(theta_min, np.amin(theta_tmp[ind_best_inds]))
                            theta_max = np.maximum(theta_max, np.amax(theta_tmp[ind_best_inds]))
                    
                        if len(self.eta1[ind_weight][ind_w].shape) == 2:
                            self.eta1[ind_weight][ind_w] = \
                                alp * np.sum(np.square(theta_tmp) * np.reshape(w_tmp, (len(w_tmp), 1, 1)),
                                             axis=0) / np.sum(w_tmp) + \
                                (1 - alp) * self.eta1[ind_weight][ind_w]
                            self.eta2[ind_weight][ind_w] = \
                                alp * np.sum(theta_tmp * np.reshape(w_tmp, (len(w_tmp), 1, 1)), axis=0) / np.sum(
                                    w_tmp) + \
                                (1 - alp) * self.eta2[ind_weight][ind_w]
                        
                            tmp = self.eta1[ind_weight][ind_w] - np.square(self.eta2[ind_weight][ind_w])
                            empty_tmp = tmp[tmp < 0]
                            if len(empty_tmp):
                                self.eta1[ind_weight][ind_w][tmp < 0] += empty_tmp + 1e-5
                    
                        elif len(self.eta1[ind_weight][ind_w].shape) == 1:
                            self.eta1[ind_weight][ind_w] = \
                                alp * np.sum(np.square(theta_tmp) * np.reshape(w_tmp, (len(w_tmp), 1)),
                                             axis=0) / np.sum(w_tmp) + \
                                (1 - alp) * self.eta1[ind_weight][ind_w]
                            self.eta2[ind_weight][ind_w] = \
                                alp * np.sum(theta_tmp * np.reshape(w_tmp, (len(w_tmp), 1)), axis=0) / np.sum(w_tmp) + \
                                (1 - alp) * self.eta2[ind_weight][ind_w]
                        
                            tmp = self.eta1[ind_weight][ind_w] - np.square(self.eta2[ind_weight][ind_w])
                            empty_tmp = tmp[tmp < 0]
                            if len(empty_tmp):
                                self.eta1[ind_weight][ind_w][tmp < 0] += empty_tmp + 1e-5
                    
                        self.std[ind_weight][ind_w] = np.maximum(self.std[ind_weight][ind_w], 1e-5)
        
            print("Checking done. ")
        
            itr_cnt += 1
            print("-----")
    
        self.policy.policy_model.save(self.log_folder + "/policy_model.h5")

    def train_optimistic_EO(self):
        """
        Train the policy distribution using the constrained cross entropy method, with optimistic assumption.
        :return:
        """
        logger.info("----- Optimistic-EO case -----")
    
        samples_G = np.zeros(self.n_samples)
        samples_H = np.zeros(self.n_samples)
        samples_H_eo = np.zeros(self.n_samples)
        z0_pos_rate = np.zeros(self.n_samples)
        z1_pos_rate = np.zeros(self.n_samples)
        sample_nn_weight_l1 = np.zeros(self.n_samples)
    
        n_best = max(1, int(self.n_samples * self.rho))
        itr_cnt = 0
        nl = self.nl
    
        "The following variables are for termination conditions"
        cnt_good_average = 0
        last_mean_H = 0
        cnt_converged = 0
    
        logger.info("The number of parameters in the policy network = " + str(self.param_cnt))
        logger.info("The number of sample policies (n_samples) = " + str(self.n_samples))
        logger.info("The number of elite samples (n_best) = " + str(n_best))
    
        # while self.ave_std > 0.01 and itr_cnt < 100:
        # while (itr_cnt < 100) or (itr_cnt < 500 and (cnt_good_average < 20 or cnt_converged < 20)):
        while itr_cnt < 80:
            # while itr_cnt < 100 or (cnt_converged < 5 or cnt_good_average < 5):
            nn_weights = []
        
            n_samples = self.n_samples
        
            logger.info("[Opt-EO Iteration " + str(itr_cnt) + " ] n_samples = " + str(n_samples))
            logger.info("average std value = " + str(self.ave_std) + ", maximum std value = " + str(self.max_std))
            logger.info("cnt_good_average = " + str(cnt_good_average))
            logger.info("cnt_converged = " + str(cnt_converged))
        
            "Try different sample policies"
            for ii_policy in range(n_samples):
                policy_sample_weight = self.sample_random_policy()
                nn_weights.append(policy_sample_weight)
                self.policy.set_nn_weights(policy_sample_weight)
            
                "Record the total absolute value of all parameter weights"
                sample_nn_weight_l1[ii_policy] = self.policy.get_nn_weight_sum(policy_sample_weight)
            
                "Evaluate the given policy in self.policy"
                samples_G[ii_policy], samples_H_eo[ii_policy], z0_pos_rate[ii_policy], z1_pos_rate[ii_policy] = \
                    self.evaluate_policy(1)
            
                "g_objective and h_objective saves the objective values for each simulation trajectory "
                pos_cnt0_tmp, pos_cnt1_tmp = 0, 0
            
                for ind_x in range(len(self.init0[0])):
                    s_input_0 = np.array([[self.init0[0][ind_x] / (self.init0[0][ind_x] + self.init0[1][ind_x]),
                                           self.init0[0][ind_x] - self.init0[1][ind_x], 0]])
                    score = self.policy.policy_model.predict(s_input_0)
                    pos_cnt0_tmp += score[0][1] * self.cond_init0[2][ind_x]
            
                for ind_x in range(len(self.init1[0])):
                    s_input_1 = np.array([[self.init1[0][ind_x] / (self.init1[0][ind_x] + self.init1[1][ind_x]),
                                           self.init1[0][ind_x] - self.init1[1][ind_x], 1]])
                    score = self.policy.policy_model.predict(s_input_1)
                    pos_cnt1_tmp += score[0][1] * self.cond_init1[2][ind_x]
            
                samples_H[ii_policy] = -np.abs(pos_cnt0_tmp - pos_cnt1_tmp)
        
            "Update the policy distribution: first select a subset of elite sample policies"
            data_theta = sorted(list(zip(samples_G,
                                         samples_H,
                                         [ii for ii in range(len(samples_H))],
                                         sample_nn_weight_l1)),
                                key=itemgetter(0))
            data_theta = sorted(data_theta, key=itemgetter(1))
        
            ind_sort = [c for a, b, c, d in data_theta]
            best_inds = ind_sort[-n_best:]
        
            "Note that samples_H is non-positive here"
            if np.amin([samples_H[ii] for ii in best_inds]) > -self.tolerance:
                min_G = np.amin([samples_G[ii] for ii in best_inds])
                ind_sort = [c for a, b, c, d in data_theta if b >= -self.tolerance and a >= min_G]
                data_theta3 = sorted([(ii, samples_G[ii]) for ii in ind_sort], key=itemgetter(1))
            
                ind_sort = [a for a, c in data_theta3]
                best_inds = ind_sort[-n_best:]
        
            logger.info("ind_sort = " + str(ind_sort))
            # logger.info("(G, H, z0_pos_rate, z1_pos_rate) = " +
            #             str([(a, b, c, d) for a, b, c, d in zip(samples_G, samples_H, z0_pos_rate, z1_pos_rate)]))
        
            logger.info("G_theta for elite samples: " + str([samples_G[ii] for ii in best_inds]))
            logger.info("H_theta for elite samples: " + str([samples_H[ii] for ii in best_inds]))
            logger.info("Average G_theta for elite samples vs all samples: " +
                        str(np.mean([samples_G[ii] for ii in best_inds])) + ", " +
                        str(np.mean(samples_G)))
            logger.info("Average H_theta for elite samples vs all samples: " +
                        str(np.mean([samples_H[ii] for ii in best_inds])) + ", " +
                        str(np.mean(samples_H)))
            logger.info("Percentage of feasible samples = " + str(np.sum(samples_H > -self.tolerance) / len(samples_H)))
        
            if np.mean(samples_H) > -self.tolerance:
                cnt_good_average += 1
            else:
                cnt_good_average = 0
        
            if np.abs(np.mean(samples_H) - last_mean_H) < 1e-3:
                cnt_converged += 1
            else:
                cnt_converged = 0
        
            "Update the distribution"
        
            elist_g_list = [samples_G[ii] for ii in best_inds]
            elist_g_min = min(elist_g_list)
        
            norm_const = (max(elist_g_list) - min(elist_g_list)) * 0.2
            if norm_const < 1e-5:
                norm_const = 1
            # norm_const = .5
        
            logger.info("norm_const = " + str(norm_const))
        
            self.csv_writer_performance.writerow([np.mean(samples_G), np.mean(samples_H),
                                                  np.mean([samples_G[ii] for ii in best_inds]),
                                                  np.mean([samples_H[ii] for ii in best_inds]),
                                                  np.std(samples_G) / np.sqrt(self.n_samples - 1),
                                                  np.std(samples_H) / np.sqrt(self.n_samples - 1),
                                                  np.mean(z0_pos_rate), np.mean(z1_pos_rate),
                                                  norm_const, np.mean(np.abs(z0_pos_rate - z1_pos_rate)),
                                                  np.mean(samples_H_eo),
                                                  np.std(samples_H_eo) / np.sqrt(self.n_samples - 1)])
            self.csv_file_performance.flush()
        
            alp = .5
            lam = 0
            # alp = self.alp / (itr_cnt/100 + 1)**0.501
            # lam = self.lam /(itr_cnt+1)**.5
        
            w_tmp = np.zeros(len(best_inds))
            for ind_best_inds in range(len(best_inds)):
                w_tmp[ind_best_inds] = np.exp((samples_G[best_inds[ind_best_inds]] - elist_g_min) / norm_const)
            print("w_tmp = " + str(w_tmp))
        
            weight_struct = nn_weights[0]
            for ind_weight in range(len(weight_struct)):
                if len(weight_struct[ind_weight]) == 0:
                    print("empty")
                    pass
                else:
                    for ind_w in range(len(weight_struct[ind_weight])):
                        theta_tmp = np.array([np.zeros_like(weight_struct[ind_weight][ind_w])
                                              for _ in range(len(best_inds))])
                        theta_min = np.inf
                        theta_max = -np.inf
                        for ind_best_inds in range(len(best_inds)):
                            theta_tmp[ind_best_inds] = nn_weights[best_inds[ind_best_inds]][ind_weight][ind_w]
                            theta_min = np.minimum(theta_min, np.amin(theta_tmp[ind_best_inds]))
                            theta_max = np.maximum(theta_max, np.amax(theta_tmp[ind_best_inds]))
                    
                        if len(self.eta1[ind_weight][ind_w].shape) == 2:
                            self.eta1[ind_weight][ind_w] = \
                                alp * np.sum(np.square(theta_tmp) * np.reshape(w_tmp, (len(w_tmp), 1, 1)),
                                             axis=0) / np.sum(w_tmp) + \
                                (1 - alp) * self.eta1[ind_weight][ind_w]
                            self.eta2[ind_weight][ind_w] = \
                                alp * np.sum(theta_tmp * np.reshape(w_tmp, (len(w_tmp), 1, 1)), axis=0) / np.sum(
                                    w_tmp) + \
                                (1 - alp) * self.eta2[ind_weight][ind_w]
                        
                            tmp = self.eta1[ind_weight][ind_w] - np.square(self.eta2[ind_weight][ind_w])
                            empty_tmp = tmp[tmp < 0]
                            if len(empty_tmp):
                                self.eta1[ind_weight][ind_w][tmp < 0] += empty_tmp + 1e-5
                    
                        elif len(self.eta1[ind_weight][ind_w].shape) == 1:
                            self.eta1[ind_weight][ind_w] = \
                                alp * np.sum(np.square(theta_tmp) * np.reshape(w_tmp, (len(w_tmp), 1)),
                                             axis=0) / np.sum(w_tmp) + \
                                (1 - alp) * self.eta1[ind_weight][ind_w]
                            self.eta2[ind_weight][ind_w] = \
                                alp * np.sum(theta_tmp * np.reshape(w_tmp, (len(w_tmp), 1)), axis=0) / np.sum(w_tmp) + \
                                (1 - alp) * self.eta2[ind_weight][ind_w]
                        
                            tmp = self.eta1[ind_weight][ind_w] - np.square(self.eta2[ind_weight][ind_w])
                            empty_tmp = tmp[tmp < 0]
                            if len(empty_tmp):
                                self.eta1[ind_weight][ind_w][tmp < 0] += empty_tmp + 1e-5
                    
                        self.std[ind_weight][ind_w] = np.maximum(self.std[ind_weight][ind_w], 1e-5)
        
            print("Checking done. ")
        
            itr_cnt += 1
            print("-----")
    
        self.policy.policy_model.save(self.log_folder + "/policy_model.h5")

    def train_race_blind(self):
        """
        Train the policy distribution using the constrained cross entropy method, with the race blind assumption.
        epsilon is set to be self.tolerance
        Equivalently, the policy for group 1 and group 0 are the same
        :return:
        """
        logger.info("----- Race blind case -----")
    
        samples_G = np.zeros(self.n_samples)
        samples_H = np.zeros(self.n_samples)
        samples_H_eo = np.zeros(self.n_samples)
        z0_pos_rate = np.zeros(self.n_samples)
        z1_pos_rate = np.zeros(self.n_samples)
        sample_nn_weight_l1 = np.zeros(self.n_samples)
    
        n_best = max(1, int(self.n_samples * self.rho))
        itr_cnt = 0
        nl = self.nl
    
        cnt_good_average = 0
        last_mean_H = 0
        cnt_converged = 0
    
        logger.info("The number of parameters in the policy network = " + str(self.param_cnt))
        logger.info("The number of sample policies (n_samples) = " + str(self.n_samples))
        logger.info("The number of elite samples (n_best) = " + str(n_best))
    
        # while self.ave_std > 0.01 and itr_cnt < 100:
        # while (itr_cnt < 100) or (itr_cnt < 500 and (cnt_good_average < 20 or cnt_converged < 20)):
        while itr_cnt < 80:
        # while itr_cnt < 100 or (cnt_converged < 5 or cnt_good_average < 5):
            nn_weights = []
        
            # n_samples = int(self.n_samples * (itr_cnt+1)**.1)
            n_samples = self.n_samples
        
            logger.info("[Rac Iteration " + str(itr_cnt) + " ] n_samples = " + str(n_samples))
            logger.info("average std value = " + str(self.ave_std) + ", maximum std value = " + str(self.max_std))
            logger.info("cnt_good_average = " + str(cnt_good_average))
            logger.info("cnt_converged = " + str(cnt_converged))
        
            "Try different sample policies"
            for ii_policy in range(n_samples):
                policy_sample_weight = self.sample_random_policy()
                nn_weights.append(policy_sample_weight)
                self.policy.set_nn_weights(policy_sample_weight)
            
                "Record the total absolute value of all parameter weights"
                sample_nn_weight_l1[ii_policy] = self.policy.get_nn_weight_sum(policy_sample_weight)

                samples_G[ii_policy], samples_H[ii_policy], z0_pos_rate[ii_policy], \
                    z1_pos_rate[ii_policy], samples_H_eo[ii_policy] = self.evaluate_race_blind_policy()
        
            "Update the policy distribution: first select a subset of elite sample policies"
            "For the conservative case, we only sort the sample policies using their G values"
            data_theta = sorted(list(zip(samples_G,
                                         samples_H,
                                         [ii for ii in range(len(samples_H))],
                                         sample_nn_weight_l1)),
                                key=itemgetter(0))
        
            ind_sort = [c for a, b, c, d in data_theta]
            best_inds = ind_sort[-n_best:]
        
            logger.info("ind_sort = " + str(ind_sort))
            # logger.info("(G, H, z0_pos_rate, z1_pos_rate) = " +
            #             str([(a, b, c, d) for a, b, c, d in zip(samples_G, samples_H, z0_pos_rate, z1_pos_rate)]))
        
            logger.info("G_theta for elite samples: " + str([samples_G[ii] for ii in best_inds]))
            logger.info("H_theta for elite samples: " + str([samples_H[ii] for ii in best_inds]))
            logger.info("Average G_theta for elite samples vs all samples: " +
                        str(np.mean([samples_G[ii] for ii in best_inds])) + ", " +
                        str(np.mean(samples_G)))
            logger.info("Average H_theta for elite samples vs all samples: " +
                        str(np.mean([samples_H[ii] for ii in best_inds])) + ", " +
                        str(np.mean(samples_H)))
            logger.info("Percentage of feasible samples = " + str(np.sum(samples_H > -self.tolerance) / len(samples_H)))
        
            if np.mean(samples_H) > -self.tolerance:
                cnt_good_average += 1
            else:
                cnt_good_average = 0
        
            if np.abs(np.mean(samples_H) - last_mean_H) < 1e-3:
                cnt_converged += 1
            else:
                cnt_converged = 0
        
            "Update the distribution"
            elist_g_list = [samples_G[ii] for ii in best_inds]
            elist_g_min = min(elist_g_list)
        
            norm_const = (max(elist_g_list) - min(elist_g_list)) * 0.2
            if norm_const < 1e-5:
                norm_const = 1
            # norm_const = .5
        
            logger.info("norm_const = " + str(norm_const))
        
            self.csv_writer_performance.writerow([np.mean(samples_G), np.mean(samples_H),
                                                  np.mean([samples_G[ii] for ii in best_inds]),
                                                  np.mean([samples_H[ii] for ii in best_inds]),
                                                  np.std(samples_G) / np.sqrt(self.n_samples - 1),
                                                  np.std(samples_H) / np.sqrt(self.n_samples - 1),
                                                  np.mean(z0_pos_rate), np.mean(z1_pos_rate),
                                                  norm_const, np.mean(np.abs(z0_pos_rate - z1_pos_rate)),
                                                  np.mean(samples_H_eo),
                                                  np.std(samples_H_eo) / np.sqrt(self.n_samples - 1)])
            self.csv_file_performance.flush()
        
            alp = .5
            lam = 0
            # alp = self.alp / (itr_cnt/100 + 1)**0.501
            # lam = self.lam /(itr_cnt+1)**.5
        
            w_tmp = np.zeros(len(best_inds))
            for ind_best_inds in range(len(best_inds)):
                w_tmp[ind_best_inds] = np.exp((samples_G[best_inds[ind_best_inds]] - elist_g_min) / norm_const)
            print("w_tmp = " + str(w_tmp))
        
            weight_struct = nn_weights[0]
            for ind_weight in range(len(weight_struct)):
                if len(weight_struct[ind_weight]) == 0:
                    print("empty")
                    pass
                else:
                    for ind_w in range(len(weight_struct[ind_weight])):
                        theta_tmp = np.array([np.zeros_like(weight_struct[ind_weight][ind_w])
                                              for _ in range(len(best_inds))])
                        theta_min = np.inf
                        theta_max = -np.inf
                        for ind_best_inds in range(len(best_inds)):
                            theta_tmp[ind_best_inds] = nn_weights[best_inds[ind_best_inds]][ind_weight][ind_w]

                            theta_min = np.minimum(theta_min, np.amin(theta_tmp[ind_best_inds]))
                            theta_max = np.maximum(theta_max, np.amax(theta_tmp[ind_best_inds]))
                    
                        if len(self.eta1[ind_weight][ind_w].shape) == 2:
                            self.eta1[ind_weight][ind_w] = \
                                alp * np.sum(np.square(theta_tmp) * np.reshape(w_tmp, (len(w_tmp), 1, 1)),
                                             axis=0) / np.sum(w_tmp) + \
                                (1 - alp) * self.eta1[ind_weight][ind_w]
                            self.eta2[ind_weight][ind_w] = \
                                alp * np.sum(theta_tmp * np.reshape(w_tmp, (len(w_tmp), 1, 1)), axis=0) / np.sum(
                                    w_tmp) + \
                                (1 - alp) * self.eta2[ind_weight][ind_w]
                        
                            tmp = self.eta1[ind_weight][ind_w] - np.square(self.eta2[ind_weight][ind_w])
                            empty_tmp = tmp[tmp < 0]
                            if len(empty_tmp):
                                self.eta1[ind_weight][ind_w][tmp < 0] += empty_tmp + 1e-5
                    
                        elif len(self.eta1[ind_weight][ind_w].shape) == 1:
                            self.eta1[ind_weight][ind_w] = \
                                alp * np.sum(np.square(theta_tmp) * np.reshape(w_tmp, (len(w_tmp), 1)),
                                             axis=0) / np.sum(w_tmp) + \
                                (1 - alp) * self.eta1[ind_weight][ind_w]
                            self.eta2[ind_weight][ind_w] = \
                                alp * np.sum(theta_tmp * np.reshape(w_tmp, (len(w_tmp), 1)), axis=0) / np.sum(w_tmp) + \
                                (1 - alp) * self.eta2[ind_weight][ind_w]
                        
                            tmp = self.eta1[ind_weight][ind_w] - np.square(self.eta2[ind_weight][ind_w])
                            empty_tmp = tmp[tmp < 0]
                            if len(empty_tmp):
                                self.eta1[ind_weight][ind_w][tmp < 0] += empty_tmp + 1e-5
                    
                        self.std[ind_weight][ind_w] = np.maximum(self.std[ind_weight][ind_w], 1e-5)
        
            print("Checking done. ")
        
            itr_cnt += 1
            print("-----")
    
        self.policy.policy_model.save(self.log_folder + "/policy_model.h5")

    def train_conservative(self):
        """
        Train the policy distribution using the constrained cross entropy method, with conservative assumption constraint.
        epsilon is set to be self.tolerance
        :return:
        """
        logger.info("----- Conservative case -----")
    
        samples_G = np.zeros(self.n_samples)
        samples_H = np.zeros(self.n_samples)
        samples_H_eo = np.zeros(self.n_samples)
        z0_pos_rate = np.zeros(self.n_samples)
        z1_pos_rate = np.zeros(self.n_samples)
    
        n_best = max(1, int(self.n_samples * self.rho))
        itr_cnt = 0
    
        cnt_good_average = 0
        last_mean_H = 0
        cnt_converged = 0
        
        mu = 0.5
        sigma = 0.2
        eta1 = mu ** 2 + sigma ** 2
        eta2 = mu
    
        logger.info("The number of sample policies (n_samples) = " + str(self.n_samples))
        logger.info("The number of elite samples (n_best) = " + str(n_best))
    
        while itr_cnt < 50 or (cnt_converged < 5 or cnt_good_average < 5) and itr_cnt < 100:
            nn_weights = np.random.normal(mu, sigma, self.n_samples)
            while (nn_weights > 1).any() or (nn_weights < 0).any():
                nn_weights[nn_weights > 1] = np.random.normal(mu, sigma, np.sum(nn_weights > 1))
                nn_weights[nn_weights < 0] = np.random.normal(mu, sigma, np.sum(nn_weights < 0))
        
            # n_samples = int(self.n_samples * (itr_cnt+1)**.1)
            n_samples = self.n_samples
        
            logger.info("[Con Iteration " + str(itr_cnt) + " ] n_samples = " + str(n_samples))
            logger.info("cnt_good_average = " + str(cnt_good_average))
            logger.info("cnt_converged = " + str(cnt_converged))
        
            "Try different sample policies"
            for ii_policy in range(n_samples):
                
                "Evaluate the given policy in self.policy"
                samples_G[ii_policy], samples_H[ii_policy], z0_pos_rate[ii_policy], z1_pos_rate[
                    ii_policy], samples_H_eo[ii_policy] = self.evaluate_conservative_policy(nn_weights[ii_policy])
        
            "Update the policy distribution: first select a subset of elite sample policies"
            data_theta = sorted(list(zip(samples_G,
                                         samples_H,
                                         [ii for ii in range(len(samples_H))])),
                                key=itemgetter(0))
            data_theta = sorted(data_theta, key=itemgetter(1))
        
            ind_sort = [c for a, b, c in data_theta]
            best_inds = ind_sort[-n_best:]
        
            "Note that samples_H is non-positive here"
            if np.amin([samples_H[ii] for ii in best_inds]) > -self.tolerance:
                min_G = np.amin([samples_G[ii] for ii in best_inds])
                ind_sort = [c for a, b, c in data_theta if b >= -self.tolerance and a >= min_G]
                data_theta3 = sorted([(ii, samples_G[ii]) for ii in ind_sort], key=itemgetter(1))
            
                ind_sort = [a for a, c in data_theta3]
                best_inds = ind_sort[-n_best:]
        
            logger.info("ind_sort = " + str(ind_sort))
            # logger.info("(G, H, z0_pos_rate, z1_pos_rate) = " +
            #             str([(a, b, c, d) for a, b, c, d in zip(samples_G, samples_H, z0_pos_rate, z1_pos_rate)]))
        
            logger.info("G_theta for elite samples: " + str([samples_G[ii] for ii in best_inds]))
            logger.info("H_theta for elite samples: " + str([samples_H[ii] for ii in best_inds]))
            logger.info("Average G_theta for elite samples vs all samples: " +
                        str(np.mean([samples_G[ii] for ii in best_inds])) + ", " +
                        str(np.mean(samples_G)))
            logger.info("Average H_theta for elite samples vs all samples: " +
                        str(np.mean([samples_H[ii] for ii in best_inds])) + ", " +
                        str(np.mean(samples_H)))
            logger.info("Percentage of feasible samples = " + str(np.sum(samples_H > -self.tolerance) / len(samples_H)))
        
            if np.mean(samples_H) > -self.tolerance:
                cnt_good_average += 1
            else:
                cnt_good_average = 0
        
            if np.abs(np.mean(samples_H) - last_mean_H) < 1e-3:
                cnt_converged += 1
            else:
                cnt_converged = 0
        
            "Update the distribution"
            elist_g_list = [samples_G[ii] for ii in best_inds]
            elist_g_min = min(elist_g_list)
        
            norm_const = (max(elist_g_list) - min(elist_g_list)) * 0.2
            if norm_const < 1e-5:
                norm_const = 1
            # norm_const = .5
        
            logger.info("norm_const = " + str(norm_const))
        
            self.csv_writer_performance.writerow([np.mean(samples_G), np.mean(samples_H),
                                                  np.mean([samples_G[ii] for ii in best_inds]),
                                                  np.mean([samples_H[ii] for ii in best_inds]),
                                                  np.std(samples_G) / np.sqrt(self.n_samples - 1),
                                                  np.std(samples_H) / np.sqrt(self.n_samples - 1),
                                                  np.mean(z0_pos_rate), np.mean(z1_pos_rate),
                                                  norm_const, np.mean(np.abs(z0_pos_rate - z1_pos_rate)),
                                                  np.mean(samples_H_eo),
                                                  np.std(samples_H_eo) / np.sqrt(self.n_samples - 1)])
            self.csv_file_performance.flush()
        
            alp = .5
            lam = 0
            # alp = self.alp / (itr_cnt/100 + 1)**0.501
            # lam = self.lam /(itr_cnt+1)**.5
        
            w_tmp = np.zeros(len(best_inds))
            theta_tmp = np.zeros(len(best_inds))
            
            for ind_best_inds in range(len(best_inds)):
                w_tmp[ind_best_inds] = np.exp((samples_G[best_inds[ind_best_inds]] - elist_g_min) / norm_const)
                theta_tmp[ind_best_inds] = nn_weights[best_inds[ind_best_inds]]
            print("w_tmp = " + str(w_tmp))
            print("theta_tmp = "+str(theta_tmp))

            eta1 = alp * np.sum(np.square(theta_tmp) * w_tmp, axis=0) / np.sum(w_tmp) + (1 - alp) * eta1
            eta2 = alp * np.sum(theta_tmp * w_tmp, axis=0) / np.sum(w_tmp) + (1 - alp) * eta2
            
            mu = eta2
            sigma = eta1 - mu**2
            if sigma < 0:
                sigma = 1e-5
            else:
                sigma = np.sqrt(sigma)
            
            print("Checking done. ")
        
            itr_cnt += 1
            print("-----")
    