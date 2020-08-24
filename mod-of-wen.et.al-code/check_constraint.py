#-*-coding:utf-8-*-

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

import time, os, sys, logging
from shutil import copy2

import policyNetwork
import constrainedCEM
import envLoan


"""Main program"""
sess = tf.InteractiveSession()
sess.run(tf.initializers.global_variables())

"""Hyperparameters"""
action_dim = 1
state_dim = 3

rho = .1

h = 50
gamm = .8

"h0 is the number of steps to simulate to get the initial distribution for group 0 (with action 1)"
"h1 is the number of steps to simulate to get the initial distribution for group 1 (with action 1)"
"For z = 0, 1, there will be (hz+1) initial states"
# h0 = 1
# h1 = 3
h0 = 7
h1 = 10

n_samples = 50
nl = 49

alpha0 = 1.4614873
beta0 = 0.51479711

alpha1 = 1.51578208
beta1 = 0.19009129

# equal_opp_thres = 0.85237955
equal_opp_thres = 0.82

pZ = 1-0.29294318
I = 0.17318629
# I = 0.041225
P = 1
lamda = .01
# lamda = 0.1
epsilon = .1 # don't change this

policy_net_struct = [25, 25]

fairness_constraint = None

"""
In version0: the policy network takes 2 inputs: alpha/(alpha+beta) and z.
In version1: the policy network takes 3 inputs: alpha/(alpha+beta), alpha-beta, z.
"""
policy = policyNetwork.PolicyNetwork(3, policy_net_struct)
env = envLoan.EnvLoan(state_dim, action_dim, h, epsilon, alpha0, alpha1, beta0, beta1, pZ, I, P, lamda, gamm, h1, h0)

policy_model = load_model("/home/min/PycharmProjects/fairness/" +
                                   "h0_7_h1_10_gamma_point8_threshold_point05_lambda_point01/" +
                                   "optimistic_EO/OptEO_Thu-Jan-17-10-43-13-2019/policy_model.h5")

# print(env.set_init_state())

init0 = env.set_init_distribution(0)
init1 = env.set_init_distribution(1)
        
m = h + np.maximum(h0, h1)
num_of_key_states = int((m + 1) * (m + 2) * (m + 3) / 6)

print("m = " + str(m) + ", num_of_key_states = " + str(num_of_key_states))

key_states0 = np.zeros((num_of_key_states, 3))
key_states1 = np.zeros((num_of_key_states, 3))

key_reward = np.zeros((num_of_key_states, 4))
key_index = np.zeros((m + 1, m + 1, m + 1))
key_trans0 = np.zeros(num_of_key_states)
key_trans1 = np.zeros(num_of_key_states)

cnt = 0
for ind_step in range(m, -1, -1):
    for na in range(ind_step + 1):
        for nb in range(ind_step - na + 1):
            key_index[ind_step, na, nb] = int(cnt)
            at = na
            bt = nb + (ind_step - na - nb) * epsilon
            a0 = at + alpha0
            b0 = bt + beta0
            key_states0[cnt, :] = np.array([a0 / (a0 + b0), a0 - b0, 0])
            a1 = at + alpha1
            b1 = bt + beta1
            key_states1[cnt, :] = np.array([a1 / (a1 + b1), a1 - b1, 1])
            
            tmp = env.reward(np.array([[a0, b0, 0], [a0, b0, 0], [a1, b1, 1], [a1, b1, 1]]),
                                  [0, 1, 0, 1])
            key_reward[cnt, :] = tmp
            
            key_trans0[cnt] = a0 / (a0 + b0)
            key_trans1[cnt] = a1 / (a1 + b1)
            
            cnt += 1

key_states1_blind = key_states1.copy()
key_states1_blind[:, 2] = 0


t_s = time.time()
max_h = np.maximum(h0, h1)
g0_mat = np.zeros((h + 2 + h0, h + 2 + h0, h + 2 + h0))
h0_mat = np.zeros((h + 2 + h0, h + 2 + h0, h + 2 + h0))
g1_mat = np.zeros((h + 2 + h1, h + 2 + h1, h + 2 + h1))
h1_mat = np.zeros((h + 2 + h1, h + 2 + h1, h + 2 + h1))

score0 = policy_model.predict(key_states0)
score1 = policy_model.predict(key_states1)

for ind_step in range(h + max_h, -1, -1):
    for na in range(ind_step + 1):
        for nb in range(ind_step - na + 1):
            "Then n0 = step_ind+ - na - nb"
            
            index = int(key_index[ind_step, na, nb])
            
            if h0 <= ind_step <= h + h0:
                g0_mat[ind_step, na, nb] = score0[index][0] * (
                        key_reward[index][0] + gamm * g0_mat[ind_step + 1, na, nb]) + \
                                       score0[index][1] * (key_reward[index][1] + gamm * (
                        key_trans0[index] * g0_mat[ind_step + 1, na + 1, nb] +
                        (1 - key_trans0[index]) * g0_mat[ind_step + 1, na, nb + 1]))
                h0_mat[ind_step, na, nb] = score0[index][1] * (
                        1 + key_trans0[index] * h0_mat[ind_step + 1, na + 1, nb] +
                        (1 - key_trans0[index]) * h0_mat[ind_step + 1, na, nb + 1]) + \
                                       score0[index][0] * h0_mat[ind_step + 1, na, nb]
            elif ind_step < h0:
                g0_mat[ind_step, na, nb] = key_trans0[index] * g0_mat[ind_step + 1, na + 1, nb] + \
                                       (1 - key_trans0[index]) * g0_mat[ind_step + 1, na, nb + 1]
                h0_mat[ind_step, na, nb] = key_trans0[index] * h0_mat[ind_step + 1, na + 1, nb] + \
                                       (1 - key_trans0[index]) * h0_mat[ind_step + 1, na, nb + 1]
            
            if h1 <= ind_step <= h + h1:
                g1_mat[ind_step, na, nb] = score1[index][0] * (
                        key_reward[index][2] + gamm * g1_mat[ind_step + 1, na, nb]) + \
                                       score1[index][1] * (key_reward[index][3] + gamm * (
                        key_trans1[index] * g1_mat[ind_step + 1, na + 1, nb] +
                        (1 - key_trans1[index]) * g1_mat[ind_step + 1, na, nb + 1]))
                h1_mat[ind_step, na, nb] = score1[index][1] * (
                        1 + key_trans1[index] * h1_mat[ind_step + 1, na + 1, nb] +
                        (1 - key_trans1[index]) * h1_mat[ind_step + 1, na, nb + 1]) + \
                                       score1[index][0] * h1_mat[ind_step + 1, na, nb]
            elif ind_step < h1:
                g1_mat[ind_step, na, nb] = key_trans1[index] * g1_mat[ind_step + 1, na + 1, nb] + \
                                       (1 - key_trans1[index]) * g1_mat[ind_step + 1, na, nb + 1]
                h1_mat[ind_step, na, nb] = key_trans1[index] * h1_mat[ind_step + 1, na + 1, nb] + \
                                       (1 - key_trans1[index]) * h1_mat[ind_step + 1, na, nb + 1]

# print(time.time() - t_s)
if fairness_constraint is None:
    "demographic parity"
    g_objective, h_objective, z0_pos, z1_pos = pZ * g1_mat[0, 0, 0] + (1 - pZ) * g0_mat[0, 0, 0], \
           -np.abs(h0_mat[0, 0, 0] - h1_mat[0, 0, 0]) / h, \
           h0_mat[0, 0, 0] / h, \
           h1_mat[0, 0, 0] / h

elif fairness_constraint == 1 and equal_opp_thres is not None:
    "equality of opportunity"
    cond_init0 = init0.copy()
    cond_init0[2] = np.array([tmp2 if tmp0 / (tmp0 + tmp1) >= equal_opp_thres else 0
                             for tmp0, tmp1, tmp2 in zip(init0[0], init0[1], init0[2])])
    if np.sum(cond_init0[2]) > 0:
        cond_init0[2] = cond_init0[2] / np.sum(cond_init0[2])

    cond_init1 = init1.copy()
    cond_init1[2] = np.array([tmp2 if tmp0 / (tmp0 + tmp1) >= equal_opp_thres else 0
                              for tmp0, tmp1, tmp2 in zip(init1[0], init1[1], init1[2])])
    if np.sum(cond_init1[2]) > 0:
        cond_init1[2] = cond_init1[2] / np.sum(cond_init1[2])
    
    h0 = int(h0)
    h1 = int(h1)
    constraint = np.matmul(cond_init0[2],
                           np.array([h0_mat[h0, ii, h0 - ii] for ii in range(h0, -1, -1)])) - \
                 np.matmul(cond_init1[2],
                           np.array([h1_mat[h1, ii, h1 - ii] for ii in range(h1, -1, -1)]))
    constraint = -np.abs(constraint) / h

    g_objective, h_objective, z0_pos, z1_pos =  pZ * g1_mat[0, 0, 0] + (1 - pZ) * g0_mat[0, 0, 0], \
           constraint, \
           h0_mat[0, 0, 0] / h, \
           h1_mat[0, 0, 0] / h

print("g_objective = "+str(g_objective))
print("h_objective = "+str(h_objective))
print("z0_pos = "+str(z0_pos))
print("z1_pos = "+str(z1_pos))

