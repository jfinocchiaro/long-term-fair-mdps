#-*-coding:utf-8-*-
"""
Solve a loan example using the constrained cross-entropy method
"""

""" Hyperparameters:
    In transition function:
        epsilon:        Parameter in transition function when a = 0
    In reward function:
        lamda:          trade-off parameter between the mean and the variance, positive
        I:              the interest collected on the loan if repaid, positive
        P:              the principal of the loan, positive
    In initial distribution:
        pZ:             the parameter in initial state distribution
        alpha0, beta0, alpha1, beta0: parameters in initial states
"""

import time, os, sys, logging
import tensorflow as tf
import numpy as np
from shutil import copy2

import policyNetwork
import constrainedCEM
import envLoan

file_name_base = time.ctime()
file_name_base = file_name_base.replace(' ', '-')
file_name_base = file_name_base.replace(':', '-')

log_folder = "logs/v0_" + file_name_base #PERSONALIZE: CHANGE FOR YOUR FILE PATH
#log_folder = "/home/min/PycharmProjects/fairness/v0_" + file_name_base #PERSONALIZE: CHANGE FOR YOUR FILE PATH

if not os.path.exists(log_folder):
    os.makedirs(log_folder)


root_folder = "venv" #PERSONALIZE: again, change for your file path
#root_folder = "/home/min/PycharmProjects/fairness/venv" #PERSONALIZE: again, change for your file path

copy2(root_folder + '/CEM_loan.py', log_folder)
copy2(root_folder + '/envLoan.py', log_folder)
copy2(root_folder + '/policyNetwork.py', log_folder)
copy2(root_folder + '/constrainedCEM.py', log_folder)

"""Set up the logging part"""
logging.getLogger().setLevel(logging.ERROR)
# logger = logging.getLogger(__name__)
logger = logging.getLogger("CEM_loan")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(funcName)s:%(lineno)i] %(message)s',
                              datefmt="%Y/%m/%d %H:%M:%S")

fh = logging.FileHandler(filename= log_folder + "/" + file_name_base + "_console.log", mode='a')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

"""Main program"""
sess = tf.InteractiveSession()
sess.run(tf.initializers.global_variables())

"""Hyperparameters"""
action_dim = 1
state_dim = 3

rho = .08
h = 50
"h0 is the number of steps to simulate to get the initial distribution for group 0 (with action 1)"
"h1 is the number of steps to simulate to get the initial distribution for group 1 (with action 1)"
"For z = 0, 1, there will be (hz+1) initial states"
# h0 = 1
# h1 = 3
h0 = 1
h1 = 5


n_samples = 100
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
lamda = 0.01
# lamda = 0.01
# lamda = .5
# .01: 332, 407; .1: 242, 316; 0.5: 51, 100; 1: 0, 31; vs 3276
# .01: 2957, 3269;.5: 472, 665 (30856)
epsilon = .1
gamm = .5

policy_net_struct = [50, 50]

logger.info("state_dim = "+str(state_dim)+", action_dim = "+str(action_dim))
logger.info("rho = "+str(rho)+", h = "+str(h)+", n_samples = "+str(n_samples)+", epsilon = "+str(epsilon)+
            ", nl = "+str(nl))
logger.info("alpha0 = "+str(alpha0)+", beta0 = "+str(beta0)+", alpha1 = "+str(alpha1)+", beta1 = "+str(beta1))
logger.info("pZ = "+str(pZ)+", I = "+str(I)+", P = "+str(P)+", lamda = "+str(lamda)+", gamm = "+str(gamm))

logger.info("The policy network: "+str(policy_net_struct))

"""
In version0: the policy network takes 2 inputs: alpha/(alpha+beta) and z.
In version1: the policy network takes 3 inputs: alpha/(alpha+beta), alpha-beta, z.
"""
policy = policyNetwork.PolicyNetwork(3, policy_net_struct)
env = envLoan.EnvLoan(state_dim, action_dim, h, epsilon, alpha0, alpha1, beta0, beta1, pZ, I, P, lamda, gamm, h1, h0)

# print(env.set_init_state())
#Jessie: missing the parameter: equal_opp_thres; plugged in epsilon, but not sure if that's right.
agent = constrainedCEM.ConstrainedCEM(env, policy, n_samples, h, rho, nl, sess, epsilon, log_folder)

agent.train_demographic_parity()
# agent.train_equality_of_opportunity(equal_opp_thres)
# agent.train_optimistic()
# agent.train_race_blind()
# agent.train_conservative()
