import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import players
import random

#distribution variables
lower, upper = 0,1
mu, sigma = 0.5, 0.1
cppA = [mu, sigma, upper, lower]
cppB = [mu, sigma, upper, lower]

# model variables
T = 20
M = 50 # M >= T
M_a, M_b = 25, 25
PAa, PAb, PBa, PBb = 0.3, 0.1, 0.05, 0.4


m = 5 #size of unit mass
v = 2 #utility for sharing, known to both user and platform
c = 1 #cost of clicking, known to both user and platform
qA = 0.7
qB = 0.6

u = []

for t in range(1,T):
	#unit mass arrives
	for i in range(m):
		g = 2 * np.random.binomial(1, float(M_a / M))- 1
		if g == 1:
			cpp = [mu,sigma,upper,lower]
		else:
			cpp = [mu,sigma,upper,lower] 
		u.append(players(group=g, clickprobparams=cpp))

	#show article for each user
	for user in u:
		if t == 1:
			# mechanism to decide which article to share
			break
		else:
			new_u = []
			if user.group == 1:
				q = qA
				cpp = cppA
			else:
				q = qB
				cpp = cppB
			if random.uniform(0,1) <= q:
				new_user = players(group=user.group, clickprobparams=cpp)
				if user.shared == True:
					new_user.article = user.article
				else:
					new_user.article = 'n' # mechanism to decide which article to share
			else:
				new_user = players(group=-1 * user.group, clickprobparams=cpp)
				# mechanism to decide which article to share.
			

			#decide if user shares article, according to PAa, PAb, etc.
			if new_user.group == 1 and new_user.article == 'a':
				if random.uniform(0,1) <= PAa:
					new_user.shared= True
			elif new_user.group == 1 and new_user.article == 'b':
				if random.uniform(0,1) <= PAb:
					new_user.shared= True
			elif new_user.group == -1 and new_user.article == 'a':
				if random.uniform(0,1) <= PBa:
					new_user.shared= True
			else:
				if random.uniform(0,1) <= PBb:
					new_user.shared= True

			#add user to list
			new_u.append(new_user)
	u = new_u

