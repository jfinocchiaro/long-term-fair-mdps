import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 
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
PAa, PAb, PBa, PBb = 0.4, 0.01, 0.05, 0.4


m = 5 #size of unit mass
v = 2 #utility for sharing, known to both user and platform
c = 1 #cost of clicking, known to both user and platform
qA = 0.8
qB = 0.8

probshowA = .5 #holder variable for now, thi will be optimized later
old_u = []
time_data_diff = []

for t in range(1,T+1):
	new_u = [] #list of new players that arrive at the timestep
	#unit mass arrives

	if t == 1: #initial mass of users arrives
		for i in range(m):
			g = 2 * np.random.binomial(1, float(M_a / M))- 1
			if g == 1:
				cpp = [mu,sigma,upper,lower]
			else:
				cpp = [mu,sigma,upper,lower] 
			old_u.append(players.players(group=g, clickprobparams=cpp, article=2 * np.random.binomial(1, probshowA)- 1))
	
	else:
		for user in old_u:
				
			#now users are replaced in place (kinda)
			if user.group == 1:
				q = qA
				cpp = cppA
			else:
				q = qB
				cpp = cppB
			if random.uniform(0,1) <= q:
				new_user = players.players(group=user.group, clickprobparams=cpp)
				if user.shared == True:
					new_user.article = user.article
				else:
					new_user.article = 2 * np.random.binomial(1, probshowA)- 1 # mechanism to decide which article to share
			else:
				new_user = players.players(group=-1 * user.group, clickprobparams=cpp)
				# mechanism to decide which article to share.
				new_user.article = 2 * np.random.binomial(1, probshowA)- 1
			

			#decide if user shares article, according to PAa, PAb, etc.
			if new_user.group == 1 and new_user.article == 1:
				if random.uniform(0,1) <= PAa:
					new_user.shared= True
			elif new_user.group == 1 and new_user.article == -1:
				if random.uniform(0,1) <= PAb:
					new_user.shared= True
			elif new_user.group == -1 and new_user.article == -1:
				if random.uniform(0,1) <= PBa:
					new_user.shared= True
			else:
				if random.uniform(0,1) <= PBb:
					new_user.shared= True

			#add user to list
			new_u.append(new_user)

		old_u = new_u
	print [user.article for user in old_u]
	print np.sum([user.article for user in old_u])
	time_data_diff.append(np.sum([user.article for user in old_u]) / float(m))

print time_data_diff
plt.plot(time_data_diff)
plt.title("Mass of articles being shown over time")
plt.ylabel("learning towards article $a$ (1) and $b$ (-1)")
plt.xlabel("timestep t")
plt.show()
plt.savefig('article_leaning_overtime.png')