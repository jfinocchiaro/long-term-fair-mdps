import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 
import scipy.stats as stats
import players
import random
import platform_opt

#distribution variables
lower, upper = 0,1
mu, sigma = 0.5, 0.1
cppA = [mu, sigma, upper, lower]
cppB = [mu, sigma, upper, lower]

# model variables
T = 200
M = 300 # M >= T
M_a, M_b = 150, 150
PAa, PAb, PBa, PBb = 0.8, 0.01, 0.05, 0.2

PLA, PLB =0.1 ,0.05 #probability of like | group membership

m = 10 #size of unit mass
v = 2 #utility for sharing, known to both user and platform *NOT USED YET*
c = 1 #cost of clicking, known to both user and platform *NOT USED YET*
qA = 0.8 #probability of transitioning to player in group A conditioned on old player being in group A
qB = 0.8 #probability of transitioning to player in group B conditioned on old player being in group B

epsilon = 0.1 #approximation parameter for approximately equal probability of showing articles |theta - 1/2| <= epsilon

probshowA = platform_opt.optimize(epsilon, M_a, M, T, PAa, PBa, PLA=PLA, PLB=PLB, muA=cppA[0], muB=cppB[0]) #platform chooses their probability for showing article a by maximizing expected clickthrough rate subject to fairness constraints

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
	time_data_diff.append(np.sum([user.article for user in old_u]) / float(m))


plt.plot(time_data_diff, color='black')
plt.title("Mass of articles being shown over time")
plt.ylabel("learning towards article $a$ (1) and $b$ (-1)")
plt.xlabel("timestep t")
plt.ylim((-1,1))
plt.axhline(y=0,color='grey')
plt.axhline(y=np.average(time_data_diff),color='blue')
plt.axhline(y=epsilon,color='red')
plt.axhline(y=-1 * epsilon,color='red')
plt.show()
plt.savefig('article_leaning_overtime.png')