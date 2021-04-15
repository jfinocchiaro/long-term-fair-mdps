import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 
import scipy.stats as stats
import players
import random
import platform_opt

#distribution variables
#lower, upper = 0,1
#mu, sigma = 0.5, 0.1
#cppA = [mu, sigma, upper, lower]
#cppB = [mu, sigma, upper, lower]

# model variables
T = 100 #number of timesteps
M = 200 # M >= T
M_a, M_b = 100, 100 #number of members in groups a and b

######### questions and comments #########
# Not a question, rather comment. Great job with the comments 
#	- ahh thanks :)
# - Is it necessary for number of users to be of the same quantity? 
# 	- no it is not.  They are currently represented by M_a and M_b above
# - From what I understand, users are either 1 or -1, right? It's a bit confusing, since it's like a label ish, I think the same applies to the group.
#	- I'm not sure what you mean by the first question.  The users are objects defined in players.py, and the attributes they havve, including "group", "clicked", and "shared" are appl +/- 1
# - I would like to clarify if PAa + PAb = 1 or if PAa + PBa = 1
# 	- great question: actually, neither.  PAb is the probability of a member of group A sharing article b given that they click on it.  Since these are independent, neither has a "sum to one" constraint.
# - Here in the code, there is still probability of clicking, shouldn't it be based on the payoff?  
# 	- Yes, good catch- I think I forgot to remove it, but believe it's not being used.
# I however, I am uncertain about the how we will estimate cost, c and value, v. Is the c the same for all agents in a group? Is it 
# lower in one group and higher for another? Does it change over time? And the v, how is estimated?
# 	- my understanding was that c should be a function of group membership and article type, and same with v.  This just got discussed in the last meeting though, and figured I would keep it simple for testing now.  Can definitely change it though.
# - In the overleaf, they seemed to say that unit mass is 1, I am not sure how this value is estimated and why it's deemed a constant. 
# 	- The main goal is to be able to have a "handoff" of articles from one agent to some "next" agent.  The unit mass is 1 really just means that it's the same number of users at each timestep.
# I however, I am certain it has an effect on the model. My suggestion would be to leave it arbitrary so that with it can change over time.
# 	- would be worth discussing in today's meeting! 


P = {('1', '1') : 0.8, ('1', '-1'): 0.01,('-1', '1'):  0.05,('-1', '-1'):  0.2} #indexed (user group, article group).  probability of like | click, user group, article group
PLnC = {('1', '1') : 0.4, ('1', '-1'): 0.005,('-1', '1'):  0.01,('-1', '-1'):  0.1} #indexed (user group, article group).  probability of like | NO click, user group, article group

probclick = {1: 0.5, -1: 0.5} #prob click | group membership


m = 20 #size of unit mass
v = {('1', '1'): 3., ('-1', '1'): 1., ('1', '-1'): 1., ('-1', '-1'): 1. } #utility for liking, known to both user and platform, varies by (article shown, user group) pair
c = {('1', '1'): 1., ('-1', '1'): 1., ('1', '-1'): 1., ('-1', '-1'): 1. } #cost of clicking, known to both user and platform, varies by (article shown, user group) pair
q = {1: 0.8, -1: 0.8} #transition probability across groups at time t+1; homophily constants

epsilon = 0.4 #approximation parameter for approximately equal probability of showing articles |theta - 1/2| <= epsilon

#probshowA = platform_opt.optimize(epsilon, M_a, M, T, P[('1', '1')], P[('-1', '1')], PLA=problike[1], PLB=problike[-1], muA = probclick[1], muB=probclick[-1]) #platform chooses their probability for showing article a by maximizing expected clickthrough rate subject to fairness constraints
probshowA = 0.2

print(probshowA)

old_u = []
time_data_diff = []

for t in range(1,T+1):
	new_u = [] #list of new players that arrive at the timestep
	#unit mass arrives


	if t == 1: #initial mass of users arrives
		for i in range(m):
			g = 2 * np.random.binomial(1, float(M_a / M))- 1
			old_u.append(players.Player(group=g, article=2 * np.random.binomial(1, probshowA)- 1))

	
	else:
		for user in old_u:
			
			#now users are replaced in place (kinda)
			if random.uniform(0,1) <= q[user.group]: #if next person is drawn by homophily
				new_user = players.Player(group=user.group)
				if user.shared == True:
					new_user.article = user.article
				else:
					new_user.article = 2 * np.random.binomial(1, probshowA)- 1 # mechanism to decide which article to share
				
			else:
				new_user = players.Player(group=-1 * user.group)
				# mechanism to decide which article to share.
				new_user.article = 2 * np.random.binomial(1, probshowA)- 1

			new_user.clicked = players.calcclick(P[(str(new_user.group), '1')],P[(str(new_user.group), '-1')],probshowA,v=v[(str(new_user.article), str(new_user.group))], c=c[(str(new_user.article), str(new_user.group))])

			#decide if user shares article, according to P.
			if new_user.group == 1 and new_user.article == 1 and new_user.clicked== 1: #this is wrong, since we want a probability of liking without clicking
				if random.uniform(0,1) <= P[('1', '1')]:
					new_user.shared= True
			elif new_user.group == 1 and new_user.article == -1  and new_user.clicked== 1:
				if random.uniform(0,1) <= P[('1', '-1')]:
					new_user.shared= True
			elif new_user.group == -1 and new_user.article == -1 and new_user.clicked== 1:
				if random.uniform(0,1) <= P[('-1', '1')]:
					new_user.shared= True
			elif new_user.clicked== 1:
				if random.uniform(0,1) <= P[('-1', '-1')]:
					new_user.shared= True
			else:
				new_user.shared = False

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
#plt.axhline(y=epsilon,color='red')
#plt.axhline(y=-1 * epsilon,color='red')
plt.show()
plt.savefig('article_leaning_overtime.png')
