import numpy as np 
import random
import pickle as pkl

#local files being imported
import platform_opt
from players import *
from statistics import variance


#runs the model given a strategy theta for showing articles at time step 1.
def runModel(theta, T, pi, M, P, beta_dist, v,c,q): 
    
    '''
    parameters:
    theta      :(dict) indexed by user group g; the proprtion of users in group g shown article A (1)
    T          :(int) max total number of timesteps
    pi         :(dict) indexed by user group; proportion of users in each group.  pi[1] = 1 - pi[-1]
    M          :(int) number of players in the mass
    P          :(dict) expected value for probably liking given a click given (article group, user group)
    beta_dist  :(dict) (alpha, beta) parameters for the beta distribution for above distribution.
    v          :(dict) value for liking an article. indexed (article group, user group)
    c          :(dict) cost for clicking on an article. indexed (article group, user group)
    q          :(dict) homophily variable.  probability of intra-group propogation at next timestep indexed by group
    
    returns:
    the number of clicked articles in the simulation.
    '''
    
    
    #tracking data; this could be way more efficient :face-palm:
    old_u = []
    time_data_diff = []
    num_players_in_model = [M]
    group_1_players = []
    showns_dict = []
    clicks_dict = []
    shares_dict = []
    
    #prob_article_A = []
    #prob_article_A_cum = []
    tot_shown_A = 0
    tot_in_model = 0
    t = 1
    pi_a = pi[-1]
    
    #dictionaries to keep track of who is shown which articles and who clicks on which articles
    #indexed (article, user)

    while (t <= T) and (t == 1 or len(old_u) > 0):
        num_shown_A = 0 #number of players at this timestep that are shown article A
        new_u = []  # list of new players that arrive at the timestep
        
        shown_dict = {(1,1):   0,
              (-1,1):  0, 
              (1,-1):  0,
              (-1,-1): 0}

        click_dict = {(1,1):   0,
              (-1,1):  0, 
              (1,-1):  0,
              (-1,-1): 0}

        share_dict = {(1,1):   0,
              (-1,1):  0, 
              (1,-1):  0,
              (-1,-1): 0}

        if t == 1:  # initial mass of users arrives
            for i in range(M): # iterating over the size of the unit mass
                tot_in_model = tot_in_model + 1
                g = coin_toss(pi[-1]) # determine players group according to the true group distribution
                a = coin_toss(theta[g]) # show article A according to the platform's policy.  (right now, this is just a placeholder)
                player = Player(group=g, article=a)
                shown_dict[(a,g)] = shown_dict[(a,g)] + 1
                if a == -1:
                    num_shown_A = num_shown_A + 1
                    player.article = -1
                else:
                    player.article = 1

                P_personal = P
                P_personal[(a,g)] = np.random.beta(*beta_dist[(a,g)])
                P_personal[(-a,g)] = np.random.beta(*beta_dist[(-a,g)])

                player.clicked = calcclickdict(player, 1, 
                                                  P_personal, 
                                                  q, 
                                                  theta,
                                                  c,
                                                  v)
                if player.clicked:  
                    click_dict[(a,g)] = click_dict[(a,g)] + 1 
                    if random.uniform(0, 1) <= P[(player.article, player.group)]:
                        player.shared = True
                        share_dict[(a,g)] = share_dict[(a,g)] + 1
                old_u.append(player)
            #print("First time step: " + str(num_shown_A) + " users shown article A") #debugging statement

        else:
            for user in old_u:

                if user.shared == 1: # new user only added to the system if the previous user shared the article
                    tot_in_model = tot_in_model + 1
                    if random.uniform(0, 1) <= q[user.group]:  # if next person is drawn by homophily
                        new_user = Player(group=user.group)
                    else:
                        new_user = Player(group=-user.group)
                        
                    # show the previous person's article, regardless of the new user's group    
                    new_user.article = user.article
                    shown_dict[(new_user.article, new_user.group)] = shown_dict[(new_user.article, new_user.group)] + 1
                    if new_user.article == -1:
                        num_shown_A = num_shown_A + 1


                    P_personal = P
                    P_personal[(a,g)] = np.random.beta(*beta_dist[(new_user.article ,new_user.group)])
                    P_personal[(-a,g)] = np.random.beta(*beta_dist[(-a,g)])
                    new_user.clicked = calcclickdict(new_user, 1, 
                                                  P_personal, 
                                                  q, 
                                                  theta,
                                                  c,
                                                  v)
                    # decide if user shares article, according to P.
                    if new_user.clicked == 1:  
                        click_dict[(new_user.article, new_user.group)] = click_dict[(new_user.article, new_user.group)] + 1
                        if random.uniform(0, 1) <= P[(new_user.article, new_user.group)]:
                            new_user.shared = True
                            share_dict[(new_user.article, new_user.group)] = share_dict[(new_user.article, new_user.group)] + 1
                    else:
                        new_user.shared = False

                    #add user to list
                    new_u.append(new_user)
                else: #only add a user to the next round if the previous user shared the article 
                    pass
            
            #tracks how many players are being shown articles at all timesteps
            num_players_in_model.append(len(new_u)) 
            showns_dict.append(shown_dict)
            clicks_dict.append(click_dict)
            shares_dict.append(share_dict)
            l = 0
            for u in new_u:
                if u.group==1:
                    l += 1 
            group_1_players.append(l)
            old_u = new_u


        t = t + 1
        tot_shown_A = tot_shown_A + num_shown_A
    
    return num_players_in_model, showns_dict, clicks_dict, shares_dict, group_1_players

def saveRuns(lst,filename):    
    tommy_pickles = open(filename, "wb") # remember to open the file in binary mode 
    pkl.dump(lst, tommy_pickles)
    
    tommy_pickles.close()
    
    
def get_params(dataset_name):
    if dataset_name == 'twitter':
        # SIMULATION PARAMS DEPENDING ON DATASET
        # parameters here come from probability_sharing_distributions.ipynb
        pi = {1: 0.43294, 
             -1: 0.56706}          # number of members in groups a and b #estimated from probability_sharing_distributions.ipynb
        pi_a = pi[-1]

        # (alpha, beta) values for the beta distribution as a function of article and user groups.
        # beta_dist indexed (article group, user group). 
        beta_dist = {(-1,-1) : (41.45784070052453, 556.8653671739492),
                    (1,-1) : (0.7519296311195025, 413.4664888783973),
                    (-1,1) : (6.096475779403813, 1519.8459882514462),
                    (1,1): (2152.960173995409, 23647.671142918956)}

        # probability of like | click, user group, article group
        # P indexed (article group, user group). expected values of above beta distribution
        #estimated from probability_sharing_distributions.ipynb
        P = {( -1,  -1):  0.0692, 
             ( 1, -1):  0.001815,
             (-1,  1):  0.003995,
             (1, 1):  0.08344} 

        # player utility for liking, known to both user and platform,
        # v indexed by (article group, user group) pair
        #unclear what these values _should_ be!
        v = {( 1,  1):   2000.,
             (-1,  1):   500.,
             ( 1, -1):   500.,
             (-1, -1):   2000. }

        # TODO: DOUBLE CHECK INDEXING IS CORRECT
        # cost of clicking, known to both user and platform,
        # c indexed by (article shown, user group)
        c = {( 1,  1):   1.,
             (-1,  1):   1.,
             ( 1, -1):   1.,
             (-1, -1):   1. }

        # transition probability across groups at time t + 1 
        # indexed by the first user's group membership
        # seems too high to be practical
        q = {-1:  0.9877, 
             1: 1.}

        
    elif dataset_name=='facebook':
        # SIMULATION PARAMS DEPENDING ON DATASET
        # A = -1
        # parameters here come from Replication Exposure.ipynb
        pi = {1: 0.5, 
             -1: 0.5}          # number of members in groups a and b #estimated from probability_sharing_distributions.ipynb
        pi_a = pi[-1]

        # (alpha, beta) values for the beta distribution as a function of article and user groups.
        # beta_dist indexed (article group, user group). 
        beta_dist = {(1,1) : (0.9541492709534125, 1.345006644515015),
                    (-1,1) : (0.1822515775580026, 2.7574965182522644),
                    (1,-1) : (0.09576097403924465, 3.09136619146736),
                    (-1,-1): (0.8828729918440646, 1.6247070146941363)}

        # probability of like | click, user group, article group
        # P indexed (article group, user group). expected values of above beta distribution
        #estimated from probability_sharing_distributions.ipynb
        P = {}
        for a in [-1,1]:
            for g in [-1,1]:
                P[(a,g)] = beta_dist[(a,g)][0] / (beta_dist[(a,g)][0] +  beta_dist[(a,g)][1])


        # player utility for liking, known to both user and platform,
        # v indexed by (article group, user group) pair
        #unclear what these values _should_ be!
        v = {( 1,  1):   1000.,
             (-1,  1):   100.,
             ( 1, -1):   100.,
             (-1, -1):   1000. }

        # cost of clicking, known to both user and platform,
        # c indexed by (article shown, user group)
        c = {( 1,  1):   1.,
             (-1,  1):   1.,
             ( 1, -1):   1.,
             (-1, -1):   1. }

        # transition probability across groups at time t + 1 
        # indexed by the first user's group membership
        q = {-1: 0.7192525700416023, 
             1: 0.6797565445480234}

        # approximation parameter for approximately equal probability
        epsilon = 0.05  

        #theta_hat and theta_tilde learned with T = 15
        theta_hat = {1: 1.0, -1: 0.0}
        theta_tilde = {1: 0.9999999999599892, -1: 4.3500351052646725e-11}

        
    return pi, beta_dist,P,v,c,q

def calc_errorbars(lst, n_std=1):
    variance_list = []
    Lst = np.transpose(np.array(lst))
    for x in Lst:
        variance_list.append(n_std * np.sqrt(variance(x)))

    #print(variance_list)
    return variance_list

def loadRuns(filename):
    infile = open(filename,'rb')
    new_dict = pkl.load(infile)
    infile.close()
    return new_dict
