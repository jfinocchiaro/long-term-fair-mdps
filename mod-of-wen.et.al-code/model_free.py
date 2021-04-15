import numpy as np
import mdptoolbox.example
import mdp

#Algorithm 2 from Wen wet al. 2019 "Fairness with Dynamics"
def modelfree(M, r=500, n=30, n_prime=10, m=15, alpha=0.5, sigma=0.5,T=5):
    #r = iterations
    #n = parameter samples
    #nprime = top n' samples considered, <= n
    #m = rollout samples
    #alpha = smoothing parameter, in (0,1)
    #sigma = tolerance

    eta_hat = 0. #line 2
    for k in range(r): #line 3
        samples_theta_unbounded = np.random.exponential((1./m) * eta_hat, size=n) #line 4: sample n samples from f_{m^-1}(eta_hat) #TODO: double check this is correct
        samples_theta = np.minimum(np.maximum(samples_theta_unbounded, 0), 1)
        print samples_theta

        for i in range(n): #line 5
            #s[t] and a[t] are determined by a sample of the MDP governed by policy pi_theta.  it's unclear what pi_theta is to me.
            s,a = M.genTsteps(samples_theta[0], T=T)
            R_hat_samples = [np.sum([M.gamma ** t * M.R[s[t]][a[t]] for t in range(T)]) for x in range(m)]
            R_hat_theta_i = np.average(R_hat_samples) #line 6: update R_hat

            rho_maj =1#TODO
            rho_min =1#TODO
            eps_hat_samples = np.abs(np.add(rho_maj, -1* rho_min) )
            eps_hat_theta_i = np.average(eps_hat_samples)#line 7: update eps_hat
        #line 9: sort theta_i in increasing eps_hat
        #line 10: i_prime is the largest i so that the tolerance is attained
        if n_prime <= i_prime: #line 11
            pass#line 12: sort to i_prime theta^i's in decreasing R_hat
        #line 14: update eta_hat
    return np.random.exponential(eta_hat) #line 16


if __name__ == '__main__':
  M = mdp.mdp()
  modelfree(M)
