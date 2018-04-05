"""
python 3.5

A set of classes and functions to perform inference in HMM models,
train with EM and return most likely sequences

Adrien Le Franc
"""

import numpy as np
import time


def min_max_scaling(X):
    """
    returns data scaled in [0, 1]
    """
    
    n, d = X.shape
    
    for i in range(d):
        
        mi = min(X[:, i])
        ma = max(X[:, i])
        
        X[:, i] = (X[:, i] - mi) / (ma - mi) * 10
    
    return X



def gaussian_multivariate(x, mu, sigma):
    """
    compute gaussian multivariate probability of x
    """
    
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
    
    x = x.reshape((-1, 1))
    
    return np.exp(-0.5 * (x-mu).T.dot(inv).dot(x-mu)) / (2*np.pi * np.sqrt(det))

def log_gaussian_multivariate(x, mu, sigma):
    """
    compute log of gaussian multivariate probability of x
    """

    det = np.linalg.det(sigma)

    if det == 0:
        print('singular matrix')
        sigma += np.eye(len(mu)) * 1e-3 ## ?
        det = np.linalg.det(sigma)

    inv = np.linalg.inv(sigma)
    x = x.reshape((-1, 1))

    return float(-0.5 * (x-mu).T.dot(inv).dot(x-mu) - np.log(2*np.pi * np.sqrt(abs(det))))

    
class HMM_inference:
    """
    provides functions to compute p(u) and p(qt|u) where u is an observed data
    sequence and qt is the state at time t. The forward-backward algorithm as 
    well as p(u) deal with log-scaled values so as to avoid underflow
    """
    
    def __init__(self, observed_data, log_pi, log_A, mu, sigma):
        
        self.K = len(log_pi)
        self.T = len(observed_data)
        
        self.u = observed_data
        
        self.log_pi = log_pi
        self.log_A = log_A
        self.mu = mu
        self.sigma = sigma
        
        alpha, beta = self.alpha_beta(observed_data)
        
        self.alpha = alpha
        self.beta = beta
        
    def alpha_beta(self, u):
        """
        log-scaled implementation of forward-backward algorithm
        returns log(alpha), log(beta)
        """
        
        alpha = np.zeros((self.T, self.K))
        beta = np.zeros((self.T, self.K))
        
        for k in range(self.K):
            
            alpha[0, k] = log_gaussian_multivariate(u[0], self.mu[k], self.sigma[k]) + self.log_pi[k]
            beta[-1] = 0.0
        
        for t in range(1, self.T):
            
            for k in range(self.K):
                
                a = np.zeros((self.K, 1))
                b = np.zeros((self.K, 1))
                
                for q in range(self.K):
                    
                    a[q] = alpha[t-1, q] + self.log_A[q, k]
                    b[q] = beta[self.T-t, q] + self.log_A[k, q] + log_gaussian_multivariate(u[self.T-t], 
                        self.mu[q], self.sigma[q])
                
                a_max = float(max(a))
                b_max = float(max(b))
                
                alpha[t, k] = a_max + np.log(np.sum(np.exp(a - a_max))) + log_gaussian_multivariate(u[t], 
                    self.mu[k], self.sigma[k])
                beta[self.T-t-1, k] = b_max + np.log(np.sum(np.exp(b - b_max)))
        
        return (alpha, beta)
    
    def log_likelihood(self):
        """
        compute log(p(u)) 
        """
        
        a = self.alpha[self.T-1]
        a_max = float(max(a))
        
        return a_max + np.log(np.sum(np.exp(a - a_max))) 
    
    def gamma(self):
        """
        compute sequence log(gamma) = log p(qt|u) for all t.
        """
        
        lp_u = self.log_likelihood()
    
        return self.alpha + self.beta - lp_u
    
    def xi(self):
        """
        compute sequence log(xi) = log p(qt, qt+1|u) for all t. 
        """
        
        log_xi = np.zeros((self.K, self.K, self.T-1))
        
        for t in range(self.T-1):
            
            for i in range(self.K):
                
                for j in range(self.K):
                    
                    log_xi[i, j, t] = (self.alpha[t, i] + self.beta[t+1, j] + self.log_A[i, j] + 
                        log_gaussian_multivariate(self.u[t+1], self.mu[j], self.sigma[j]))
        
        lp_u = self.log_likelihood()
        
        return log_xi - lp_u


class HMM_EM:
    """
    EM algorithm to learn the parameters pi, A, mu and sigma
    of the HMM with an observed data sequence
    """
    
    def __init__(self, observed_data, log_pi, log_A, mu, sigma):
        
        self.K = len(log_pi)
        self.T = len(observed_data)
        self.dim = len(mu[0])
        
        self.u = observed_data
        
        self.log_pi = log_pi
        self.log_A = log_A
        self.mu = mu
        self.sigma = sigma
        
    def E_step(self):
        
        Inference = HMM_inference(self.u, self.log_pi, self.log_A, self.mu, self.sigma)
        
        log_gamma = Inference.gamma()
        log_xi = Inference.xi()
        log_p = Inference.log_likelihood()
        
        #sum_xi = np.sum(xi, axis=2)
        
        #print('gamma min {} max {}'.format(np.min(log_gamma), np.max(log_gamma)))
        #print('log_xi min {} max {}'.format(np.min(log_xi), np.max(log_xi)))
        #print('logp {}'.format(log_p))

        return (log_gamma, log_xi, log_p)
    
    def M_step(self, log_gamma, log_xi):

        #sum_gamma = np.sum(gamma, axis=0).reshape((-1, 1)) 
        
        self.log_pi = log_gamma[0] 


        #self.A = sum_xi / sum_gamma.dot(np.ones((1, self.K))) 

        #print(np.min(self.pi), np.min(self.A))
        
        new_mu = np.zeros((self.K, self.dim))
        new_sigma = np.zeros((self.K, self.dim, self.dim))
        new_log_A = np.zeros((self.K, self.K))
        log_sum_gamma = np.zeros((self.K, 1))
        
        for k in range(self.K):
            for p in range(self.K):

                log_xi_max = float(max(log_xi[k, p, :]))
                log_sum_xi = log_xi_max + np.log(np.sum(np.exp(log_xi[k, p, :] - log_xi_max)))
                new_log_A[k, p] = log_sum_xi

                if p == 0:

                    log_gamma_max = float(max(log_gamma[:, k]))
                    log_sum_gamma[k] = log_gamma_max + np.log(np.sum(np.exp(log_gamma[:, k] - log_gamma_max)))

                    sum_u = np.sum(np.exp(log_gamma[:, k] - log_gamma_max).reshape((-1, 1)).dot(np.ones((1, self.dim))) * self.u, axis=0)
                    signe = (sum_u > 0)
                    new_mu[k, :] = np.exp(log_gamma_max + np.log(abs(sum_u)) - log_sum_gamma[k]) * (signe - (1-signe))

                    print(new_mu)
                    '''

                    print(log_gamma_max + np.log(np.sum(np.exp(log_gamma[:, k] - log_gamma_max).reshape((-1, 1)).dot(np.ones((1, self.dim))) * self.u, axis=0)) - log_sum_gamma[k])

                    log_mu[k, :] = log_gamma_max + np.log(np.sum(np.exp(log_gamma[:, k] - log_gamma_max).reshape((-1, 1)).dot(
                        np.ones((1, self.dim))) * self.u, axis=0)) - log_sum_gamma[k]
                    '''

        
        self.log_A = new_log_A - log_sum_gamma.dot(np.ones((1, self.K)))
        #new_mu = np.exp(log_mu) * (signe - (1-signe))

        #print(new_mu)

        for k in range(self.K):

            log_gamma_max = float(max(log_gamma[:, k]))
            cov = np.zeros((self.dim, self.dim))

            for t in range(self.T):

                x = (self.u[t] - new_mu[k, :]).reshape((-1, 1))
                cov += np.exp(log_gamma[t, k] - log_gamma_max) * x.dot(x.T)

            signe = (cov > 0)
            new_sigma[k, :, :] = np.exp(log_gamma_max + np.log(abs(cov)) - log_sum_gamma[k]) * (signe - (1-signe))

        #new_sigma = np.exp(log_sigma) * (signe - (1-signe))
        
        self.mu = {i:new_mu[i, :].reshape((-1, 1)) for i in range(self.K)}
        self.sigma = {i:new_sigma[i, :, :] for i in range(self.K)}

        print('pi min {} max {}'.format(np.min(self.log_pi), np.max(self.log_pi)))
        print('A min {} max {}'.format(np.min(self.log_A), np.max(self.log_A)))

    
    def run(self, steps=100, eps=1e-5):
        """
        launch training. Saves values of the log-likelihood.
        """
        
        lp = np.zeros((steps, 1))
        
        for s in range(steps):
            
            print('it #{}, E step'.format(s))
            log_gamma, log_xi, log_p = self.E_step()

            print('it #{}, M step'.format(s))
            self.M_step(log_gamma, log_xi)
            
            lp[s] = log_p
            print('logp step {}: {}'.format(s, log_p))
        
            if s == 0:
                continue
                
            if lp[s] < lp[s-1] - eps:
                print("log-likelihood goes down at step {}".format(s))
                self.log_likelihood = lp[0: s+1]
                self.log_gamma = log_gamma
                break
                    
            if lp[s] < lp[s-1] + eps:
                print("converged in {} steps".format(s))
                self.log_likelihood = lp[0: s+1]
                self.log_gamma = log_gamma
                break


def viterbi(log_pi, log_A, observed_data, mu, sigma):
    """
    compute and return most likely sequence of states according to Viterbi algorithm
    """
    
    K = len(log_pi)
    T = len(observed_data)
    
    T_1 = np.zeros((T, K))
    T_2 = np.zeros((T, K))
    q = np.zeros((T, 1))
    
    for j in range(K):
        
        T_1[0, j] = log_pi[j] + log_gaussian_multivariate(observed_data[0], mu[j], sigma[j])
        T_2[0, j] = 0
    
    for t in range(T):
        
        for j in range(K):
            
            lp_ut_given_qt = log_gaussian_multivariate(observed_data[t], mu[j], sigma[j])
            p = np.array([T_1[t-1, k] + log_A[k, j] + lp_ut_given_qt for k in range(K)])
            
            k = np.argmax(p)
            T_1[t, j] = p[k]
            T_2[t, j] = k
    
    p = np.array([T_1[T-1, k] for k in range(K)])
    q[T-1] = np.argmax(p)
    
    for t in range(1, T):
         
            s = T-t
            q[s-1] = T_2[s, int(q[s])] 
    
    return q