import numpy as np

import multiprocessing
from functools import partial
from .utils import count_word_ocurrences, count_subword_ocurrences

class EMSolver:
    
    @classmethod
    def __initialize_params(cls):
        
        # Initialize initial distribution pi
        pi = np.random.uniform(
            0, 1,
            size=(cls.n_hidden_states,)
        )
        pi = pi/np.sum(pi)
        cls.pi=pi
        
        # Initialize hidden state transition probability matrix P
        P = np.random.uniform(
            0,1,
            size=(cls.n_hidden_states,cls.n_hidden_states)
        )
        for i in range(P.shape[0]):
            P[i,:]=P[i,:]/np.sum(P[i,:])
        cls.P=P
            
        # Initialize hidden state to observation probability matrix Q
        Q = np.random.uniform(
            0,1,
            size=(cls.n_hidden_states,cls.observed_vocabulary_size)
        )
        for i in range(Q.shape[0]):
            Q[i,:]=Q[i,:]/np.sum(Q[i,:])
        cls.Q=Q
        
    @classmethod
    def __get_alpha(cls):
        
        # First alpha
        alpha_0 = [
            cls.pi[i]*cls.Q[i,cls.y[0]] for i in cls.hidden_states
        ]
        alpha = [alpha_0]
        
        # Recursively calculate alphas in time
        for n in range(1,cls.N):
            alpha_n=[]
            for j in cls.hidden_states:
                alpha_n.append(
                    np.sum([
                        alpha[n-1][i]*cls.P[i,j] for i in cls.hidden_states
                    ])*cls.Q[j,cls.y[n]]
                )
            alpha.append(alpha_n)
            
        return np.array([np.array(alpha_i) for alpha_i in alpha])
    
    @classmethod
    def __get_beta(cls):
        
        # Initialize beta vector and get beta for last time step
        beta = np.empty(shape=(cls.N, cls.n_hidden_states))
        beta[cls.N-1,:] = np.ones(shape=(cls.n_hidden_states,))
        
        # Moving backwards in time, recursively calculate betas
        for n in range(cls.N-2, -1, -1):
            beta[n,:] = np.array([
                np.sum([
                    cls.P[i,j]*cls.Q[j,cls.y[n+1]]*beta[n+1,j] for j in cls.hidden_states
                ]) for i in cls.hidden_states
            ])
        return beta
    
    @classmethod
    def __get_gamma(
        cls,
        alpha,
        beta,
    ):
        
        gamma = np.array([
            np.array([
                alpha[n,i]*beta[n,i]/np.dot(alpha[n,:],beta[n,:]) for i in cls.hidden_states
            ]) for n in range(cls.N)
        ])
        
        return gamma
    
    @classmethod
    def __get_epsilon(
        cls,
        alpha,
        beta
    ):
        # Initialize array to keep eps(i,j)
        epsilon = np.empty(
            shape=(
                cls.N-1,
                cls.n_hidden_states,
                cls.n_hidden_states
            )
        )
        
        # Calculate epsilons
        for n in range(cls.N-1):
            for i, j in np.ndindex(
                (
                    cls.n_hidden_states,
                    cls.n_hidden_states
                )
            ):
                epsilon[n,i,j] = alpha[n,i]*cls.P[i,j]*cls.Q[j,cls.y[n+1]]*beta[n+1,j]/np.sum([
                    np.sum([
                        alpha[n,i]*cls.P[i,j]*cls.Q[j,cls.y[n+1]]*beta[n+1,j] for j in cls.hidden_states
                    ]) for i in cls.hidden_states
                ])
        
        return epsilon
    
    @classmethod
    def __update_model_parameters(
        cls,
        gamma,
        epsilon
    ):
        # Update pi using gamma
        pi = gamma[0,:]
        
        # Update P using epsilon and gamma
        P = np.empty(
            shape=(
                cls.n_hidden_states,
                cls.n_hidden_states
            )
        )
        for i, j in np.ndindex(
            (
                cls.n_hidden_states,
                cls.n_hidden_states
            )
        ):
            P[i,j] = np.sum(epsilon[:,i,j])/np.sum(gamma[:-1,i])
            
        # Update Q using gamma
        Q=np.empty(
            shape=(
                cls.n_hidden_states,
                cls.observed_vocabulary_size
            )
        )
        for i, j in np.ndindex(
            (
                cls.n_hidden_states,
                cls.observed_vocabulary_size
            )
        ):
            Q[i,j] = np.sum([
                gamma[n, i] for n in np.where(cls.y==j)[0]
            ])/np.sum(gamma[:,i])
                
        return pi, P, Q
    
    @classmethod
    def __get_estimated_hidden_states(cls):
        
        # Initialize delta and phi
        delta = np.empty(shape=(cls.N, cls.n_hidden_states))
        phi = np.empty(shape=(cls.N, cls.n_hidden_states))
        delta[0,:]=np.array([
            cls.pi[i]*cls.Q[i,cls.y[0]] for i in cls.hidden_states
        ])
        phi[0,:]=np.zeros(shape=(cls.n_hidden_states,))
        
        for n in range(1,cls.N):
            for j in cls.hidden_states:
                delta[n,j]=np.max([
                    delta[n-1,i]*cls.P[i,j]*cls.Q[j,cls.y[n]] for i in cls.hidden_states
                ])
                phi[n,j] = np.argmax([
                    delta[n-1,i]*cls.P[i,j]*cls.Q[j,cls.y[n]] for i in cls.hidden_states
                ])
        
        best_P, best_XN = np.max(delta[-1,:]), np.argmax(delta[-1,:])
        X = np.empty(dtype=int, shape=(cls.N,))
        X[-1] = int(best_XN)
        for n in range(cls.N-2, -1, -1):
            X[n]=int(phi[n+1, X[n+1]])
            
        return X, best_P
        
        
    
    
    @classmethod
    def fit(
        cls,
        hmm,
        y,
        thresh
    ):
        cls.y=np.array(y)
        cls.N=len(y)
        cls.n_hidden_states=hmm.n_hidden_states
        cls.hidden_states = hmm.hidden_states
        cls.thresh=thresh
        
        if hmm.observed_vocabulary is None:
            cls.observed_vocabulary=np.unique(y)
            hmm.observed_vocabulary=cls.observed_vocabulary
        else:
            cls.observed_vocabulary=hmm.observed_vocabulary
            
        cls.observed_vocabulary_size=len(cls.observed_vocabulary)
        hmm.observed_vocabulary_size=cls.observed_vocabulary_size
        
        cls.__initialize_params()
        
        P_model=[]
        last_P=0
        current_P=0
        finish_algorithm=False
        i=0
        while not finish_algorithm:
            alpha = cls.__get_alpha()
            # import pdb;pdb.set_trace()
            beta = cls.__get_beta()
            # import pdb;pdb.set_trace()
            gamma = cls.__get_gamma(
                alpha=alpha,
                beta=beta
            )
            epsilon = cls.__get_epsilon(
                alpha=alpha,
                beta=beta
            )
            X_est, current_P = cls.__get_estimated_hidden_states()
            P_model.append(current_P)
            i+=1
            if last_P==0:
                finish_algorithm=False
            else:
                if (current_P-last_P)/last_P > cls.thresh:
                    finish_algorithm=False
                else:
                    finish_algorithm=True
                    continue
            
            last_P = current_P
            
            cls.pi, cls.P, cls.Q = cls.__update_model_parameters(
                gamma=gamma,
                epsilon=epsilon
            )
            
        hmm.alpha=alpha
        hmm.beta=beta
        hmm.gamma=gamma
        hmm.best_model_prob = current_P
        hmm.prob_evolution=P_model
        hmm.pi = cls.pi
        hmm.P = cls.P
        hmm.Q = cls.Q
        hmm.X = X_est
        hmm.nit = i
        hmm.y=y
        
        
        
        
