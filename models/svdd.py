# Reference: https://zephyrus1111.tistory.com/471#c3
import numpy as np
try:
    from cvxopt import matrix, solvers
except ModuleNotFoundError:
    import warnings
    warnings.warn("Since the package 'cvxopt' is not installed, it will be installed using [pip install cvxopt].", ImportWarning)
    import os
    os.system("pip install cvxopt")
    from cvxopt import matrix, solvers

 
class SVDD:
    def __init__(self, C=1, s=0.1):
        self.C = C ## regularization parameter
        self.s = s ## parameter for rbf kernel
        self.alpha = None ## alpha
        self.R2 = None ## radius
        self.Q = None ## kernel matrix
        self.X = None
        
    def rbf_kernel(self, x, y, s):
        return np.exp(-np.sum(np.square(x-y))/(s**2))
    
    def make_kernel_matrix(self, X, s):
        n = X.shape[0]
        Q = np.zeros((n,n))
        q_list = []
        for i in range(n):
            for j in range(i, n):
                q_list.append(self.rbf_kernel(X[i, ], X[j, ], s))
        
        Q_idx = np.triu_indices(len(Q))
        Q[Q_idx] = q_list
        Q = Q.T + Q - np.diag(np.diag(Q))
        return Q
    
    def fit(self, X):
        C = self.C
        s = self.s
        Q = self.make_kernel_matrix(X, s)
        n = X.shape[0]
        P = matrix(2*Q)
        q = np.array([self.rbf_kernel(x, x, s) for x in X])
        q = matrix(q.reshape(-1, 1))
        G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
        h = matrix(np.hstack([np.zeros(n), np.ones(n)*C]))
        A = matrix(np.ones((1, n)))
        b = matrix(np.ones(1))
        
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])
        
        rho = 0
        alphas = alphas.flatten()
        S = ((alphas>1e-3)&(alphas<C))
        S_idx = np.where(S)[0]
        
        R2 = alphas.dot(Q.dot(alphas))
        
        for si in S_idx:
            temp_vector = np.array([-2*alphas[i]*self.rbf_kernel(X[i, ], X[si, ], s) for i in range(n)])
            R2 += (self.rbf_kernel(X[si, ], X[si, ], s) + np.sum(temp_vector))/len(S_idx)
            
        self.R2 = R2
        self.alphas = alphas
        self.X = X
        self.Q = Q
        return self
    
    def predict(self, X):
        return np.array([np.sign(self._predict(x)) for x in X])
    
    def _predict(self, x):
        X = self.X
        n = X.shape[0]
        alphas = self.alphas
        R2 = self.R2
        s = self.s
        Q = self.Q
        first_term = self.rbf_kernel(x, x, s)
        second_term = np.sum([2 * alphas[i] * self.rbf_kernel(x, X[i, ], s) for i in range(n)])
        thrid_term = alphas.dot(Q.dot(alphas))
        return R2-first_term+second_term-thrid_term