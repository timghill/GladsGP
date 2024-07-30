"""

Implementation of randomized fixed-rank SVD

Halko et al. (2011)
https://doi.org/10.1137/090771806

"""

import numpy as np

def randomized_svd(X, p, k=None, q=1, return_error=False):
    """
    Parameters
    ----------
    X : (m, n) array-like
        Data to compute SVD
    
    p : int
        Fixed rank for truncated SVD

    k : int
        Oversampling rank
    
    q : int
        Number of power iterations

    return_error : bool
                   Return an estimate of svd error
    
    Returns
    -------
    svd : (U, S, Vh) truncated singular value arrays
          U : (m, p)
          S : (p,)
          Vh: (p, n)
    
    error : float
            Error upper bound
    """
    # Default: compute 2*k rank SVD then truncate
    if k is None:
        k = p
    
    # Random matrix with oversampling
    omega = np.random.normal(size=(X.shape[1], p+k)).astype(np.float32)
    Y = X @ omega

    # Power iterations to reduce error
    for i in range(q):
        Y = X @ X.T @ Y

    # Orthgonal matrix decomposition on the smaller matrix Y
    Q,R = np.linalg.qr(Y, mode='reduced')
    B = Q.T @ X

    # Now do the smaller SVD problem and truncate
    U,S,V = np.linalg.svd(B, full_matrices=False)
    U = Q @ U

    U = U[:, :p]
    S = S[:p]
    V = V[:p,:]
    svd = (U, S, V)

    if return_error:
        # Compute error bound
        try:
            next_sv = S[p]
        except IndexError:
            next_sv = 0
        l = min(*X.shape)
        error_bound = next_sv*(1 + (1 + 4*np.sqrt(2*l/(p-1)))**(1/(2*q-1)))
        return svd, error_bound
    
    else:
        return svd
