"""

Implementation of randomized SVD

Halko et al. (2011)
https://doi.org/10.1137/090771806

"""

import time

import numpy as np
import netCDF4 as nc
import cmocean

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib import colors

# Some data
data = np.load('../data/data.npy').T
mean = data.mean(axis=0)
sd = data.std(axis=0)
sd[sd<1e-3] = 1e-3
data = (data-mean)/sd
print('data.shape:', data.shape)

with nc.Dataset('../data/mesh.nc', 'r') as dmesh:
    nodes = dmesh['tri/nodes'][:].data.T
    connect = dmesh['tri/connect'][:].data.T.astype(int)-1

mtri = Triangulation(nodes[:, 0]/1e3, nodes[:, 1]/1e3, connect)

fig, axs = plt.subplots(nrows=2)
ax1,ax2 = axs
ax1.tripcolor(mtri, data[0, :], vmin=-1, vmax=1)
ax2.tripcolor(mtri, data[1, :], vmin=-1, vmax=1)

for ax in axs:
    ax.set_aspect('equal')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 25])

# Do the usual PCA
p = 5
t0 = time.perf_counter()
U,S,VT = np.linalg.svd(data, full_matrices=False)
t1 = time.perf_counter()
print('Exact SVD:', t1-t0)
ind_var = S**2/np.sum(S**2)
sum_var = np.cumsum(ind_var)

K = VT[:p, :]
print('K.shape', K.shape)

fig, axs = plt.subplots(nrows=3, ncols=2)
for i in range(p):
    ax = axs.flat[i]
    ax.tripcolor(mtri, K[i,:]/np.std(K[i,:]), norm=colors.CenteredNorm(vcenter=0), cmap=cmocean.cm.balance)
    ax.set_aspect('equal')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 25])
axs.flat[-1].set_visible(False)

fig.suptitle('Exact PCA')

fig, screeax = plt.subplots()
screeax.plot(sum_var)
screeax.set_yscale('log')
screeax.grid(linestyle=':', linewidth=0.5)
fig.suptitle('Exact PCA')

def randomized_svd(X, p, k=None, q=1):
    if k is None:
        k = p
    
    omega = np.random.normal(size=(X.shape[1], p+k))
    Y = X @ omega
    for i in range(q):
        Y = X @ X.T @ Y

    Q,R = np.linalg.qr(Y, mode='reduced')
    B = Q.T @ X

    # Now do the smaller SVD problem then truncate
    U2,S2,V2 = np.linalg.svd(B, full_matrices=False)
    # print(S2.shape)
    Unew = Q @ U2

    U = Unew[:, :p]
    S = S2[:p]
    V = V2[:p,:]
    svd = (U, S, V)

    # print(S2)
    next_sv = S2[p]

    l = min(*X.shape)
    error_bound = next_sv*(1 + (1 + 4*np.sqrt(2*l/(p-1)))**(1/(2*q-1)))
    return svd, error_bound


# Start doing some random vectors
k = 2
q = 1
t0 = time.perf_counter()
(Unew, S2, V2), eb = randomized_svd(data,p=p, k=None, q=q)
t1 = time.perf_counter()
print('Random SVD:', t1-t0)

Krand = V2[:p, :]
flip_signs = np.sign(np.nanmean(Krand/K, axis=1))
print(flip_signs)
print(Krand.shape)
Krand *= np.vstack(flip_signs)

fig, axs = plt.subplots(nrows=3, ncols=2)
for i in range(p):
    ax = axs.flat[i]
    ax.tripcolor(mtri, Krand[i,:]/np.std(Krand[i,:]), norm=colors.CenteredNorm(vcenter=0), cmap=cmocean.cm.balance)
    ax.set_aspect('equal')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 25])
axs.flat[-1].set_visible(False)

fig.suptitle('Randomized PCA')

fig, ax = plt.subplots()
ax.plot(np.cumsum(S2**2/np.sum(S2**2)))
ax.set_yscale('log')
ax.grid(linestyle=':', linewidth=0.5)
fig.suptitle('Exact PCA')


basis_error = np.linalg.norm(K - Krand, ord='fro')
rel_basis_error = basis_error/np.linalg.norm(K, ord='fro')
print('Relative basis error:', rel_basis_error)

fig, ax = plt.subplots()
ax.scatter(K.flatten(), Krand.flatten())
ax.grid(linestyle=':', linewidth=0.5)

p_vals = np.arange(2, int(data.shape[0]/2))
err_bounds = np.zeros(p_vals.shape)
err_actual = np.zeros(p_vals.shape)
err_exact = np.zeros(p_vals.shape)
data_norm = np.linalg.norm(data, ord='fro')
# data_norm = 1
print(data_norm)
for i in range(len(p_vals)):
    (u,s,v), eb = randomized_svd(data, p_vals[i], k=2)
    # data_norm = 10
    err_bounds[i] = eb/data_norm
    err_actual[i] = np.linalg.norm(data - u@np.diag(s)@v, ord='fro')/data_norm
    err_exact[i] = np.linalg.norm(data - U[:, :p_vals[i]] @ np.diag(S[:p_vals[i]]) @ VT[:p_vals[i]], ord='fro')/data_norm

print(err_actual)
print(err_exact)

fig, ax = plt.subplots()
ax.plot(p_vals, err_bounds, 'k:', label='Error bound', linewidth=2)
ax.plot(p_vals, err_actual, 'k-', label='Randomized (k={}, q={})'.format(k, q), linewidth=1.5)
ax.plot(p_vals, err_exact, 'r--', label='Exact', linewidth=1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of PCs ($p$)')
ax.set_ylabel('Relative error')
ax.legend()
ax.grid(linestyle=':', linewidth=0.5)

plt.show()
