import numpy as np
from matplotlib import pyplot as plt

S = np.load('synthetic_S.npy')
dS = S[:, 1, :] - S[:, 0, :]
dS_thresh = 1e-2

count = np.ones(dS.shape)
count[dS<dS_thresh] = 0

prop_channels = np.sum(count, axis=0)/count.shape[0]
print(prop_channels)

nonconverged_sims = np.where(prop_channels>1e-2)[0]
print(nonconverged_sims)
print(len(nonconverged_sims))


fig, (ax,ax2) = plt.subplots(ncols=2, figsize=(8, 3))
S[S<1e-15] = np.nan
ax.hist(np.log10(S[:, 0].flatten()), range=(-15, 2), bins=100, 
    histtype='step', color='g', label='t = 1 a')
ax.hist(np.log10(S[:, 1].flatten()), range=(-15, 2), bins=100, 
    histtype='step', color='m', label='t = 2 a')
# ax.set_yscale('log')
ax.legend()
ax.spines[['right', 'top']].set_visible(False)
ax.grid(linestyle=':')

dS[dS<1e-15] = np.nan
ax2.hist(np.log10(dS.flatten()), bins=100, range=(-15, 2), color='gray')
ax2.spines[['right', 'top']].set_visible(False)
ax2.grid(linestyle=':')

ax_inset = ax2.inset_axes((0.8, 0.3, 0.65, 0.65))
ax_inset.hist(np.log10(dS.flatten()), bins=100, range=(-15, 2), color='gray')
ax_inset.set_xlim([-2, 2])
ax_inset.set_ylim([0, 4e3])
ax_inset.grid(linestyle=':')

ax.set_xlabel('log $S$ (m$^2$)')
ax2.set_xlabel('log $\Delta S$ (m$^2$)')
ax_inset.set_xlabel('log $\Delta S$ (m$^2$)')

ax.set_ylabel('Number of channel segments')

ax.text(-0.025, 1, 'a', transform=ax.transAxes,
    fontweight='bold', ha='right', va='bottom')

ax2.text(-0.025, 1, 'b', transform=ax2.transAxes,
    fontweight='bold', ha='right', va='bottom')
ax_inset.text(0.025, 0.975, 'c', transform=ax_inset.transAxes,
    fontweight='bold', ha='left', va='top')

fig.subplots_adjust(left=0.125, right=0.825, bottom=0.175, top=0.95)

fig.savefig('channel_convergence.png', dpi=400)