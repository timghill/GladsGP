import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec
import cmocean



Y = np.load('../issm/test/greenland_ff.npy')

nt = 365
nx = int(Y.shape[0]/365)

Y = Y.reshape((nx, nt, -1))
ywinter = Y[:, 100]

ymed = np.median(ywinter, axis=-1)
ymin = np.min(ywinter, axis=-1)

mesh = np.load('../issm/data/geom/IS_mesh.pkl', allow_pickle=True)

mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)
fig,ax = plt.subplots()
pc = ax.tripcolor(mtri, ymed, vmin=-1, vmax=1, cmap=cmocean.cm.curl)
ax.tricontour(mtri, ymed, levels=[0], colors='black')
cbar = fig.colorbar(pc, location='top', label='Min winter flotation fraction')
ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.set_xlim([-236, -200])
ax.set_ylim([-2510, -2485])
fig.savefig('figures/median_winter_flot_map.png', dpi=400)


fig = plt.figure(figsize=(6,6))
gs = GridSpec(4, 1, height_ratios=(10, 10, 10, 100),
    hspace=0.05, bottom=0.05, top=0.9, right=0.95, left=0.05)
ax = fig.add_subplot(gs[-1,0])
cax1 = fig.add_subplot(gs[0,0])
cax2 = fig.add_subplot(gs[1,0])
add_pos = 2
add_neg = 0
# nticks_pos = logmax - logabsmin
# nticks_neg = logmin - logabsmin

ylog_pos = np.nan*np.zeros(ymed.shape)
ylog_pos[ymed>0] = np.log10(ymed[ymed>0])

ylog_neg = np.nan*np.zeros(ymed.shape)
ylog_neg[ymed<0] = np.log10(-ymed[ymed<0])

# cticks = np.arange(-add_pos, 0.1)
# cticklabels = [r'10$^{{{}}}$'.format(int(x)) for x in cticks]

pc = ax.tripcolor(mtri, 10**ylog_pos, cmap='Reds', vmin=0, vmax=1)
cbar_pos = fig.colorbar(pc, cax=cax1, label=r'Median winter flotation fraction $f_{\rm{w}}>0$',
    orientation='horizontal')
# cax1.set_xticks(cticks, cticklabels)
cax1.xaxis.tick_top()
cax1.xaxis.set_label_position('top')


cticks = np.arange(-1, 2)
cticklabels = [r'10$^{{{}}}$'.format(x) for x in cticks]
print(cticklabels)
pc = ax.tripcolor(mtri, ylog_neg, cmap='Blues', vmin=-1, vmax=1)
cbar_neg = fig.colorbar(pc, cax=cax2, label=r'Median winter flotation fraction $f_{\rm{w}}<0$',
    orientation='horizontal')
cax2.set_xticks(cticks, cticklabels)

# ax.tripcolor(mtri, ylog_neg, cmap=cmocean.cm.rain)
ax.tricontour(mtri, ymin, levels=[0], colors='black', linestyles='solid')
ax.tricontour(mtri, ymed, levels=[0], colors='gray', linestyles='solid')
ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.set_xlim([-234, -200])
ax.set_ylim([-2510, -2485])

# mock_cmap = np.zeros((256, 4))
# mock_cmap[:128] = 
fig.savefig('figures/median_winter_flot_map_divlog.png', dpi=400)


fig,ax = plt.subplots()
ax.hist(ymed, bins=50)
ax.set_yscale('log')
ax.set_ylim([1, 5000])
ax.set_xlabel('Min winter flotation fraction')
ax.set_ylabel('Count')
fig.savefig('figures/median_winter_flot_hist.png', dpi=400)


print('Min flot frac:', np.min(ymed))

ymed_element = np.mean(ymed[mesh['elements']-1], axis=1)
print('ymed_element:', ymed_element.shape)

ymin_element = np.mean(ymin[mesh['elements']-1], axis=1)
print('ymin_element:', ymin_element.shape)

A_combined = mesh['area'].copy()
A_combined[ymed_element<0] = np.nan
# A_combined[ymin_element<-1./3.] = np.nan

A_neg = np.nansum(A_combined)
A_tot = np.nansum(mesh['area'])

A_removed = A_tot - A_neg
A_removed_prop = A_removed/A_tot

N_neg = len(A_combined[np.isnan(A_combined)])
N_neg_prop = N_neg / len(A_combined)
print(f'Removed nodes: {N_neg}  ({N_neg_prop:.3%})')
print(f'Removed area: {A_removed:.3e}  ({A_removed_prop:.3%})')

# plt.show()
