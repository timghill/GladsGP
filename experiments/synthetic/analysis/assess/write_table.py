import numpy as np

npc = np.arange(1, 11+1)
nsim = np.array([512, 256, 128, 64, 32, 16])

# npc = np.arange(2, 5+1)
# nsim = np.array([64, 32, 16])

xx,yy = np.meshgrid(npc, nsim)

npc = xx.flatten()
nsim = yy.flatten()

with open('table.dat', 'w') as table:
    for i in range(len(npc)):
        line = '{id} python -u compute_test_preds.py ../train_config.py ../test_config.py --npc {pc} --nsim {num} -t'.format(id=i+1, pc=npc[i], num=nsim[i])
        table.write(line + '\n')

