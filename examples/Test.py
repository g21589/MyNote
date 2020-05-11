# -*- coding: utf-8 -*-

import timeit
import matplotlib
matplotlib.use('Agg')
matplotlib.rcsetup.interactive_bk
matplotlib.rcsetup.non_interactive_bk
matplotlib.rcsetup.all_backends

import matplotlib.pyplot as plt

N = 500

stats = [
    {'med': 5, 'q1': 2, 'q3': 6, 'whislo': 1, 'whishi': 8} for i in range(N)
]

_, ax = plt.subplots(1,1, figsize=(20, 5), dpi=150)
plt.autoscale(False)
#ax.use_sticky_edges = False

st = timeit.default_timer()
ax.bxp(stats, showbox=True, showfliers=False)
print("Time: {:.4f} sec".format(timeit.default_timer() - st))

plt.xlim(0, N+1)
plt.ylim(0, 10)

plt.savefig(
    'test.png', 
#    bbox_inches='tight'
)

plt.cla()
