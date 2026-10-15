import networkx as nx
from egonets_wl.kernels.wl_subtree import WLSubtreeKernel
from egonets_wl.kernels.rarity_weighted_wl import RarityWeightedWLKernel

g1 = nx.path_graph(3)
g2 = nx.cycle_graph(3)
g3 = nx.path_graph(4)
graphs = [g1, g2, g3]

# Standard WL kernel
wl = WLSubtreeKernel(n_iter=2)
K_wl = wl.fit_transform(graphs)
print('Standard WL kernel matrix:')
print(K_wl)

# Rarity weighted WL kernel
rw = RarityWeightedWLKernel(n_iter=2)
K_rw = rw.fit_transform(graphs)
print('Rarity-weighted WL kernel matrix:')
print(K_rw)
