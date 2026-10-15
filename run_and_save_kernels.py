import os
import sys
import numpy as np

# Ensure the src directory is on the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from egonets_wl.kernels.wl_subtree import WLSubtreeKernel
from egonets_wl.kernels.rarity_weighted_wl import RarityWeightedWLKernel
import networkx as nx

# Define some sample graphs
graphs = [nx.path_graph(3), nx.cycle_graph(3), nx.path_graph(4)]

# Compute standard WL kernel matrix
wl = WLSubtreeKernel(n_iter=2)
K_wl = wl.fit_transform(graphs)

# Compute rarity‑weighted WL kernel matrix
rw = RarityWeightedWLKernel(n_iter=2)
K_rw = rw.fit_transform(graphs)

# Print matrices (optional)
print("Standard WL kernel matrix:")
print(K_wl)
print("Rarity‑weighted WL kernel matrix:")
print(K_rw)

# Save the matrices to files for later inspection
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "standard_wl.npy"), K_wl)
np.save(os.path.join(output_dir, "rarity_weighted_wl.npy"), K_rw)
print(f"Saved standard WL matrix to {os.path.join(output_dir, 'standard_wl.npy')}")
print(f"Saved rarity‑weighted WL matrix to {os.path.join(output_dir, 'rarity_weighted_wl.npy')}")
