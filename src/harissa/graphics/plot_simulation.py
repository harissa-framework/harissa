"""
Plotting simulations
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_simulation(sim):
    """
    Basic plotting function for simulations.
    """
    plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(2,1)
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    n = sim.protein_levels.shape[1]
    # Plot proteins
    for i in range(n):
        ax1.plot(sim.time_points, sim.protein_levels[:,i], label=f'$P_{{{i+1}}}$')
        ax1.set_xlim(sim.time_points[0], sim.time_points[-1])
        ax1.set_ylim(0, np.max([1.2*np.max(sim.protein_levels), 1]))
        ax1.tick_params(axis='x', labelbottom=False)
        ax1.legend(loc='upper left', ncol=4, borderaxespad=0, frameon=False)
    # Plot mRNA
    for i in range(n):
        ax2.plot(sim.time_points, sim.rna_levels[:,i], label=f'$M_{{{i+1}}}$')
        ax2.set_xlim(sim.time_points[0], sim.time_points[-1])
        ax2.set_ylim(0, 1.2*np.max(sim.rna_levels))
        ax2.legend(loc='upper left', ncol=4, borderaxespad=0, frameon=False)
