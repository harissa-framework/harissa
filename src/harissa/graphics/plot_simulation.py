"""
Plotting simulations
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from harissa.core.simulation import Simulation

def plot_simulation(sim: Simulation.Result, plot_stimulus: bool = False):
    """
    Basic plotting function for simulations.
    """
    plt.figure(figsize=(10,4))
    gs = gridspec.GridSpec(2 + plot_stimulus, 1)
    if plot_stimulus:
        ax_stim = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0 + plot_stimulus, 0])
    ax2 = plt.subplot(gs[1 + plot_stimulus, 0])
    n = sim.protein_levels.shape[1]
    y_scale = 1.4

    if plot_stimulus:
        ax_stim.plot(sim.time_points, sim.stimulus_levels, label='$Stimulus$')
        ax_stim.set_xlim(sim.time_points[0], sim.time_points[-1])
        ax_stim.set_ylim(1.0-y_scale, y_scale)
        ax_stim.tick_params(axis='x', labelbottom=False)
        ax_stim.legend(loc='upper left', ncol=1, borderaxespad=0, frameon=False)    

    # Plot proteins
    for i in range(1, n):
        ax1.plot(sim.time_points, sim.protein_levels[:,i], label=f'$P_{{{i}}}$')
    ax1.set_xlim(sim.time_points[0], sim.time_points[-1])
    ax1.set_ylim(0, y_scale * np.max([np.max(sim.protein_levels), 1]))
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.legend(loc='upper left', ncol=4, borderaxespad=0, frameon=False)
    # Plot mRNA
    for i in range(1, n):
        ax2.plot(sim.time_points, sim.rna_levels[:,i], label=f'$M_{{{i}}}$')
    ax2.set_xlim(sim.time_points[0], sim.time_points[-1])
    ax2.set_ylim(0, y_scale * np.max(sim.rna_levels))
    ax2.legend(loc='upper left', ncol=4, borderaxespad=0, frameon=False)
    # Return the current figure
    return plt.gcf()
