# Basic repressilator network (3 genes)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from harissa import NetworkModel, NetworkParameter
from harissa.simulation import ApproxODE

# Model
param = NetworkParameter(3)
model = NetworkModel(param)
param.degradation_rna[:] = 1
param.degradation_protein[:] = 0.2
param.basal[1] = 5
param.basal[2] = 5
param.basal[3] = 5
param.interaction[1,2] = -10
param.interaction[2,3] = -10
param.interaction[3,1] = -10
scale = param.burst_size_inv / param.burst_frequency_max
param.creation_rna[:] = param.degradation_rna * scale 
param.creation_protein[:] = param.degradation_protein * scale

# Time points
time = np.linspace(0,100,1000)

# Simulation of the PDMP model
sim = model.simulate(time)

# Simulation of the ODE model (slow-fast limit)
model.simulation = ApproxODE()
sim_ode = model.simulate(time, P0=[0,0,0.1,0.2])

# Figure
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(3, 1, hspace=0.6)

# Plot mRNA levels
ax = plt.subplot(gs[0,0])
ax.set_title(f'mRNA levels ($d_0 = {param.degradation_rna.mean()}$)')
ax.set_xlim(sim.time_points[0], sim.time_points[-1])
ax.set_ylim(0, 1.2*np.max(sim.rna_levels))
for i in range(3):
    ax.plot(sim.time_points, sim.rna_levels[:, i], label=f'$M_{{{i+1}}}$')
ax.legend(loc='upper left', ncol=4, borderaxespad=0, frameon=False)

# Plot protein levels
ax = plt.subplot(gs[1,0])
ax.set_title(f'Protein levels ($d_1 = {param.degradation_protein.mean()}$)')
ax.set_xlim(sim.time_points[0], sim.time_points[-1])
ax.set_ylim(0, np.max([1.2*np.max(sim.protein_levels), 1]))
for i in range(3):
    ax.plot(sim.time_points, sim.protein_levels[:, i], label=f'$P_{{{i+1}}}$')
ax.legend(loc='upper left', ncol=4, borderaxespad=0, frameon=False)

# Plot protein levels (ODE model)
ax = plt.subplot(gs[2,0])
ax.set_title(r'Protein levels - ODE model ($d_0/d_1\to\infty$)')
ax.set_xlim(sim_ode.time_points[0], sim_ode.time_points[-1])
ax.set_ylim(0, 1)
for i in range(3):
    ax.plot(sim_ode.time_points, 
            sim_ode.protein_levels[:,i], 
            label=f'$P_{{{i+1}}}$')
ax.legend(loc='upper left', ncol=4, borderaxespad=0, frameon=False)    


fig.savefig('repressilator.pdf', bbox_inches='tight')
