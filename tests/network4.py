# Basic 4-gene network with stimulus and feedback loop
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path += ['../harissa/model']
sys.path += ['../harissa/utils']
from model import NetworkModel
from harissa.utils import build_pos, plot_network
#### Simulate scRNA-seq data ####

# Number of cells
C = 100

# Set the time points
k = np.linspace(0, C, 11, dtype='int')
t = np.linspace(0, 20, 10, dtype='int')
time = np.zeros(C, dtype='int')
for i in range(10):
    time[k[i]:k[i+1]] = t[i]
print(f'Times points ({t.size}): {t}')

# Number of genes
G = 4

# Prepare data
data = np.zeros((C,G+1), dtype='int')
data[:,0] = time # Time points
    
# Initialize the model
model = NetworkModel(G)
model.degradation_rna[:] = 1
model.degradation_protein[:] = 0.2
model.basal[1:] = -5
model.interaction[0,1] = 10
model.interaction[1,2] = 10
model.interaction[1,3] = 10
model.interaction[3,4] = 10
model.interaction[4,1] = -10
model.interaction[2,2] = 10
model.interaction[3,3] = 10

# Generate data
for k in range(C):
    print(f'* Cell {k+1} (t = {time[k]})')
    sim = model.simulate(time[k], burn_in=5)
    data[k,1:] = np.random.poisson(sim.rna_levels[0])

# Save data in basic format
np.savetxt('network4_data.txt', data, fmt='%d', delimiter='\t')

#### Plot mean trajectory ####

# Import time points
time = np.sort(list(set(data[:,0])))
T = np.size(time)

# Average for each time point
trajectory = np.zeros((T,G))
for k, t in enumerate(time):
    trajectory[k] = np.mean(data[data[:,0]==t,1:], axis=0)

# Draw trajectory and export figure
fig = plt.figure(figsize=(8,2))
labels = [rf'$\langle M_{i+1} \rangle$' for i in range(G)]
plt.plot(time, trajectory, label=labels)
ax = plt.gca()
ax.set_xlim(time[0], time[-1])
ax.set_ylim(0, 1.2*np.max(trajectory))
ax.set_xticks(time)
ax.set_title(f'Bulk-average trajectory ({int(C/T)} cells per time point)')
ax.legend(loc='upper left', ncol=G, borderaxespad=0, frameon=False)
fig.savefig('network4_mean.pdf', bbox_inches='tight')


#### Plot the network ####

# Node labels and positions
names = [''] + [f'{i+1}' for i in range(G)]
pos = build_pos(model.interaction)

# Draw network and export figure
fig = plt.figure(figsize=(5,5))
plot_network(model.interaction, pos, axes=fig.gca(), names=names, scale=2)
fig.savefig('network4_graph.pdf', bbox_inches='tight')


#### Perform network inference ####

# Load the data
x = np.loadtxt('network4_data.txt', dtype=int, delimiter='\t')

# Calibrate the model
model = NetworkModel()
model.fit(x)

# Export interaction matrix
np.savetxt('network4_inter.txt', model.interaction, delimiter='\t')
