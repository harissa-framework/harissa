# Branching 4-gene pathways with stimulus
import numpy as np
import matplotlib.pyplot as plt
from harissa import NetworkModel
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
data1 = np.zeros((C,G+1), dtype='int')
data1[:,0] = time # Time points
data2 = data1.copy()

# Model 1
model1 = NetworkModel(G)
model1.degradation_rna[:] = 1
model1.degradation_protein[:] = 0.2
model1.basal[1:] = -5
model1.interaction[0,1] = 10
model1.interaction[1,2] = 10
model1.interaction[1,3] = 10
model1.interaction[2,4] = 10

# Model 2
model2 = NetworkModel(G)
model2.degradation_rna[:] = 1
model2.degradation_protein[:] = 0.2
model2.basal[1:] = -5
model2.interaction[0,1] = 10
model2.interaction[1,2] = 10
model2.interaction[1,3] = 10
model2.interaction[3,4] = 10

# Generate data
for k in range(C):
    # Data for model 1
    sim1 = model1.simulate(time[k], burn_in=5)
    data1[k, 1:] = np.random.poisson(sim1.rna_levels[0])
    # Data for model 2
    sim2 = model2.simulate(time[k], burn_in=5)
    data2[k, 1:] = np.random.poisson(sim2.rna_levels[0])

# Save data in basic format
np.savetxt('pathways_data1.txt', data1, fmt='%d', delimiter='\t')
np.savetxt('pathways_data2.txt', data2, fmt='%d', delimiter='\t')


#### Plot mean trajectories ####

for i, data in [(1,data1),(2,data2)]:
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
    fig.savefig(f'pathways_mean{i}.pdf', bbox_inches='tight')


#### Plot the networks ####

# Node labels
names = [''] + [f'{i+1}' for i in range(G)]

# Draw networks and export figures
for i, model in [(1,model1),(2,model2)]:
    pos = build_pos(model.interaction)
    fig = plt.figure(figsize=(5,5))
    plot_network(model.interaction, pos, axes=fig.gca(), names=names, scale=2)
    fig.savefig(f'pathways_graph{i}.pdf', bbox_inches='tight')


#### Perform network inference ####

for i in [1,2]:
    # Load the data
    x = np.loadtxt(f'pathways_data{i}.txt', dtype=int, delimiter='\t')
    # Calibrate the model
    model = NetworkModel()
    model.fit(x)
    # Export interaction matrix
    np.savetxt(f'pathways_inter{i}.txt', model.interaction, delimiter='\t')
