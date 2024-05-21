from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import ks_2samp as ks

from harissa.core.dataset import Dataset

def plot_average_traj(
    data: Dataset, 
    path: Optional[str]=None, 
    times_unique=None
):
    if times_unique is None:
        times_unique = np.unique(data.time_points)
    T = np.size(times_unique)
    C, G = data.count_matrix.shape
    # Average for each time point
    traj = np.zeros((T,G-1))
    for k, t in enumerate(times_unique):
        traj[k] = np.mean(data.count_matrix[data.time_points==t, 1:], axis=0)
    # Draw trajectory and export figure
    fig = plt.figure(figsize=(8,2))
    labels = [rf'$\langle M_{g} \rangle$' for g in range(1,G)]
    ax = fig.add_subplot()
    ax.plot(times_unique, traj, label=labels)
    ax.set_xlim(times_unique[0], times_unique[-1])
    ax.set_ylim(0, 1.2*np.max(traj))
    ax.set_xticks(times_unique)
    ax.set_title(f'Bulk-average trajectory ({int(C/T)} cells per time point)')
    ax.legend(loc='upper left', ncol=G, borderaxespad=0, frameon=False)
    
    if path is None:
        fig.show(warn=False)
    else:
        fig.savefig(path)

def plot_data_distrib(dataset_ref: Dataset, 
                      dataset_sim: Dataset, 
                      path: str,
                      t_ref: Optional[np.ndarray] = None,
                      t_sim: Optional[np.ndarray] = None):
    data_ref = dataset_ref.count_matrix[:, 1:]
    data_sim = dataset_sim.count_matrix[:, 1:]

    if t_ref is None:
        t_ref = np.unique(dataset_ref.time_points)
    if t_sim is None:
        t_sim = np.unique(dataset_sim.time_points)

    if not np.array_equal(t_ref, t_sim):
        raise RuntimeError('Time points are not the same !')

    rat = 5
    nb_genes_by_pages = 10
    nb_genes = data_ref.shape[1]
    nb_pages = int(nb_genes / nb_genes_by_pages) + 1

    if dataset_ref.gene_names is None:
        names = np.array([f'g_{i+1}' for i in range(nb_genes)], dtype=str)
    else:
        names = dataset_ref.gene_names[1:]

    param_hist_ref = {
        'density': True,
        'color': 'grey', 
        'histtype': 'bar', 
        'alpha': 0.7    
    }
    
    param_hist_sim = {
        'density': True,
        'ec': 'red', 
        'histtype': u'step', 
        'alpha': 1, 
        'linewidth': 2
    }

    with PdfPages(path) as pdf:
        for page in range(nb_pages):
            nb_rows, nb_cols = t_sim.size, min(nb_genes_by_pages, nb_genes)
            fig, axs = plt.subplots(
                nb_rows, 
                nb_cols,
                figsize=(nb_cols * rat, nb_rows * rat)
            )

            gene_offset = page * nb_genes_by_pages
            remaining_genes = nb_genes - gene_offset
            if (remaining_genes < nb_genes_by_pages 
                and nb_genes_by_pages < nb_genes):
                axs[:, remaining_genes:nb_genes_by_pages].set_axis_off()
                # for i in range(t_ref.size):
                #     for j in range(remaining_genes, nb_genes_by_pages):
                #         axs[i, j].set_axis_off()

            for g in range(gene_offset, 
                           min((page+1) * nb_genes_by_pages, nb_genes)):
                n_max = max(
                    np.quantile(data_ref[:, g], 1), 
                    np.quantile(data_sim[:, g], 1)
                ) + 1
                n_bins = min(int(n_max / 2) + 1, 25)
                bins = np.linspace(0, n_max, n_bins)
                for i, time in enumerate(t_ref):
                    is_cell_time = dataset_ref.time_points == time
                    data_tmp_ref = data_ref[is_cell_time, g]
                    data_tmp_sim = data_sim[is_cell_time, g]
                    if time == t_ref[-1]: 
                        axs[-1, g].set_xlabel(
                            'mRNA (copies per cell)', 
                            fontsize=20
                        )
                    if time == t_ref[0]: 
                        axs[i, g].set_title(
                            names[g], 
                            fontweight="bold", 
                            fontsize=30
                        )

                    ax = axs[i, g]
                    ax.hist(data_tmp_ref, bins=bins, **param_hist_ref)
                    ax.hist(data_tmp_sim, bins=bins, **param_hist_sim)
                    ax.legend(
                        labels=[f'Model (t = {time}h)', f'Data (t = {time}h)']
                    )
            pdf.savefig(fig)
            plt.close()

def compare_marginals(dataset_ref: Dataset,
                      dataset_sim: Dataset, 
                      path: str,
                      t_ref: Optional[np.ndarray] = None, 
                      t_sim: Optional[np.ndarray] = None):

    data_ref = dataset_ref.count_matrix[:, 1:]
    data_sim = dataset_sim.count_matrix[:, 1:]

    if t_ref is None:
        t_ref = np.unique(dataset_ref.time_points)
    if t_sim is None:
        t_sim = np.unique(dataset_sim.time_points)

    if not np.array_equal(t_ref, t_sim):
        raise RuntimeError('Time points are not the same !')
    
    T = t_ref.size
    nb_genes = data_ref.shape[1]

    if dataset_ref.gene_names is None:
        names = np.array([f'g_{i+1}' for i in range(nb_genes)], dtype=str)
    else:
        names = dataset_ref.gene_names[1:]

    pval_netw = np.ones((T, nb_genes))
    for t, time in enumerate(t_ref):
        is_cell_time = dataset_ref.time_points == time
        data_tmp_real = data_ref[is_cell_time, :]
        data_tmp_netw = data_sim[is_cell_time, :]
        for g in range(nb_genes):
            stat_tmp = ks(data_tmp_real[:, g], data_tmp_netw[:, g])
            pval_netw[t, g] = stat_tmp[1]

    # Figure
    fig = plt.figure(figsize=(4*nb_genes, 4.1*nb_genes))
    grid = gs.GridSpec(6, 4, wspace=0, hspace=0,
        width_ratios=[0.09,1.48,0.32,1],
        height_ratios=[0.49,0.2,0.031,0.85,0.22,0.516])
    panelA = grid[0,:]
    # Panel settings
    opt = {'xy': (0,1), 'xycoords': 'axes fraction', 'fontsize': 10,
        'textcoords': 'offset points', 'annotation_clip': False}

    # Color settings
    colors = ['#d73027','#f46d43','#fdae61','#fee08b','#ffffbf',
        '#d9ef8b','#a6d96a','#66bd63','#1a9850']

    # A. KS test p-values
    axA = plt.subplot(panelA)
    axA.annotate('A', xytext=(-14,6), fontweight='bold', **opt)
    axA.annotate('KS test p-values', xytext=(0,6), **opt)
    # axA.set_title('KS test p-values', fontsize=10)
    cmap = LinearSegmentedColormap.from_list('pvalue', colors)
    norm = Normalize(vmin=0, vmax=0.1)
    # Plot the heatmap
    im = axA.imshow(pval_netw, cmap=cmap, norm=norm)
    axA.set_aspect('equal','box')
    axA.set_xlim(-0.5, nb_genes - 0.5)
    axA.set_ylim(T-0.5,-0.5)
    # Create colorbar
    divider = make_axes_locatable(axA)
    cax = divider.append_axes('right', '1.5%', pad='2%')
    cbar = axA.figure.colorbar(im, cax=cax, extend='max')
    pticks = np.array([0,1,3,5,7,9])
    cbar.set_ticks(pticks/100 + 0.0007)
    cbar.ax.set_yticklabels([0]+[f'{p}%' for p in pticks[1:]], fontsize=6)
    cbar.ax.spines[:].set_visible(False)
    cbar.ax.tick_params(axis='y',direction='out', length=1.5, pad=1.5)
    axA.set_xticks(np.arange(nb_genes))
    axA.set_yticks(np.arange(T))
    axA.set_xticklabels(names, rotation=45, ha='right', rotation_mode='anchor',
        fontsize=3)
    axA.set_yticklabels([f'{t}h' for t in t_ref], fontsize=6.5)
    axA.spines[:].set_visible(False)
    axA.set_xticks(np.arange(nb_genes+1) - 0.5, minor=True)
    axA.set_yticks(np.arange(T+1) - 0.5, minor=True)
    axA.grid(which='minor', color='w', linestyle='-', linewidth=1)
    axA.tick_params(which='minor', bottom=False, left=False)
    axA.tick_params(which='major', bottom=False, left=False)
    axA.tick_params(axis='x',direction='out', pad=-0.1)
    axA.tick_params(axis='y',direction='out', pad=-0.1)

    # Export the figure
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.02)

def _configure(ax):
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('UMAP1', fontsize=7, weight='bold')
    ax.set_ylabel('UMAP2', fontsize=7, weight='bold')
    ax.tick_params(axis='both', labelsize=7)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.spines[['top', 'right']].set_visible(False)

def plot_data_umap(dataset_ref: Dataset, 
                    dataset_sim: Dataset,
                    path: str, 
                    t_ref: Optional[np.ndarray] = None,
                    t_sim: Optional[np.ndarray] = None):
    
    try:
        # import here because it takes forever to import umap
        from umap import UMAP
    except ImportError:
        import sys
        print('umap-learn not found. It wont plot the umap. '
              'Please install umap-learn.',
              file=sys.stderr)
        return
    
    # Remove stimulus
    data_ref = dataset_ref.count_matrix[:, 1:]
    data_sim = dataset_sim.count_matrix[:, 1:]

    if t_ref is None:
        t_ref = np.unique(dataset_ref.time_points)
    if t_sim is None:
        t_sim = np.unique(dataset_sim.time_points)

    if not np.array_equal(t_ref, t_sim):
        raise RuntimeError('Time points are not the same !')
    
    # Compute the UMAP projection
    reducer = UMAP(random_state=42, min_dist=0.7)
    proj = reducer.fit(data_ref)
    x_ref = proj.transform(data_ref)
    x_sim = proj.transform(data_sim)

    # Figure
    fig = plt.figure(figsize=(10, 4))
    grid = gs.GridSpec(2, 2, height_ratios=[1, 0.05], wspace=0.3)
    ax0 = plt.subplot(grid[0, 0])
    ax1 = plt.subplot(grid[0, 1])
    ax3 = plt.subplot(grid[1, :])

    # Panel settings
    opt = {'xy': (0, 1), 'xycoords': 'axes fraction', 'fontsize': 10,
        'textcoords': 'offset points', 'annotation_clip': False}

    # Timepoint colors
    T = t_ref.size
    cmap = [plt.get_cmap('viridis', T)(i) for i in range(T)]
    c_ref = [cmap[np.argwhere(t_ref==t)[0,0]] for t in dataset_ref.time_points]
    c_sim = [cmap[np.argwhere(t_sim==t)[0,0]] for t in dataset_sim.time_points]

    # A. Original data
    _configure(ax0)
    title = 'Original data'
    ax0.annotate('A', xytext=(-11, 6), fontweight='bold', **opt)
    ax0.annotate(title, xytext=(3, 6), **opt)
    ax0.scatter(x_ref[:, 0], x_ref[:, 1], c=c_ref, s=2)

    # B. Inferred network
    _configure(ax1)
    title = 'Inferred network'
    ax1.annotate('B', xytext=(-11, 6), fontweight='bold', **opt)
    ax1.annotate(title, xytext=(3, 6), **opt)
    ax1.scatter(x_sim[:, 0], x_sim[:, 1], c=c_sim, s=2)
    ax1.set(xlim=ax0.get_xlim(), ylim=ax0.get_ylim())

    # Legend panel
    labels = [f'{int(t_ref[k])}h' for k in range(T)]
    lines = [Line2D([0], [0], color=cmap[k], lw=5) for k in range(T)]
    ax3.legend(lines, labels, ncol=T, frameon=False, borderaxespad=0,
            loc='lower right', handlelength=1, fontsize=8.5)
    ax3.text(-0.02, 0.8, 'Timepoints:', transform=ax3.transAxes, fontsize=8.5)
    ax3.axis('off')

    # Export the figure
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.02)