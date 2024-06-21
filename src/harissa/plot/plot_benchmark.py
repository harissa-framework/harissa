"""Some utility plot functions for benchmarking Harissa"""

from __future__ import annotations
from typing import List, Tuple, Optional

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from harissa.core import NetworkParameter, Inference
from harissa.plot.plot_network import build_pos, plot_network

class DirectedPlotter:
    def __init__(self,
        inferences_order: List[str],
        network: Optional[NetworkParameter] = None, 
        alpha_curve_std: float = 0.2
    ) -> None:
        self._inferences_order = inferences_order
        self._truth = None
        if network is None:
            self._network = network
        else:
            self.network = network
        self.alpha_curve_std = alpha_curve_std

    @property
    def network(self):
        return self._network
    
    @network.setter
    def network(self, network: NetworkParameter):
        if network is None:
            raise TypeError('Cannot assign None to network.')
        if not isinstance(network, NetworkParameter):
            raise TypeError('network must be a NetworkParameter object.')
        
        self._network = network
        self._truth = self._prepare_truth(network.interaction)

    @property
    def truth(self) -> npt.NDArray[np.float64]:
        return self._truth    

    def _prepare_score(self, 
        matrix: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        n = matrix.shape[0]
        # remove first column and the diagonal
        mask = ~np.hstack((
            np.ones((n, 1), dtype=bool), 
            np.eye(n, dtype=bool)[:, 1:]
        ))

        # assert matrix[mask].shape == (n*(n - 2) + 1,)

        return np.abs(matrix[mask])

    def _prepare_truth(self, matrix):
        return 1.0 * (self._prepare_score(matrix) > 0)
    
    def _accept_inference(self, inference: Inference) -> bool:
        return inference.directed
    
    
    def plot_network(self, **kwargs):
        if self.network is not None:
            if self.network.layout is None:
                layout = build_pos(self.network.interaction)
            else:
                layout = self.network.layout

            if 'names' not in kwargs:
                kwargs['names'] = self.network.genes_names

            plot_network(self.network.interaction, layout, **kwargs)


    def roc(self, 
        score: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute a receiver operating characteristic (ROC) curve.
        Here score and inter are arrays of shape (G,G) where:
        * score[i,j] is the estimated score of interaction i -> j
        * inter[i,j] = 1 if i -> j is present and 0 otherwise.
        """
        x, y, _ = roc_curve(self.truth, self._prepare_score(score))
        return x, y

    def auroc(self, 
        score:npt.NDArray[np.float64]
    ) -> float:
        """
        Area under ROC curve (see function `roc`).
        """
        x, y = self.roc(score)
        return auc(x, y) 
    
    def plot_roc_curves(self,
        ax: plt.Axes
    ):
        """
        Plot mutiple ROC curves (see function `roc`).
        """
        y_start, y_end = 0.0, 1.0
        try:
            yield from self._plot_curves(ax, self.roc, y_start, y_end)
        finally:
            ax.plot(
                [y_start, y_end], 
                [y_start, y_end], 
                color='lightgray',
                ls='--',
                label='Random (0.50)'
            )
            ax.set_xlabel('False positive rate')
            ax.set_ylabel('True positive rate')
            ax.legend(loc='lower right')
            # ax.set_xlim(0,1)
            # ax.set_ylim(0)

    def plot_roc_boxes(self, ax: plt.Axes):
        try:
            yield from self._plot_boxes_auc(ax, self.auroc)
        finally:
            left, right = ax.get_xlim()
            ax.plot(
                [left, right], 
                [0.5,0.5], 
                color='lightgray', 
                ls='--' 
                # label=f'Random ({b:.2f})'
            )
            
            ax.set_xlim(left, right)
            # ax.set_ylim(0,1)
            ax.set_ylabel('AUROC')

    def pr(self,
        score: npt.NDArray[np.float64]
    ) ->  Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute a precision recall (PR) curve.
        Here score and inter are arrays of shape (G,G) where:
        * score[i,j] is the estimated score of interaction i -> j
        * inter[i,j] = 1 if i -> j is present and 0 otherwise.
        """
        y, x, _ = precision_recall_curve(
            self.truth, 
            self._prepare_score(score)
        )

        return np.flip(x), np.flip(y)

    def aupr(self, 
        score: npt.NDArray[np.float64]
    ) -> float:
        """
        Area under PR curve (see function `pr`).
        """
        x, y = self.pr(score)
        return auc(x,y)

    def plot_pr_curves(self, ax: plt.Axes):
        """
        Plot multiple PR curves (see function `pr`).
        """
        try:
            yield from self._plot_curves(ax, self.pr, 1.0)
        finally:
            b = np.mean(self.truth)
            ax.plot(
                [0,1], 
                [b,b], 
                color='lightgray', 
                ls='--', 
                label=f'Random ({b:.2f})'
            )
            
            # ax.set_xlim(0,1)
            # ax.set_ylim(0,1)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.legend(loc='lower right')

    def plot_pr_boxes(self, ax: plt.Axes):
        try:
            yield from self._plot_boxes_auc(ax, self.aupr)
        finally:
            b = np.mean(self.truth)
            left, right = ax.get_xlim()
            ax.plot(
                [left, right], 
                [b,b], 
                color='lightgray', 
                ls='--' 
                # label=f'Random ({b:.2f})'
            )
            
            ax.set_xlim(left, right)
            # ax.set_ylim(0,1)
            ax.set_ylabel('AUPR')

    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    def _plot_curves(self, ax, curve_fn, y_start, y_end=None):
        ys_per_inf = {}
        x = np.linspace(0, 1, 1000)
        try:
            while True:
                params = yield 
                if params is not None:
                    inf_name, (inference, colors), result = params
                    if self._accept_inference(inference):
                        curve = curve_fn(result.parameter.interaction)
                        y = np.interp(x, *curve)
                        y[0] = y_start

                        if inf_name not in ys_per_inf:
                            ys_per_inf[inf_name] = ([], colors)

                        ys_per_inf[inf_name][0].append(y)
                yield
        finally:
            # reorder inferences
            ys_per_inf = {
                inf_name:ys_per_inf[inf_name] 
                for inf_name in self._inferences_order 
                if inf_name in ys_per_inf
            }

            for inf, (ys, colors) in ys_per_inf.items():
                ys = np.array(ys)
                y = np.mean(ys, axis=0)
                if y_end is not None:
                    y[-1] = y_end
                std_y = np.std(ys, axis=0)
                ax.plot(
                    x, 
                    y,
                    color=colors[0], 
                    label=f'{inf} ({auc(x, y):.2f})'
                )
                ax.fill_between(
                    x, 
                    np.maximum(y - std_y, 0.0), 
                    np.minimum(y + std_y, 1.0),
                    color=colors[1],
                    alpha=self.alpha_curve_std
                )

    def _plot_boxes_auc(self, ax, auc_fn):
        aucs_per_inf = {}
        try:
            while True:
                params = yield
                if params is not None:
                    inf_name, (inference, colors), result = params
                    if self._accept_inference(inference):
                        auc = auc_fn(result.parameter.interaction)

                        if inf_name not in aucs_per_inf:
                            aucs_per_inf[inf_name] = ([], colors)

                        aucs_per_inf[inf_name][0].append(auc)
                yield
        finally:
            # reorder inferences
            aucs_per_inf = {
                inf_name:aucs_per_inf[inf_name] 
                for inf_name in self._inferences_order
                if inf_name in aucs_per_inf
            }

            for i, (aucs, colors) in enumerate(aucs_per_inf.values()):
                box = ax.boxplot(
                    [np.array(aucs)], 
                    positions=[i+0.5], 
                    patch_artist= True,
                    widths= [.25]
                )

                w = 0.8
                for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                    plt.setp(box[item], color=colors[0], lw=w)
                plt.setp(box['boxes'], facecolor=colors[1])
                plt.setp(
                    box['fliers'], 
                    markeredgecolor=colors[0], 
                    ms=3, 
                    markerfacecolor=colors[1],
                    markeredgewidth=w
                )

            ax.set_xticklabels(list(aucs_per_inf.keys()))
            # w = 0.7
            # ax.tick_params(direction='out', length=3, width=w)
            # ax.tick_params(axis='x', pad=2, labelsize=5.5)
            # ax.tick_params(axis='y', pad=0.5, labelsize=5.5)
            # for x in ['top','bottom','left','right']:
            #     ax.spines[x].set_linewidth(w)


class UnDirectedPlotter(DirectedPlotter):
    def __init__(self,
        inferences_order: List[str], 
        network: Optional[NetworkParameter] = None,
        alpha_curve_std: float = 0.2
        ) -> None:
        super().__init__(inferences_order, network, alpha_curve_std)

    def _prepare_score(self, 
        matrix: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        abs_matrix = np.abs(matrix.filled(0.0))
        # remove lower triangle
        mask = ~np.tri(*abs_matrix.shape, dtype=bool)

        return np.maximum(abs_matrix[mask], abs_matrix.T[mask])

    def _accept_inference(self, inference: Inference) -> bool:
        return True
    
def plot_benchmark(
    benchmark, 
    networks_order, 
    inferences_order, 
    show_networks=False
):
    plotters_per_networks = {
        net_name:(
            DirectedPlotter(inferences_order), 
            UnDirectedPlotter(inferences_order)
        )
        for net_name in networks_order
    }
    nb_networks = len(plotters_per_networks)
    nb_colum = 4 + show_networks
    scale = 4
    figs = [
        plt.figure(figsize=(18, 10), layout="constrained"),
        plt.figure(figsize=(scale*nb_colum, scale*nb_networks), layout="constrained"),
        plt.figure(figsize=(scale*nb_colum, scale*nb_networks), layout="constrained")
    ]
    titles = ['general', 'directed', 'undirected']
    for fig, title in zip(figs, titles):
        fig.suptitle(title)

    grid = gs.GridSpec(2, 4, figure=figs[0])
    plotters = [
        DirectedPlotter(inferences_order), 
        UnDirectedPlotter(inferences_order)
    ]
    for i, (plotter, title) in enumerate(zip(plotters, titles[1:])):
        plotter.axs = [figs[0].add_subplot(grid[i, j]) for j in range(0,4)]
        plotter.plots = None
        plotter.axs[0].text(
            -0.195, 
            0.875, 
            title, 
            bbox={
                'boxstyle':'round,pad=0.2',
                'fc':'none',
                'ec':'lightgray',
                'lw':0.8
            },
            fontsize=9,
            transform=plotter.axs[0].transAxes,
            ha='right'
        )

    for k in range(1, 3):
        fig = figs[k]
        grid = gs.GridSpec(nb_networks, nb_colum, figure=fig)
        for i, network_name in enumerate(plotters_per_networks):
            # prepare axes
            axs = [fig.add_subplot(grid[i, j]) for j in range(nb_colum)]
            axs[0].text(
                -0.095, 
                0.875, 
                network_name, 
                bbox={
                    'boxstyle':'round,pad=0.2',
                    'fc':'none',
                    'ec':'lightgray',
                    'lw':0.8
                },
                fontsize=9,
                transform=axs[0].transAxes,
                ha='right'
            )
            
            plotter = plotters_per_networks[network_name][k-1]
            plotter.axs = axs
            plotter.plots = None
    
    for key, value in benchmark.items():
        network_name, inf_name = key[0], key[1]
        network, inf, result = value[0], value[1], value[3]
        params = (inf_name, inf, result)

        for plotter in plotters:
            plotter.network = network
            if plotter.plots is None:
                plotter.plots = [
                    plotter.plot_roc_curves(plotter.axs[0]),
                    plotter.plot_roc_boxes(plotter.axs[1]),
                    plotter.plot_pr_curves(plotter.axs[2]),
                    plotter.plot_pr_boxes(plotter.axs[3])
                ]
            
            for plot in plotter.plots:
                next(plot)
                plot.send(params)

        for plotter in plotters_per_networks[network_name]:
            plotter.network = network    
            if plotter.plots is None:
                if show_networks:
                    ax_pos = plotter.axs[0].get_position()
                    scale = 1.1 / np.min([ax_pos.width,ax_pos.height])
                    plotter.plot_network(axes=plotter.axs[0], scale=scale)
                plotter.plots = [
                    plotter.plot_roc_curves(plotter.axs[show_networks+0]),
                    plotter.plot_roc_boxes(plotter.axs[show_networks+1]),
                    plotter.plot_pr_curves(plotter.axs[show_networks+2]),
                    plotter.plot_pr_boxes(plotter.axs[show_networks+3])
                ]
            
            for plot in plotter.plots:
                next(plot)
                plot.send(params)

    for i, plotter in enumerate(plotters):
        for plot in plotter.plots:
            plot.close()
        
        for j in range(1, 4):
            plotter.axs[j].sharey(plotter.axs[0])
            plt.setp(plotter.axs[j].get_yticklabels(), visible=False)
        
        if i < 1:
            for j, ax in enumerate(plotter.axs):
                ax.set_xlabel('')
                if j % 2 == 0:
                    plt.setp(ax.get_xticklabels(), visible=False)

    for i, plotters in enumerate(plotters_per_networks.values()):
        for plotter in plotters:
            for plot in plotter.plots:
                plot.close()

            for j in range(show_networks + 1, nb_colum):
                plotter.axs[j].sharey(plotter.axs[show_networks])
                plt.setp(plotter.axs[j].get_yticklabels(), visible=False)
                
            if i < nb_networks - 1:
                for ax in plotter.axs:
                    ax.set_xlabel('')
                    plt.setp(ax.get_xticklabels(), visible=False)
             
    return figs