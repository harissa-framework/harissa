"""
Main class for network inference
"""
from pathlib import Path
from typing import Dict, Union
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from harissa.core.parameter import NetworkParameter
from harissa.core.inference import Inference
from harissa.core.dataset import Dataset
from harissa.inference.hartree.base import infer_kinetics
from harissa.inference.cardamom.utils import (
    core_basins_binary, 
    # estim_gamma_poisson,
    # build_cnt
)
from harissa.utils.npz_io import save_npz, save_dir

# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# def infer_kinetics(x, times, tol=1e-5, max_iter=1000, verb=False):
#     """
#     Infer parameters a[0], ..., a[m-1] and b of a Gamma-Poisson model
#     with time-dependant a and constant b for a given gene at m time points.

#     Parameters
#     ----------
#     x[k] = gene expression in cell k
#     times[k] = time point of cell k
#     """

#     t = np.sort(list(set(times)))
#     m = t.size
#     n = np.zeros(m) # Number of cells for each time point
#     a = np.zeros(m)
#     b = np.zeros(m)
#     # Initialization of a and b
#     for i in range(m):
#         cells = (times == t[i])
#         n[i] = np.sum(cells)
#         a[i], b[i] = estim_gamma_poisson(x[cells])
#     b = np.mean(b)
#     # Newton-like method
#     k, c = 0, 0
#     sx = np.sum(x)
#     while (k == 0) or (k < max_iter and c > tol):
#         da = np.zeros(m)
#         for i in range(m):
#             if a[i] > 0:
#                 cells = (times == t[i])
#                 z = a[i] + x[cells]
#                 p0 = np.sum(psi(z))
#                 p1 = np.sum(polygamma(1, z))
#                 d = n[i]*(np.log(b)-np.log(b+1)-psi(a[i])) + p0
#                 h = p1 - n[i]*polygamma(1, a[i])
#                 da[i] = -d/h
#         anew = a + da
#         if np.sum(anew < 0) == 0: 
#             a[:] = anew
#         else:
#             max_test = 5
#             test = 0
#             da *= 0.5
#             while (np.sum(a + da < 0) > 0) and (test < max_test):
#                 da *= 0.5
#                 test += 1
#             if test < max_test: 
#                 a[:] = a + da
#             else: 
#                 print('Warning: parameter a not improved')
#         if np.sum(a == 0) == 0:
#             b = np.sum(n*a)/sx
#         else: 
#             b = b
#         c = np.max(np.abs(da))
#         k += 1
#     if (k == max_iter) and (c > tol):
#         print('Warning: bad convergence (b = {})'.format(b))
#         a, b = a/b, 1
#     if verb: 
#         print(f'Estimation done in {k} iterations')
#     if np.sum(a < 0) > 0: 
#         print('WARNING: a < 0')
#     if b < 0: 
#         print('WARNING: b < 0')
#     if np.all(a == 0): 
#         print('WARNING: a == 0')
#     # if k > 20 and np.max(a/b) > 2: print(k, np.max(a/b))

#     return a, b


def penalization_l1(x, s):
    return (x-s/2)*(x>s) - (x+s/2)*(-x>s) + ((x**2)/(2*s))*(x<=s and -x<=s)

def grad_penalization_l1(x, s):
    return 1*(x>s) - 1*(-x>s) + (x/s)*(x<=s and -x<=s)

def penalization(Q, X, X_init, time_init, l, sc, G, cnt_move, j, p):

    l_inter, l_diag = l[0], l[1]
    if not time_init:
        Q += penalization_l1(X[-1] - X_init[j, -1], sc) * l_inter * p
    if time_init == 1:
        Q += penalization_l1(X[0] - X_init[j, 0], sc) * l_inter * p
    tmp_list = list(range(1, j)) + list(range(j + 1, G))
    for i in tmp_list:
        tmp = (
            cnt_move[i] * (1 + abs(X_init[i, j]) / (1 + abs(X_init[j, i])))
        )
        Q += penalization_l1(X[i] - X_init[j, i], sc) * l_inter * tmp
    tmp_diag = (
        cnt_move[j] / (1 + np.sum(np.abs(X_init[j, :-1])) 
        - abs(X_init[j, j]))
    )
    Q += (
        l_diag * (penalization_l1(X[j] - X_init[j, j], sc) 
        + (X[j] - X_init[j, j]) ** 2) * tmp_diag
    )
    return Q
    
def grad_penalization(dq, X, X_init, time_init, l, sc, G, cnt_move, j, p):

    l_inter, l_diag = l[0], l[1]
    if not time_init: 
        dq[-1] += (
            grad_penalization_l1(X[-1] - X_init[j, -1], sc) * l_inter * p
        )
    else: 
        dq[-1] = 0
    if time_init == 1:
        dq[0] += (
            grad_penalization_l1(X[0] - X_init[j, 0], sc) * l_inter * p
        )
    else:
        dq[0] = 0
    tmp_list = list(range(1, j)) + list(range(j + 1, G))
    for i in tmp_list:
        tmp = (
            cnt_move[i] * (1 + abs(X_init[i, j]) / (1 + abs(X_init[j, i])))
        )
        dq[i] += (
            grad_penalization_l1(X[i] - X_init[j, i], sc) * l_inter * tmp
        )
    tmp_diag = (
        cnt_move[j] / (1 + np.sum(np.abs(X_init[j, :-1])) 
        - abs(X_init[j, j]))
    )
    dq[j] += (
        l_diag * (grad_penalization_l1(X[j] - X_init[j, j], sc) 
        + 2 * (X[j] - X_init[j, j])) * tmp_diag
    )
    return dq

def objective(
    X, y, vect_kon, ko, X_init, time_init, l, sc, G, cnt_move, j, p
):
    """
    Objective function to be maximized 
    (all cells, one gene, all timepoints).
    """
    sigma = expit(X[-1] + y @ X[:-1])
    Q = np.sum((ko + (1 - ko) * sigma - vect_kon) ** 2)
    return penalization(Q, X, X_init, time_init, l, sc, G, cnt_move, j, p)
    
def grad_theta(
    X, y, vect_kon, ko, X_init, time_init, l, sc, G, cnt_move, j, p
):
    """
    Objective gradient for gene i for all cells.
    """

    dq = np.zeros(G + 1)
    sigma = expit(X[-1] + y @ X[:-1])
    tmp = (
        2 * (ko + (1 - ko) * sigma - vect_kon) 
        * (1 - ko) * sigma * (1 - sigma)
    )
    for i in range(0, G):
        dq[i] += np.sum(y[:, i] * tmp)
    dq[-1] += np.sum(tmp)

    return grad_penalization(
        dq, X, X_init, time_init, l, sc, G, cnt_move, j, p
    )

def core_inference(
    variations,
    time_points, 
    times_unique, 
    y, 
    vect_kon, 
    ko, 
    Xo, 
    l, 
    sl, 
    G, 
    cnt_init,
    cnt_end,
    p,
    tolerance
):
    cnt_move = np.ones((G, G))
    theta_t = np.zeros((cnt_end - cnt_init, G, G + 1))
    for cnt, time in enumerate(times_unique[cnt_init:cnt_end]):
        penalization_strength = l * np.sum(time_points == time) * 2.5 / 100
        # if cnt + cnt_init > 1:
        #     cnt_move = build_cnt(
        #         cnt + cnt_init, 
        #         cnt_move, 
        #         vect_kon, 
        #         vect_t, 
        #         times, 
        #         G,
        #         p
        #     )

        # Optimization parameters
        optimization_params = {
            'fun': objective,
            'jac': grad_theta,
            'method': 'L-BFGS-B', 
            'tol': tolerance
        }

        X_init = Xo.copy()
        for j in range(1, G):
            res = minimize(
                **optimization_params,
                x0=X_init[j, :],
                args=(
                    y[time_points == time, :],
                    vect_kon[time_points == time, j],
                    ko[j], X_init, cnt + cnt_init, penalization_strength,
                    sl, G, cnt_move[j], j, p
                )
            )

            if not res.success: 
                print(f'Warning, minimization time {time} failed')
            variations[cnt][j, :] += np.abs(res.x[:-1] - Xo[j, :-1])
            Xo[j, :] = res.x[:]
            theta_t[cnt, j, :] = Xo[j, :]

    return Xo, variations, theta_t

def inference_optim(
        time_points, times_unique, y, vect_kon, ko, sl, p, tolerance
):
    """
    Network inference procedure.
    Return the inferred network (basal + network) and the time at which 
    each edge has been detected with strongest intensity.
    """
    G = np.size(y, 1)
    inter = np.zeros((G, G))
    basal = np.zeros(G)
    inter_t = np.zeros((times_unique.size, G, G))
    basal_t = np.zeros((times_unique.size, G))
    variations = np.zeros((times_unique.size, G, G))
    time_variations = np.zeros((G, G))

    # penalites
    l = np.array([1, np.sqrt(G - 1)])  # l_inter, l_basal, l_diag

    Xo = np.zeros((G, G + 1))
    Xo, variations, theta_t = core_inference(
        variations,
        time_points,
        times_unique, 
        y, 
        vect_kon, 
        ko, 
        Xo, 
        l, 
        sl, 
        G, 
        0, 
        times_unique.size,
        p,
        tolerance
    )

    for i in range(0, G):
        for j in range(0, G):
            time_variations[i, j] = np.argmax(variations[1:, i, j])

    inter[:, :] = Xo[:, :-1]
    basal[:] = Xo[:, -1]
    inter_t[:, :, :] = theta_t[:, :, :-1]
    basal_t[:, :] = theta_t[:, :, -1]

    return basal, inter, time_variations, basal_t, inter_t

def core_optim(x, time_points, times_unique, n_genes_stim, sl, p, tolerance):
    """
    Fit the network model to the data.
    """

    # Initialization
    k_max = np.max(x, 0)
    ko = np.min(x, 0) / k_max
    y, vect_kon = x / k_max, x / k_max

    variations_tot = np.zeros((n_genes_stim, n_genes_stim))
    basal_tot = np.zeros(n_genes_stim)
    inter_tot = np.zeros((n_genes_stim, n_genes_stim))
    basal_tot_t = np.zeros((times_unique.size, n_genes_stim))
    inter_tot_t = np.zeros((times_unique.size, n_genes_stim, n_genes_stim))

    # Inference procedure
    basal, inter, variations_time, basal_t, inter_t = inference_optim(
        time_points, 
        times_unique, 
        y, 
        vect_kon, 
        ko,
        sl,
        p,
        tolerance
    )

    # Build the results
    cnt_i, cnt_j = 0, 0
    for i in range(0, n_genes_stim):
        cnt_j = 0
        for j in range(0, n_genes_stim):
            inter_tot[i, j] = inter[cnt_j, cnt_i]
            inter_tot_t[:, i, j] = inter_t[:, cnt_j, cnt_i]
            variations_tot[i, j] = variations_time[cnt_j, cnt_i]
            cnt_j += 1
        basal_tot[i] = basal[cnt_i]
        basal_tot_t[:, i] = basal_t[:, cnt_i]
        cnt_i += 1

    return basal_tot, inter_tot, variations_tot, basal_tot_t, inter_tot_t

_numba_functions = {
    False : {
        'penalization_l1': penalization_l1,
        'grad_penalization_l1': grad_penalization_l1,
        'penalization': penalization,
        'grad_penalization' : grad_penalization
    },
    True: None
}

class Cardamom(Inference):

    class Result(Inference.Result):
        def __init__(self, 
            parameter: NetworkParameter,
            variations: np.ndarray,
            basal_time: Dict[float, np.ndarray],
            interaction_time: Dict[float, np.ndarray],
            data_bool: np.ndarray
        ) -> None:
            super().__init__(
                parameter, 
                variations=variations, 
                basal_time=basal_time,
                interaction_time=interaction_time,
                data_bool=data_bool
            )

        @classmethod
        def load_txt(cls, path: Union[str, Path], load_extra: bool = False):
            if load_extra:
                path_extra = Path(path) / 'extra'
                return cls(
                    NetworkParameter.load_txt(path),
                    np.loadtxt(path_extra / 'variations.txt'), 
                    {
                        f.stem.split('_')[1]:np.loadtxt(f) 
                        for f in (path_extra / 'basal_time').iterdir()
                    }, 
                    {
                        f.stem.split('_')[1]:np.loadtxt(f) 
                        for f in (path_extra / 'interaction_time').iterdir()
                    }, 
                    np.loadtxt(path_extra / 'data_bool.txt')
                )
            else:
                return super().load_txt(path)
            
        @classmethod
        def load(cls, path: Union[str, Path], load_extra: bool = False):
            if load_extra:
                param = NetworkParameter.load(path)
                path = f'{Path(path).with_suffix("")}_extra'
                basal_time = {}
                inter_time = {}

                with np.load(path + '_basal_time.npz') as data:
                    for k, v in dict(data).items():
                        basal_time[float(k.split('_')[1])] = v
                with np.load(path + '_interaction_time.npz') as data:
                    for k, v in dict(data).items():
                        inter_time[float(k.split('_')[1])] = v
                with np.load(path + '_variations.npz') as data:
                    variations = data['variations']
                with np.load(path + '_data_bool.npz') as data:
                    data_bool = data['data_bool']
                
                return cls(
                    param, 
                    variations, 
                    basal_time, 
                    inter_time, 
                    data_bool
                )
            else:
                return super().load(path)

        def save_extra_txt(self, path: Union[str, Path]):
            path = Path(path) / 'extra'
            basal_time = {f't_{t}':v for t,v in self.basal_time.items()}
            inter_time = {f't_{t}':v for t,v in self.interaction_time.items()}

            save_dir(path / 'basal_time', basal_time)
            save_dir(path / 'interaction_time', inter_time)
            np.savetxt(path / 'variations.txt', self.variations)
            np.savetxt(path / 'data_bool.txt', self.data_bool)
        
        def save_extra(self, path: Union[str, Path]):
            path = f'{Path(path).with_suffix("")}_extra'
            basal_time = {f't_{t}':v for t,v in self.basal_time.items()}
            inter_time = {f't_{t}':v for t,v in self.interaction_time.items()}
            
            save_npz(path + '_basal_time', basal_time)
            save_npz(path + '_interaction_time', inter_time)
            save_npz(path + '_variations', {'variations': self.variations})
            save_npz(path + '_data_bool', {'data_bool': self.data_bool})

    def __init__(self, 
        threshold: float = 1e-3,
        pseudo_l1_coeff: float = 5e-3, # sl
        penalization: float = 0.4, # p
        tolerance: float = 1e-5,
        max_iteration: int = 1000,
        verbose: bool = False,
        use_numba: bool = True
    ) -> None:
        self.threshold: float = threshold
        self.pseudo_l1_coeff: float = pseudo_l1_coeff
        self.penalization: float = penalization
        self.tolerance: float = tolerance
        self.max_iteration: int = max_iteration
        self.verbose: bool = verbose
        self._use_numba = False
        self.use_numba: bool = use_numba

    @property
    def use_numba(self) -> bool:
        return self._use_numba
    
    @use_numba.setter
    def use_numba(self, use_numba: bool) -> None:
        global _numba_functions

        if self._use_numba != use_numba:
            if use_numba and _numba_functions[True] is None:
                from numba import njit
                _numba_functions[True] = {}
                for name, f in _numba_functions[False].items():
                    jited_f = njit()(f)
                    _numba_functions[True][name] = jited_f
                    globals()[name] = jited_f
            
            for fname, f in _numba_functions[use_numba].items():
                globals()[fname] = f
            
            self._use_numba = use_numba

    @property
    def directed(self) -> bool:
        return True
    
    def run(self, data: Dataset, param: NetworkParameter) -> Inference.Result:
        """
        Fit a network parameter to the data.
        Return the list of successive objective function values.
        """
        n_cells, n_genes_stim = data.count_matrix.shape
        times_unique = np.unique(data.time_points)

        # Get kinetic parameters
        data_bool = np.ones_like(data.count_matrix, dtype='float')
        data_bool[data.time_points == 0.0, 0] = 0.0
        weight = np.zeros((n_cells, n_genes_stim, 2))
        a = np.ones((3, n_genes_stim))
        for g in range(1, n_genes_stim):
            x = data.count_matrix[:, g]
            at, a[-1, g], k = infer_kinetics(
                x, 
                data.time_points,
                times_unique,
                self.tolerance,
                self.max_iteration
            )
            a[0, g] = max(np.min(at), self.threshold)
            a[1, g] = max(np.max(at), self.threshold)
            if self.verbose:
                print(f'Estimation done in {k} iterations') 
                print(f'Gene {g} calibrated...', a[:, g])
                

            core_basins_binary(
                x, 
                data_bool[:, g], 
                a[:-1, g], 
                a[-1, g], 
                weight[:, g, :]
            )
        
        # # Remove genes with too small variations
        # mask = np.ones(n_genes_stim, dtype='bool')
        # for g in range(1, n_genes_stim):
        #     mean_g = [
        #         np.mean(data.count_matrix[data.time_points == time, g]) 
        #         for time in times
        #     ]
        #     if np.max(mean_g) - np.min(mean_g) < 0.1:
        #         mask[g] = False

        # if self.verbose:
        #     print('number genes of interest', np.sum(mask))
        # for attr in ['a', 'd', 's', 'basal', 'interaction']:
        #     getattr(param, attr).mask[..., ~mask] = True

        basal, inter, variations, basal_t, inter_t = core_optim(
            data_bool, 
            data.time_points, 
            times_unique, 
            n_genes_stim,
            self.pseudo_l1_coeff,
            self.penalization,
            self.tolerance
        )

        param.burst_frequency_min[:] = a[0] * param.degradation_rna
        param.burst_frequency_max[:] = a[1] * param.degradation_rna
        param.burst_size_inv[:] = a[2]
        param.creation_rna[:] = param.degradation_rna * param.rna_scale()
        param.creation_protein[:] = param.degradation_protein * param.protein_scale()
        param.basal[:] = basal
        param.interaction[:] = inter

        if self.verbose:
            print('TOT', param.interaction, param.basal)

        basal_t = {t:v for t,v in zip(times_unique, basal_t)}
        inter_t = {t:v for t,v in zip(times_unique, inter_t)}

        return self.Result(param, variations, basal_t, inter_t, data_bool)
