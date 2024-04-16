"""
Main class for network inference
"""
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

def _create_penalization(penalization_l1):
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
    
    return penalization

penalization = _create_penalization(penalization_l1)

def _create_grad_penalization(grad_penalization_l1):
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
    
    return grad_penalization

grad_penalization = _create_penalization(grad_penalization_l1)
_grad_penalization_jit = None

def _create_objective(penalization):
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
    
    return objective

objective = _create_objective(penalization)

def _create_grad_theta(grad_penalization):
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

    return grad_theta

grad_theta = _create_grad_theta(grad_penalization)


def _create_core_inference(objective, grad_theta):
    def core_inference(
        variations, 
        times, 
        vect_t, 
        y, 
        vect_kon, 
        ko, 
        Xo, 
        l, 
        sl, 
        G, 
        cnt_init,
        cnt_end,
        p
    ):
        cnt_move = np.ones((G, G))
        theta_t = np.zeros((cnt_end - cnt_init, G, G + 1))
        for cnt, time in enumerate(times[cnt_init:cnt_end]):
            penalization_strength = l * np.sum(vect_t == time) * 2.5 / 100
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

            X_init = Xo.copy()
            for j in range(1, G):
                res = minimize(
                    objective, X_init[j, :],
                    args=(
                        y[vect_t == time, :],
                        vect_kon[vect_t == time, j],
                        ko[j], X_init, cnt + cnt_init, penalization_strength,
                        sl, G, cnt_move[j], j, p
                    ),
                    jac=grad_theta,
                    method='L-BFGS-B',
                    tol=1e-5
                )

                if not res.success: 
                    print(f'Warning, minimization time {time} failed')
                variations[cnt][j, :] += np.abs(res.x[:-1] - Xo[j, :-1])
                Xo[j, :] = res.x[:]
                theta_t[cnt, j, :] = Xo[j, :]

        return Xo, variations, theta_t

    return core_inference

core_inference = _create_core_inference(objective, grad_theta)

def _create_inference_optim(core_inference):
    def inference_optim(vect_t, times, y, vect_kon, ko, sl, p):
        """
        Network inference procedure.
        Return the inferred network (basal + network) and the time at which 
        each edge has been detected with strongest intensity.
        """
        G = np.size(y, 1)
        inter = np.zeros((G, G))
        basal = np.zeros(G)
        inter_t = np.zeros((len(times), G, G))
        basal_t = np.zeros((len(times), G))
        variations = np.zeros((len(times), G, G))
        time_variations = np.zeros((G, G))

        # penalites
        l = np.array([1, np.sqrt(G - 1)])  # l_inter, l_basal, l_diag

        Xo = np.zeros((G, G + 1))
        Xo, variations, theta_t = core_inference(
            variations, 
            times, 
            vect_t, 
            y, 
            vect_kon, 
            ko, 
            Xo, 
            l, 
            sl, 
            G, 
            0, 
            len(times),
            p
        )

        for i in range(0, G):
            for j in range(0, G):
                time_variations[i, j] = np.argmax(variations[1:, i, j])

        inter[:, :] = Xo[:, :-1]
        basal[:] = Xo[:, -1]
        inter_t[:, :, :] = theta_t[:, :, :-1]
        basal_t[:, :] = theta_t[:, :, -1]

        return basal, inter, time_variations, basal_t, inter_t

    return inference_optim

inference_optim = _create_inference_optim(core_inference)

def _create_core_optim(inference_optim):
    def core_optim(x, time_points, times_unique, n_genes_stim, sl, p):
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
            p
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
    
    return core_optim

core_optim = _create_core_optim(inference_optim)
_core_optim_jit = None

class Cardamom(Inference):
    def __init__(self, 
        threshold: float = 1e-3,
        verbose: bool = False,
        use_numba: bool = True
    ) -> None:
        self.threshold = threshold
        self.verbose = verbose
        # TODO Better names
        self.sl = 5e-3 # pseudo l1 coefficient
        self.p = 0.4

        self._use_numba, self._core_optim = False, core_optim
        self.use_numba: bool = use_numba

    @property
    def use_numba(self) -> bool:
        return self._use_numba
    
    @use_numba.setter
    def use_numba(self, use_numba: bool) -> None:
        global _core_optim_jit

        if self._use_numba != use_numba:
            if use_numba:
                if _core_optim_jit is None:
                    from numba import njit
                    p1_jit = njit()(penalization_l1)
                    grad_p1_jit = njit()(grad_penalization_l1)
                    penalization_jit = njit()(_create_penalization(p1_jit))
                    grad_penalization_jit = njit()(
                        _create_grad_penalization(grad_p1_jit)
                    )
                    objective_jit = _create_objective(penalization_jit)
                    grad_theta_jit= _create_grad_theta(grad_penalization_jit)
                    core_inference_jit = _create_core_inference(
                        objective_jit, 
                        grad_theta_jit
                    )
                    inference_optim_jit = _create_inference_optim(
                        core_inference_jit
                    )
                    _core_optim_jit = _create_core_optim(inference_optim_jit)
                self._core_optim = _core_optim_jit
            else:
                self._core_optim = core_optim
            
            self._use_numba = use_numba

    def run(self, data: Dataset) -> Inference.Result:
        """
        Fit a network parameter to the data.
        Return the list of successive objective function values.
        """
        n_cells, n_genes_stim = data.count_matrix.shape
        times = np.unique(data.time_points)

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
                times,
                1e-5,
                1000
            )
            a[0, g] = max(np.min(at), self.threshold)
            a[1, g] = max(np.max(at), self.threshold)
            if self.verbose: 
                print(f'Gene {g} calibrated...', a[:, g])
                print(f'Estimation done in {k} iterations')

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
          
        param = NetworkParameter(n_genes_stim - 1)
        # for attr in ['a', 'd', 's', 'basal', 'interaction']:
        #     getattr(param, attr).mask[..., ~mask] = True

        basal, inter, variations, basal_t, inter_t = self._core_optim(
            data_bool, 
            data.time_points, 
            times, 
            n_genes_stim,
            self.sl,
            self.p
        )

        param.a[:] = a
        param.basal[:] = basal
        param.interaction[:] = inter

        if self.verbose:
            print('TOT', param.interaction, param.basal)

        return Inference.Result(
            param, 
            variations=variations, 
            basal_t=basal_t,
            interaction_t=inter_t,
            data_bool=data_bool
        )


# class NetworkModel:
#     """
#     Handle networks within the package.
#     """
#     def __init__(self, n_genes=None, times=None):
#         # Kinetic parameters
#         self.d = None
#         # Mixture parameters
#         self.data_bool = None
#         # Optim parameters
#         self.n_train = None
#         # Network parameters
#         self.mask = None
#         self.a = None
#         self.basal = None
#         self.inter = None
#         self.basal_t = None
#         self.inter_t = None
#         self.variations = None
#         # Default behaviour
#         if n_genes is not None:
#             G = n_genes + 1 # Genes plus stimulus
#             # Default degradation rates
#             self.d = np.zeros((2,G))
#             self.d[0] = np.log(2)/9 # mRNA degradation rates
#             self.d[1] = np.log(2)/46 # protein degradation rates
#             # Default network parameters
#             self.basal = np.zeros(G)
#             self.inter = np.zeros((G,G))
#             self.variations = np.zeros((G, G))


#     def core_basins_binary(self, data, alph, b, g):
#         """
#         Compute the basal parameters of filtered genes.
#         """
#         for n, c in enumerate(data):
#             for z, k in enumerate(alph):
#                 self.weight[n, g, z] = log_gamma_poisson_pdf(c, k, b)
#             self.data_bool[n, g] = alph[np.argmax(self.weight[n, g, :])]


#     def core_optim(self, x, vect_t, times, G_tot):
#         """
#         Fit the network model to the data.
#         Return the list of successive objective function values.
#         """

#         # Initialization
#         k_max = np.max(x, 0)
#         ko = np.min(x, 0) / k_max
#         y, vect_kon = x / k_max, x / k_max

#         variations_tot = np.zeros((G_tot, G_tot))
#         basal_tot = np.zeros(G_tot)
#         inter_tot = np.zeros((G_tot, G_tot))
#         basal_tot_t = np.zeros((len(times), G_tot))
#         inter_tot_t = np.zeros((len(times), G_tot, G_tot))

#         # Inference procedure
#         basal, inter, variations_time, basal_t, inter_t = inference_optim(vect_t, times, y, vect_kon, ko)

#         # Build the results
#         cnt_i, cnt_j = 0, 0
#         for i in range(0, G_tot):
#             cnt_j = 0
#             if self.mask[i]:
#                 for j in range(0, G_tot):
#                     if self.mask[j]:
#                         inter_tot[i, j] = inter[cnt_j, cnt_i]
#                         inter_tot_t[:, i, j] = inter_t[:, cnt_j, cnt_i]
#                         variations_tot[i, j] = variations_time[cnt_j, cnt_i]
#                         cnt_j += 1
#                 basal_tot[i] = basal[cnt_i]
#                 basal_tot_t[:, i] = basal_t[:, cnt_i]
#                 cnt_i += 1

#         return basal_tot, inter_tot, variations_tot, basal_tot_t, inter_tot_t



#     def fit(self, data, verb=False):
#         """
#         Fit the network model to the data.
#         Return the list of successive objective function values.
#         """
#         C, G_tot = data.shape
#         vect_t = data[:, 0]
#         times = list(set(vect_t))
#         times.sort()

#         # Get kinetic parameters
#         seuil = 1e-3
#         self.data_bool = np.ones_like(data, dtype='float')
#         self.data_bool[vect_t == 0, 0] = 0
#         self.weight = np.zeros((C, G_tot, 2), dtype='float')
#         a = np.ones((3, G_tot))
#         for g in range(1, G_tot):
#             x = data[:, g]
#             at, a[-1, g] = infer_kinetics(x, vect_t, verb=verb)
#             a[0, g] = max(np.min(at), seuil)
#             a[1, g] = max(np.max(at), seuil)
#             if verb: print('Gene {} calibrated...'.format(g), a[:, g])
#             self.core_basins_binary(x, a[:-1, g], a[-1, g], g)
#         self.a = a
        
#         # Remove genes with too small variations
#         self.mask = np.ones(G_tot, dtype='bool')
#         for g in range(1, G_tot):
#             meang = [np.mean(data[vect_t == time, g]) for time in times]
#             if np.max(meang) - np.min(meang) < .1:
#                 self.mask[g] = 0

#         if verb: print('number genes of interest', np.sum(self.mask))
#         self.d = self.d[:, self.mask]
#         data_bool = self.data_bool[:, self.mask]

#         self.basal, self.inter, self.variations, self.basal_t, self.inter_t = \
#             self.core_optim(data_bool, data[:, 0], times, G_tot)

#         if verb: print('TOT', self.inter.T, self.basal)
