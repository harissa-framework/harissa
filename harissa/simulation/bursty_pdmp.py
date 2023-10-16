"""
Perform simulations using the PDMP model
"""
from .simulation import Simulation
from ..utils.math import kon, kon_bound, flow
import numpy as np

class BurstyPDMP(Simulation):
    """
    Bursty PDMP version of the network model (promoters not described)
    """

    def __init__(self, M0=None, P0=None, burnin=None, 
                 thin_adapt=True, verbose=False):
        self.M0 : np.ndarray | None = M0
        self.P0 : np.ndarray | None = P0
        self.burn_in : float | None = burnin
        self.state : np.ndarray | None = None
        self.thin_adapt : bool  = thin_adapt
        self.is_verbose : bool  = verbose
    
    def run(self, 
            time_points: np.ndarray, 
            burst_frequency_min: np.ndarray, 
            burst_frequency_max: np.ndarray, 
            burst_size: np.ndarray, 
            degradation_rna: np.ndarray, 
            degradation_protein: np.ndarray,
            basal: np.ndarray, 
            interaction: np.ndarray) -> Simulation.Result:
        """
        Perform simulation of the network model (bursty PDMP version).
        """
        k0 = burst_frequency_min * degradation_rna
        k1 = burst_frequency_max * degradation_rna

        # Normalize protein scales
        s1 = degradation_protein * burst_size / burst_frequency_max

        # Thinning parameter
        tau = np.sum(k1[1:])

        nb_genes = basal.size

        # Default state
        types = [('M','float'), ('P','float')]
        self.state = np.array([(0,0) for i in range(nb_genes)], dtype=types)
        # state  = np.zeros((2, nb_genes))

        # Burnin simulation without stimulus
        if self.M0 is not None:
            self.state['M'][1:] = self.M0[1:]
        if self.P0 is not None: 
            self.state['P'][1:] = self.P0[1:]
        if self.burn_in is not None: 
            self._simulation(time_points=np.array([self.burn_in]),
                             basal=basal,
                             inter=interaction,
                             d0=degradation_rna,
                             d1=degradation_protein,
                             s1=s1,
                             k0=k0,
                             k1=k1,
                             b=burst_size,
                             tau=tau)
        
        # Activate the stimulus
        self.state['P'][0] = 1
        # Final simulation with stimulus
        res = self._simulation(time_points=time_points,
                               basal=basal,
                               inter=interaction,
                               d0=degradation_rna,
                               d1=degradation_protein,
                               s1=s1,
                               k0=k0,
                               k1=k1,
                               b=burst_size,
                               tau=tau)
        
        return Simulation.Result(time_points, res['M'], res['P'])
        # return Simulation.Result(time_points, res[:, 0], res[:, 1])

    def _simulation(self, time_points, basal, inter, 
                    d0, d1, s1, k0, k1, b, tau):
        """
        Exact simulation of the network in the bursty PDMP case.
        """
        types = [('M','float'), ('P','float')]
        nb_genes = basal.size
        # states = np.empty((time_points.size, 2, nb_genes - 1))
        sim = [] # List of states to be recorded
        phantom_jump_count, true_jump_count = 0, 0
        t = 0
        # Core loop for simulation
        for time_point in time_points:
            # Recording
            M, P = flow(time_point - t, self.state, d0, d1, s1)
            sim += [np.array([(M[i],P[i]) for i in range(1, nb_genes)], 
                             dtype=types)]
            # states[time_point, 0] = M[1:]
            # states[time_point, 1] = P[1:]

            while t < time_point:
                U, jump = self._step(basal, inter, d0, d1, s1, k0, k1, b, tau)
                t += U
                if jump:
                    true_jump_count += 1
                else: 
                    phantom_jump_count += 1

        # Display info about jumps
        if self.is_verbose:
            total_jump = phantom_jump_count + true_jump_count
            if total_jump > 0:
                print(f'Exact simulation used {total_jump} jumps ' 
                      f'including {phantom_jump_count} phantom jumps ' 
                      f'({100*phantom_jump_count/total_jump : .2f}%)')
            else: 
                print('Exact simulation used no jump')

        return np.array(sim)

    def _step(self, basal, inter, d0, d1, s1, k0, k1, b, tau):
        """
        Compute the next jump and the next step of the
        thinning method, in the case of the bursty model.
        """
        if self.thin_adapt:
            # Adaptive thinning parameter
            tau = np.sum(kon_bound(self.state, basal, inter, 
                                   d0, d1, s1, k0, k1))
        jump = False # Test if the jump is a true or phantom jump
        
        # 0. Draw the waiting time before the next jump
        U = np.random.exponential(scale=1/tau)
        
        # 1. Update the continuous states
        self.state['M'], self.state['P'] = flow(U, self.state, d0, d1, s1)
        
        # 2. Compute the next jump
        G = basal.size # Genes plus stimulus
        v = kon(self.state['P'], k0, k1) / tau # i = 1, ..., G-1 : burst of mRNA i
        v[0] = 1 - np.sum(v[1:]) # i = 0 : no change (phantom jump)
        i = np.random.choice(G, p=v)
        if i > 0:
            r = b[i]
            self.state['M'][i] += np.random.exponential(1/r)
            jump = True
        
        return U, jump
        