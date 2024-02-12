import pytest
import numpy as np
from harissa import NetworkParameter, Simulation

def test_simulation_instance():
    with pytest.raises(TypeError):
        Simulation()

class SimulationMissingRun(Simulation):
    def __init__(self):
        ...

def test_simulation_missing_run():
    with pytest.raises(TypeError):
        SimulationMissingRun()

class SimulationSuperRun(Simulation):
    def __init__(self):
        ...

    def run(self, 
            initial_state: np.ndarray, 
            time_points: np.ndarray, 
            parameter: NetworkParameter) -> Simulation.Result:
        return super().run(initial_state,time_points, parameter)
    
def test_simulation_super_run():
    sim = SimulationSuperRun()

    with pytest.raises(NotImplementedError):
        sim.run(np.empty(1), np.empty(1), None)