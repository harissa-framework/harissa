import pytest
import numpy as np
from dataclasses import is_dataclass
from harissa.core import NetworkParameter, Simulation

class SimulationMissingRun(Simulation):
    def __init__(self):
        ...

class SimulationSuperRun(Simulation):
    def __init__(self):
        ...

    def run(self,
            time_points: np.ndarray,  
            initial_state: np.ndarray, 
            parameter: NetworkParameter) -> Simulation.Result:
        return super().run(time_points, initial_state, parameter)

class TestSimulation:
    def test_instance(self):
        with pytest.raises(TypeError):
            Simulation()

    def test_missing_run(self):
        with pytest.raises(TypeError):
            SimulationMissingRun()

        
    def test_super_run(self):
        sim = SimulationSuperRun()

        with pytest.raises(NotImplementedError):
            sim.run(np.empty(1), np.empty((1, 2)), None)

class TestSimulationResult:
    def test_init(self):
        time_points = np.zeros(1)
        rna_levels = np.zeros((1, 2))
        protein_levels = np.zeros((1, 2))
        
        res = Simulation.Result(time_points, rna_levels, protein_levels)

        for param_name in Simulation.Result.param_names: 
                assert(hasattr(res, param_name))

        assert is_dataclass(res)
        assert np.array_equal(res.time_points, time_points)
        assert np.array_equal(res.rna_levels, rna_levels)
        assert np.array_equal(res.protein_levels, protein_levels)

        assert res.time_points.shape[0] == res.rna_levels.shape[0]
        assert res.time_points.shape[0] == res.protein_levels.shape[0]
        assert res.rna_levels.shape == res.protein_levels.shape


    @pytest.mark.parametrize('times_shape,rna_shape,protein_shape',[
        ((1,),  (1,2),  (2,2)),
        ((1,),  (2,2),  (1,2)),
        ((1,),  (1,2),  (1,3)),
        ((1,),  (1,3),  (1,2)),
        ((1,2), (1,2),  (1,2)),
        ((1,2), (1,),   (1,2)),
        ((1,2), (1,2),  (1,))
    ])
    def test_init_wrong_shape(self, times_shape, rna_shape, protein_shape):
        with pytest.raises(TypeError):
            Simulation.Result(
                np.zeros(times_shape), 
                np.zeros(rna_shape), 
                np.zeros(protein_shape)
            )

