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

@pytest.fixture(scope='module')
def time_points():
    return np.zeros(2)

@pytest.fixture(scope='module')
def rna_levels():
    return np.zeros((2, 2))

@pytest.fixture(scope='module')
def protein_levels():
    return np.zeros((2, 2))

@pytest.fixture(scope='module')
def simulation_res(time_points, rna_levels, protein_levels):
    return Simulation.Result(time_points, rna_levels, protein_levels)

@pytest.fixture(scope='module')
def npz_file(tmp_path_factory, simulation_res):
    path = tmp_path_factory.mktemp('simu_res') / 'simu_res.npz'
    np.savez_compressed(
        path,
        time_points=simulation_res.time_points,
        rna_levels=simulation_res.rna_levels,
        protein_levels=simulation_res.protein_levels
    )
    return path

@pytest.fixture(scope='module')
def txt_dir(tmp_path_factory, simulation_res):
    path = tmp_path_factory.mktemp('simu_res') / 'simu_res'
    path.mkdir()

    np.savetxt(path/'time_points.txt', simulation_res.time_points)
    np.savetxt(path/'rna_levels.txt', simulation_res.rna_levels)
    np.savetxt(path/'protein_levels.txt', simulation_res.protein_levels)
    
    return path

class TestSimulationResult:
    def test_init(self, time_points, rna_levels, protein_levels):
        
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

    def test_load(self, npz_file, time_points, rna_levels, protein_levels):
        res = Simulation.Result.load(npz_file)
        
        assert np.array_equal(res.time_points, time_points)
        assert np.array_equal(res.rna_levels, rna_levels)
        assert np.array_equal(res.protein_levels, protein_levels)

    def test_load_txt(self, txt_dir, time_points, rna_levels, protein_levels):
        res = Simulation.Result.load_txt(txt_dir)
        
        assert np.array_equal(res.time_points, time_points)
        assert np.array_equal(res.rna_levels, rna_levels)
        assert np.array_equal(res.protein_levels, protein_levels)

    def test_save(self, tmp_path, simulation_res):

        path = simulation_res.save(tmp_path / 'res.npz')
        data = np.load(path)

        assert np.array_equal(simulation_res.time_points, data['time_points'])
        assert np.array_equal(simulation_res.rna_levels, data['rna_levels'])
        assert np.array_equal(simulation_res.protein_levels, data['protein_levels'])

    def test_save_txt(self, tmp_path, simulation_res):
        
        path = simulation_res.save_txt(tmp_path / 'res')

        assert np.array_equal(
            simulation_res.time_points, 
            np.loadtxt(path/'time_points.txt', ndmin=1)
        )
        assert np.array_equal(
            simulation_res.rna_levels,
            np.loadtxt(path/'rna_levels.txt', ndmin=2)
        )
        assert np.array_equal(
            simulation_res.protein_levels, 
            np.loadtxt(path/'protein_levels.txt', ndmin=2)
        )

