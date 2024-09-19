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
            stimulus: np.ndarray,
            parameter: NetworkParameter) -> Simulation.Result:
        return super().run(time_points, initial_state, stimulus, parameter)

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
            sim.run(np.empty(1), np.empty((1, 2)), np.empty(1), None)

@pytest.fixture(scope='module')
def time_points():
    return np.arange(2, dtype=np.float64)

@pytest.fixture(scope='module')
def rna_levels():
    return np.zeros((2, 2))

@pytest.fixture(scope='module')
def protein_levels():
    return np.ones((2, 2))

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
        assert res.rna_levels.shape == res.protein_levels.shape

        assert np.array_equal(res.stimulus_levels, protein_levels[:, 0])
        assert np.array_equal(
            res.final_state,
            np.vstack((rna_levels[-1], protein_levels[-1]))
        )


    @pytest.mark.parametrize('times,rna,protein',[
        ([2.5], np.zeros((1,2)),  np.zeros((1,2))),
        (np.zeros(1), [0.5, 1.8], np.zeros((1,2))),
        (np.zeros(1), np.zeros((1,2)),  [0.5, 1.8]),
    ])
    def test_init_wrong_type(self, times, rna, protein):
        with pytest.raises(TypeError):
            Simulation.Result(times, rna, protein)

    @pytest.mark.parametrize('times_shape,rna_shape,protein_shape',[
        ((1,2), (1,2),  (1,2)),
        ((1,), (1,),   (1,2)),
        ((1,), (1,2),  (1,)),
        ((1,), (1,2),  (1,3,4))
    ])
    def test_init_wrong_dim(self, times_shape, rna_shape, protein_shape):
        with pytest.raises(TypeError):
            Simulation.Result(
                np.zeros(times_shape),
                np.zeros(rna_shape),
                np.zeros(protein_shape)
            )

    @pytest.mark.parametrize('times_shape,rna_shape,protein_shape',[
        ((1,),  (1,2),  (2,2)),
        ((1,),  (2,2),  (1,2)),
        ((1,),  (1,2),  (1,3)),
        ((1,),  (1,3),  (1,2))
    ])
    def test_init_wrong_shape(self, times_shape, rna_shape, protein_shape):
        with pytest.raises(ValueError):
            Simulation.Result(
                np.zeros(times_shape),
                np.zeros(rna_shape),
                np.zeros(protein_shape)
            )

    @pytest.mark.parametrize('times', [
        np.array([0.0, 2.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 2.0, 1.0])
    ])
    def test_init_times_not_unique(self, times):
        n_genes_stim = 2
        with pytest.raises(ValueError):
            Simulation.Result(
                times,
                np.zeros((times.size, n_genes_stim)),
                np.zeros((times.size, n_genes_stim))
            )

    def test_add(self, simulation_res):
        rna_levels = np.array([[0.0,2.0],[2.0,2.0]])
        protein_levels = np.array([[1.0,3.0],[3.0,3.0]])
        time_points = (simulation_res.time_points[-1]
                       + np.arange(1, 3, dtype=np.float64))

        sim = (simulation_res 
               + Simulation.Result(time_points, rna_levels, protein_levels))

        for key in ['time_points', 'rna_levels', 'protein_levels']:
            s_arr = getattr(sim, key)
            s_res_arr = getattr(simulation_res, key)
            l_arr = locals()[key]

            assert s_arr.size == s_res_arr.size + l_arr.size
            assert np.array_equal(s_arr[:s_res_arr.shape[0]], s_res_arr)
            assert np.array_equal(s_arr[s_res_arr.shape[0]:], l_arr)

    @pytest.mark.parametrize('sim', [1, 3.0, 'foo'])
    def test_add_wrong_type(self, simulation_res, sim):
        with pytest.raises(NotImplementedError):
            simulation_res + sim

    @pytest.mark.parametrize('sim', [
        Simulation.Result(
            np.arange(-1, 1, dtype=np.float64),
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ),
        Simulation.Result(
            np.arange(3, 5, dtype=np.float64),
            np.zeros((2, 3)),
            np.zeros((2, 3))
        ),
        None
    ])
    def test_add_wrong_values(self, simulation_res, sim):
        if sim is None:
            sim = simulation_res

        with pytest.raises(ValueError):
            simulation_res + sim
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
