import pytest
import numpy as np
from harissa.core import Inference, Dataset, NetworkParameter
from harissa.benchmark.generators.networks import bn8

@pytest.fixture(scope='module')
def network_parameter():
    return bn8()

@pytest.fixture(scope='module')
def res_npz(tmp_path_factory, network_parameter):
    path = tmp_path_factory.mktemp('inference_result') / 'result.npz'
    network_parameter.save(path)

    return path

@pytest.fixture(scope='module')
def res_txt(tmp_path_factory, network_parameter):
    path = tmp_path_factory.mktemp('inference_result') / 'result'
    network_parameter.save_txt(path)

    return path


class InferenceMissing(Inference):
    pass

class InferenceSuper(Inference):
    @property
    def directed(self):
        return super().directed

    def run(self, data: Dataset, param: NetworkParameter) -> Inference.Result:
        return super().run(data, param)


class TestInference:
    def test_inference_instance(self):
        with pytest.raises(TypeError):
            Inference()

    def test_inference_missing_run(self):
        with pytest.raises(TypeError):
            InferenceMissing()

    def test_inference_missing_directed(self):
        with pytest.raises(TypeError):
            InferenceMissing()

    def test_inference_super_run(self):
        inf = InferenceSuper()
        net = NetworkParameter(1)
        dataset = Dataset(np.empty(1), np.empty((1, 2), dtype=np.uint))

        with pytest.raises(NotImplementedError):
            inf.run(dataset, net)

    def test_inference_super_directed(self):
        inf = InferenceSuper()
        with pytest.raises(NotImplementedError):
            inf.directed

class TestInferenceResult:
    def test_init(self):
        param = NetworkParameter(1)
        res = Inference.Result(param)

        assert res.parameter == param

        res = Inference.Result(param, foo='bar')

        assert res.parameter == param
        assert res.foo == 'bar'

    def test_init_wrong_type(self):
        with pytest.raises(TypeError):
            Inference.Result(1)

    def test_load(self, res_npz, network_parameter):
        res = Inference.Result.load(res_npz)

        assert res.parameter == network_parameter

    def test_load_txt(self, res_txt, network_parameter):
        res = Inference.Result.load_txt(res_txt)

        assert res.parameter == network_parameter

    def test_save(self, tmp_path):
        res = Inference.Result(NetworkParameter(1))

        path = res.save(tmp_path / 'res.npz')
        param = NetworkParameter.load(path)

        assert res.parameter == param

        res = Inference.Result(NetworkParameter(1), foo='bar')

        path = res.save(tmp_path / 'res.npz', save_extra=True)
        param = NetworkParameter.load(path)

        assert res.parameter == param

    def test_save_txt(self, tmp_path):
        res = Inference.Result(NetworkParameter(1))

        path = res.save_txt(tmp_path / 'res')
        param = NetworkParameter.load_txt(path)

        assert res.parameter == param

        res = Inference.Result(NetworkParameter(1), foo='bar')

        path = res.save_txt(tmp_path / 'res', save_extra=True)
        param = NetworkParameter.load_txt(path)

        assert res.parameter == param
