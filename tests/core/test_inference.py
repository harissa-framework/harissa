import pytest
import numpy as np
from json import dump
import sys
from harissa.core import Inference, Dataset, NetworkParameter
from harissa.benchmark.generators.networks import bn8

@pytest.fixture(scope='module')
def network_parameter():
    return bn8()

@pytest.fixture(scope='module')
def inf_error_json(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('inference')
    sys_path = sys.path
    sys.path.append(str(tmp_path))

    inf_path = tmp_path / 'my_inference.py'
    path = tmp_path / 'inf_error.json'

    with open(inf_path, 'w') as fp:
        fp.write('class MyInference:\n\tpass')

    with open(path, 'w') as fp:
        dump({
            'classname': 'MyInference',
            'module' : f'{inf_path.stem}',
            'kwargs': {}
        }, fp)

    yield path

    sys.path = sys_path

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

@pytest.fixture(scope='module')
def res_json(tmp_path_factory, network_parameter):
    path = tmp_path_factory.mktemp('inference_result') / 'result.json'
    network_parameter.save_json(path)

    return path

class MyInference:
    pass

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

    def test_load_not_Inference(self, inf_error_json):

        with pytest.raises(RuntimeError):
            Inference.load_json(inf_error_json)

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

    def test_load_json(self, res_json, network_parameter):
        res = Inference.Result.load_json(res_json)

        assert res.parameter == network_parameter

    def test_save(self, tmp_path):
        res = Inference.Result(NetworkParameter(2))

        path = res.save(tmp_path / 'res.npz')
        param = NetworkParameter.load(path)

        assert res.parameter == param

        res = Inference.Result(NetworkParameter(2), foo='bar')

        path = res.save(tmp_path / 'res.npz', save_extra=True)
        param = NetworkParameter.load(path)

        assert res.parameter == param

    def test_save_txt(self, tmp_path):
        res = Inference.Result(NetworkParameter(2))

        path = res.save_txt(tmp_path / 'res')
        param = NetworkParameter.load_txt(path)

        assert res.parameter == param

        res = Inference.Result(NetworkParameter(2), foo='bar')

        path = res.save_txt(tmp_path / 'res', save_extra=True)
        param = NetworkParameter.load_txt(path)

        assert res.parameter == param

    def test_save_json(self, tmp_path):
        res = Inference.Result(NetworkParameter(2))

        path = res.save_json(tmp_path / 'res.json')
        param = NetworkParameter.load_json(path)

        assert res.parameter == param

        res = Inference.Result(NetworkParameter(2), foo='bar')

        path = res.save_json(tmp_path / 'res.json', save_extra=True)
        param = NetworkParameter.load_json(path)

        assert res.parameter == param
