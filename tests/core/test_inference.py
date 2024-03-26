import pytest
import numpy as np
from harissa.core import Inference, Dataset, NetworkParameter

class InferenceMissingRun(Inference):
    def __init__(self):
        ...

class InferenceSuperRun(Inference):
    def __init__(self):
        ...

    def run(self, data: Dataset) -> Inference.Result:
        return super().run(data)


class TestInference:
    def test_inference_instance(self):
        with pytest.raises(TypeError):
            Inference()

    def test_inference_missing_run(self):
        with pytest.raises(TypeError):
            InferenceMissingRun()
        
    def test_inference_super_run(self):
        inf = InferenceSuperRun()

        with pytest.raises(NotImplementedError):
            inf.run(Dataset(np.empty(1), np.empty((1, 2), dtype=np.uint)))

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