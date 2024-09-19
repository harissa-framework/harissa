from harissa.inference.pearson import Pearson

def test_is_not_directed():
    inf = Pearson()

    assert not inf.directed
