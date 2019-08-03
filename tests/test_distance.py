import pytest
from pepdist import distance
import random
import numpy as np



def get_random_index():
    aa = distance.Aaindex()
    aalist = list(aa.aaindex_dic.values())
    return aalist[random.randint(0, len(aalist)) - 1]


@pytest.fixture
def get_test_data():
    return ["AAAA", "AYAY", "GAAG", "YYYY", "WWWW", "WWYW"]


@pytest.mark.parametrize("x", [get_random_index() for i in range(10)])
def test_get_index(x):
    for key, val in x.items():
        assert key in {"A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"}
        assert isinstance(val, float)
        assert not val == "NA"

def test_descriptor(get_test_data):
    dis = distance.descriptor(get_test_data, [get_random_index() for i in range(3)])
    assert dis.sequences == get_test_data
    assert len(dis.descriptors) == len(get_test_data)
    assert len(dis.descriptors[0]) == 12
    dis.change_norm_method(distance.z_normalize)
    dis.change_norm_method(distance.max_normalize)
    for desc in dis.descriptors:
        assert (desc <= 1).all() and (desc >= 0).all()

        
def test_hash_table(get_test_data):
    pass
 
        

