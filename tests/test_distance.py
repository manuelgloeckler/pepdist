import pytest
from pepdist import distance
import random
import numpy as np
from math import isclose



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

def test_positivize():
    pos_index = distance.positivize(get_random_index())
    for val in pos_index.values():
        assert val >= 0
    assert len(pos_index) == len(get_random_index())

def test_max_normalize():
    norm_index = distance.max_normalize(get_random_index())
    for val in norm_index.values():
        assert val >= 0
        assert val <= 1
    assert len(norm_index) == len(get_random_index())

def test_z_normalize():
    norm_index = distance.z_normalize(get_random_index())
    assert len(norm_index) == len(get_random_index())


def test_descriptor(get_test_data):
    indices = [get_random_index() for i in range(3)]
    dis = distance.IndexDescriptor(get_test_data, indices)
    assert dis.sequences == get_test_data
    assert len(dis.descriptors) == len(get_test_data)
    assert len(dis.descriptors[0]) == 12
    dis.change_norm_method(distance.z_normalize)
    dis.change_norm_method(distance.max_normalize)
    for desc in dis.descriptors:
        assert (desc <= 1).all() and (desc >= 0).all()
    dis.change_norm_method(lambda x: x)
    dis.add_seqs(["GGGG"])
    assert "GGGG" in dis.sequences
    dis.remove_seqs(["GGGG"])
    assert "GGGG" not in dis.sequences


def test_naive_nearest_neighbour(get_test_data):
    dis = distance.IndexDescriptor(get_test_data, [get_random_index()])
    for i in range(len(get_test_data)):
        assert distance.naive_nearest_neighbour(dis.sequences, dis.descriptors, dis.descriptors[i])[1] == 0.0
        
def test_hash_table(get_test_data):
    table = distance.HashTable(2,2,2)
    assert table.k_dot_products == 2
    assert table.inp_dimensions == 2
    assert table.bin_width == 2
    table[np.asarray([1,2])] = "AA"
    assert len(table.keys()) == 1
    assert len(table.values()) == 1

def test_LSH(get_test_data):
    lsh = distance.LSH(4)
    dis = distance.IndexDescriptor(get_test_data, [get_random_index()])
    lsh.add(dis.sequences, dis.descriptors)
    for i in range(len(get_test_data)):
        assert lsh.nearest_neighbour(dis.descriptors[i])[1] == 0.0
 
        

