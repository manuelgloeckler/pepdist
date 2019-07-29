import pytest
import pepdist
import random
import numpy as np



def get_random_index():
    aa = pepdist.Aaindex1()
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
        
def test_seq(get_test_data):
    for word in get_test_data:
        randvec = np.random.randn(len(word))
        seq = pepdist.Seq(word, randvec)
        assert seq.seq == word
        assert np.array_equal(seq.descriptor,randvec)
        
def test_hash_table(get_test_data):
    pass
 
        

