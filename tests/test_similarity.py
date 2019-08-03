import pytest
import random
from pepdist import similarity
from math import isclose


@pytest.fixture
def get_trie_data():
    return ["AAAA", "AYAY", "GAAG", "YYYY", "WWWW", "WWYW"]


@pytest.fixture
def get_matrix():
    return {("A", "B"): -1, ("B", "B"): 1, ("A", "A"): 1}


@pytest.fixture
def random_sample():
    sample_length = 1000
    peptides = []
    for i in range(sample_length):
        peptide = ''.join(random.choice('ACDEFGHIKLMNPQRSTVWY') for i in range(9))
        peptides.append(peptide)
    return peptides


def test_symmetrize(get_matrix):
    sym = similarity.symmetrize(get_matrix)
    assert ("B", "A") in sym
    assert sym[("B", "A")] == -1


def test_positivize(get_matrix):
    pos = similarity.positivize(get_matrix)
    for val in pos.values():
        assert val >= 0


def test_max_normalize(get_matrix):
    norm = similarity.max_normalize(get_matrix)
    for val in norm.values():
        assert val >= 0
        assert val <= 1


def test_trie_creation():
    trie = similarity.Trie()
    assert trie.root.char == ""
    assert trie.root.depth == 0
    assert not trie.root.word_finished
    assert trie.root.children == {}
    assert trie.alphabet == set()


def test_add_find_word(get_trie_data):
    trie = similarity.Trie()
    trie.add(get_trie_data)
    assert trie.alphabet == {"A", "Y", "G", "W"}
    for data in get_trie_data:
        assert trie.find_word(data)
        assert trie.get_prefix(data) == data
        assert trie.get_prefix(data[:2] + "XX") == data[:2]


def test_nearest_neighbour(get_trie_data):
    trie = similarity.Trie()
    trie.add(get_trie_data)
    for data in get_trie_data:
        assert trie.k_nearest_neighbour(data, similarity.blosum62) == [(data, 1.0)]


def test_k_nearest_neighbour(get_trie_data):
    trie = similarity.Trie()
    trie.add(get_trie_data)
    for data in get_trie_data:
        assert trie.k_nearest_neighbour(data, similarity.blosum62, k=1) == [(data, 1.0)]
        assert trie.k_nearest_neighbour(data, similarity.blosum62, k=2)[0] == (data, 1.0)
    assert len(trie.k_nearest_neighbour("AAAY", similarity.blosum62, k=3)) == 3
    assert trie.k_nearest_neighbour("WWYY", similarity.blosum62, k=2)[0][0] == "WWYW"
    assert trie.k_nearest_neighbour("WWYY", similarity.blosum62, k=2)[1][0] == "WWWW"


def test_random_nearest_neighbour(random_sample):
    trie = similarity.Trie()
    trie.add(random_sample)
    for data in random_sample:
        assert trie.k_nearest_neighbour(data, similarity.blosum62)[0][0] == data
    test_peptide = ''.join(random.choice('ACDEFGHIKLMNPQRSTVWY') for i in range(9))
    trie_solution = trie.k_nearest_neighbour(test_peptide, similarity.blosum62)
    naive_solution = similarity.naive_nearest_neighbour(random_sample, test_peptide, similarity.blosum62)
    assert isclose(trie_solution[0][1], naive_solution[1], rel_tol=0.001)


def test_kmer_trie_creation():
    lengths = [3, 4]
    trie = similarity.KmerTrie(lengths)
    assert trie.root.char == ""
    assert trie.root.depth == 0
    assert not trie.root.word_finished
    assert trie.root.children == {}
    assert trie.root.sequences == []
    assert trie.kmer_length == lengths
    assert trie.alphabet == set()

def test_add_find_word_kmertrie(get_trie_data):
    lengths = [3, 4]
    trie = similarity.KmerTrie(lengths)
    trie.add(get_trie_data)
    assert trie.alphabet == {"A", "Y", "G", "W"}
    for data in get_trie_data:
        assert trie.find_word(data)
        assert trie.get_prefix(data) == data
        assert trie.get_prefix(data[:2] + "XX") == data[:2]


def test_k_nearest_neighbour(get_trie_data):
    lengths = [3]
    trie = similarity.KmerTrie(lengths)
    trie.add(get_trie_data)
    for data in get_trie_data:
        assert trie.k_nearest_neighbour(data, similarity.blosum62, k=1) == [(data, 1.0,[data])]
        assert trie.k_nearest_neighbour(data, similarity.blosum62, k=2)[0] == (data, 1.0, [data])
    assert len(trie.k_nearest_neighbour("AAAY", similarity.blosum62, k=3)) == 3
    assert trie.k_nearest_neighbour("WWYY", similarity.blosum62, k=2)[0][0] == "WWY"
    assert trie.k_nearest_neighbour("WWYY", similarity.blosum62, k=2)[1][0] == "WWYW"

