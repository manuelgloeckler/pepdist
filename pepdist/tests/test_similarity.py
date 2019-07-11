import pytest
import pepdist


@pytest.fixture
def get_trie_data():
    return ["AAAA", "AYAY", "GAAG", "YYYY", "WWWW", "WWYW"]


def test_trie_creation():
    trie = pepdist.Trie()
    assert trie.root.prefix == ""
    assert trie.root.depth == 0
    assert not trie.root.word_finished
    assert trie.root.children == {}


def test_add_find_word(get_trie_data):
    trie = pepdist.Trie()
    trie.add(get_trie_data)
    assert trie.alphabet == {"A", "Y", "G", "W"}
    for data in get_trie_data:
        assert trie.find_word(data)
        assert trie.get_prefix(data) == data
        assert trie.get_prefix(data[:2] + "XX") == data[:2]


def test_nearest_neighbour(get_trie_data):
    trie = pepdist.Trie()
    trie.add(get_trie_data)
    for data in get_trie_data:
        assert pepdist.nearest_neighbour(trie, data) == (data, 1.0)
    assert pepdist.nearest_neighbour(trie, "AAAY")[0] == "AYAY"
    assert pepdist.nearest_neighbour(trie, "WWYY")[0] == "WWYW"


def test_k_nearest_neighbour(get_trie_data):
    trie = pepdist.Trie()
    trie.add(get_trie_data)
    for data in get_trie_data:
        assert pepdist.k_nearest_neighbour(trie, data, k=1) == [(data, 1.0)]
        assert pepdist.k_nearest_neighbour(trie, data, k=2)[0] == (data, 1.0)
    assert len(pepdist.k_nearest_neighbour(trie, "AAAY", k=3)) == 3
    assert pepdist.k_nearest_neighbour(trie, "WWYY", k=2)[0][0] == "WWYW"
    assert pepdist.k_nearest_neighbour(trie, "WWYY", k=2)[1][0] == "WWWW"
