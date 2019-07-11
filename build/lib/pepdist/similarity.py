import numpy as np
from multiprocess import Pool
from .matrix_score import blosum62
import warnings


class Trie():
    """ A trie/prefix-tree data structure.

        A prefix tree is a tree, where strings can be added
        and each node represent a prefix of this string.

    Attributes
    ----------
    root : TrieNode
        Description of `attr1`.
        alphabet : Set
                Set of the characters used in the Trie.

    """

    def __init__(self):
        self.root = TrieNode("")
        self.alphabet = set()

    def add(self, words: list):
        """ Adds given words into the Trie """
        for word in words:
            node = self.root
            for i in range(len(word)):
                                # Adds new letters to alphabet
                if word[i] not in self.alphabet:
                    self.alphabet.add(word[i])
                prefix = word[:i + 1]
                if prefix not in node.children:
                    new_node = TrieNode(prefix)
                    node.children[prefix] = new_node
                    node = new_node
                else:
                    node = node.children[prefix]
            node.word_finished = True

    def find_word(self, word: str) -> bool:
        """ Returns true if the word is in the Trie """
        node = self.root
        for i in range(len(word)):
            prefix = word[:i + 1]
            if prefix in node.children:
                node = node.children[prefix]
            else:
                                # The word is not present in the Trie
                return False
        return node.word_finished

    def get_prefix(self, word: str) -> str:
        """ Returns the common prefix of the word with the Trie """
        node = self.root
        for i in range(len(word)):
            prefix = word[:i + 1]
            if prefix in node.children:
                node = node.children[prefix]
            else:
                                # The rest of the word is not present in the
                                # Trie
                return prefix[:i]
        return word


class TrieNode(object):
    """ A node of a Trie

    Attributes
    ----------
    prefix : str
                Prefix, which is represented by this node
    depth : int
                Depth of the node, which represents the length of a
                prefix.
    children : dict
                Dictionary of children TrieNodes.
    word_finished : bool
                End of a word, represent's a leaf in the Tree
    """

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.depth = len(prefix)
        self.children = {}
        self.word_finished = False

    def __str__(self):
        return self.prefix


def nearest_neighbour(trie: Trie, word: str, score: dict = blosum62, maxlen=9):
    """ Computes the nearest neighbour of a given strings

    Attributes
    ----------
    trie : Trie
                The Trie which represent the dataset to search
    word : str
                The query word
    score (dict):
                Scoring matrix, standard is blosum62 substitution matrix
    maxlen : int
                Length of the corresponding strings

    Returns
    -------
    tuple : (str, double)
                Where the string represents the nearest neighbour
        and the double the corresponding score.

    """

    root = trie.root

    # Check for equal alphabet
    score_alphabet = set()
    for k in score:
        k1, k2 = k
        score_alphabet.add(k1)
        score_alphabet.add(k2)

    if not trie.alphabet.issubset(score_alphabet):
        warnings.warn(
            "The Scoring matrix don't has the same alphabet as the Trie. Characters that are not mapped are scored with 0. The following substitution is problematic:" +
            str(
                trie.alphabet.symmetric_difference(score_alphabet)))

    word_alphabet = set(list(word))
    if not word_alphabet.issubset(score_alphabet):
        warnings.warn(
            "The query word don't has the same alphabet as the Scoring Matrix. Chacters that are not mapped are scored with 0. The following chars are problematic: " +
            str(
                word_alphabet.symmetric_difference(score_alphabet)))

    # Trie Search for equal strings is fast
    if trie.find_word(word):
        return (word, 1.0)

    # Compute minimum match score
    min_score = np.inf
    for k1, k2 in score.keys():
        if k1 == k2 and not k1 == "X":
            s = score.get((k1, k2), 0)
            if s < min_score:
                min_score = s

    word_length = len(word)
    bound = -np.inf
    best = ""
    nodes = list(root.children.items())
    sc = [0] * (maxlen + 1)
    s = [0] * (maxlen + 1)

    # Score of the query word
    self_score = [0] * (word_length + 1)
    for i in range(word_length):
        self_score[i + 1] = self_score[i] + \
            score.get((word[word_length - 1 - i], word[word_length - 1 - i]), 0)

    while not nodes == []:
        k, node = nodes.pop()
        length = node.depth
        index = length - 1
        sc[length] = sc[index] + score.get((k[index], word[index]), 0)
        s[length] = s[index] + score.get((k[index], k[index]), 0)
        # Bounding
        if (sc[length] + self_score[word_length - length])**2 / ((s[length] +
                                                                  min_score * (word_length - length)) * self_score[word_length]) <= bound:
            continue
        # Check if the bound gets better
        if node.word_finished:
            if sc[length] < 0:
                scc = -sc[length]**2 / (s[length] * self_score[word_length])
            else:
                scc = sc[length]**2 / (s[length] * self_score[word_length])
            if scc > bound:
                bound = scc
                best = k
                continue
        # If not bounded or leaf than check all children.
        nodes.extend(node.children.items())
    return (best, np.sqrt(bound))


def k_nearest_neighbour(
        trie: Trie,
        word: str,
        score: dict = blosum62,
        maxlen=9,
        k=5):
    """ Computes the nearest neighbour of a given strings

    Attributes
    ----------
    trie : Trie
                The Trie which represent the dataset to search
    word : str
                The query word
    score (dict):
                Scoring matrix, standard is blosum62 substitution matrix
    maxlen : int
                Length of the corresponding strings
        k : int
                Number of nearest neighbours to find

    Returns
    -------
    list : [(str, double)]
                Ordered list of k near neighbours, represented as tuples of
                strings, corresponding to their sequence, and doubles,
                corresponding to their score.

    """
    root = trie.root
    results = []

    # Check for equal alphabet
    score_alphabet = set()
    for key in score:
        key1, key2 = key
        score_alphabet.add(key1)
        score_alphabet.add(key2)

    print(score_alphabet)

    if not trie.alphabet.issubset(score_alphabet):
        warnings.warn(
            "The Scoring matrix don't has the same alphabet as the Trie. Characters that are not mapped are scored with 0. The following substitution is problematic:" +
            str(
                trie.alphabet.symmetric_difference(score_alphabet)))

    word_alphabet = set(list(word))
    if not word_alphabet.issubset(score_alphabet):
        warnings.warn(
            "The query word don't has the same alphabet as the Scoring Matrix. Chacters that are not mapped are scored with 0. The following chars are problematic: " +
            str(
                word_alphabet.symmetric_difference(score_alphabet)))

    # Compute minimum match score
    min_score = np.inf
    for k1, k2 in score.keys():
        if k1 == k2 and not k1 == "X":
            s = score[(k1, k2)]
            if s < min_score:
                min_score = s
    word_length = len(word)

    # Score of the query word
    self_score = [0] * (word_length + 1)
    for i in range(word_length):
        self_score[i + 1] = self_score[i] + \
            score[word[word_length - 1 - i], word[word_length - 1 - i]]

    bounds = [-np.inf] * k

    for i in range(k):

        # Trie Search for equal strings is fast
        if trie.find_word(word):
            results.append((word, 1.0))
            continue

        bound = bounds.pop()
        best = ""
        nodes = list(root.children.items())
        sc = [0] * (maxlen + 1)
        s = [0] * (maxlen + 1)

        while not nodes == []:
            k, node = nodes.pop()
            length = node.depth
            index = length - 1
            sc[length] = sc[index] + score[(k[index], word[index])]
            s[length] = s[index] + score[(k[index], k[index])]
            if (sc[length] + self_score[word_length - length])**2 / ((s[length] +
                                                                      min_score * (word_length - length)) * self_score[word_length]) < bound:
                continue
            if node.word_finished:
                # Already found
                if k in list(map(lambda x: x[0], results)):
                    continue

                if sc[length] < 0:
                    scc = -sc[length]**2 / \
                        (s[length] * self_score[word_length])
                else:
                    scc = sc[length]**2 / (s[length] * self_score[word_length])

                if scc >= bound:
                    bound = scc
                    bounds.append(scc)
                    best = k
                    continue
            nodes.extend(node.children.items())

        results.append((best, np.sqrt(bounds.pop())))
    return results


def compute_neighbours(trie, data, func=nearest_neighbour, cpus=4):
    """ Multithreaded nearest neighbour search for big datasets

    Attributes
    ----------
    trie : Trie
        The Trie which represent the dataset to search
    data : list
        The list of query words
    func : function
        The nearest neighbour method
    cpus : int
        Number of cpus

    Returns
    -------
    list : [[(str, double)]]
        List of the resulting nearest neighbour search.

    """
    def func(x): return nearest_neighbour(trie, x)
    pool = Pool(cpus)
    results = pool.map(func, data)
    pool.close()
    pool.join()
    return results
