import numpy as np
from multiprocess import Pool
import warnings
import gc
import _pickle as cPickle
from Bio.SubsMat import MatrixInfo
import copy


def word_score(word1: str, word2: str, score: dict) -> float:
    """ Computes the score between two words by the given scoring matrix"""
    sc = 0
    for i in range(min(len(word1), len(word2))):
        key = (word1[i], word2[i])
        sc += score[key]
    return sc


def squared_root_similarity(word1: str, word2: str, score: dict) -> float:
    """ Computes the squared root normalized score of thwo words"""
    bl_ab = word_score(word1, word2, score=score)
    bl_aa = word_score(word1, word1, score=score)
    bl_bb = word_score(word2, word2, score=score)

    return bl_ab / np.sqrt(bl_aa * bl_bb)


def naive_nearest_neighbour(data, word, score):
    """ Computes the nearest neighbor in a naive fashion"""
    max_match = ""
    max_score = -np.inf
    for seq in data:
        score = squared_root_similarity(word, seq, score)
        if score > max_score:
            max_score = score
            max_match = seq

    return max_match, max_score


class Trie(object):
    """ A trie/prefix-tree data structure.

        A prefix tree is a tree, where strings can be added
        and each node represent a prefix of this string.

    Attributes
    ----------
    root : TrieNode
        root of the Trie
    alphabet : Set
        Set of the characters used in the Trie.
    lengths : Set
        Set of word lengths saved in the Trie.

    """

    def __init__(self, data=None):
        self.root = TrieNode("", 0)
        self.alphabet = set()
        self.lengths = set()
        if data is not None:
            self.add(data)

    def add(self, words: list):
        """ Adds given words into the Trie """
        for word in words:
            self.lengths.add(len(word))
            node = self.root
            for i in range(len(word)):
                char = word[i]
                # Adds new letters to alphabet
                if char not in self.alphabet:
                    self.alphabet.add(char)
                if char not in node.children:
                    new_node = TrieNode(char, i + 1)
                    new_node.add_maxdepth(len(word))
                    node.children[char] = new_node
                    node = new_node
                else:
                    new_node = node.children[char]
                    new_node.add_maxdepth(len(word))
                    node = new_node
            node.word_finished = True

    def find_word(self, word: str) -> bool:
        """ Returns true if the word is in the Trie """
        node = self.root
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                # The word is not present in the Trie
                return False
        return node.word_finished

    def get_prefix(self, word: str) -> str:
        """ Returns the common prefix of the word with the Trie """
        node = self.root
        prefix = ""
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                # The rest of the word is not present in the
                # Trie
                return prefix
            prefix += char
        return prefix

    def k_nearest_neighbour(
            self,
            word: str,
            score: dict,
            k: int = 1,
            weights: list = None) -> list:
        """ Computes the nearest neighbour of a given strings

        Attributes
        ----------
        word : str
                    The query word
        score (dict):
                    Scoring matrix, standard is blosum62 substitution matrix
        k : int
                    Number of nearest neighbours to find
        weights : list
                    list of integers, that weight the position of the corresponding word.

        Returns
        -------
        list : [(str, double)]
                    Ordered list of k near neighbours, represented as tuples of
                    strings, corresponding to their sequence, and doubles,
                    corresponding to their score.

        """
        root = self.root
        word_length = len(word)
        results = []

        self._check_scoring_matrix(score)

        # Set the weights if not specified
        if weights is None:
            weights = [1] * (word_length + 1)

        # Compute minimum match score
        min_score = np.inf
        for k1, k2 in score.keys():
            if k1 == k2 and not k1 == "X":
                s = score[(k1, k2)]
                if s < min_score:
                    min_score = s

        min_scores = list(map(lambda x: sum(min_score * np.array(weights[x + 1:])), list(range(word_length))))

        # Score of the query word
        self_score = [0] * (word_length + 1)
        for i in range(word_length):
            self_score[i + 1] = self_score[i] + \
                                weights[word_length - 1 - i] * score[
                                    word[word_length - 1 - i], word[word_length - 1 - i]]

        bounds = [-np.inf] * k

        for i in range(k):

            # Trie Search for equal strings is fast
            if self.find_word(word) and all(v == 1 for v in weights):
                results.append((word, 1.0))
                continue

            bound = bounds.pop()
            best = ""
            nodes = list(root.children.items())
            prefix = [""] * word_length
            sc = [0] * (word_length + 1)
            s = [0] * (word_length + 1)

            while not nodes == []:
                char, node = nodes.pop()
                length = node.depth
                index = length - 1

                if length > word_length or word_length not in node.maxdepth:
                    # The word is to long or no equal length words are in this branch
                    continue

                prefix[index] = char
                sc[length] = sc[index] + weights[index] * score[(char, word[index])]
                s[length] = s[index] + weights[index] * score[(char, char)]

                if (sc[length] + self_score[word_length - length]) ** 2 / ((s[length] +
                                                                            min_scores[index]) * self_score[
                                                                               word_length]) < bound:
                    continue
                if length == word_length and node.word_finished:
                    # Already found
                    if "".join(prefix) in list(map(lambda x: x[0], results)):
                        continue

                    if sc[length] < 0:
                        scc = -sc[length] ** 2 / \
                              (s[length] * self_score[word_length])
                    else:
                        scc = sc[length] ** 2 / (s[length] * self_score[word_length])

                    if scc >= bound:
                        bound = scc
                        bounds.append(scc)
                        best = "".join(prefix)
                        continue
                nodes.extend(node.children.items())

            if bound > 0:
                results.append((best, np.sqrt(bounds.pop())))
            else:
                if bound == -np.inf:
                    results.append(("", -1))
                else:
                    results.append((best, -np.sqrt(abs(bounds.pop()))))
        return results

    """    
    def k_nearest_subwords(
            self,
            word: str,
            score: dict = blosum62,
            k=1):
            
        spaced_seeds_leq = [[1]*len(word)]
        for length in [7,8,9]:
            if length < len(word):
                for i in range(len(word)-length+1):
                    seed_start = [0]*i
                    seed_mid = [1]*length
                    seed_end = [0]*(len(word)-length-i)
                    seed_mid.extend(seed_end)
                    spaced_seeds_leq.append(seed_start.extend(seed_mid))
            if length > len(word):
                pass
        results = []      
        for spaced_seed in spaced_seeds_leq:
            results.extend(self.k_nearest_neighbour(word, score=score, k=k, weights=spaced_seed))
            
        return results
        """

    def compute_neighbours(self, words: str, score: dict, k: int = 1, weights: list = None, cpu: int = 2) -> list:
        """ Computes nearest neighbours in a multiprocessing way. """
        pool = Pool(cpu)
        result = pool.map(lambda x: self.k_nearest_neighbour(x, score, k, weights), words)
        pool.close()
        pool.join()

        return result

    def save_trie(self, path: str):
        """ Saves the Trie structere"""
        file = open(path, "wb")
        cPickle.dump(self, file, protocol=-1)
        file.close()

    def load_trie(self, path: str):
        """ Loades a Trie structure"""
        # Garbage Collector slows down the loading significant and is
        # therefore excluded.
        gc.disable()
        file = open(path, "rb")
        trie = cPickle.load(file)
        file.close()
        self.root = trie.root
        self.alphabet = trie.alphabet
        self.lengths = trie.lengths
        gc.enable()

    def _check_scoring_matrix(self, score: dict):
        pass

    """
    # TODO fertig machen!
        score_alphabet = set()
        for key1, key2 in score:
            if not (key2, key1) in score:
                raise ValueError("The scoring matrix have to be symmetric, this is not the case for: "
            score_alphabet.add(key1)
            score_alphabet.add(key2)

        if not self.alphabet.issubset(score_alphabet): warnings.warn( "The Scoring matrix don't has the same alphabet 
        as the Trie. Characters that are not mapped are scored with 0. The following substitution is problematic:" + 
        str( self.alphabet.symmetric_difference(score_alphabet))) 

        word_alphabet = set(list(word)) if not word_alphabet.issubset(score_alphabet): warnings.warn( "The query word 
        don't has the same alphabet as the Scoring Matrix. Chacters that are not mapped are scored with 0. The 
        following chars are problematic: " + str( word_alphabet.symmetric_difference(score_alphabet))) """


def load_trie(path: str) -> Trie:
    """ Loads a trie file and returns a Trie object."""
    gc.disable()
    file = open(path, "rb")
    trie = cPickle.load(file)
    file.close()
    gc.enable()

    return trie


class TrieNode(object):
    """ A node of a Trie

    Attributes
    ----------
    char : str
                Prefix, which is represented by this node
    depth : int
                Depth of the node, which represents the length of a
                prefix.
    children : dict
                Dictionary of children TrieNodes.
    word_finished : bool
                End of a word, represent's a leaf in the Tree
    """

    def __init__(self, char: str, depth: int):
        self.char = char
        self.depth = depth
        self.children = {}
        self.maxdepth = set()
        self.word_finished = False

    def add_maxdepth(self, depth: int):
        self.maxdepth.add(depth)

    def __str__(self):
        return self.char


def kmer_count(k, word, count=False):
    """ Creates all possible unique k-mer in a given word"""
    f = {}
    for x in range(len(word) + 1 - k):
        kmer = word[x:x + k]
        f[kmer] = f.get(kmer, 0) + 1
    if not count:
        return list(f.keys())
    else:
        return f


class KmerTrie(Trie):
    """ A trie/prefix-tree data structure.

        A prefix tree is a tree, where strings can be added
        and each node represent a prefix of this string.

    Attributes
    ----------
    root : TrieNode
        root of the Trie
    alphabet : Set
        Set of the characters used in the Trie.
    lengths : Set
        Set of word lengths saved in the Trie.

    """

    def __init__(self, kmer_length):
        super().__init__()
        self.root = KmerTrieNode("", 0)
        self.alphabet = set()
        self.lengths = set()
        self.kmer_length = kmer_length

    def add(self, words: list):
        """ Adds given words into the Trie """
        for word in words:
            self.lengths.add(len(word))
            kmers = [word]
            for k in self.kmer_length:
                kmers.extend(kmer_count(k, word))

            for kmer in kmers:
                node = self.root
                for i in range(len(kmer)):
                    char = kmer[i]
                    # Adds new letters to alphabet
                    if char not in self.alphabet:
                        self.alphabet.add(char)
                    if char not in node.children:
                        new_node = KmerTrieNode(char, i + 1)
                        new_node.add_maxdepth(len(kmer))
                        node.children[char] = new_node
                        node = new_node
                    else:
                        new_node = node.children[char]
                        new_node.add_maxdepth(len(kmer))
                        node = new_node
                node.word_finished = True
                node.sequences.append(word)

    def k_nearest_neighbour(self, word: str, score: dict, k=1, **kwargs):
        """ Computes the nearest neighbour of a given strings

        Attributes
        ----------
        word : str
                    The query word
        score (dict):
                    Scoring matrix, standard is blosum62 substitution matrix
        k : int
                    Number of nearest neighbours to find

        Returns
        -------
        list : [(str, double)]
                    Ordered list of k near neighbours, represented as tuples of
                    strings, corresponding to their sequence, and doubles,
                    corresponding to their score.
        """
        root = self.root
        results = []

        kmers = [word]
        for k_len in self.kmer_length:
            if k_len < len(word):
                kmers.extend(kmer_count(k_len, word))
        kmers = list(set(kmers))

        self._check_scoring_matrix(score)

        # Compute minimum match score
        min_score = np.inf
        for k1, k2 in score.keys():
            if k1 == k2 and not k1 == "X":
                s = score[(k1, k2)]
                if s < min_score:
                    min_score = s
        temp_results = []
        for kmer in kmers:
            word_length = len(kmer)
            weights = [1] * word_length

            min_scores = list(map(lambda x: sum(min_score * np.array(weights[x + 1:])), list(range(word_length))))

            # Score of the query word
            self_score = [0] * (word_length + 1)
            for i in range(word_length):
                self_score[i + 1] = self_score[i] + \
                                    weights[word_length - 1 - i] * score[
                                        kmer[word_length - 1 - i], kmer[word_length - 1 - i]]

            # Trie Search for equal strings is fast
            if self.find_word(kmer) and all(v == 1 for v in weights):
                temp_results.append((word, 1.0, self.__get_last_node(word).sequences))
                continue

            bounds = [-np.inf] * k

            for i in range(k):

                bound = bounds.pop()
                best = ""
                best_word_origin = []
                nodes = list(root.children.items())
                prefix = [""] * word_length
                sc = [0] * (word_length + 1)
                s = [0] * (word_length + 1)

                while not nodes == []:
                    char, node = nodes.pop()
                    length = node.depth
                    index = length - 1

                    if length > word_length or word_length not in node.maxdepth:
                        # The word is to long or no equal length words are in this branch
                        continue

                    prefix[index] = char
                    sc[length] = sc[index] + weights[index] * score[(char, kmer[index])]
                    s[length] = s[index] + weights[index] * score[(char, char)]

                    if (sc[length] + self_score[word_length - length]) ** 2 / ((s[length] +
                                                                                min_scores[index]) * self_score[
                                                                                   word_length]) < bound:
                        continue
                    if length == word_length and node.word_finished:
                        # Already found
                        if "".join(prefix) in list(map(lambda x: x[0], temp_results)):
                            for temp_result in temp_results:
                                if temp_result[0] == "".join(prefix):
                                    temp_result[2].extend([seq for seq in node.sequences if seq not in temp_result[2]])
                            continue

                        if sc[length] < 0:
                            scc = -sc[length] ** 2 / \
                                  (s[length] * self_score[word_length])
                        else:
                            scc = sc[length] ** 2 / (s[length] * self_score[word_length])

                        if scc >= bound:
                            bound = scc
                            bounds.append(scc)
                            best = "".join(prefix)
                            # here have to be a copy
                            best_word_origin = copy.deepcopy(node.sequences)
                            continue
                    nodes.extend(node.children.items())

                if bound > 0:
                    temp_results.append((best, np.sqrt(bounds.pop()), best_word_origin))
                else:
                    if bound == -np.inf:
                        temp_results.append(("", -1, []))
                    else:
                        temp_results.append((best, -np.sqrt(abs(bounds.pop())), best_word_origin))
        for i in range(k):
            if not temp_results:
                results.append(('', -1, []))
                continue
            max_score = max(temp_results, key=lambda item: item[1])

            # for res in temp_results:
            # if res[0] == max_score[0]:
            # for seq in res[2]:
            #    if seq not in max_score[2]:
            #        max_score[2].append(seq)
            # temp_results.remove(res)

            results.append(max_score)
            temp_results.remove(max_score)

        return results

    def compute_neighbours(self, words: list, score: dict, k: int = 1, cpu: int = 2, **kwargs) -> list:
        """ Computes nearest neighbours in a multiprocessing way.
        :param **kwargs:
        """
        pool = Pool(cpu)
        result = pool.map(lambda x: self.k_nearest_neighbour(x, score, k, ), words)
        pool.close()
        pool.join()

        return result

    def load_trie(self, path):
        """ Loades a Trie structure"""
        # Garbage Collector slows down the loading significant and is
        # therefore excluded.
        gc.disable()
        file = open(path, "rb")
        trie = cPickle.load(file)
        file.close()
        self.root = trie.root
        self.alphabet = trie.alphabet
        self.lengths = trie.lengths
        self.kmer_length = trie.kmer_length
        gc.enable()

    def __get_last_node(self, word: str):
        """ Returns true if the word is in the Trie """
        node = self.root
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                # The word is not present in the Trie
                return KmerTrieNode("", 0)
        return node

    def _check_scoring_matrix(self, score: dict):
        pass

    """
    # TODO fertig machen!
        score_alphabet = set()
        for key1, key2 in score:
            if not (key2, key1) in score:
                raise ValueError("The scoring matrix have to be symmetric, this is not the case for: "
            score_alphabet.add(key1)
            score_alphabet.add(key2)

        if not self.alphabet.issubset(score_alphabet):
            warnings.warn(
                "The Scoring matrix don't has the same alphabet as the Trie. Characters that are not mapped are scored with 0. The following substitution is problematic:" +
                str(
                    self.alphabet.symmetric_difference(score_alphabet)))

        word_alphabet = set(list(word))
        if not word_alphabet.issubset(score_alphabet):
            warnings.warn(
                "The query word don't has the same alphabet as the Scoring Matrix. Chacters that are not mapped are scored with 0. The following chars are problematic: " +
                str(
                    word_alphabet.symmetric_difference(score_alphabet)))
    """


class KmerTrieNode(TrieNode):
    """ A node of a Trie

    Attributes
    ----------
    char : str
                Prefix, which is represented by this node
    depth : int
                Depth of the node, which represents the length of a
                prefix.
    children : dict
                Dictionary of children TrieNodes.
    word_finished : bool
                End of a word, represent's a leaf in the Tree
    """

    def __init__(self, char: str, depth: int):
        super().__init__(char, depth)
        self.sequences = []
