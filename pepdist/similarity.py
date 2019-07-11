import numpy as np
from multiprocess import Pool
from pepdist import blosum62
import warnings
import gc
import _pickle as cPickle


class Trie():
    """ A trie/prefix-tree data structure.

        A prefix tree is a tree, where strings can be added
        and each node represent a prefix of this string.

    Attributes
    ----------
    root : TrieNode
        root of the Trie
    alphabet : Set
        Set of the characters used in the Trie.

    """

    def __init__(self):
        self.root = TrieNode("", 0)
        self.alphabet = set()

    def add(self, words: list):
        """ Adds given words into the Trie """
        for word in words:
            node = self.root
            for i in range(len(word)):
                char = word[i]
                # Adds new letters to alphabet
                if char not in self.alphabet:
                    self.alphabet.add(char)
                if char not in node.children:
                    new_node = TrieNode(char, i+1)
                    node.children[char] = new_node
                    node = new_node
                else:
                    node = node.children[char]
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
            score: dict = blosum62,
            k=1,
            normed = True,
            weights = None):
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
        root = self.root
        word_length = len(word)
        results = []

        # Check for equal alphabet
        score_alphabet = set()
        for key in score:
            key1, key2 = key
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
                    
        # Set the weights if not specified
        if weights == None:
            weights = [1]*(word_length+1)

        # Compute minimum match score
        min_score = np.inf
        for k1, k2 in score.keys():
            if k1 == k2 and not k1 == "X":
                s = score[(k1, k2)]
                if s < min_score:
                    min_score = s
                   
        min_scores = list(map(lambda x: sum(min_score  * np.array(weights[x+1:])), list(range(word_length))))




        # Score of the query word
        self_score = [0] * (word_length + 1)
        for i in range(word_length):
            self_score[i + 1] = self_score[i] + \
                weights[word_length - 1 - i] * score[word[word_length - 1 - i], word[word_length - 1 - i]]

        bounds = [-np.inf] * k

        for i in range(k):

            # Trie Search for equal strings is fast
            if self.find_word(word) and all(v == 1 for v in weights):
                results.append((word, 1.0))
                continue

            bound = bounds.pop()
            best = ""
            nodes = list(root.children.items())
            prefix = [""] * (word_length)
            sc = [0] * (word_length + 1)
            s = [0] * (word_length + 1)

            while not nodes == []:
                char, node = nodes.pop()
                length = node.depth
                index = length - 1
                
                if length > word_length:
                    # The word is to long
                    continue
                
                prefix[index] = char
                sc[length] = sc[index] + weights[index] * score[(char, word[index])]
                s[length] = s[index] + weights[index] * score[(char, char)]
                
                if (sc[length] + self_score[word_length - length])**2 / ((s[length] +
                                                                          min_scores[index]) * self_score[word_length]) < bound:
                    continue
                if length == word_length and node.word_finished:
                    # Already found
                    if "".join(prefix) in list(map(lambda x: x[0], results)):
                        continue

                    if sc[length] < 0:
                        scc = -sc[length]**2 / \
                            (s[length] * self_score[word_length])
                    else:
                        scc = sc[length]**2 / (s[length] * self_score[word_length])

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
        
    def compute_neighbours(self, words, cpus = 2):
        pool = Pool(cpus)
        result = pool.map(self.k_nearest_neighbour, peptides)
        pool.close()
        pool.join()
            
        return result
            
    def save_trie(self, path):
        file = open(path, "wb")
        cPickle.dump(self, file, protocol=-1)
        file.close()
            
    def load_trie(self, path):
        gc.disable()
        file = open(path, "rb")
        trie = cPickle.load(file)
        file.close()
        self.root = trie.root
        self.alphabet = trie.alphabet
        gc.enable()
        
      
def load_trie(path):
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

    def __init__(self, char: str, depth:int):
        self.char = char
        self.depth = depth
        self.children = {}
        self.word_finished = False

    def __str__(self):
        return self.char
