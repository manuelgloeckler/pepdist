""" This module contains a locally sensitiv hashing class for fast nearest neighbour computation.
Additionaly it contains functions to represent a peptide/sequence as a feature vector 
based on AAindex or other metrics.
"""

import numpy as np
from scipy.stats import norm
from scipy.spatial import KDTree
import time
import _pickle
import gc


def naive_nearest_neighbour(data, inp_vec):
    """ Naive nearest neighbour search """
    min_dist = np.inf
    min_match = ""

    for seq in data:
        # All rows represent a index, the mean of all index row distances is the overall
        # distance for one sequence.
        dist = np.linalg.norm(seq.descriptor - inp_vec)
        if dist < min_dist:
            min_dist = dist
            min_match = seq.seq

    return min_match, min_dist


class descriptor:
    """ Amino acid descriptor designed to describe peptides with the AAIndex.

    Parameters
    ----------
    seqs
        List of sequences that should be described
    indices
        List of indices, for example AAIndex indices, that should describe the sequences.
    norm_method
        A function that should normalize the values. Default value is no normalization.

    Attributes
    ----------
    sequences : list
        List of sequences contained.
    descriptors : list
        List of arrays that describe the sequences.
    """

    def __init__(self, seqs: list, indices: list = [], norm_method=lambda x: x):
        self.sequences = seqs
        self.descriptors = []
        self.indices = indices
        self.normalized_indices = []
        self.norm_method = norm_method

        self.normalize_all()
        self.calculate_all()

    def __len__(self):
        return len(self.sequences)

    def add_seqs(self, seqs):
        """ Add sequences to the descriptor. They will be automatically described by..."""
        if seqs is isinstance(str):
            self.sequences.append(seqs)
            self.descriptors.append(self.translate(seqs))
        else:
            self.sequences.extend(seqs)
            self.sequences.extend(list(map(lambda x: self.translate(x, self.normalized_indices), seqs)))

    def add_index(self, index):
        """ Add another index that should also describe the sequence. The descriptors are automatically updated."""
        self.indices.append(index)
        self.normalized_indices.append(self.norm_method(index))
        self.calculate_all()

    def change_norm_method(self, norm_method):
        """" Change to another normalization method, the descriptors are automatically updated."""
        self.norm_method = norm_method
        self.normalize_all()
        self.calculate_all()

    def normalize_all(self):
        for i in range(len(self.indices)):
            if i > len(self.descriptors)-1:
                self.normalized_indices.append(self.norm_method(self.indices[i]))
            else:
                self.normalized_indices[i] = self.norm_method(self.indices[i])

    def calculate_all(self):
        for i in range(len(self.sequences)):
            if i > len(self.descriptors)-1:
                self.descriptors.append(self.translate(self.sequences[i], self.normalized_indices))
            else:
                self.descriptors[i] = self.translate(self.sequences[i], self.normalized_indices)

    @ staticmethod
    def translate(self, word: str, indices: list):
        """ Translate a string to feature vectors
            Parameters
            ----------
            word:
                A sequence as string
            indices
                A List of dictionary which represent a aaindex or another descriptor!
            Returns
            -------
            numpy array
                Concatenated translation of the word by all indices.
        """
        vec = []
        for index in indices:
            for char in word:
                vec.append(index[char])
        return np.array(vec)


class HashTable:
    """Hash Table Implementation for random projection binning

    N given input vectors are projected by k independent gausian normal distributed
    projection vectors and than placed in quantization bin's of widht w.

    Parameters
    ----------
    k_dot_products : int
        k independet hash functions are used for binning.
    bin_widht : float
        Width of a quantization bin.
    inp_dimension : int
        Dimension of hashable input vectors.

    Attributes
    ----------
    k_dot_products : int
        Hashfunction uses a projection of k dot products.
    bin_widht : float
        Width of a quantization bin.
    b : array(float)
        Uniformly distributed random variabales in [0,bin_width]
    inp_dimension : int
        Dimension of hashable input vectors.
    hash_table : dict
        Dictionary which is used as hash table.
    projections : ndarray
        Normal distributed N(0,1) projection matrix.
    """

    def __init__(self, inp_dimensions: int, k_dot_products: int, bin_width: float):
        self.k_dot_products = k_dot_products
        self.bin_width = bin_width
        self.b = np.random.uniform(low=0, high=bin_width, size=k_dot_products)
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = np.random.randn(
            self.k_dot_products, self.inp_dimensions)

    def generate_hash(self, inp_vec):
        return tuple(np.floor(
            (np.dot(inp_vec, self.projections.T) + self.b) / self.bin_width))

    def __setitem__(self, inp_vec, label):
        hash_value = self.generate_hash(inp_vec)
        self.hash_table[hash_value] = self.hash_table \
                                          .get(hash_value, list()) + [label]

    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, [])

    def values(self):
        return self.hash_table.values()

    def keys(self):
        return self.hash_table.keys()



class LSH:
    """Locally sensitv hashing

    Attributes
    ----------
    num_projections : int
        Number of projection done for each binning
    k_dot_products : int
        Hashfunction uses a projection of k dot products.
    bin_width : float
        Width of a quantization bin.
    inp_dimensions :int
        Dimension of hashable input vectors.
    hash_tables : list
        List of Hashtable, which represents the num_projections times
        created hash tables

    """

    def __init__(
            self,
            inp_dimensions: int,
            num_projections: int = 10,
            k_dot_products: int = 3,
            bin_width: float = None):
        self.num_projections = num_projections
        self.k_dot_products = k_dot_products
        if bin_width is None:
            self.bin_width = inp_dimensions/2
        else:
            self.bin_width = bin_width
        self.inp_dimensions = inp_dimensions
        self.hash_tables = list()
        for i in range(self.num_projections):
            self.hash_tables.append(
                HashTable(
                    self.inp_dimensions,
                    self.k_dot_products,
                    self.bin_width))

    def __setitem__(self, inp_vec, label):
        for table in self.hash_tables:
            table[inp_vec] = (inp_vec, label)

    def __getitem__(self, inp_vec):
        results = list()
        for table in self.hash_tables:
            results.extend(table[inp_vec])
        return list(results)

    def add(self, inp_vecs, labels):
        if len(inp_vecs) != len(labels):
            raise SystemExit("Length of descriptors and labels is not the same!")
        else:
            for i in range(len(inp_vecs)):
                self.__setitem__(inp_vecs[i], labels[i])

    def nearest_neighbour(self, inp_vec):
        """ Find the nearest_neighbour by hashing the inp_vec and find the nearest neighbour in the
            corresponding bin."""
        hash_bin = self.__getitem__(inp_vec)
        if not hash_bin:
            hash_bin = [item for sublist in self.hash_tables[0].hash_table.values() for item in sublist]
        min_dist = np.inf
        min_match = ""
        for seq in hash_bin:
            dist = np.linalg.norm(seq[0] - inp_vec)
            if dist < min_dist:
                min_dist = dist
                min_match = seq[1]

        return (min_match, min_dist)

    @ staticmethod
    def p1(self, bin_width):
        """ Probability that two points with ||p-q|| < bin_width Fall into the same bucket """
        return 1 - 2 * norm.cdf(-bin_width) - \
               2 / (np.sqrt(2 * np.pi) * bin_width) * \
               (1 - np.exp(-(bin_width ** 2 / 2)))

    @ staticmethod
    def p2(self, bin_width, scale=2):
        """ Probability that two points ||p-q|| > scale * bin_width Fall into the same bucket."""
        return 1 - 2 * norm.cdf(-bin_width / scale) - \
               2 / (np.sqrt(2 * np.pi) * bin_width / scale) * \
               (1 - np.exp(-(bin_width ** 2 / (2 * scale ** 2))))

    @ staticmethod
    def compute_num_projections(self, bin_width, k_dot_products, fail_prob=0.1):
        """ L needed to have a probability smaller than fail_prob that nearest neighbor is not in the same bucket. """
        return int(np.ceil(np.log(fail_prob) / np.log(1 - self.p1(bin_width) ** k_dot_products)))

    def mean_buckets(self):
        """ Returns the mean number of buckets in each hashtable """
        buckets = 0
        for hash_table in self.hash_tables:
            buckets += len(hash_table.values())
        return int(buckets / self.num_projections)

    def mean_bucket_size(self):
        """ Returns the mean number of elements in a bucket. """
        bucket_size = 0
        for hash_table in self.hash_tables:
            hash_bucket_sizes = 0
            num_buckets = len(list(hash_table.values()))
            for buckets in list(hash_table.values()):
                hash_bucket_sizes += len(buckets)
            bucket_size += hash_bucket_sizes / num_buckets
        return int(bucket_size / self.num_projections)

    def train_LSH(self, data, query_sample, tolerance=1.5, maxk=10, fail_prob=0.1):
        kdtree = KDTree(data)
        best = -np.inf
        for q in query_sample:
            dist = kdtree.query(q)[0]
            if dist > best:
                best = dist

        bin_widht = best + tolerance
        best_k = 1
        best_L = 1
        best_time = np.inf
        for k in range(1, maxk):
            L = self.compute_num_projections(bin_widht, k, fail_prob=fail_prob)
            self.__init__(self.inp_dimensions, num_projections=L, k_dot_products=k, bin_width=bin_widht)
            self.add(data, data)
            times = []
            for sample in query_sample:
                start_time = time.time()
                self.nearest_neighbour(sample)
                times.append(time.time()-start_time)
            if np.mean(times) < best_time:
                best_k = k
                best_L = L

        self.__init__(self.inp_dimensions, num_projections=best_L, k_dot_products=best_k, bin_width=bin_widht)

    def save_lsh(self, path: str):
        """ Saves the LSH structure at the given path. The path also defines the file name."""
        file = open(path, "wb")
        _pickle.dump(self, file, protocol=-1)
        file.close()

    @ staticmethod
    def load_lsh(self, path: str):
        """ Load's a Trie structure from the given path."""
        # Garbage Collector slows down the loading significant and is
        # therefore excluded.
        gc.disable()
        file = open(path, "rb")
        lsh = _pickle.load(file)
        file.close()
        gc.enable()
        return lsh

