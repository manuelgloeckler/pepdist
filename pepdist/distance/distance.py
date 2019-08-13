""" This module contains a locally sensitiv hashing class for fast nearest neighbour computation.
Additionaly it contains functions to represent a peptide/sequence as a feature vector 
based on AAindex or other metrics.
"""

import numpy as np
from scipy.stats import norm
from scipy.spatial import cKDTree
import time
import _pickle
import gc


def naive_nearest_neighbour(labels: list, descriptors: list, query: np.array)-> tuple:
    """ Naive nearest neighbour search

    Parameters
    ----------
    labels
        List of sequences represented for example as list of strings.
    descriptors
        List of descriptors represented as list of numpy arrays.
    query
        A numpy array that represents the descriptor for a given sequence.

    Returns
    -------
    (min_dist, min_match)
        A tuple containing the minimum distance and the corresponding label.

    Raises
    ------
    ValueError:
        The label list and descriptor list must have the same length.

    """
    min_dist = np.inf
    min_match = ""

    if len(labels) != len(descriptors):
        raise ValueError("labels and vectors have to be the same length! And ordered in the same way.")

    for i in range(len(labels)):
        # All rows represent a index, the mean of all index row distances is the overall
        # distance for one sequence.
        dist = np.linalg.norm(descriptors[i] - query)
        if dist < min_dist:
            min_dist = dist
            min_match = labels[i]

    return min_match, min_dist


class IndexDescriptor:
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
    indices : list
        List of indices used to describe the sequences.

    Examples
    --------
    >>> from pepdist import distance
    >>> seq = ["AAGG", "WYWW"]
    >>> aa = distance.Aaindex()
    >>> desc = distance.IndexDescriptor(seq, [aa['GEIM800103']])
    >>> desc.sequences
    ['AAGG', 'WYWW']
    >>> desc.descriptors
    [array([1.55, 1.55, 0.59, 0.59]), array([1.86, 1.08, 1.86, 1.86])]
    """

    def __init__(self, seqs: list, indices: list = [], norm_method=lambda x: x):
        self.sequences = seqs
        self.descriptor = []
        self.indices = indices
        self.__normalized_indices = []
        self.__norm_method = norm_method

        self.__normalize_all()
        self.calculate_all()

    def __len__(self):
        return len(self.sequences)

    def add_seqs(self, seqs):
        """ Add sequences to the descriptor. They will be automatically described by..."""
        if isinstance(seqs, str):
            self.sequences.append(seqs)
            self.descriptor.append(self.translate(seqs, self.__normalized_indices))
        else:
            self.sequences.extend(seqs)
            self.descriptor.extend(list(map(lambda x: self.translate(x, self.__normalized_indices), seqs)))

    def remove_seqs(self, seqs: list):
        """ Removes a sequence of multiple sequences from the descriptor class. """
        # If string is the input...
        if isinstance(seqs, str):
            index = self.sequences.index(seqs)
            del self.sequences[index]
            del self.descriptor[index]
        else:
            for seq in seqs:
                index = self.sequences.index(seq)
                del self.sequences[index]
                del self.descriptor[index]

    def add_indices(self, indices: list):
        """ Add another index that should also describe the sequence. The descriptors are automatically updated."""
        # If only one is inputed...
        if isinstance(indices, dict):
            indices = [indices]
        for index in indices:
            self.indices.append(index)
            self.__normalized_indices.append(self.__norm_method(index))
            self.calculate_all()

    def remove_index(self, indices: list):
        """ Removes a index or multiple indices. """
        # If only one is inputed...
        if isinstance(indices, dict):
            index = self.indices.index(indices)
            del self.indices[index]
            del self.__normalized_indices[index]
        else:
            for i in indices:
                index = self.indices.index(i)
                del self.indices[index]
                del self.__normalized_indices[index]
        self.calculate_all()

    def change_norm_method(self, norm_method):
        """" Change to another normalization method, the descriptors are automatically updated."""
        self.__norm_method = norm_method
        self.__normalize_all()
        self.calculate_all()

    def __normalize_all(self):
        """ Normalize all sequence by the defined method """
        for i in range(len(self.indices)):
            if i > len(self.descriptor)-1:
                self.__normalized_indices.append(self.__norm_method(self.indices[i]))
            else:
                self.__normalized_indices[i] = self.__norm_method(self.indices[i])

    def calculate_all(self):
        """ Translate all sequences by the defined indices. """
        for i in range(len(self.sequences)):
            if i > len(self.descriptor)-1:
                self.descriptor.append(self.translate(self.sequences[i], self.__normalized_indices))
            else:
                self.descriptor[i] = self.translate(self.sequences[i], self.__normalized_indices)

    @ staticmethod
    def translate(word: str, indices: list) -> np.array:
        """ Translate a string to feature vectors
            Parameters
            ----------
            word:
                A sequence as string
            indices
                A List of dictionary which represent a aaindex or another descriptor! Only one index is also allowed.
            Returns
            -------
            numpy array
                Concatenated translation of the word by all indices.
        """
        if isinstance(indices, dict):
            indices = [indices]

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
    bin_width : float
        Width of a quantization bin.
    inp_dimensions : int
        Dimension of hashable input vectors.

    Attributes
    ----------
    inp_dimensions : int
        Dimension of hashable input vectors.
    k_dot_products : int
        Hashfunction uses a projection of k dot products.
    bin_width : float
        Width of a quantization bin.

    Notes
    -----
    Input dimensions have to be the same!!!
    """

    def __init__(self, inp_dimensions: int, k_dot_products: int, bin_width: float):
        self.k_dot_products = k_dot_products
        self.bin_width = bin_width
        self.__b = np.random.uniform(low=0, high=bin_width, size=k_dot_products)
        self.inp_dimensions = inp_dimensions
        self.__hash_table = dict()
        self.__projections = np.random.randn(self.k_dot_products, self.inp_dimensions)

    def generate_hash(self, descriptor):
        return tuple(np.floor((np.dot(descriptor, self.__projections.T) + self.__b) / self.bin_width))

    def __setitem__(self, descriptor, label):
        hash_value = self.generate_hash(descriptor)
        if hash_value in self.__hash_table.keys():
            self.__hash_table[hash_value].add(label, descriptor)
        else:
            new_bin = Bin()
            new_bin.add(label, descriptor)
            self.__hash_table[hash_value] = new_bin

    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return self.__hash_table.get(hash_value, Bin())

    def values(self):
        return self.__hash_table.values()

    def keys(self):
        return self.__hash_table.keys()


class Bin:
    """ Hash Table bin"""

    def __init__(self):
        self.sequences = []
        self.vectors = []
        self.kdtree = None

    def create_kdtree(self):
        self.kdtree = cKDTree(self.vectors)

    def nearest_neighbour(self, inp_vec):
        if self.kdtree is not None:
            dist, label = self.kdtree.query(inp_vec)
            return self.sequences[label], dist
        else:
            min_dist = np.inf
            min_match = ""
            for i in range(len(self.sequences)):
                dist = np.linalg.norm(self.vectors[i] - inp_vec)
                if dist < min_dist:
                    min_dist = dist
                    min_match = self.sequences[i]
            return min_match, min_dist

    def add(self, label, descriptor):
        self.sequences.append(label)
        self.vectors.append(descriptor)
        if self.kdtree is not None:
            self.kdtree = cKDTree(self.vectors)




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
            bin_width: float = None,
            kd_tree: bool = True):
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
        self.kd_tree = kd_tree
        if self.kd_tree:
            for table in self.hash_tables:
                for bins in table.values():
                    bins.create_kdtree()

    def __setitem__(self, descriptor, label):
        for table in self.hash_tables:
            table[descriptor] = label

    def __getitem__(self, inp_vec):
        results = list()
        for table in self.hash_tables:
            results.append(table[inp_vec])
        return results

    def add(self, labels, descriptors):
        if len(descriptors) != len(labels):
            raise SystemExit("Length of descriptors and labels is not the same!")
        else:
            for i in range(len(descriptors)):
                self.__setitem__(np.asarray(descriptors[i]), labels[i])

        if self.kd_tree:
            for table in self.hash_tables:
                for bins in table.values():
                    bins.create_kdtree()

    def nearest_neighbour(self, descriptor):
        """ Find the nearest_neighbour by hashing the inp_vec and find the nearest neighbour in the
            corresponding bin."""
        hash_bin = self.__getitem__(descriptor)

        results = []
        for bins in hash_bin:
            results.append(bins.nearest_neighbour(np.asarray(descriptor)))

        if not results:
            hash_bin = [item for sublist in self.hash_tables[0].values() for item in sublist]
            for bins in hash_bin:
                results.append(bins.nearest_neighbour(np.asarray(descriptor)))
                results.append(bins.nearest_neighbour(np.asarray(descriptor)))

        minimum = min(results, key=lambda x: x[1])
        min_match = minimum[0]
        min_dist = minimum[1]

        return min_match, min_dist

    @ staticmethod
    def p1(bin_width):
        """ Probability that two points with ||p-q|| < bin_width Fall into the same bucket """
        return 1 - 2 * norm.cdf(-bin_width) - \
               2 / (np.sqrt(2 * np.pi) * bin_width) * \
               (1 - np.exp(-(bin_width ** 2 / 2)))

    @ staticmethod
    def p2(bin_width, scale=2):
        """ Probability that two points ||p-q|| > scale * bin_width Fall into the same bucket."""
        return 1 - 2 * norm.cdf(-bin_width / scale) - \
               2 / (np.sqrt(2 * np.pi) * bin_width / scale) * \
               (1 - np.exp(-(bin_width ** 2 / (2 * scale ** 2))))

    @ staticmethod
    def compute_num_projections(bin_width, k_dot_products, fail_prob=0.1):
        """ L needed to have a probability smaller than fail_prob that nearest neighbor is not in the same bucket. """
        return int(np.ceil(np.log(fail_prob) / np.log(1 - LSH.p1(bin_width) ** k_dot_products)))

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
        kdtree = cKDTree(data)
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

