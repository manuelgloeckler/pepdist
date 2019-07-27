"""@package distance
This module contains a locally sensitiv hashing class for fast nearest neighbour computation.
Additionaly it contains functions to represent a peptide/sequence as a feature vector 
based on AAindex or other metrics.
"""

import numpy as np
import os
from scipy.stats import norm

immu_indices = [
    "GEIM800103",
    "OOBM770104",
    "PALJ810115",
    "QIAN880132",
    "OOBM850102",
    "NADH010106",
    "RADA880106",
    "QIAN880112",
    "WEBA780101",
    "QIAN880125",
    "JOND750101",
    "QIAN880124",
    "MUNV940101",
    "HUTJ700102",
    "MITS020101",
    "KARP850103",
    "FAUJ880113",
    "ISOY800106",
    "RACS820113",
    "GEOR030105",
    "QIAN880114",
    "DIGM050101",
    "MIYS850101"]
"""list: List of Aaindex ID's that are associated with immunogenicity

The creators of the immunogenicity predictor POPI used a advanced
generic algorithmen to find aimno acid indices in the AAindex databese,
that are associated with immunogenicity. These are listed here.
"""


class Aaindex():
    """Parse the AAindex1 databse into a dictionary of id -> dict.

    The AAindex1 consits of indices, which translate a given amino
    acid into a real value. This value should represent a special
    proberty of this amino acid, for example hydrophobisity.

    Attributes
    ----------
    aaindex_dic : dict
        A dictionary which maps the ID of an AAindex to the
        corresponding index represented by a dict.

    """

    def __init__(self):
        """The __init__ method parse the aaindex1.txt file to a dict.

        The AAindex is represented in the module as local aaindex1.txt
        file, which is parsed to a dictionary here. That maps the ID of
        a index to the index, represnted as a dictionary that maps aaindex
        amino acid to a real value.

        Notes
        ----------
        A index with NA values is discarded, beacause they can't be used
        for distance computation.

        """
        global key
        path = os.path.dirname(os.path.abspath(
            __file__)) + "/aaindex/aaindex1.txt"
        with open(path) as aaindex1:
            aaindex_dic = {}
            for line in aaindex1:
                dic = {}
                # line starting with "H" contains the ID
                if line[0] == "H":
                    key = line[2:len(line) - 1]
                # line starting with "I" contains the index.
                if line[0] == "I":
                    key1 = ("A", "R", "N", "D", "C", "Q", "E", "G", "H", "I")
                    key2 = ("L", "K", "M", "F", "P", "S", "T", "W", "Y", "V")
                    line1 = aaindex1.readline().split()
                    line2 = aaindex1.readline().split()
                    # Index with NA values are discarded, if not they are added to
                    # the dictionary.
                    na = False
                    for i in range(len(line1)):
                        if line1[i] != "NA":
                            dic[key1[i]] = float(line1[i].strip())
                        else:
                            na = True
                            continue
                    for i in range(len(line2)):
                        if line2[i] != "NA":
                            dic[key2[i]] = float(line2[i].strip())
                        else:
                            na = True
                            continue
                    if na:
                        continue
                    aaindex_dic[key] = dic
            self.aaindex_dic = aaindex_dic

    def __len__(self):
        return len(self.aaindex_dic)

    def __getitem__(self, key):
        try: 
            value = self.aaindex_dic[key]
        except KeyError as e:
            raise KeyError("The ID " + str(e) + " is not in the AAIndex or contains NA values") from e
        
        return value
        
        

    def __contains__(self, item):
        return item in self.aaindex_dic

    def __iter__(self):
        return iter(self.aaindex_dic.keys())

    def make_positive(self, index):
        """ Turns an a index of real numbers to a index of positve real numbers """
        positive_index = {}
        for i in index:
            positive_index[i] = index[i] + abs(min(index.values()))

        return positive_index

    def max_normalize(self, index):
        """ Normalize a index of real numbers to a normalized index between [0,1] """
        pos_index = self.make_positive(index)
        normalized_index = {}
        maximum = max(pos_index.values())
        for i in index:
            normalized_index[i] = pos_index[i] / maximum

        return normalized_index
        
    def z_normalize(self, index):
        mean = np.mean(index.values())
        sigma = np.std(index.values())
        normalized_index = {}
        for i in index:
            normalized_index[i] = (index[i] - mean)/sigma
        return normalized_index

        
     


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

    return (min_match, min_dist)
    
class descriptor():
    """  """
    def __init__(self, seqs, indices=[]):
        self.sequences = seqs
        self.descriptors = []
        self.indices = indices
        self.calculate_all()
        
    def __len__(self):
        return len(self.sequences)
        
    def addSeqs(self, seqs):
        self.sequences.extend(seqs)
        self.calculate_all()
    
    def addIndex(self, index):
        self.indices.append(index)
        self.calculate_all()
    
    def calculate_all(self):
        """ """
        for index in self.indices:
            for i in range(len(self.sequences)):
                self.descriptors[i] = self.translate(self.sequences, self.indices)

    def translate(self, word: str, indices):
        """ Translate a string to feature vectors
            Parameters:
                word (str): A sequence as string
                indices (list): A List of dictionary which represent a aaindex
            Returns:
                numpy ndarray of feature vectors
        """
        vec = []
        for index in indices:
            for char in word:
                vec.append(index[char])
        return np.ndarray(
            (int(
                len(vec) /
                len(word)),
                len(word)),
            buffer=np.array(vec))
            
            
        


class HashTable:
    """Hash Table Implementation for random projection binning

    N given input vectors are projected by k independent gausian nornmal distributed
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

    def __init__(self, k_dot_prodcucts, bin_width, inp_dimensions):
        self.k_dot_prodcucts = k_dot_prodcucts
        self.bin_width = bin_width
        self.b = np.random.uniform(low=0, high=bin_width, size=k_dot_prodcucts)
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = np.random.randn(
            self.k_dot_prodcucts, self.inp_dimensions)

    def generate_hash(self, inp_vec):
        return tuple(np.floor(
            (np.dot(inp_vec, self.projections.T) + self.b) / self.bin_width))

    def __setitem__(self, inp_vec, label):
        hash_value = self.generate_hash(inp_vec)
        self.hash_table[hash_value] = self.hash_table\
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
    bin_widht : float
        Width of a quantization bin.
    inp_dimension :int
        Dimension of hashable input vectors.
    hash_tables : list
        List of Hashtable, which represents the num_projections times
        created hash tables

    """

    def __init__(
            self,
            num_projections,
            k_dot_products,
            bin_width,
            inp_dimensions):
        self.num_projections = num_projections
        self.k_dot_products = k_dot_products
        self.bin_width = bin_width
        self.inp_dimensions = inp_dimensions
        self.hash_tables = list()
        for i in range(self.num_projections):
            self.hash_tables.append(
                HashTable(
                    self.k_dot_products,
                    self.bin_width,
                    self.inp_dimensions))

    def __setitem__(self, inp_vec, label):
        for table in self.hash_tables:
            table[inp_vec] = label

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
                self.__setitem__(self, inp_vecs[i], labels[i])
        
        
    

    def nearest_neighbour(self, inp_vec):
        """ Find the nearest_neighbour by hashing the inp_vec and find the nearest neighbour in the
            corresponding bin."""
        hash_bin = self.__getitem__(inp_vec)
        if not hash_bin:
            hash_bin = [item for sublist in self.hash_tables[0].hash_table.values() for item in sublist]
        min_dist = np.inf
        min_match = ""
        for seq in hash_bin:
            dist = np.linalg.norm(seq.descriptor - inp_vec)
            if dist < min_dist:
                min_dist = dist
                min_match = seq.seq

        return (min_match, min_dist)
        
    def p1(self):
        """ Probability that two points with ||p-q|| < bin_width Fall into the same bucket """
        return 1-2*norm.cdf(-self.bin_width) - \
               2/(np.sqrt(2*np.pi)*self.bin_width) *\
               (1-np.exp(-(self.bin_width**2/2)))
               
    def p2(self, scale = 2):
        """ Probability that two points ||p-q|| > scale * bin_width Fall into the same bucket."""
        return 1-2*norm.cdf(-self.bin_width/scale) - \
               2/(np.sqrt(2*np.pi)*self.bin_width/scale) *\
               (1-np.exp(-(self.bin_width**2/(2*scale**2))))     

    def compute_num_projections(self, fail_prob = 0.01):
        """ L needed to have a probability smaller than fail_prob that nearest neighbor is not in the same bucket. """
        return np.ceil(np.log(fail_prob)/np.log(1-self.p1()**self.k_dot_products))
        
    def mean_buckets(self):
        buckets = 0
        for hash_table in self.hash_tables:
            buckets += len(hash_table.values())
        return int(buckets/self.num_projections) 
        
    def mean_bucket_size(self):
        bucket_size = 0
        for hash_table in self.hash_tables:
            hash_bucket_sizes = 0
            num_buckets = len(list(hash_table.values()))
            for buckets in list(hash_table.values()):
                hash_bucket_sizes += len(buckets)
            bucket_size += hash_bucket_sizes/num_buckets
        return int(bucket_size/self.num_projections)
        
        
    def train_LSH(self, data, query_sample):
        pass
        
    
