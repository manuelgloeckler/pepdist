from .scoring_matrix import blosum62
from .scoring_matrix import positivize
from .scoring_matrix import symmetrize
from .scoring_matrix import max_normalize


from .similarity import Trie
from .similarity import KmerTrie
from .similarity import kmer_count
from .similarity import load_trie
from .similarity import word_score
from .similarity import squared_root_similarity
from .similarity import naive_nearest_neighbour
from .similarity import multiprocess_data

from.ga import GeneticAlgorithm