.. pepdist documentation master file, created by
   sphinx-quickstart on Mon Jul 29 21:05:56 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pepdist's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Installation
============
The Package can easily installed via pip. Just enter:

.. code-block:: ruby

    pip install pepdist

and the package and all dependencies should be installed.

To use the package just import it with:

.. code-block:: python

    import pepdist
    from pepdist import distance, similarity

Getting started
===============
The package consists of two major sub-packages:

* `Similarity`_ : This package uses a substitution- or other scoring matrices two obtain a measurement of peptide similarity. The nearest neighbor of a given query peptide in a big set of peptides can then be computed by a fast trie-based branch and bound algorithm.

* `Distance`_ : This package uses a feature vector representation of the peptides given for example by the AAindex database. The distance between peptides can then be defined as the euclidean distance between these vectors. A locally sensitive hashing can then obtain the approximated nearest neighbor.

While the code was designed for peptide comparison, it can also be used for other purposes.

Similarity
----------
The function's and classes are documented via autodoc here :mod:`pepdist.similarity`. The main classes are :class:`pepdist.similarity.Trie` and :class:`pepdist.similarity.KmerTrie`
, which are both prefix trie data structures and both have the capability to search the nearest neighbor fast with a normalized score given a substitution/scoring matrix. Given a BLOSUM matrix, this normalization
is known as squared root blosum score, and for two peptides A and B of length n it is given as:

.. math::

    score(A,B) = \frac{\sum_i^n bl(a_i,b_i)}{\sqrt{\sum_i^n bl(a_i,a_i) \cdot bl(b_i,b_i)}}
    
The Trie can be filled with words of different length, but the nearest neighbor search only compare sequences with the same length. Additionaly the positions of each position can be weighted by a real
value.

.. code-block:: python

    >>> from pepdist import similarity
    
    >>> x = ["AAAGG", "AGA", "WWYYA", "WWWW", "YYY", "AYA"]
    >>> trie = similarity.Trie(x)
    
    >>> trie.k_nearest_neighbour("AAA", similarity.blosum62)
    [('AGA', 0.6172133998483676)]
    >>> trie.k_nearest_neighbour("WWYYY", similarity.blosum62)
    [('WWYYA', 0.8198127976897006)]
    >>> trie.k_nearest_neighbour("WWYYY", similarity.blosum62, weights=[1,1,1,1,0])
    [('WWYYA', 1.0)]

The kmerTrie can be used in a similar way. However given a number k, it also saves all k-mers for all word
in the trie. This can be used to not only score equal length peptides, rather you can find the best scoring
substring of length k.

Distance
--------
The function's and classes are documented via autodoc here :mod:`pepdist.distance`. The main classes are :class:`pepdist.distance.Aaindex`, :class:`pepdist.distance.descriptor` and
:class:`pepdist.distance.LSH`. The Aaindex class parse the AAindex database into python dictionaries. This indices map an amino acid to a real value and therefore can be used to map peptides to a feature
vector representing a physical or chemical property. The distance between two peptides can then be represented as the euclidean distance of the feature vectors. However, encoding a peptide with multiple
indices can lead to very high dimensional feature vectors. For an efficient approximate nearest neighbor search, a locally sensitive hashing was implemented, which allows blazing fast search times.
Here are some example of how to use it:

.. code-block:: python

    >>> from pepdist import distance
    >>> x = ["AAAGG", "AGAAA", "WWYYA", "WWWWA", "YYYWW", "AYAWW"]
    >>> AA = distance.Aaindex()
    >>> desc = distance.IndexDescriptor(x, [AA["GEIM800103"]])
    >>> desc.
    >>> desc.sequences
    ['AAAGG', 'AGAAA', 'WWYYA', 'WWWWA', 'YYYWW', 'AYAWW']
    >>> desc.descriptors
    [array([1.55, 1.55, 1.55, 0.59, 0.59]), array([1.55, 0.59, 1.55, 1.55, 1.55]), array([1.86, 1.86, 1.08, 1.08, 1.55]), array([1.86, 1.86, 1.86, 1.86, 1.55]), array([1.08, 1.08, 1.08, 1.86, 1.86]), array([1.55, 1.08, 1.55, 1.86, 1.86])]
    >>> desc.change_norm_method(distance.z_normalize)
    >>> desc.descriptors
    [array([ 1.02132811,  1.02132811,  1.02132811, -1.01708142, -1.01708142]), array([ 1.02132811, -1.01708142,  1.02132811,  1.02132811,  1.02132811]), array([1.67956452, 1.67956452, 0.02335678, 0.02335678, 1.02132811]), array([1.67956452, 1.67956452, 1.67956452, 1.67956452, 1.02132811]), array([0.02335678, 0.02335678, 0.02335678, 1.67956452, 1.67956452]), array([1.02132811, 0.02335678, 1.02132811, 1.67956452, 1.67956452])]
    >>> lsh = distance.LSH(5)
    >>> lsh.add(desc.sequences, desc.descriptors)
    >>> lsh.nearest_neighbour(desc.descriptors[0])
    ('AAAGG', 0.0)

