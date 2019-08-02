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
The package consists of two subpackages:

* `Similarity`_ : This package uses a substitution- or other scoring matrices two obtain a measurement of peptide similarity. The nearest neighbor of a given query peptide in a big set of peptides can then be computed by a fast trie-based branchand bound algorithm.

* `Distance`_ : This package uses a feature vector representation of the peptides given for example by the AAindex database. The distance between peptides can then be defined as the euclidean distance between this vectors. The approximated nearest neighbor can then be obtained by a locally sensitve hashing.

While the code was designed for peptide comparision it can also be used for other purposes.

Similarity
----------
The autodoc documenteded function's and classes are documented via autodoc here :mod:`pepdist.similarity`. The main classes are :class:`pepdist.similarity.Trie` and :class:`pepdist.similarity.KmerTrie`
which are both prefix trie datastructures and both have the capability to fast nearest neighbor search with normalized score given a substitution/scoring matrix. Given a blosum Matrix, this normalization
is known as squared root blossum score, and for two peptides A and B of length n it is given as:

.. math::

    score(A,B) = \frac{\sum_i^n bl(a_i,b_i)}{\sqrt{\sum_i^n bl(a_i,a_i) \cdot bl(b_i,b_i)}}
    
The Trie can be filled with words of different length, but the nearest neighbor search only compare sequences with the same length. Additionaly the positions of each position can be weighted by a real
value.

.. code-block:: python

    >>>from pepdist import similarity
    
    >>>x = ["AAAGG", "AGA", "WWYYA", "WWWW", "YYY", "AYA"]
    >>>trie = similarity.Trie(x)
    
    >>>trie.k_nearest_neighbour("AAA", similarity.blosum62)
    [('AGA', 0.6172133998483676)]
    >>>trie.k_nearest_neighbour("WWYYY", similarity.blosum62)
    [('WWYYA', 0.8198127976897006)]
    >>>trie.k_nearest_neighbour("WWYYY", similarity.blosum62, weights=[1,1,1,1,0])
    [('WWYYA', 1.0)]

The kmerTrie can be used to find to highest scoring kmer between peptides:

Distance
--------



