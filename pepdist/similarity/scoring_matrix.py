# -*- coding: utf-8 -*-
"""@module scoring_matrix

This module consist the blosum62 substitution matrix in a symmetric version. Additionally their are some scoring matrix
function for linear transformation techniques.
"""
blosum62 = {('A', 'A'): 4, ('A', 'B'): -2, ('A', 'C'): 0, ('A', 'D'): -2, ('A', 'E'): -1, ('A', 'F'): -2, ('A', 'G'): 0,
            ('A', 'H'): -2, ('A', 'I'): -1, ('A', 'K'): -1, ('A', 'L'): -1, ('A', 'M'): -1, ('A', 'N'): -2, ('A', 'P'): -1,
            ('A', 'Q'): -1, ('A', 'R'): -1, ('A', 'S'): 1, ('A', 'T'): 0, ('A', 'V'): 0, ('A', 'W'): -3, ('A', 'X'): 0,
            ('A', 'Y'): -2, ('A', 'Z'): -1, ('B', 'A'): -2, ('B', 'B'): 4, ('B', 'C'): -3, ('B', 'D'): 4, ('B', 'E'): 1,
            ('B', 'F'): -3, ('B', 'G'): -1, ('B', 'H'): 0, ('B', 'I'): -3, ('B', 'K'): 0, ('B', 'L'): -4, ('B', 'M'): -3,
            ('B', 'N'): 3, ('B', 'P'): -2, ('B', 'Q'): 0, ('B', 'R'): -1, ('B', 'S'): 0, ('B', 'T'): -1, ('B', 'V'): -3,
            ('B', 'W'): -4, ('B', 'X'): -1, ('B', 'Y'): -3, ('B', 'Z'): 1, ('C', 'A'): 0, ('C', 'B'): -3, ('C', 'C'): 9,
            ('C', 'D'): -3, ('C', 'E'): -4, ('C', 'F'): -2, ('C', 'G'): -3, ('C', 'H'): -3, ('C', 'I'): -1, ('C', 'K'): -3,
            ('C', 'L'): -1, ('C', 'M'): -1, ('C', 'N'): -3, ('C', 'P'): -3, ('C', 'Q'): -3, ('C', 'R'): -3, ('C', 'S'): -1,
            ('C', 'T'): -1, ('C', 'V'): -1, ('C', 'W'): -2, ('C', 'X'): -2, ('C', 'Y'): -2, ('C', 'Z'): -3, ('D', 'A'): -2,
            ('D', 'B'): 4, ('D', 'C'): -3, ('D', 'D'): 6, ('D', 'E'): 2, ('D', 'F'): -3, ('D', 'G'): -1, ('D', 'H'): -1,
            ('D', 'I'): -3, ('D', 'K'): -1, ('D', 'L'): -4, ('D', 'M'): -3, ('D', 'N'): 1, ('D', 'P'): -1, ('D', 'Q'): 0,
            ('D', 'R'): -2, ('D', 'S'): 0, ('D', 'T'): -1, ('D', 'V'): -3, ('D', 'W'): -4, ('D', 'X'): -1, ('D', 'Y'): -3,
            ('D', 'Z'): 1, ('E', 'A'): -1, ('E', 'B'): 1, ('E', 'C'): -4, ('E', 'D'): 2, ('E', 'E'): 5, ('E', 'F'): -3,
            ('E', 'G'): -2, ('E', 'H'): 0, ('E', 'I'): -3, ('E', 'K'): 1, ('E', 'L'): -3, ('E', 'M'): -2, ('E', 'N'): 0,
            ('E', 'P'): -1, ('E', 'Q'): 2, ('E', 'R'): 0, ('E', 'S'): 0, ('E', 'T'): -1, ('E', 'V'): -2, ('E', 'W'): -3,
            ('E', 'X'): -1, ('E', 'Y'): -2, ('E', 'Z'): 4, ('F', 'A'): -2, ('F', 'B'): -3, ('F', 'C'): -2, ('F', 'D'): -3,
            ('F', 'E'): -3, ('F', 'F'): 6, ('F', 'G'): -3, ('F', 'H'): -1, ('F', 'I'): 0, ('F', 'K'): -3, ('F', 'L'): 0,
            ('F', 'M'): 0, ('F', 'N'): -3, ('F', 'P'): -4, ('F', 'Q'): -3, ('F', 'R'): -3, ('F', 'S'): -2, ('F', 'T'): -2,
            ('F', 'V'): -1, ('F', 'W'): 1, ('F', 'X'): -1, ('F', 'Y'): 3, ('F', 'Z'): -3, ('G', 'A'): 0, ('G', 'B'): -1,
            ('G', 'C'): -3, ('G', 'D'): -1, ('G', 'E'): -2, ('G', 'F'): -3, ('G', 'G'): 6, ('G', 'H'): -2, ('G', 'I'): -4,
            ('G', 'K'): -2, ('G', 'L'): -4, ('G', 'M'): -3, ('G', 'N'): 0, ('G', 'P'): -2, ('G', 'Q'): -2, ('G', 'R'): -2,
            ('G', 'S'): 0, ('G', 'T'): -2, ('G', 'V'): -3, ('G', 'W'): -2, ('G', 'X'): -1, ('G', 'Y'): -3, ('G', 'Z'): -2,
            ('H', 'A'): -2, ('H', 'B'): 0, ('H', 'C'): -3, ('H', 'D'): -1, ('H', 'E'): 0, ('H', 'F'): -1, ('H', 'G'): -2,
            ('H', 'H'): 8, ('H', 'I'): -3, ('H', 'K'): -1, ('H', 'L'): -3, ('H', 'M'): -2, ('H', 'N'): 1, ('H', 'P'): -2,
            ('H', 'Q'): 0, ('H', 'R'): 0, ('H', 'S'): -1, ('H', 'T'): -2, ('H', 'V'): -3, ('H', 'W'): -2, ('H', 'X'): -1,
            ('H', 'Y'): 2, ('H', 'Z'): 0, ('I', 'A'): -1, ('I', 'B'): -3, ('I', 'C'): -1, ('I', 'D'): -3, ('I', 'E'): -3,
            ('I', 'F'): 0, ('I', 'G'): -4, ('I', 'H'): -3, ('I', 'I'): 4, ('I', 'K'): -3, ('I', 'L'): 2, ('I', 'M'): 1,
            ('I', 'N'): -3, ('I', 'P'): -3, ('I', 'Q'): -3, ('I', 'R'): -3, ('I', 'S'): -2, ('I', 'T'): -1, ('I', 'V'): 3,
            ('I', 'W'): -3, ('I', 'X'): -1, ('I', 'Y'): -1, ('I', 'Z'): -3, ('K', 'A'): -1, ('K', 'B'): 0, ('K', 'C'): -3,
            ('K', 'D'): -1, ('K', 'E'): 1, ('K', 'F'): -3, ('K', 'G'): -2, ('K', 'H'): -1, ('K', 'I'): -3, ('K', 'K'): 5,
            ('K', 'L'): -2, ('K', 'M'): -1, ('K', 'N'): 0, ('K', 'P'): -1, ('K', 'Q'): 1, ('K', 'R'): 2, ('K', 'S'): 0,
            ('K', 'T'): -1, ('K', 'V'): -2, ('K', 'W'): -3, ('K', 'X'): -1, ('K', 'Y'): -2, ('K', 'Z'): 1, ('L', 'A'): -1,
            ('L', 'B'): -4, ('L', 'C'): -1, ('L', 'D'): -4, ('L', 'E'): -3, ('L', 'F'): 0, ('L', 'G'): -4, ('L', 'H'): -3,
            ('L', 'I'): 2, ('L', 'K'): -2, ('L', 'L'): 4, ('L', 'M'): 2, ('L', 'N'): -3, ('L', 'P'): -3, ('L', 'Q'): -2,
            ('L', 'R'): -2, ('L', 'S'): -2, ('L', 'T'): -1, ('L', 'V'): 1, ('L', 'W'): -2, ('L', 'X'): -1, ('L', 'Y'): -1,
            ('L', 'Z'): -3, ('M', 'A'): -1, ('M', 'B'): -3, ('M', 'C'): -1, ('M', 'D'): -3, ('M', 'E'): -2, ('M', 'F'): 0,
            ('M', 'G'): -3, ('M', 'H'): -2, ('M', 'I'): 1, ('M', 'K'): -1, ('M', 'L'): 2, ('M', 'M'): 5, ('M', 'N'): -2,
            ('M', 'P'): -2, ('M', 'Q'): 0, ('M', 'R'): -1, ('M', 'S'): -1, ('M', 'T'): -1, ('M', 'V'): 1, ('M', 'W'): -1,
            ('M', 'X'): -1, ('M', 'Y'): -1, ('M', 'Z'): -1, ('N', 'A'): -2, ('N', 'B'): 3, ('N', 'C'): -3, ('N', 'D'): 1,
            ('N', 'E'): 0, ('N', 'F'): -3, ('N', 'G'): 0, ('N', 'H'): 1, ('N', 'I'): -3, ('N', 'K'): 0, ('N', 'L'): -3,
            ('N', 'M'): -2, ('N', 'N'): 6, ('N', 'P'): -2, ('N', 'Q'): 0, ('N', 'R'): 0, ('N', 'S'): 1, ('N', 'T'): 0,
            ('N', 'V'): -3, ('N', 'W'): -4, ('N', 'X'): -1, ('N', 'Y'): -2, ('N', 'Z'): 0, ('P', 'A'): -1, ('P', 'B'): -2,
            ('P', 'C'): -3, ('P', 'D'): -1, ('P', 'E'): -1, ('P', 'F'): -4, ('P', 'G'): -2, ('P', 'H'): -2, ('P', 'I'): -3,
            ('P', 'K'): -1, ('P', 'L'): -3, ('P', 'M'): -2, ('P', 'N'): -2, ('P', 'P'): 7, ('P', 'Q'): -1, ('P', 'R'): -2,
            ('P', 'S'): -1, ('P', 'T'): -1, ('P', 'V'): -2, ('P', 'W'): -4, ('P', 'X'): -2, ('P', 'Y'): -3, ('P', 'Z'): -1,
            ('Q', 'A'): -1, ('Q', 'B'): 0, ('Q', 'C'): -3, ('Q', 'D'): 0, ('Q', 'E'): 2, ('Q', 'F'): -3, ('Q', 'G'): -2,
            ('Q', 'H'): 0, ('Q', 'I'): -3, ('Q', 'K'): 1, ('Q', 'L'): -2, ('Q', 'M'): 0, ('Q', 'N'): 0, ('Q', 'P'): -1,
            ('Q', 'Q'): 5, ('Q', 'R'): 1, ('Q', 'S'): 0, ('Q', 'T'): -1, ('Q', 'V'): -2, ('Q', 'W'): -2, ('Q', 'X'): -1,
            ('Q', 'Y'): -1, ('Q', 'Z'): 3, ('R', 'A'): -1, ('R', 'B'): -1, ('R', 'C'): -3, ('R', 'D'): -2, ('R', 'E'): 0,
            ('R', 'F'): -3, ('R', 'G'): -2, ('R', 'H'): 0, ('R', 'I'): -3, ('R', 'K'): 2, ('R', 'L'): -2, ('R', 'M'): -1,
            ('R', 'N'): 0, ('R', 'P'): -2, ('R', 'Q'): 1, ('R', 'R'): 5, ('R', 'S'): -1, ('R', 'T'): -1, ('R', 'V'): -3,
            ('R', 'W'): -3, ('R', 'X'): -1, ('R', 'Y'): -2, ('R', 'Z'): 0, ('S', 'A'): 1, ('S', 'B'): 0, ('S', 'C'): -1,
            ('S', 'D'): 0, ('S', 'E'): 0, ('S', 'F'): -2, ('S', 'G'): 0, ('S', 'H'): -1, ('S', 'I'): -2, ('S', 'K'): 0,
            ('S', 'L'): -2, ('S', 'M'): -1, ('S', 'N'): 1, ('S', 'P'): -1, ('S', 'Q'): 0, ('S', 'R'): -1, ('S', 'S'): 4,
            ('S', 'T'): 1, ('S', 'V'): -2, ('S', 'W'): -3, ('S', 'X'): 0, ('S', 'Y'): -2, ('S', 'Z'): 0, ('T', 'A'): 0,
            ('T', 'B'): -1, ('T', 'C'): -1, ('T', 'D'): -1, ('T', 'E'): -1, ('T', 'F'): -2, ('T', 'G'): -2, ('T', 'H'): -2,
            ('T', 'I'): -1, ('T', 'K'): -1, ('T', 'L'): -1, ('T', 'M'): -1, ('T', 'N'): 0, ('T', 'P'): -1, ('T', 'Q'): -1,
            ('T', 'R'): -1, ('T', 'S'): 1, ('T', 'T'): 5, ('T', 'V'): 0, ('T', 'W'): -2, ('T', 'X'): 0, ('T', 'Y'): -2,
            ('T', 'Z'): -1, ('V', 'A'): 0, ('V', 'B'): -3, ('V', 'C'): -1, ('V', 'D'): -3, ('V', 'E'): -2, ('V', 'F'): -1,
            ('V', 'G'): -3, ('V', 'H'): -3, ('V', 'I'): 3, ('V', 'K'): -2, ('V', 'L'): 1, ('V', 'M'): 1, ('V', 'N'): -3,
            ('V', 'P'): -2, ('V', 'Q'): -2, ('V', 'R'): -3, ('V', 'S'): -2, ('V', 'T'): 0, ('V', 'V'): 4, ('V', 'W'): -3,
            ('V', 'X'): -1, ('V', 'Y'): -1, ('V', 'Z'): -2, ('W', 'A'): -3, ('W', 'B'): -4, ('W', 'C'): -2, ('W', 'D'): -4,
            ('W', 'E'): -3, ('W', 'F'): 1, ('W', 'G'): -2, ('W', 'H'): -2, ('W', 'I'): -3, ('W', 'K'): -3, ('W', 'L'): -2,
            ('W', 'M'): -1, ('W', 'N'): -4, ('W', 'P'): -4, ('W', 'Q'): -2, ('W', 'R'): -3, ('W', 'S'): -3, ('W', 'T'): -2,
            ('W', 'V'): -3, ('W', 'W'): 11, ('W', 'X'): -2, ('W', 'Y'): 2, ('W', 'Z'): -3, ('X', 'A'): 0, ('X', 'B'): -1,
            ('X', 'C'): -2, ('X', 'D'): -1, ('X', 'E'): -1, ('X', 'F'): -1, ('X', 'G'): -1, ('X', 'H'): -1, ('X', 'I'): -1,
            ('X', 'K'): -1, ('X', 'L'): -1, ('X', 'M'): -1, ('X', 'N'): -1, ('X', 'P'): -2, ('X', 'Q'): -1, ('X', 'R'): -1,
            ('X', 'S'): 0, ('X', 'T'): 0, ('X', 'V'): -1, ('X', 'W'): -2, ('X', 'X'): -1, ('X', 'Y'): -1, ('X', 'Z'): -1,
            ('Y', 'A'): -2, ('Y', 'B'): -3, ('Y', 'C'): -2, ('Y', 'D'): -3, ('Y', 'E'): -2, ('Y', 'F'): 3, ('Y', 'G'): -3,
            ('Y', 'H'): 2, ('Y', 'I'): -1, ('Y', 'K'): -2, ('Y', 'L'): -1, ('Y', 'M'): -1, ('Y', 'N'): -2, ('Y', 'P'): -3,
            ('Y', 'Q'): -1, ('Y', 'R'): -2, ('Y', 'S'): -2, ('Y', 'T'): -2, ('Y', 'V'): -1, ('Y', 'W'): 2, ('Y', 'X'): -1,
            ('Y', 'Y'): 7, ('Y', 'Z'): -2, ('Z', 'A'): -1, ('Z', 'B'): 1, ('Z', 'C'): -3, ('Z', 'D'): 1, ('Z', 'E'): 4,
            ('Z', 'F'): -3, ('Z', 'G'): -2, ('Z', 'H'): 0, ('Z', 'I'): -3, ('Z', 'K'): 1, ('Z', 'L'): -3, ('Z', 'M'): -1,
            ('Z', 'N'): 0, ('Z', 'P'): -1, ('Z', 'Q'): 3, ('Z', 'R'): 0, ('Z', 'S'): 0, ('Z', 'T'): -1, ('Z', 'V'): -2,
            ('Z', 'W'): -3, ('Z', 'X'): -1, ('Z', 'Y'): -2, ('Z', 'Z'): 4}
""" The symmetrized blosum62 substitutionmatrix obtained from biopython package."""


def symmetrize(self, matrix: dict, subst: bool = False) -> dict:
    """ This function symmetrize a given matrix.

    Attributes
    ----------
    matrix: dict {(i,j): int}
                A scoring matrix given as dictionary.
    subst: bool
                In a substitution matrix both cases (i,j) or (j,i) are included in one key. If you want to divide
                the score of mismatches by 2 then set this to True.

    Returns
    -------
    dict : {(i,j): int]
                Returns the same scoring matrix with symmetrized keys.

    Notes
    -----
    Biopython's or most other substitution matrices are not symmetric and should be symmetrized by this method or
    manuall.
    """

    if subst:
        new_matrix = {}
        for k, v in matrix.items():
            if k[0] == k[1]:
                new_matrix[k] = v
            else:
                new_matrix[k] = v / 2
                new_matrix[tuple(reversed(k))] = v / 2
    else:
        new_matrix = {}
        for k, v in matrix.items():
            new_matrix[k] = v
            new_matrix[tuple(reversed(k))] = v
        return new_matrix


def positivize(self, matrix: dict) -> dict:
    """ This function transforms all score values linearly to be greater equal than zero

    Attributes
    ----------
    matrix: dict {(i,j): int}
                A scoring matrix given as dictionary.
    Returns
    -------
    dict : {(i,j): int]
                Returns the same scoring matrix with positive score values
    """
    new_matrix = {}
    matrix_min = min(matrix.values())
    if matrix_min >= 0:
        return matrix

    for k, v in matrix.items():
        new_matrix[k] = v + abs(matrix_min)
    return new_matrix


def max_normalize(self, matrix: dict):
    """ This function normalize the score between 0 and 1, by dividung throught the maximum value.

    Attributes
    ----------
    matrix: dict {(i,j): int}
                A scoring matrix given as dictionary.
    Returns
    -------
    dict : {(i,j): int]
                Returns the same scoring matrix with normalized score values between 0 and 1.
    """
    matrix = self.positivize(matrix)
    new_matrix = {}
    matrix_max = max(matrix.values())
    for k, v in matrix.items():
        new_matrix[k] = v / matrix_max
    return new_matrix
