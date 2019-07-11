from Bio.SubsMat import MatrixInfo


class MatrixScore():
    def __init__(self):
        self.matrices = {}
        matrices = MatrixInfo.available_matrices
        for matrix in matrices:
            self.matrices[matrix] = vars(MatrixInfo)[matrix]

    def sym(self, matrix: dict, subst=False):
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

    def positivize(self, matrix: dict):
        new_matrix = {}
        matrix_min = min(matrix.values())
        for k, v in matrix.items():
            new_matrix[k] = v + abs(matrix_min)
        return new_matrix

    def normalize(self, matrix: dict):
        new_matrix = {}
        matrix_max = max(matrix.values())
        for k, v in matrix.items():
            new_matrix[k] = v / matrix_max
        return new_matrix

    def distance_transformation(self, matrix: dict):
        new_matrix = {}
        for k, v in matrix.items():
            new_matrix[k] = (1 - v)
        return new_matrix

    def get(self, matrix: str, method=sym):
        # symetric matrices are needed
        return method(self, self.matrices[matrix])


matrices = MatrixScore()
blosum62 = matrices.get("blosum62")


def blosum_score(word1, word2, matrix=blosum62):
    score = 0
    for i in range(min(len(word1), len(word2))):
        key = (word1[i], word2[i])
        score += matrix[key]
    return score


def blosum_similarity(seq1, seq2, matrix=blosum62):
    bl_ab = blossum_score(seq1, seq2, matrix=blosum62)
    bl_aa = blossum_score(seq1, seq1, matrix=blosum62)
    bl_bb = blossum_score(seq2, seq2, matrix=blosum62)

    return bl_ab / np.sqrt(bl_aa * bl_bb)
