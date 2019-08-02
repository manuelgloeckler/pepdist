import os
import numpy as np


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
    """Parse the AAindex1 datbase into a dictionary with id as key and the index as value.

    The AAindex1 consits of indices, which translate a given amino
    acid into a real value. This value should represent a special
    property of this amino acid, for example hydrophobisity.

    Attributes
    ----------
    aaindex_dic : dict
        A dictionary which maps the ID of an AAindex to the
        corresponding index, represented as a dictionary.

    Notes
    -----
    A index with NA values is discarded, beacause they can't be used
    for distance computation.

    """

    def __init__(self):
        # parsing of the AAindex format...
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))
        with open(os.path.join(__location__,"aaindex1.txt")) as aaindex1:
            key = ""
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


def positivize(index):
    """ Turns an a index of real numbers to a index of positve real numbers. """
    positive_index = {}
    for i in index:
        positive_index[i] = index[i] + abs(min(index.values()))

    return positive_index


def max_normalize(index):
    """ Normalize a index of real numbers to a normalized index between [0,1]. """
    pos_index = positivize(index)
    normalized_index = {}
    maximum = max(pos_index.values())
    for i in index:
        normalized_index[i] = pos_index[i] / maximum

    return normalized_index


def z_normalize(index):
    """ Normalize a index by with the z-score normalization method."""
    mean = np.mean(index.values())
    sigma = np.std(index.values())
    normalized_index = {}
    for i in index:
        normalized_index[i] = (index[i] - mean) / sigma
    return normalized_index