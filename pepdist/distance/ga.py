import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp, mannwhitneyu
import pandas as pd
from pepdist.distance import Aaindex
import random
import multiprocess
import copy




class GeneticAlgorithm(object):

    def __init__(self, data1, data2, reference_data, index_db, chromosom_length, cpus=10):
        self.data1 = data1
        self.data2 = data2
        self.reference_data = reference_data
        self.index_db = index_db
        self.chromosom_length = chromosom_length
        self.population = []
        self.scores = []
        self.pool = multiprocess.Pool(cpus)
        self.fittness = []
        self.fittness_function = self.fittness_min_mean

    def set_fittnes_function(self, func):
        self.fittness_function = func

    def create_starting_population(self, popSize):
        population = list()
        while len(population) < popSize:
            chromosom = list(np.random.choice(list(self.index_db.keys()), size=self.chromosom_length, replace=False))
            if chromosom not in population:
                population.append(chromosom)

        self.population = population
        self.rank_population()

    def translate(self, data, chromosom):
        translated_data = []
        for d in data:
            vec = []
            for gen in chromosom:
                index = self.index_db[gen]
                vec.extend(list(map(lambda x: index[x], d)))
            translated_data.append(np.array(vec))
        return translated_data

    def fittness_min_mean(self, chromosom, remove_equal = True):
        trie = cKDTree(np.array(self.translate(self.reference_data, chromosom)))
        score_data1 = []
        for data in self.translate(self.data1, chromosom):
            score = trie.query(data)[0]
            if not remove_equal or score != 0:
                score_data1.append(score)

        score_data2 = []
        for data in self.translate(self.data2, chromosom):
            score = trie.query(data)[0]
            if not remove_equal or score != 0:
                score_data2.append(trie.query(data)[0])

        return mannwhitneyu(score_data1, score_data2, alternative = "greater")[1]

    def fittness_kl_div(self, chromosom, remove_equal = True):
        trie = cKDTree(np.array(self.translate(self.reference_data, chromosom)))
        score_data1 = []
        for data in self.translate(self.data1, chromosom):
            score = trie.query(data)[0]
            if not remove_equal or score != 0:
                score_data1.append(score)

        score_data2 = []
        for data in self.translate(self.data2, chromosom):
            score = trie.query(data)[0]
            if not remove_equal or score != 0:
                score_data2.append(trie.query(data)[0])

        return ks_2samp(score_data1, score_data2)[1]

    def rank_population(self):
        scores = self.pool.map(lambda x: self.fittness_function(x), self.population)

        self.population = [x for x,_ in sorted(zip(self.population, scores), key = lambda x: x[1])]
        self.scores = sorted(scores)

    def tournament_selection(self, selectionSize, tournamentSize):
        selection_result = self.pool.map(lambda x: self.tournament(tournamentSize, self.population), list(range(selectionSize)))

        return sorted(selection_result, key = lambda x: self.scores[self.population.index(x)])

    def tournament(self, tournamentSize, population):
        tournament = random.sample(population, tournamentSize)

        return min(tournament, key=lambda x: self.scores[self.population.index(x)])

    def uniform_cross_over(self, individual1, individual2):
        child = []
        for i in range(len(individual1)):
            if int(100*np.random.rand()) < 50:
                child.append(individual1[i])
            else:
                child.append(individual2[i])
        return child

    def one_point_cross_over(self, individual1, individual2, crossover_probability):
        for i in range(len(individual1)):
            if np.random.rand() < crossover_probability:
                child = individual1[:i]
                child.append(individual2[i:])
                return child
            else:
                np.random.choice([individual1], [individual2])

    def breedPopulation(self, mating_pool, eliteSize):
        children = []
        pool1 = random.sample(mating_pool[:len(self.population)], len(self.population)-eliteSize)
        pool2 = random.sample(mating_pool[len(self.population):], len(self.population)-eliteSize)

        for i in range(0, eliteSize):
            children.append(self.population[i])

        bad_children = self.pool.map(lambda x: self.uniform_cross_over(x[0], x[1]), list(zip(pool1, pool2)))
        children.extend(bad_children)

        return children

    def mutate(self, individual, mutationRate, indices):
        mutated_individual = []
        for i in range(len(individual)):
            if np.random.rand() < mutationRate:
                mutated_individual.append(random.choice(list(indices.keys())))
            else:
                mutated_individual.append(individual[i])
        return mutated_individual

    def mutatePopulation(self, mutationRate):
        mutated_population = self.pool.map(lambda x: self.mutate(x, mutationRate, self.index_db), self.population)
        self.population = mutated_population

    def nextGeneration(self, tournament_size=10, eliteSize=0, mutation_rate = 0.01):

        mating_pool = self.tournament_selection(2*len(self.population), tournament_size)
        new_population = self.breedPopulation(mating_pool, eliteSize)
        self.population = new_population
        self.mutatePopulation(mutation_rate)
        self.rank_population()

        self.fittness.append(self.scores[0])

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

