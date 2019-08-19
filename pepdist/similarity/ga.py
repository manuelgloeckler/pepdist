import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu
import random
import multiprocess




class GeneticAlgorithm(object):

    def __init__(self, data1, data2, trie, similarity_score, weights, chromosom_length, cpus=10):
        self.data1 = data1
        self.data2 = data2
        self.reference_data = trie
        self.similarity_score = similarity_score
        self.weights = weights
        self.chromosom_length = chromosom_length
        self.population = []
        self.scores = []
        self.pool = multiprocess.Pool(cpus)
        self.fittness = []
        self.fittness_function = self.fittness_min_mean

    def set_fittnes_function(self, func):
        self.fittness_function = func

    def create_starting_population(self, popSize):
        """ Creates a random starting populaion with size of popSize."""
        population = list()
        while len(population) < popSize:
            chromosom = list(np.random.choice(self.weights, size=self.chromosom_length, replace=True))
            if chromosom not in population:
                population.append(chromosom)

        self.population = population
        self.rank_population()

    def fittness_min_mean(self, chromosom, remove_equal = True, alternative="less"):
        """ For a given chromosom it minimize the p-value of Mann Whitney U test if the score distribtuions of
        data1 is less then that of data2"""
        score_data1 = []
        for data in self.data1:
            score = self.reference_data.k_nearest_neighbour(data, self.similarity_score, weights=chromosom, score_only=True)
            if not remove_equal or score != 1:
                score_data1.append(score)

        score_data2 = []
        for data in self.data2:
            score = self.reference_data.k_nearest_neighbour(data, self.similarity_score, weights=chromosom, score_only=True)
            if not remove_equal or score != 1:
                score_data2.append(score)

        if score_data1 == score_data2:
            return 1

        return mannwhitneyu(score_data1, score_data2, alternative = alternative)[1]

    def fittness_kl_div(self, chromosom, remove_equal = True):
        """ Returns p-value of a KS-test, it is minimized for different distribtuions."""
        score_data1 = []
        for data in self.data1:
            score = self.reference_data.k_nearest_neighbour(data, self.similarity_score, weights=chromosom, score_only=True)
            if not remove_equal or score != 1:
                score_data1.append(score)

        score_data2 = []
        for data in self.data2:
            score = self.reference_data.k_nearest_neighbour(data, self.similarity_score, weights=chromosom, score_only=True)
            if not remove_equal or score != 1:
                score_data2.append(score)

        return ks_2samp(score_data1, score_data2)[1]

    def rank_population(self):
        """ Rank population according to score."""
        scores = self.pool.map(lambda x: self.fittness_function(x), self.population)

        self.population = [x for x,_ in sorted(zip(self.population, scores), key = lambda x: x[1])]
        self.scores = sorted(scores)

    def tournament_selection(self, selectionSize, tournamentSize):
        """ Selection """
        selection_result = self.pool.map(lambda x: self.tournament(tournamentSize, self.population), list(range(selectionSize)))

        return sorted(selection_result, key = lambda x: self.scores[self.population.index(x)])

    def tournament(self, tournamentSize, population):
        """ Simple tournament"""
        tournament = random.sample(population, tournamentSize)

        return min(tournament, key=lambda x: self.scores[self.population.index(x)])

    def uniform_cross_over(self, individual1, individual2):
        """ Cross over which selects genes from parents uniformly. """
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
        """ Breed the selected mating_pool """
        children = []
        pool1 = random.sample(mating_pool[:len(self.population)], len(self.population)-eliteSize)
        pool2 = random.sample(mating_pool[len(self.population):], len(self.population)-eliteSize)

        for i in range(0, eliteSize):
            children.append(self.population[i])

        bad_children = self.pool.map(lambda x: self.uniform_cross_over(x[0], x[1]), list(zip(pool1, pool2)))
        children.extend(bad_children)

        return children

    def mutate(self, individual, mutationRate, indices):
        """ Mutates an individual """
        mutated_individual = []
        for i in range(len(individual)):
            if np.random.rand() < mutationRate:
                mutated_individual.append(random.choice(indices))
            else:
                mutated_individual.append(individual[i])
        return mutated_individual

    def mutatePopulation(self, mutationRate):
        """ Mutate the whole population"""
        mutated_population = self.pool.map(lambda x: self.mutate(x, mutationRate, self.weights), self.population)
        self.population = mutated_population

    def nextGeneration(self, tournament_size=3, eliteSize=0, mutation_rate = 0.01):
        """ Generates the next generation """
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

