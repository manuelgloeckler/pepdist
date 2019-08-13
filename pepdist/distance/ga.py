import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp
import pandas as pd
from pepdist.distance import Aaindex
import random
import multiprocess



class GeneticAlgorithm(object):

    def __init__(self, data1, data2, reference_data, index_db, chromosom_length):
        self.data1 = data1
        self.data2 = data2
        self.reference_data = reference_data
        self.index_db = index_db
        self.chromosom_length = chromosom_length
        self.population = []
        self.scores = []

    def create_starting_population(self, popSize):
        population = list()
        while len(population) < popSize:
            chromosom = list(np.random.choice(list(self.index_db.keys()), size=self.chromosom_length, replace=False))
            if chromosom not in population:
                population.append(chromosom)

        self.population = population

    def translate(self, data, chromosom):
        translated_data = []
        for d in data:
            vec = []
            for gen in chromosom:
                index = self.index_db[gen]
                vec.extend(list(map(lambda x: index[x], d)))
            translated_data.append(np.array(vec))
        return np.array(translated_data)

    def fittness_kl_div(self, chromosom):
        trie = cKDTree(np.array(self.translate(self.reference_data, chromosom)))
        score_data1 = []
        for data in self.translate(self.data1, chromosom):
            score_data1.append(trie.query(data)[0])

        score_data2 = []
        for data in self.translate(self.data2, chromosom):
            score_data2.append(trie.query(data)[0])

        # TODO estimate distributions...
        return ks_2samp(score_data1, score_data2)[1]

    def rank_population(self, fittnes_function=fittness_kl_div):
        pool = multiprocess.Pool(10)
        scores = pool.map(lambda x: fittnes_function(self, x), self.population)

        pool.close()
        pool.join()

        self.population = [x for x,_ in sorted(zip(self.population, scores), key = lambda x: x[1])]
        self.scores = sorted(scores)

    def fitness_proportinate_selection(self, eliteSize):
        selection_result = []
        for i in range(0, eliteSize):
            selection_result.append(self.population[i])

        for i in range(eliteSize, ):
            pass

    def tournament_selection(self, selectionSize, tournamentSize):
        selection_result = []
        while len(selection_result) <= selectionSize:
            tournament = []
            for i in range(tournamentSize):
                tournament.append(random.choice(list(zip(self.population, self.scores))))
            selection_result.append(min(tournament, key=lambda x: x[1]))

        return list(map(lambda x: x[0],sorted(selection_result, key=lambda x: x[1])))

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
        length = len(self.population)-eliteSize
        pool = random.sample(mating_pool, len(mating_pool))

        for i in range(0, eliteSize):
            children.append(mating_pool[i])

        for i in range(0,length):
            child = self.uniform_cross_over(pool[i], pool[len(pool)-i-1])
            children.append(child)

        return children

    def mutate(self, individual, mutationRate):
        mutated_individual = []
        for i in range(len(individual)):
            if np.random.rand() < mutationRate:
                mutated_individual.append(random.choice(list(self.index_db.keys())))
            else:
                mutated_individual.append(individual[i])
        return mutated_individual

    def mutatePopulation(self, population, mutationRate):
        mutated_population = []
        for individual in population:
            mutated_population.append(self.mutate(individual, mutationRate))
        return mutated_population

    def nextGeneration(self, tournament_size, eliteSize=0, mutation_rate = 0.01):
        n = len(self.population)
        if eliteSize is None:
            eliteSize = int(0.01*n)

        self.rank_population()
        mating_pool = self.tournament_selection(2*len(self.population), tournament_size)
        next_generation = self.breedPopulation(mating_pool, eliteSize)
        next_generation = self.mutatePopulation(next_generation, mutation_rate)

        self.population = next_generation

