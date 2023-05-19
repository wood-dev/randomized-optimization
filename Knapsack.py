import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from time import time
from util import getFullFilePath

class Knapsack:

    RANDOM_SEED = 100
    TOP_MAX_ATTEMPTS = 200
    title = ''
    no_of_items = 30
    current_fitness_mark = 0
    fitness_list = []
    weights = []
    values = []

    def __init__(self):
        self.title = self.__class__.__name__

    def init_custom_fitness(self):
        self.current_fitness_mark = 0
        self.fitness_list = []

    def evaluateRandomHillClimb(self, problem, color_list):
        ################ Random Hill Climb Evaluation #############
        max_attempts_list = [10, 50, 100, 150]
        fig = plt.figure()
        ax = fig.gca()
        i = 0
        for max_attempts_value in max_attempts_list:
            _, best_fitness, curve = mlrose.random_hill_climb(problem, max_attempts=max_attempts_value, curve=True, random_state=self.RANDOM_SEED)
            ax.plot(curve, linewidth=8-i*2, label='max_attemps='+str(max_attempts_value), alpha=0.7, color=color_list[i])
            i = i + 1
        plt.title('{title} - RHC - Fitness vs Iteration with various Max Attempts'.format(title=self.title))
        plt.xlabel('Iteration')
        plt.ylabel("Fitness")
        plt.legend()
        filename = '{title}-RHC-Fitness-vs-Iteration-MaxAttempt.png'.format(title=self.title)
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()

    def evaluateSimulatedAnnealing(self, problem, color_list):
        ################ Simulated Annealing Evaluation #############
        decays_list = [0.99, 0.7, 0.5, 0.3]
        fig = plt.figure()
        ax = fig.gca()
        i = 0
        for decays_list_value in decays_list:
            schedule = mlrose.GeomDecay(decay=decays_list_value)
            _, best_fitness, curve = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=self.TOP_MAX_ATTEMPTS, curve=True, random_state=self.RANDOM_SEED)
            ax.plot(curve, linewidth=8-i*2, label='decays='+str(decays_list_value), alpha=0.7, color=color_list[i])
            i = i + 1
        plt.title('{title} - SA - Fitness vs Iteration with different Decays Rate'.format(title=self.title))
        plt.xlabel('Iteration')
        plt.ylabel("Fitness")
        plt.legend()
        filename = '{title}-SA-Fitness-vs-Iteration-DecaysRate.png'.format(title=self.title)
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()


    def evaluateGeneticAlgorithm(self, problem, color_list):

        ################ Genetic Algorithm Evaluation #############
        pop_size_list = [100, 200, 300]
        fig = plt.figure()
        ax = fig.gca()
        i = 0
        for pop_size_value in pop_size_list:
            _, best_fitness, curve = mlrose.genetic_alg(problem, pop_size=pop_size_value, max_attempts=self.TOP_MAX_ATTEMPTS, curve=True, random_state=self.RANDOM_SEED)
            ax.plot(curve, linewidth=8-i*2, label='pop_size='+str(pop_size_value), alpha=0.7, color=color_list[i])
            i = i + 1
        plt.title('{title} - GA - Fitness vs Iteration with various Pop Size'.format(title=self.title))
        plt.xlabel('Iteration')
        plt.ylabel("Fitness")
        plt.legend()
        filename = '{title}-GA-Fitness-vs-Iteration-PopSize.png'.format(title=self.title)
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()

        mutation_prob_list = [0.05, 0.1, 0.3, 0.5]
        fig = plt.figure()
        ax = fig.gca()
        i = 0
        for mutation_prob_value in mutation_prob_list:
            _, best_fitness, curve = mlrose.genetic_alg(problem, mutation_prob=mutation_prob_value, max_attempts=self.TOP_MAX_ATTEMPTS, curve=True, random_state=self.RANDOM_SEED)
            ax.plot(curve, linewidth=8-i*2, label='mutation_prob='+str(mutation_prob_value), alpha=0.7, color=color_list[i])
            i = i + 1
        plt.title('{title} - GA - Fitness vs Iteration with various Mutation Probability'.format(title=self.title))
        plt.xlabel('Iteration')
        plt.ylabel("Fitness")
        plt.legend()
        filename = '{title}-GA-Fitness-vs-Iteration-MutationProb.png'.format(title=self.title)
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()

    def evaluateMimic(self, problem, color_list):

        ################ Mimic Evaluation #############
        pop_size_list = [100, 200, 300]
        fig = plt.figure()
        ax = fig.gca()
        i = 0
        for pop_size_value in pop_size_list:
            _, best_fitness, curve = mlrose.mimic(problem, pop_size=pop_size_value, max_attempts=self.TOP_MAX_ATTEMPTS, curve=True, random_state=self.RANDOM_SEED)
            ax.plot(curve, linewidth=8-i*2, label='pop_size='+str(pop_size_value), alpha=0.7, color=color_list[i])
            i = i + 1
        plt.title('{title} - Mimic - Fitness vs Iteration with various Pop Size'.format(title=self.title))
        plt.xlabel('Iteration')
        plt.ylabel("Fitness")
        plt.legend()
        filename = '{title}-Mimic-Fitness-vs-Iteration-PopSize.png'.format(title=self.title)
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()

        keep_pct_list = [0.15, 0.2, 0.4, 0.6]
        fig = plt.figure()
        ax = fig.gca()
        i = 0
        for keep_pct_value in keep_pct_list:
            _, best_fitness, curve = mlrose.mimic(problem, keep_pct=keep_pct_value, max_attempts=self.TOP_MAX_ATTEMPTS, curve=True, random_state=self.RANDOM_SEED)
            ax.plot(curve, linewidth=8-i*2, label='keep_pct='+str(keep_pct_value), alpha=0.7, color=color_list[i])
            i = i + 1
        plt.title('{title} - Mimic - Fitness vs Iteration with various Keep Pct'.format(title=self.title))
        plt.xlabel('Iteration')
        plt.ylabel("Fitness")
        plt.legend()
        filename = '{title}-Mimic-Fitness-vs-Iteration-KeepPct.png'.format(title=self.title)
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()

    def comparePerformance(self, problem, color_list, size):

        time_list = []
        fitness_list = []
        minimum_iteration_list = []

        fig = plt.figure(1)
        ax = fig.gca()

        fig2 = plt.figure(2)
        ax2 = fig2.gca()

        i = 0

        # rhc
        self.init_custom_fitness()
        time_begin = time()
        _, best_fitness, curve = mlrose.random_hill_climb(problem, max_attempts=self.TOP_MAX_ATTEMPTS, curve=True, random_state=self.RANDOM_SEED)
        time_end = time()
        ax.plot(curve, linewidth=8-i*2, label='Random Hill Climb', alpha=0.7, color=color_list[i])
        ax2.plot(self.fitness_list, label='Random Hill Climb', alpha=0.4, color=color_list[i])
        time_list.append((time_end - time_begin) / len(curve))
        fitness_list.append(best_fitness)
        index_max = np.argmax(curve)
        minimum_iteration_list.append(index_max)
        i = i + 1

        # sa
        self.init_custom_fitness()
        schedule = mlrose.GeomDecay(decay=0.7)
        time_begin = time()
        _, best_fitness, curve = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=self.TOP_MAX_ATTEMPTS, curve=True, random_state=self.RANDOM_SEED)
        time_end = time()
        ax.plot(curve, linewidth=8-i*2, label='Stimmulated Annealing', alpha=0.7, color=color_list[i])
        ax2.plot(self.fitness_list, label='Stimmulated Annealing', alpha=0.4, color=color_list[i])
        time_list.append((time_end - time_begin) / len(curve))
        fitness_list.append(best_fitness)
        index_max = np.argmax(curve)
        minimum_iteration_list.append(index_max)
        i = i + 1

        # ga
        self.init_custom_fitness()
        time_begin = time()
        _, best_fitness, curve = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.05, max_attempts=self.TOP_MAX_ATTEMPTS, curve=True, random_state=self.RANDOM_SEED)
        time_end = time()
        ax.plot(curve, linewidth=8-i*2, label='Genetic Algorithm', alpha=0.7, color=color_list[i])
        ax2.plot(self.fitness_list, label='Genetic Algorithm', alpha=0.4, color=color_list[i])
        time_list.append((time_end - time_begin) / len(curve))
        fitness_list.append(best_fitness)
        index_max = np.argmax(curve)
        minimum_iteration_list.append(index_max)
        i = i + 1

        # mimic
        self.init_custom_fitness()
        time_begin = time()
        _, best_fitness, curve = mlrose.mimic(problem, pop_size=300, keep_pct=0.15, max_attempts=self.TOP_MAX_ATTEMPTS, curve=True, random_state=self.RANDOM_SEED)
        time_end = time()
        ax.plot(curve, linewidth=8-i*2, label='Mimic', alpha=0.7, color=color_list[i])
        ax2.plot(self.fitness_list, label='Mimic', alpha=0.4, color=color_list[i])
        time_list.append((time_end - time_begin) / len(curve))
        fitness_list.append(best_fitness)
        index_max = np.argmax(curve)
        minimum_iteration_list.append(index_max)

        fig = plt.figure(1)
        plt.title('{title} - Comparison between Algorithms at size {size}'.format(title=self.title, size=size))
        plt.xlabel('Iteration')
        plt.ylabel("Fitness")
        plt.legend()
        filename = '{title}-Algorithms-Comparison-{size}.png'.format(title=self.title, size=size)
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()

        fig = plt.figure(2)
        plt.title('{title} - Comparison (evaluation) between Algorithms at problem size {size}'.format(title=self.title, size=size))
        plt.xlabel('Evaluation')
        plt.xscale('log')
        plt.ylabel("Fitness")
        plt.legend()
        filename = '{title}-Algorithms-Comparison-Evaulation-{size}.png'.format(title=self.title, size=size)
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()


        print("-----------" + self.title)
        print("Runtime / Iteration: ")
        print(time_list)
        print("minimum required iteration:")
        print(minimum_iteration_list)
        print("Runtime for minimum required iteration: ")
        print([a * b for a, b in zip(time_list, minimum_iteration_list)])
        print("Best Fitness:")
        print(fitness_list)



    def run(self):

        #init
        random.seed(self.RANDOM_SEED)

        self.no_of_items = 30
        self.weights = [random.randint(1, 20) for k in range(self.no_of_items)]
        self.values = [random.randint(1, 50) for k in range(self.no_of_items)]
        fitness = mlrose.CustomFitness(self.custom_function)
        problem = mlrose.DiscreteOpt(self.no_of_items, fitness)
        color_list = ['yellow','red','green','black']

        # different steps of analyze
        self.evaluateRandomHillClimb(problem, color_list)
        self.evaluateSimulatedAnnealing(problem, color_list)
        self.evaluateGeneticAlgorithm(problem, color_list)
        self.evaluateMimic(problem, color_list)
        self.comparePerformance(problem, color_list, self.no_of_items)

        # size vs performance at size 5
        self.no_of_items = 5
        self.weights = [random.randint(1, 20) for k in range(self.no_of_items)]
        self.values = [random.randint(1, 50) for k in range(self.no_of_items)]
        fitness = mlrose.CustomFitness(self.custom_function)
        problem = mlrose.DiscreteOpt(self.no_of_items, fitness)
        self.comparePerformance(problem, color_list, no_of_items)

        # size vs performance at size 10
        self.no_of_items = 10
        self.weights = [random.randint(1, 20) for k in range(self.no_of_items)]
        self.values = [random.randint(1, 50) for k in range(self.no_of_items)]
        fitness = mlrose.CustomFitness(self.custom_function)
        problem = mlrose.DiscreteOpt(self.no_of_items, fitness)
        self.comparePerformance(problem, color_list, no_of_items)

        # size vs performance at size 15
        self.no_of_items = 15
        self.weights = [random.randint(1, 20) for k in range(self.no_of_items)]
        self.values = [random.randint(1, 50) for k in range(self.no_of_items)]
        fitness = mlrose.CustomFitness(self.custom_function)
        problem = mlrose.DiscreteOpt(self.no_of_items, fitness)
        self.comparePerformance(problem, color_list, no_of_items)



    def custom_function(self, state):
        fitness_fn = mlrose.Knapsack(self.weights, self.values, 0.6)
        fitness = fitness_fn.evaluate(state)
        if fitness > self.current_fitness_mark:
            self.current_fitness_mark = fitness
        self.fitness_list.append(self.current_fitness_mark)
        return fitness

