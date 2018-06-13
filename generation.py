from network import Network
from simulator import Simulator
import numpy as np
import random
import copy

class Generation:
    def __init__(self):
        self.__genomes = [Network() for i in range(12)]
        self.__best_genomes = []
        self.generation = 0
        self.sim = Simulator(render= False) #set render True if you need to watch the simulation, will slow down
        self.expected_fitness =10000

    def execute(self):
        self.generation += 1
        
        for _g, genome in enumerate(self.__genomes):
            i = 0
            while i < self.expected_fitness * 1.5:
                observation = self.sim.get_obs()
                inputs = observation
                outputs = genome.forward(np.array(inputs, dtype=float))
                if outputs[0] > 0.55:
                    action = 1
                else:
                    action = 0
                self.sim.next_step(action)
                if self.sim.get_status():
                    break
                i += 1
            genome.fitness = self.sim.get_fitness()
            print('Generation {} genome {} fitness {}'.format(self.generation, _g+1, self.sim.get_fitness()))
            self.sim.reset()

    def keep_best_genomes(self):
        self.__genomes.sort(key=lambda x: x.fitness, reverse=True)
        self.__genomes = self.__genomes[:4]
        self.__best_genomes = self.__genomes[:]

    def mutations(self):
        while len(self.__genomes) < 10:
            genome1 = random.choice(self.__best_genomes)
            genome2 = random.choice(self.__best_genomes)
            self.__genomes.append(self.mutate(self.cross_over(genome1, genome2)))
        while len(self.__genomes) < 12:
            genome = random.choice(self.__best_genomes)
            self.__genomes.append(self.mutate(genome))

    def cross_over(self, genome1, genome2):
        new_genome = copy.deepcopy(genome1)
        other_genome = copy.deepcopy(genome2)
        cut_location = int(len(new_genome.W1) * random.uniform(0, 1))
        for i in range(cut_location):
            new_genome.W1[i], other_genome.W1[i] = other_genome.W1[i], new_genome.W1[i]
        cut_location = int(len(new_genome.W2) * random.uniform(0, 1))
        for i in range(cut_location):
            new_genome.W2[i], other_genome.W2[i] = other_genome.W2[i], new_genome.W2[i]
        return new_genome

    def __mutate_weights(self, weights):
        if random.uniform(0, 1) < 0.2:
            return weights * (random.uniform(0, 1) - 0.5) * 3 + (random.uniform(0, 1) - 0.5)
        else:
            return 0

    def mutate(self, genome):
        new_genome = copy.deepcopy(genome)
        new_genome.W1 += self.__mutate_weights(new_genome.W1)
        new_genome.W2 += self.__mutate_weights(new_genome.W2)
        return new_genome
