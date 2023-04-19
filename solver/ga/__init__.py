import random


class Problem:
    def __init__(self, num_dim, fitness, range_max=1):
        self.num_dim = num_dim
        self.fitness = fitness
        self.range_max = range_max


class GA:
    class Chromosome:
        def __init__(self, problem: Problem):
            self.problem = problem
            self.genes = []
            self.fitness = None
            self.init_genes()
            self.calculate_fitness()

        def init_genes(self):
            self.genes = [random.randint(0, self.problem.range_max) for _ in range(self.problem.num_dim)]

        def calculate_fitness(self):
            self.fitness = self.problem.fitness(self.genes)

        def mutate(self, mutation_rate):
            for i in range(self.problem.num_dim):
                if random.random() < mutation_rate:
                    self.genes[i] = random.randint(0, self.problem.range_max)
            self.calculate_fitness()

        def crossover(self, other):
            child1 = GA.Chromosome(self.problem)
            child2 = GA.Chromosome(self.problem)
            for i in range(self.problem.num_dim):
                if random.random() < 0.5:
                    child1.genes[i] = self.genes[i]
                    child2.genes[i] = other.genes[i]
                else:
                    child1.genes[i] = other.genes[i]
                    child2.genes[i] = self.genes[i]
            child1.calculate_fitness()
            child2.calculate_fitness()
            return child1, child2

    def __init__(self, population_size, mutation_rate, crossover_rate, tournament_size, generations, problem):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.generations = generations
        self.problem = problem
        self.population = []
        self.best_solution = None

    def init_population(self):
        self.population = [GA.Chromosome(self.problem) for _ in range(self.population_size)]

    def run(self):
        self.init_population()
        for generation in range(self.generations):
            self.population = self.selection()
            self.crossover()
            self.mutation()
            self.best_solution = self.get_best_solution()
            # print("Generation: %d, Best solution: %s" % (generation, self.best_solution.fitness))
            # print(self.best_solution.genes)
        return self.best_solution.genes, self.best_solution.fitness

    def selection(self):
        #self.population.sort(key=lambda x: x.fitness, reverse=True)
        new_population = [] + self.population[:2]
        while len(new_population) < self.population_size:
            tournament = self.tournament_selection()
            if tournament not in new_population:
                new_population.append(tournament)
        return new_population

    def tournament_selection(self):
        tournament = []
        for _ in range(self.tournament_size):
            index = random.randint(0, self.population_size - 1)
            tournament.append(self.population[index])
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[0]

    def crossover(self):
        new_population = [] + self.population
        while len(new_population) < self.population_size * 2:
            index1 = random.randint(0, self.population_size - 1)
            index2 = random.randint(0, self.population_size - 1)
            child1, child2 = self.population[index1].crossover(self.population[index2])
            new_population.append(child1)
            new_population.append(child2)

        self.population = new_population

    def mutation(self):
        for i in range(2, self.population_size):
            self.population[i].mutate(self.mutation_rate)

    def get_best_solution(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[0]


def run_test():
    problem = Problem(100, lambda x: sum(x), range_max=4)
    ga = GA(population_size=100, mutation_rate=0.2, crossover_rate=0.2, tournament_size=2, generations=1000,
            problem=problem)
    ga.run()


if __name__ == '__main__':
    run_test()
