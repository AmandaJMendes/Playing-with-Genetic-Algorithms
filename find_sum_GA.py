import numpy as np

def initialize(population_size, chromossome_dim, data_type, min_value, max_value):
    if data_type == int:
        return np.random.randint(min_value, max_value+1, size = (population_size, chromossome_dim))
    elif data_type == float:
        return np.random.uniform(min_value, max_value, size = (population_size, chromossome_dim))
    else:
        raise Excpetion(f"Unsupported data type {data_type}")

def evaluate_fitness(population, target):
    return np.abs(target-np.sum(population, axis = 1))

def roulette_selection(population, proabilities, parents_size):
    parents_idx  = np.random.choice(population.shape[0], size = parents_size,
                                    replace = True, p = proabilities)
    parents      = population[parents_idx] 
    return parents

def select(population, scores, elite_p = 0.0, random_p = 0.0, mode = "max", selection_algorithm = roulette_selection):
    elite_length, random_length = int(population.shape[0]*elite_p), 0
    all_parents = []
    if elite_p: 
        if mode == "min":
            elite, others = np.split(population[np.argpartition(scores, elite_length)], [elite_length])
        else:
            others, elite = np.split(population[np.argpartition(scores, -elite_length)], [-elite_length])
        all_parents.append(elite)
    if random_p:
        random = others[np.random.uniform(size = len(others))<random_p]
        random_length = len(random)
        all_parents.append(random)
    scores = scores if mode == "max" else scores.max()-scores+scores.min()
    normalized_scores = normalize_fitness(scores)
    selected_parents = selection_algorithm(population, normalized_scores, len(population)-(elite_length+random_length))
    all_parents.append(selected_parents)
    return np.concatenate(all_parents)

def normalize_fitness(scores):
    return (scores+1E-10)/(scores+1E-10).sum()

def one_point_crossover(parents):
    offspring = []
    n_couples = int(np.ceil(parents.shape[0]/2))
    crossover_idxs = np.random.randint(1, parents.shape[1], size = n_couples)
    couples = np.random.randint(1, parents.shape[0], size = (n_couples, 2))

    for i in range(parents.shape[0]//2):
        parentA, parentB = parents[couples[i]]
        childA, childB   = parentA.copy(), parentB.copy()
        childA[crossover_idxs[i]:] = parentB[crossover_idxs[i]:]
        childB[crossover_idxs[i]:] = parentA[crossover_idxs[i]:]
        offspring += [childA, childB]

    offspring = offspring[:-1] if parents.shape[0] % 2 else offspring
    return np.array(offspring)

def mutate(population, mutation_prob, min_value, max_value):
    if mutation_prob:
        mutation_mask      = np.random.uniform(size = population.shape[0]) <= mutation_prob
        children_to_mutate = mutation_mask.nonzero()[0]
        gens_to_mutate     = np.random.randint(0, population.shape[1], size = len(children_to_mutate))

        if population.dtype == int:
            new_gens = np.random.randint(min_value, max_value+1, size = len(children_to_mutate))
        else:
            new_gens = np.random.uniform(min_value, max_value,   size = len(children_to_mutate))
        population[children_to_mutate, gens_to_mutate] = new_gens
    return population

if __name__ == "__main__":
    import matplotlib.pyplot as plt 

    POPULATION_SIZE = 100       # Number of individuals in each generation 
    CHRMOSSOME_SIZE = 10        # Number of gens in each individual
    MIN, MAX        = -10, 1000 # Range 
    DTYPE           = float     # Either float or int
    N_GENERATIONS   = 800       # Number of generations / epochs
    TARGET          = 3700      # Target sum
    ELITE_P         = 0.1       # Float in range [0, 1]: percentage of the population selected through elitism
    RANDOM_RATE     = 0.001     # Chance of selecting an individual that was not selected through elitism
    MUTATION_RATE   = 0.01      # Chance of mutating each individual

    initial_population = initialize(POPULATION_SIZE, CHRMOSSOME_SIZE, DTYPE, MIN, MAX)
    
    offspring   = initial_population
    avg_fitness = []
    for i in range(N_GENERATIONS):
        fitness   = evaluate_fitness(offspring, TARGET)
        selected  = select(offspring, fitness, ELITE_P, RANDOM_RATE, "min", roulette_selection)
        offspring = one_point_crossover(selected)
        offspring = mutate(offspring, MUTATION_RATE, MIN, MAX)
        avg_fitness.append(fitness.mean())
    final_fitness = evaluate_fitness(offspring, TARGET).mean()
    avg_fitness.append(final_fitness)

    solutions = np.unique(offspring, axis = 0, return_counts = True)
    for solution, count in zip(solutions[0], solutions[1]):
        print(solution, ' | Occurences: ', count, ' | Sum: ', solution.sum())

    plt.plot(list(range(N_GENERATIONS+1)), avg_fitness)
    plt.show()