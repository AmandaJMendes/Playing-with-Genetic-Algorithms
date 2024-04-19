import numpy as np

def initialize(population_size, chromossome_dim, data_type, min_value, max_value):
    if data_type == int:
        return np.random.randint(min_value, max_value+1, size = (population_size, chromossome_dim))
    elif data_type == float:
        return np.random.uniform(min_value, max_value+1, size = (population_size, chromossome_dim))
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
    original_population = population.copy()
    elite_length    = int(population.shape[0]*elite_p)
    random_length   = 0
    if elite_p:  
        if mode == "min":
            elite_idx, population_idx = np.split(np.argpartition(scores, elite_length), [elite_length])
        else:
            population_idx, elite_idx = np.split(np.argpartition(scores, -elite_length), [-elite_length])
        population = original_population[population_idx]
        elite      = original_population[elite_idx]
    if random_p:
        random = population[np.random.uniform(size = len(population))<random_p]
        random_length = len(random)

    scores = scores if mode == "max" else scores.max()-scores+scores.min()
    normalized_scores = normalize_fitness(scores)

    selected_parents = selection_algorithm(original_population, normalized_scores,
                                           len(original_population)-(elite_length+random_length))

    all_parents = selected_parents
    if elite_length:
        all_parents = np.concatenate([all_parents, elite])
    if random_length:
        all_parents = np.concatenate([all_parents, random])
    return all_parents

def normalize_fitness(scores):
    return (scores+1E-10)/(scores+1E-10).sum()

def one_point_crossover(parents, mutation_prob, min_value, max_value):
    offspring = []
    for i in range(0, parents.shape[0], 2):
        childA = parents[i].copy()
        childB = parents[i+1].copy()

        crossover_idx = np.random.randint(1, parents.shape[1])
        
        childA[crossover_idx:] = parents[i+1][crossover_idx:]
        childB[crossover_idx:] = parents[i][crossover_idx:]

        mutate_children = np.random.uniform(size = 2) <= mutation_prob
        if mutate_children[0]:
            childA[np.random.randint(parents.shape[1])] = np.random.randint(min_value, max_value+1)
        if mutate_children[1]:
            childB[np.random.randint(parents.shape[1])] = np.random.randint(min_value, max_value+1)

        offspring += [childA, childB]

    return np.array(offspring)


def mutate(parents, p, min_value, max_value):

    return []

if __name__ == "__main__":
    initial_population = initialize(100, 30, float, -5, 100)
    offspring = initial_population
    for i in range(3000):
        fitness = evaluate_fitness(offspring, 3700)
        selected = select(offspring, fitness, 0.01, 0.9, mode = "max")
        offspring = one_point_crossover(selected, 0.01, -5, 1000)
    print(offspring.sum(axis=1))
    print((offspring==offspring[0, :]).all())



    





