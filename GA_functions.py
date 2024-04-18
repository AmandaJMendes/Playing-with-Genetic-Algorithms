import numpy as np

def initialize(population_size, chromossome_dim, data_type, min_value, max_value):
    if data_type == int:
        return np.random.randint(min_value, max_value+1, size = (population_size, chromossome_dim))
    elif data_type == float:
        return np.random.uniform(min_value, max_value+1, size = (population_size, chromossome_dim))
    else:
        raise Excpetion(f"Unsupported data type {data_type}")

def evaluate(population, target):
    return np.abs(target-np.sum(population, axis = 1))

def roulette_selection(population, proabilities):
    parents_idx  = np.random.choice(population.shape[0], size = population.shape[0],
                                    replace = True, p = proabilities)
    parents      = population[parents_idx] 
    return parents

def select(population, scores, mode = "max", selection_algorithm = roulette_selection):
    scores = scores if mode == "max" else scores.max()-scores+scores.min()
    normalized_scores = normalize_fitness(scores)
    parents = selection_algorithm(population, normalized_scores)
    return parents

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
    initial_population = initialize(100, 1000, int, -5, 5)
    offspring = initial_population
    for i in range(800):
        fitness = evaluate(offspring, 723)
        print(i ,fitness.mean())
        selected = select(offspring, fitness, mode = "min")
        offspring = one_point_crossover(selected, 0.008, -5, 5)
    print(offspring.sum(axis=1))
    print((offspring==offspring[0, :]).all())



    





