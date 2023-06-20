import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from deap import creator, base, tools, algorithms

data = pd.read_excel('project_3_evolution_algorithms\Project3_DistancesMatrix.xlsx', header=None) 

distances = data.values[1:, 1:]  

number_of_ecopoints = len(distances) - 1

def evalTSP(individual: list)-> tuple: 
    """
    Fitness of an individual. The fitness is the total distance of the route.

    Args:
        individual (list): Individual to evaluate

    Returns:
        tuple: Tuple with a float value of Fitness of the individual
    """    

    total_distance = sum(distances[individual[i-1] + 1][individual[i] + 1] for i in range(1, len(individual)))
    total_distance += distances[0][individual[0] + 1]
    total_distance += distances[individual[-1] + 1][0]
    
    # Return a tuple with a single value because it's a requisite of DEAP
    return (total_distance,)  



creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("indices", np.random.permutation, range(number_of_ecopoints)) 
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxOrdered) # cxOrdered - order crossover CANNOT BE USED
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalTSP)



n_populations = 10 * 1000 
n_generations = 10 * n_populations 
prob_mutation = 0.01 
prob_crossover = 0.8 
convergence_generations = n_generations//10   # Number of generations over which to check for convergence
convergence_threshold = 0.0001  # The relative change in best fitness that signifies convergence

# Population
pop = toolbox.population(n=n_populations)

# Hall of fame of 1 best individual
hof = tools.HallOfFame(1)

# Lists to store the fitness evolution
best_fitness_evolution = []
avg_fitness_evolution = []
min_fitness_evolution = []
max_fitness_evolution = []

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

print("Start of evolution")

start_time = time.time()
timeout = 19 * 60  # minutes x seconds

# Run the evolution
gen = 0
converged = False
while time.time() - start_time < timeout and gen < n_generations and not converged:
    gen += 1
    # Select and clone the next generation individuals
    offspring = map(toolbox.clone, toolbox.select(pop, len(pop)))
    
    # Apply crossover and mutation on the offspring
    offspring = algorithms.varAnd(offspring, toolbox, cxpb=prob_crossover, mutpb=prob_mutation)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    # Replace the current population by the offspring
    pop[:] = offspring

    # Update the hall of fame with the generated individuals
    hof.update(pop)

    # Record the stats of the current generation
    record = stats.compile(pop)
    print(f"Generation {gen}: {record}")

    # Record the best, avg, min and max fitness from this generation to the lists
    best_fitness_evolution.append(hof[0].fitness.values[0])
    avg_fitness_evolution.append(record["avg"])
    min_fitness_evolution.append(record["min"])
    max_fitness_evolution.append(record["max"])

    # Check for convergence: if the best fitness has not improved significantly for a certain number of generations
    if gen >= convergence_generations:
        relative_change = (best_fitness_evolution[-convergence_generations] - best_fitness_evolution[-1]) / best_fitness_evolution[-convergence_generations]
        if abs(relative_change) < convergence_threshold:
            print("Convergence reached")
            converged = True

# Best individual
print(f"Best individual is: {hof[0]}, {len(hof[0])} \nwith fitness: {hof[0].fitness}")
print(time.time() - start_time)