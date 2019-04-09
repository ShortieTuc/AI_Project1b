import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Any


def create_starting_population(individuals, length_x, length_y):

    pop = np.random.randint(0, 4, size=(individuals, length_y, length_x))

    return pop


def feasibility_check(population,pop_size, length_x, length_y):

    fitness_table = np.zeros(pop_size)

    hc = np.zeros((3, 14))
    for i in range(3):
        for j in range(14):
            if ((i == 0 and j == 0) or (i == 0 and j == 1) or (i == 0 and j == 7) or (i == 0 and j == 8) or (
                    i == 1 and j == 0) or (i == 1 and j == 1) or (i == 1 and j == 2) or (i == 1 and j == 4) or (
                    i == 1 and j == 7) or (i == 1 and j == 8) or (i == 1 and j == 9) or (i == 1 and j == 11)):
                hc[i][j] = 10
            else:
                hc[i][j] = 5

    ones = 0
    twos = 0
    threes = 0

    for k in range(pop_size):  # population size
        for j in range(length_x):  # days
            for i in range(length_y):  # employees
                if population[k, i, j] == 1:
                    ones += 1
                elif population[k, i, j] == 2:
                    twos += 1
                elif population[k, i, j] == 3:
                    threes += 1
            if hc[j, 0] == ones and hc[j, 1] == twos and hc[j, 2] == threes:
                ones = 0
                twos = 0
                threes = 0
                fitness_table[k] = 1
            else:
                break

    return fitness_table


def select_individual_by_tournament(population, scores):
    # Get population size
    population_size = len(scores)

    # Pick individuals for tournament
    fighter_1 = random.randint(0, population_size - 1)
    fighter_2 = random.randint(0, population_size - 1)

    # Get fitness score for each
    fighter_1_fitness = scores[fighter_1]
    fighter_2_fitness = scores[fighter_2]

    # Identify undividual with highest fitness
    # Fighter 1 will win if score are equal
    if fighter_1_fitness >= fighter_2_fitness:
        winner = fighter_1
    else:
        winner = fighter_2

    # Return the chromsome of the winner
    return population[winner, :]

'''
# Set up and score population
reference = create_reference_solution(10)
population = create_starting_population(6, 10)
scores = calculate_fitness(reference, population)

# Pick two parents and dispplay
parent_1 = select_individual_by_tournament(population, scores)
parent_2 = select_individual_by_tournament(population, scores)
print (parent_1)
print (parent_2)
'''


def breed_by_crossover(parent_1, parent_2):
    # Get length of chromosome
    chromosome_length = len(parent_1)

    # Pick crossover point, avoding ends of chromsome
    crossover_point = random.randint(1, chromosome_length - 1)

    # Create children. np.hstack joins two arrays
    child_1 = np.hstack((parent_1[0:crossover_point],
                         parent_2[crossover_point:]))

    child_2 = np.hstack((parent_2[0:crossover_point],
                         parent_1[crossover_point:]))

    # Return children
    return child_1, child_2

'''
# Set up and score population
reference = create_reference_solution(15)
population = create_starting_population(100, 15)
scores = calculate_fitness(reference, population)

# Pick two parents and dispplay
parent_1 = select_individual_by_tournament(population, scores)
parent_2 = select_individual_by_tournament(population, scores)

# Get children
child_1, child_2 = breed_by_crossover(parent_1, parent_2)

# Show output
print ('Parents')
print (parent_1)
print (parent_2)
print ('Children')
print (child_1)
print (child_2)
'''


def randomly_mutate_population(population, mutation_probability):
    # Apply random mutation
    random_mutation_array = np.random.random(
        size=(population.shape))

    random_mutation_boolean = \
        random_mutation_array <= mutation_probability

    population[random_mutation_boolean] = np.logical_not(population[random_mutation_boolean])

    # Return mutation population
    return population

'''
# Set up and score population
reference = create_reference_solution(15)
population = create_starting_population(100, 15)
scores = calculate_fitness(reference, population)

# Pick two parents and display
parent_1 = select_individual_by_tournament(population, scores)
parent_2 = select_individual_by_tournament(population, scores)

# Get children and make new population 
child_1, child_2 = breed_by_crossover(parent_1, parent_2)
population = np.stack((child_1, child_2))

# Mutate population
mutation_probability = 0.25
print ("Population before mutation")
print (population)
population = randomly_mutate_population(population, mutation_probability)
print ("Population after mutation")
print (population)
'''

# Set general parameters
chromosome_length_x = 14  # parallelism with days
chromosome_length_y = 30  # parallelism with employees
population_size = 200
maximum_generation = 30
best_score_progress = []  # Tracks progress

# Create starting population
population = create_starting_population(population_size, chromosome_length_x,chromosome_length_y)
#print(population)

# Display best score in starting population
check_table = feasibility_check(population, population_size, chromosome_length_x, chromosome_length_y)
print(check_table)
'''
best_score = np.max(scores) / chromosome_length * 100
print('Starting best score, percent target: %.1f' % best_score)

# Add starting best score to progress tracker
best_score_progress.append(best_score)

# Now we'll go through the generations of genetic algorithm
for generation in range(maximum_generation):
    # Create an empty list for new population
    new_population = []

    # Create new popualtion generating two children at a time
    for i in range(int(population_size / 2)):
        parent_1 = select_individual_by_tournament(population, scores)
        parent_2 = select_individual_by_tournament(population, scores)
        child_1, child_2 = breed_by_crossover(parent_1, parent_2)
        new_population.append(child_1)
        new_population.append(child_2)

    # Replace the old population with the new one
    population = np.array(new_population)

    # Score best solution, and add to tracker
    scores = calculate_fitness(reference, population)
    best_score = np.max(scores) / chromosome_length * 100
    best_score_progress.append(best_score)

# GA has completed required generation
print('End best score, percent target: %.1f' % best_score)

# Plot progress
plt.figure()
plt.plot(best_score_progress)
plt.xlabel('Generation')
plt.ylabel('Best score (% target)')
plt.show()
'''