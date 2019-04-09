import random
import numpy as np
import matplotlib.pyplot as plt


def create_starting_population(individuals, length_x, length_y):

    pop = np.random.randint(0, 4, size=(individuals, length_y, length_x))

    return pop


def feasibility_check(pop, pop_size, length_x, length_y):

    fitness_table = np.zeros(pop_size)  # feasible table will be marked as '1', others '0'

    # Table 6 (Hard Constraint) in page 14 from assignment
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

    # Access 3-D Table of Population
    for k in range(pop_size):  # population size
        for j in range(length_x):  # days
            for i in range(length_y):  # employees
                if pop[k, i, j] == 1:
                    ones += 1
                elif pop[k, i, j] == 2:
                    twos += 1
                elif pop[k, i, j] == 3:
                    threes += 1
            if hc[0, j] == ones and hc[1, j] == twos and hc[2, j] == threes:
                fitness_table[k] = 1
            else:
                ones = 0
                twos = 0
                threes = 0
                break
    return fitness_table


# Set general parameters
chromosome_length_x = 14   # parallelism with days
chromosome_length_y = 30   # parallelism with employees
population_size = 1000000  # 1 million (!!!)
maximum_generation = 30
best_score_progress = []   # Tracks progress

# Create starting population
population = create_starting_population(population_size, chromosome_length_x, chromosome_length_y)
# print(population)

# Make Hard Constraint Check
check_table = feasibility_check(population, population_size, chromosome_length_x, chromosome_length_y)
# print(check_table)

count = 0
for i in range(population_size):
    if check_table[i] == 1:
        count += 1
print(count)
