import random
import numpy as np
import matplotlib.pyplot as plt


def create_starting_population(pop_size, length_x, length_y):
    pop = np.random.randint(0, 4, size=(pop_size, length_y, length_x))

    return pop


def feasibility_check(pop, pop_size, length_x, length_y):
    fitness_table = np.zeros(pop_size)  # feasible table will be marked as '1', others '0'

    # Table 6 (Hard Constraint) in page 14 from assignment
    hc = np.zeros((3, 14))
    for i in range(3):
        for j in range(14):
            if ((i == 0 and j == 0) or (i == 0 and j == 1) or (i == 0 and j == 7) or (i == 0 and j == 8) or
                    (i == 1 and j == 0) or (i == 1 and j == 1) or (i == 1 and j == 2) or (i == 1 and j == 4) or
                    (i == 1 and j == 7) or (i == 1 and j == 8) or (i == 1 and j == 9) or (i == 1 and j == 11)):
                hc[i][j] = 10
            else:
                hc[i][j] = 5

    ones = 0
    twos = 0
    threes = 0

    # Access 3-D Table of Population
    for kk in range(pop_size):  # population size
        for jj in range(length_x):  # days
            for ii in range(length_y):  # employees
                if pop[kk, ii, jj] == 1:
                    ones += 1
                elif pop[kk, ii, jj] == 2:
                    twos += 1
                elif pop[kk, ii, jj] == 3:
                    threes += 1
            if hc[0, jj] == ones and hc[1, jj] == twos and hc[2, jj] == threes:
                fitness_table[kk] = 1
            else:
                ones = 0
                twos = 0
                threes = 0
                break
    return fitness_table


def fitness_check(fitness_table, pop, pop_size, length_x, length_y):
    # score__table Array for fitness check
    score__table = np.zeros(pop_size)
    hours = 0
    n_days_off = 0
    d_days_off = 0
    for k in range(pop_size):  # population size
        score__table[k] = 0
        if fitness_table[k] == 1:
            for i in range(length_y):  # employees
                if hours >= 70:  # Max 70 hours
                    score__table[k] += 1000
                if n_days_off < 2:  # 2 Days off after 4 consecutive night shifts
                    score__table[k] += 100
                if d_days_off < 2:  # 2 Days off after 7 consecutive workdays
                    score__table[k] += 1
                hours = 0
                consecutive_days = 0
                consecutive_nights = 0
                night_flag = 0
                evening_flag = 0
                nights = 0
                n_days_off = 0
                workdays = 0
                d_days_off = 0
                day_off_flag = 0
                for j in range(length_x):  # days
                    if pop[k, i, j] == 1:  # morning_shift
                        hours += 8
                        consecutive_days += 1
                        consecutive_nights = 0
                        if night_flag == 1:  # Morning shift after night shift
                            night_flag = 0
                            score__table[k] += 1000
                        if evening_flag == 1:  # Morning shift after evening shift
                            evening_flag = 0
                            score__table[k] += 800
                        workdays += 1
                        if consecutive_days == 1 and day_off_flag == 1:  # Workday -> Day off -> Workday
                            score__table[k] += 1
                            day_off_flag = 0
                    elif pop[k, i, j] == 2:  # evening_shift
                        hours += 8
                        consecutive_days += 1
                        consecutive_nights = 0
                        if night_flag == 1:
                            night_flag = 0
                            score__table[k] += 600
                        evening_flag = 1
                        workdays += 1
                        if consecutive_days == 1 and day_off_flag == 1:  # Workday -> Day off -> Workday
                            score__table[k] += 1
                            day_off_flag = 0
                    elif pop[k, i, j] == 3:  # night_shift
                        hours += 10
                        consecutive_days += 1
                        consecutive_nights += 1
                        consecutive_nights = 0
                        night_flag = 1
                        nights += 1
                        workdays += 1
                        if consecutive_days == 1 and day_off_flag == 1:  # Workday -> Day off -> Workday
                            score__table[k] += 1
                            day_off_flag = 0
                    else:  # day off
                        if night_flag == 1:
                            night_flag = 0
                        if evening_flag == 1:
                            evening_flag = 0
                        if nights >= 4:
                            n_days_off += 1
                        if workdays >= 7:
                            d_days_off += 1
                        if consecutive_days == 1:
                            if day_off_flag == 0:  # Day off -> Workday -> Day off
                                score__table[k] += 1
                            day_off_flag = 1
                            consecutive_days = 0
                        else:
                            consecutive_days = 0
                        if j == 6 or j == 13:  # Worked Saturday but not Sunday
                            if pop[k, i, j - 1] != 0:
                                score__table[k] += 1
                        if j == 5 or j == 12:  # Worked Sunday but not Saturday
                            if pop[k, i, j + 1] != 0:
                                score__table[k] += 1
                    if consecutive_days > 7:  # Worked more than 7 days in a row
                        score__table[k] += 1000
                        consecutive_days = 0
                        consecutive_nights = 0
                    if consecutive_nights > 4:  # Worked more than 4 nights in a row
                        score__table[k] += 1000
                        consecutive_nights = 0
                    if j == 13:  # Worked both weekends
                        if pop[k, i, j] != 0 and pop[k, i, j - 1] != 0:
                            if pop[k, i, j - 7] != 0 and pop[k, i, j - 8] != 0:
                                score__table[k] += 1
    return score__table


def roulette_selection(passed_chr, score__table):
    sum = 0
    keys = []
    for ii in range(len(passed_chr)):
        keys.append(sum)
        sum += score__table[passed_chr[ii]]

    max = sum
    key = np.random.randint(0, max)

    for ii in range(len(keys)):
        if keys[ii] > key:
            return passed_chr[ii]


def one_point_crossover(parent1, parent2, len_x):

    # print('\nParent 1: \n', parent1)
    # print('\nParent 2: \n', parent2)

    # Take a random pick from x axis
    random_crossover_point_x = np.random.randint(1, len_x-1)

    # print('\nRand x: ', random_crossover_point_x)
    # print('\n')

    child_ = np.hstack((parent1[:, 0:random_crossover_point_x], parent2[:, random_crossover_point_x:]))
    return child_


def multi_point_crossover(parent1, parent2, len_x):

    # print('\nParent 1: \n', parent1)
    # print('\nParent 2: \n', parent2)

    # Array of crossover points
    random_crossover_point_x = []

    # Generate the crossover points
    for i in range (3):
        # Take a random pick from x axis
        random_crossover_point_x.append(np.random.randint(1, len_x - 1))

    # Sort the points to make the join right
    random_crossover_point_x.sort()

    child__ = np.hstack((parent1[:, 0:random_crossover_point_x[0]], parent2[:, random_crossover_point_x[0]:random_crossover_point_x[1]], parent1[:, random_crossover_point_x[1]:random_crossover_point_x[2]], parent2[:,random_crossover_point_x[2]:]))
    return child__


def mutation_by_transposition(child_):

    nr = child_.shape[0]  # number of rows
    nc = child_.shape[1]  # number of columns
    kx = 7
    ky = 15

    l1 = child_[:ky, :kx]
    l2 = child_[:ky, kx:nc]
    l3 = child_[ky:nr, :kx]
    l4 = child_[ky:nr, kx:nc]

    mutated_child_1 = np.hstack((l4, l3))
    mutated_child_2 = np.hstack((l2, l1))

    mutated_child_ = np.vstack((mutated_child_1, mutated_child_2))

    return mutated_child_


def mutation_by_gene(child_, p_mut):

    nr = child_.shape[0]  # number of rows
    nc = child_.shape[1]  # number of columns

    for ii in range(nr):
        for jj in range(nc):
            roll_ = np.random.random()
            if roll_ > p_mut:
                if child_[ii][jj] == 1:
                    child_[ii][jj] = 2
                elif child_[ii][jj] == 2:
                    child_[ii][jj] = 3
                elif child_[ii][jj] == 3:
                    child_[ii][jj] = 0
                else:
                    child_[ii][jj] = 1
    return child_


# Set general parameters
chromosome_length_x = 14   # parallelism with days
chromosome_length_y = 30   # parallelism with employees
population_size = 10000  # 1 million
maximum_generations = 5
p_sel = 0.05    # Probability of selection
p_cross = 0.3  # Probability of crossover
p_mut_t = 0.3  # Probability of mutation by transposition
p_mut_g = 0.3  # Probability of mutation by gene
best_score_progress = []  # Tracks progress

# Create starting population
population = create_starting_population(population_size, chromosome_length_x, chromosome_length_y)

# Make Hard Constraint Check
check_table = feasibility_check(population, population_size, chromosome_length_x, chromosome_length_y)
# print(check_table)

passed_chromosomes = []

# Take indices of passed chromosomes from "Hard Constrains Check"
for i in range(population_size):
    if check_table[i] == 1:
        passed_chromosomes.append(i)

print('\nPassed: ', len(passed_chromosomes))

# Make Soft Constraint Check and Calculate Score
score_table = fitness_check(check_table, population, population_size, chromosome_length_x, chromosome_length_y)

# Take best score of starting population and put it in best_score_progress table
best_score = np.max(score_table)
best_score_progress.append(best_score)

# This loop is also the termination criterion of our genetic algorithm
for k in range(maximum_generations):

    print('Generation: ', k+1)

    new_population = []

    # Create new population generating one child at a time
    for i in range(int(population_size / 2)):
        # Select two valid chromosomes via weighted roulette
        p_sel_roll = np.random.random()  # Roll for selection
        if p_sel_roll > p_sel:  # Select Passed
            parent_1_idx = roulette_selection(passed_chromosomes, score_table)
            parent_2_idx = roulette_selection(passed_chromosomes, score_table)
            # We don't want duplicates!
            while parent_1_idx == parent_2_idx:
                parent_2_idx = roulette_selection(passed_chromosomes, score_table)

            p_cross_roll = np.random.random()  # Roll for crossover
            if p_cross_roll > p_cross:  # Crossover passed
                # One-Point Crossover by column
                if parent_1_idx is not None and parent_2_idx is not None:
                    child = one_point_crossover(population[parent_1_idx], population[parent_2_idx], chromosome_length_x)
                    new_population.append(child)
                # Multi-Point Crossover by column
                # child = multi_point_crossover(population[parent_1_idx], population[parent_2_idx], chromosome_length_x)
                # print('\nChild: \n', child)
                roll = np.random.random()  # Roll for mutation
                if roll > p_mut_t:
                    # Mutation by transposition
                    mutated_child = mutation_by_transposition(child)
                    # print('\nMutated Child: \n', mutated_child)
                # Mutation by gene
                # mutated_child = mutation_by_gene(child, p_mut_g)
                # print('\nMutated Child: \n', mutated_child)

    population = np.array(new_population)
    # Make Hard Constraint Check
    check_table = feasibility_check(population, len(population), chromosome_length_x, chromosome_length_y)
    # print(check_table)

    passed_chromosomes = []

    # Take indices of passed chromosomes from "Hard Constrains Check"
    for i in range(len(population)):
        if check_table[i] == 1:
            passed_chromosomes.append(i)

    print('\nPassed: ', len(passed_chromosomes))

    # Make Soft Constraint Check and Calculate Score
    score_table = fitness_check(check_table, population, len(population), chromosome_length_x, chromosome_length_y)

    # Take best score of new population and put it in best_score_progress table
    best_score = np.max(score_table)
    best_score_progress.append(best_score)

# Plot progress
plt.plot(best_score_progress)
plt.xlabel('Generation')
plt.ylabel('Best score (% target)')
plt.show()
