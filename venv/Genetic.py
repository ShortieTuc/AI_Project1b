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


def fitness_check(fitness_table, pop, pop_size, length_x, length_y):
    # Score Array for fitness check
    score = np.zeros(pop_size)
    hours = 0
    n_days_off = 0
    d_days_off = 0
    for k in range(pop_size):  # population size
        score[k] = 0
        if fitness_table[k] == 1:
            for i in range(length_y):  # employees
                if hours >= 70:  # Max 70 hours
                    score[k] += 1000
                if n_days_off < 2:  # 2 Days off after 4 consecutive night shifts
                    score[k] += 100
                if d_days_off < 2:  # 2 Days off after 7 consecutive workdays
                    score[k] += 1
                hours = 0
                consecutive_days = 0
                consecutive_nights = 0
                night_flag = 0
                evening_flag = 0
                nights = 0
                n_days_off = 0
                workdays = 0
                d_days_off = 0
                dayoff_flag = 0

                for j in range(length_x):  # days
                    if pop[k, i, j] == 1:  # morning_shift
                        hours += 8
                        consecutive_days += 1
                        consecutive_nights = 0
                        if night_flag == 1:  # Morning shift after night shift
                            night_flag = 0
                            score[k] += 1000
                        if evening_flag == 1:  # Morning shift after evening shift
                            evening_flag = 0
                            score[k] += 800
                        workdays += 1
                        if consecutive_days == 1 and dayoff_flag == 1:  # Workday - Day off - Workday
                            score[k] += 1
                            dayoff_flag = 0
                    elif pop[k, i, j] == 2:  # evening_shift
                        hours += 8
                        consecutive_days += 1
                        consecutive_nights = 0
                        if night_flag == 1:
                            night_flag = 0
                            score[k] += 600
                        evening_flag = 1
                        workdays += 1
                        if consecutive_days == 1 and dayoff_flag == 1:  # Workday - Day off - Workday
                            score[k] += 1
                            dayoff_flag = 0

                    elif pop[k, i, j] == 3:  # night_shift
                        hours += 10
                        consecutive_days += 1
                        consecutive_nights += 1
                        consecutive_nights = 0
                        night_flag = 1
                        nights += 1
                        workdays += 1
                        if consecutive_days == 1 and dayoff_flag == 1:  # Workday - Day off - Workday
                            score[k] += 1
                            dayoff_flag = 0

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
                            if  dayoff_flag == 0:  # Day off - Workday - Day off
                                score[k] += 1
                            dayoff_flag = 1
                            consecutive_days = 0
                        else:
                            consecutive_days = 0
                        if j == 6 or j == 13:  # Worked Saturday but not Sunday
                            if pop[k, i, j-1] != 0:
                                score[k] += 1
                        if j == 5 or j == 12:  # Worked Sunday but not Saturday
                            if pop[k, i, j+1] != 0:
                                score += 1

                    if consecutive_days > 7:  # Worked more than 7 days in a row
                        score[k] += 1000
                        consecutive_days = 0
                        consecutive_nights = 0

                    if consecutive_nights > 4:  # Worked more than 4 nights in a row
                        score[k] += 1000
                        consecutive_nights = 0

                    if j == 13:  # Worked both weekends
                        if pop[k, i, j] != 0 and  pop[k, i, j-1] != 0:
                            if pop[k, i, j-7] != 0 and pop[k, i, j-8] != 0:
                                score[k] += 1
    return score


# Set general parameters
chromosome_length_x = 14   # parallelism with days
chromosome_length_y = 30   # parallelism with employees
population_size = 100000  # 1 million (!!!)
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
