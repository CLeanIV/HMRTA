import numpy as np
import json
from MRTA import MultiRobotTaskAllocation
from genetic_algorithm.main import genetic_optimisation

np.set_printoptions(precision=2)


def objfun(alpha, beta, gamma, zeta):
    robots = np.load("random_initial_robots.npy")
    tasks = np.load("random_initial_tasks.npy")
    MRTA = MultiRobotTaskAllocation(robots, tasks)
    func = -MRTA.run(alpha, beta, gamma, zeta)
    return func


def fitness(params):
    return objfun(**params)


low = [-10, -10, -10, -10]
high = [10, 10, 10, 10]

param_space = {"alpha": [low[0], high[0]], "beta": [low[1], high[1]], "gamma": [low[2], high[2]], "zeta": [low[3], high[3]]}

sol = genetic_optimisation(input_model=fitness, param_space=param_space, pop_size=1000, num_parents=8,
                     max_num_generations=100, mutation_prob=0.35, stoping_rounds=15, integer_params=[])

list_key_value = list([k] for k in sol.items())
print(list_key_value)
np.savetxt('GA_results_1.txt', list_key_value)
