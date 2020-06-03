import numpy as np
from MRTA import MultiRobotTaskAllocation
from scipy.optimize import minimize

np.set_printoptions(precision=2)

sample_robots = np.load("random_initial_robots.npy")
sample_tasks = np.load("random_initial_tasks.npy")


def fun(x, *arg):
    robots = arg[0]
    tasks = arg[1]
    alpha = x[0]
    beta = x[1]
    gamma = x[2]
    zeta = [3]
    MRTA = MultiRobotTaskAllocation(robots, tasks)
    func = MRTA.run(alpha, beta, gamma, zeta)
    return func


low = [-10, -10, -10, -10]
high = [10, 10, 10, 10]
x0 = np.random.uniform(low, high, (1, 4))
# print("initial swarm: \n", initial_swarm)

"""

"""

nelder_mead_options = {'maxiter': None, 'maxfev': None, 'disp': True, 'return_all': True,
                       'initial_simplex': None, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': False}

sol = minimize(fun, x0, args=(sample_robots, sample_tasks), method='Nelder-Mead', tol=1e-6, callback=None, options=nelder_mead_options)

print(sol)
