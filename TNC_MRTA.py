import numpy as np
from MRTA import MultiRobotTaskAllocation
from scipy.optimize import minimize

np.set_printoptions(precision=2)

sample_robots = np.load("random_initial_robots.npy")
sample_tasks = np.load("random_initial_tasks.npy")
print("sample robots: \n", sample_robots)
print("sample tasks: \n", sample_tasks)


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

TNC_options = {'eps': 1e-08, 'scale': None, 'offset': None, 'mesg_num': None, 'maxCGit': 100, 'maxiter': 1000,
               'eta': 0.25, 'stepmx': 10, 'accuracy': 0.00001, 'minfev': 0, 'ftol': 1e-6, 'xtol': 1e-6, 'gtol': 1e-10,
               'rescale': -1, 'disp': True}

sol = minimize(fun, x0, args=(sample_robots, sample_tasks), method='TNC', jac=None, bounds=None, tol=None, callback=None, options=TNC_options)

print(sol)
