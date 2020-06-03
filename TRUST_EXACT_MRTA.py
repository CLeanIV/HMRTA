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


sol = minimize(fun, x0, args=(sample_robots, sample_tasks), method='trust-exact', jac=None, hess=None, tol=None, callback=None, options={})

print(sol)
