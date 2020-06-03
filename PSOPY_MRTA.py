import numpy as np
from MRTA import MultiRobotTaskAllocation
from psopy import minimize

np.set_printoptions(precision=2)

sample_robots = np.load("random_initial_robots.npy")
sample_tasks = np.load("random_initial_tasks.npy")
print("sample robots: \n", sample_robots)
print("sample tasks: \n", sample_tasks)


def objfun(x, *arg):
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
initial_swarm = np.random.uniform(low, high, (100, 4))
# print("initial swarm: \n", initial_swarm)

"""
options : dict, optional
    A dictionary of solver options:
        friction : float, optional
            Velocity is scaled by friction before updating, default 0.8.
        g_rate : float, optional
            Global learning rate, default 0.8.
        l_rate : float, optional
            Local (or particle) learning rate, default 0.5.
        max_velocity : float, optional
            Threshold for velocity, default 5.0.
        max_iter : int, optional
            Maximum iterations to wait for convergence, default 1000.
        stable_iter : int, optional
            Number of iterations to wait before Swarm is declared stable,
            default 100.
        ptol : float, optional
            Change in position should be greater than ``ptol`` to update,
            otherwise particle is considered stable, default 1e-6.
        ctol : float, optional
            Acceptable error in constraint satisfaction, default 1e-6.
        sttol : float, optional
            Tolerance to convert strict inequalities to non-strict
            inequalities, default 1e-6.
        eqtol : float, optional
            Tolerance to convert equalities to non-strict inequalities,
            default 1e-7.
        verbose : bool, optional
            Set True to display convergence messages.
        savefile : string or None, optional
            File to save global best solution vector and its corresponding
            function value for each iteration as a csv file. If None, no
            data is saved.
"""

pso_options = {"friction": 0.75, "max_velocity": 5.0, "g_rate": 0, "l_rate": 1, "max_iter": 1000, "stable_iter": 15,
               "ptol": 0.001, "ctol": 1e-6, "verbose": True, "savefile": "100_particle_solution_12.csv"}

sol = minimize(objfun, x0=initial_swarm, args=(sample_robots, sample_tasks), tol=1e-6, callback=None,
               options=pso_options)

print(sol)
