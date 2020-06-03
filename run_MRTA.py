import numpy as np
from MRTA import MultiRobotTaskAllocation

np.set_printoptions(precision=2)
num_bots = 10
num_tasks = 20
num_types = 4
home_base = np.array([0, 0])  # position of home base
uav_efficiency = 0.5

# initial robots:
sample_robots = np.array([
    [0, 0, 0, 100, uav_efficiency, 1],
    [1, 1, 0, 100, uav_efficiency, 1],
    [2, 2, 0, 100, uav_efficiency, 1],
    [3, 3, 0, 100, uav_efficiency, 1],
    [4, 4, 0, 100, uav_efficiency, 1],
    [5, 5, 0, 100, uav_efficiency, 1],
    [6, 6, 0, 100, uav_efficiency, 1],
    [7, 7, 0, 100, uav_efficiency, 1],
    [8, 8, 0, 100, uav_efficiency, 1],
    [9, 9, 0, 100, uav_efficiency, 1],

])

# sample_robots = np.transpose([np.linspace(0, num_bots-1, num=num_bots, axis=0)])  # robot index
# sample_robots = np.concatenate((sample_robots, np.random.rand(num_bots, 2)*10), axis=1)  # position information
# sample_robots = np.concatenate((sample_robots, np.ones((num_bots, 1))*100), axis=1)  # battery level information
# sample_robots = np.concatenate((sample_robots, np.random.randint(1, 3, (num_bots, 1))*0.1), axis=1)  # energy efficiency information


sample_robots = np.concatenate((sample_robots, np.random.randint(0, 2, (num_bots, num_types))), axis=1)  # type information

# sample_robots = np.insert(sample_robots, 4, 1, axis=1)  # charging station task
# np.save("random_initial_robots", sample_robots)
# np.savetxt("random_initial_robots", sample_robots)
sample_robots = np.load("random_initial_robots.npy")
# np.transpose(sample_robots)[3] = 1000
print(sample_robots)

# hand crafted objectives:
# sample_tasks = np.array([
#     [0, 0, 50, 0, 0, 0, 0, 0, 0, 0],
#     [1, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [2, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [3, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [4, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [5, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [6, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [7, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [8, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [9, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [0, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [0, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [0, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [0, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [0, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [0, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [0, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [0, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [0, 23, 10, 0, 0, 0, 0, 0, 0, 0],
#     [0, 23, 10, 0, 0, 0, 0, 0, 0, 0]
#     ])

# # randomly generated objectives:
# sample_tasks = np.transpose([np.linspace(0, num_tasks-1, num=num_tasks, axis=0)])  # task index
# # print(sample_tasks)
# sample_tasks = np.concatenate((sample_tasks, np.random.rand(num_tasks, 2)*10), axis=1)  # position information
# sample_tasks = np.concatenate((sample_tasks, np.random.rand(num_tasks, 1)*100), axis=1)  # battery level information
# sample_tasks = np.concatenate((sample_tasks, np.random.randint(1, 3, (num_tasks, 1))*0.1), axis=1)  # energy efficiency information
# sample_tasks = np.concatenate((sample_tasks, np.random.randint(0, num_types, (num_tasks, num_types))), axis=1)  # type information
# sample_tasks = np.insert(sample_tasks, 4, 0, axis=1)  # charging station task
# print(sample_tasks)
# np.save("random_initial_tasks", sample_tasks)
sample_tasks = np.load("random_initial_tasks.npy")
# print(sample_tasks)
# np.savetxt("random_initial_tasks.txt", sample_tasks)
# print(sample_tasks)
# aergvare

print("sample robots: \n", sample_robots)
print("sample tasks: \n", sample_tasks)

MRTA = MultiRobotTaskAllocation(sample_robots, sample_tasks, save_data=True)
MRTA.run(2.2867975297601477, 1.2172393070417535, -1.2983878440623928, -8.620057169901632)


