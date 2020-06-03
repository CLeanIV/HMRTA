import numpy as np
np.set_printoptions(precision=2)
num_bots = 5
num_tasks = 1
num_types = 2
home_base = np.array([0, 0])  # position of home base

sample = np.random.rand(num_bots, 2)*100  # position information
sample = np.concatenate((sample,np.random.rand(num_bots, 1)*100), axis=1)  # battery level information
sample = np.concatenate((sample,np.random.randint(1, 3, (num_bots, 1))*0.1), axis=1)  # energy efficiency information
sample = np.concatenate((sample,np.random.randint(0, 2, (num_bots, num_types))), axis=1)  # type information

# hand crafted objectives:
# sample_centers = np.array([[10, 23, 10, 0], [35, 76, 15, 0]])

# randomly generated objectives:
sample_centers = np.random.rand(num_tasks, 2)*100  # position information
sample_centers = np.concatenate((sample_centers,np.random.rand(num_tasks, 1)*100), axis=1)  # battery level information
sample_centers = np.concatenate((sample_centers,np.random.randint(1, 3, (num_tasks, 1))*0.1), axis=1)  # energy efficiency information
sample_centers = np.concatenate((sample_centers,np.random.randint(0, 2, (num_tasks, num_types))), axis=1)  # type information

print("sample data: \n", sample)
print("sample centers: \n", sample_centers)