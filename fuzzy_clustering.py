import numpy as np
np.set_printoptions(precision=2)
num_bots = 5
num_tasks = 3
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

print("sample centers: \n", sample_centers)

master_task = []
for task in sample_centers:
    task_types = task[4:]
    for i in range(len(task_types)):
        entry = []
        if task_types[i] == 1:
            entry = np.zeros((1, len(task_types)))
            entry[0][i] = 1
            np.concatenate((task[:4],entry),axis=1)
            master_task.append(entry)
print(np.array(master_task))

revsjn
print("sample data: \n", sample)

dtyjh
def fuzzy(data, centers):
    fuzz = []
    for point in data:
        entry = []
        for center in centers:
            entry.append(np.linalg.norm(point-center))
        entry_sum = sum(entry)
        for i in range(len(entry)):
            entry[i] = entry[i]/entry_sum
        fuzz.append(entry)
    return np.array(fuzz)


def bipartite_graph(data, centers, fuzzy_weights=1, home=[0, 0], hetero=False):
    # Column 0: x position
    # Column 1: y position
    # Column 2: Battery level/cost
    # Column 3: Energy per unit distance
    # Column 4: Type1
    # Column 5: Type2
    # Column 6: Type3
    # .
    # .
    # .
    # Column n: Typen

    graph = []
    count = 0
    for point in data:
        entry = []
        for center in centers:
            distance_ij = np.linalg.norm(home[0:1] - center[0:1])  # norm of distance between the point and center
            distance_j_home = np.linalg.norm(point[0:1] - center[0:1])  # norm of distance between the point and center
            cost_ij = point[2] - center[2] - (distance_ij + distance_j_home) * point[3]
            if cost_ij <= 0:
                weight = np.Infinity
            else:
                weight = distance_ij + cost_ij
            if hetero:
                for i in range(len(point[4:])):
                    print("gejov", center[i] - point[i])
                    # if not any(typ - point[4:] == 0):
                        # print(typ, point[4:])
                    #     weight = 1e11
            entry.append(weight)
        graph.append(entry)
    return np.array(graph)


cluster_values = fuzzy(sample, sample_centers)
print("cluster weight outputs: \n", cluster_values)

bi_graph = bipartite_graph(sample, sample_centers, hetero=True)
print("weighted bipartite graph: \n", bi_graph)
