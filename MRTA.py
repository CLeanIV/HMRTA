import numpy as np
import time
import math
# import networkx as nx
# from networkx.algorithms import bipartite
np.set_printoptions(precision=3)

class MultiRobotTaskAllocation:
    def __init__(self, robots, tasks, base=[0, 0, 0, 0, 0, 10, 1], save_data=None):
        self.t = 0
        base = np.array(base)
        self.subtask_index = 6
        base = [np.concatenate((base, np.zeros(len(tasks[0][self.subtask_index:-1]))))]
        self.tasks = np.concatenate((tasks, np.zeros((len(tasks), 1))), axis=1)
        self.all_tasks = np.concatenate((base, tasks))
        self.base = base[0]
        self.graph = []
        self.distance_graph = []
        self.robots = robots
        self.robots = np.concatenate((self.robots, np.zeros((len(robots), 1)) + np.Inf), axis=1)
        self.velocity = 0.75
        self.distance_tolerance = 1
        self.saved_states = np.concatenate((self.robots, self.tasks))
        self.tasks_completed = 0
        self.subtasks_completed = 0
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.zeta = None
        self.save_data = save_data
        self.t_a = 150
        # print(self.saved_states)
        # Column 0: index
        # Column 1: x position
        # Column 2: y position
        # Column 3: Battery level/cost
        # Column 4: Energy per unit distance
        # Column 5: subtask0: Go to charging dock
        # Column 6: subtask1
        # Column 7: subtask2
        # Column 8: subtask3
        # .
        # .
        # Column n-1: subtask p
        # Column n: current task/task priority rating

    def make_graph(self):
        self.graph = []
        self.distance_graph = []
        for robot in self.robots:
            # entry = [robot[3] - np.linalg.norm(robot[1:3] - self.base[1:3]) * robot[4]]
            entry = []
            distance_entry = []
            for task in self.tasks:
                distance_j_home = self.base[1:3] - task[1:3]  # norm of distance between the base and task
                distance_j_home = math.sqrt(distance_j_home[0]**2 + distance_j_home[1]**2)
                distance_ij = robot[1:3] - task[1:3]  # norm of distance between the robot and task
                distance_ij = math.sqrt(distance_ij[0] ** 2 + distance_ij[1] ** 2)
                required_energy = (task[3] + (distance_ij + distance_j_home) * robot[4])*0
                if required_energy > robot[3]:
                    weight = np.Inf
                    # print(1)
                elif all(task[self.subtask_index:-1] == 0):
                    weight = np.Inf
                    # print(2)
                elif task[-1] == np.inf:
                    weight = np.Inf
                    # print(3)
                elif all(task[self.subtask_index:-1] * robot[self.subtask_index:-1] == 0):
                    weight = np.Inf
                    # print(4)
                elif all(task[self.subtask_index:-1] == 0):
                    weight = np.Inf
                    # print(5)
                elif robot[-1] != np.Inf:
                    weight = np.Inf
                    # print(6)
                else:
                    weight = task[3] + distance_ij * robot[4]

                if weight == np.Inf:
                    distance_entry.append(np.Inf)
                else:
                    distance_entry.append(distance_ij)
                entry.append(weight)
            self.distance_graph.append(distance_entry)
            self.graph.append(entry)
        self.distance_graph = np.array(self.distance_graph)
        self.graph = np.array(self.graph)

    def prioritize(self):
        subtask_number_priority = []
        available_robot_types = [sum(x) for x in zip(*self.robots)][self.subtask_index:-1]

        for task in self.tasks:
            subtask_number_priority.append(sum(task[self.subtask_index:-1]))
            entry = np.array([available_robot_types - task[self.subtask_index:-1]]) * self.alpha
            try:
                availability_priority = np.concatenate((availability_priority, entry), axis=0)
            except UnboundLocalError:
                availability_priority = entry * self.beta
        holder = availability_priority
        availability_priority = []
        for i in range(len(holder)):
            if self.tasks[i][-1] == np.inf:
                availability_priority.append(np.NINF)
            else:
                availability_priority.append(sum(holder[i]))

        distance_priority = self.zeta / (self.distance_graph ** self.gamma + 1)
        # distance_priority = self.distance_graph * self.gamma
        distance_priority = np.array([sum(x) for x in zip(*distance_priority)])

        priority_ratings = np.array(subtask_number_priority) + np.array(availability_priority) - np.array(distance_priority)
        for i in range(len(self.tasks)):
            if self.tasks[i][-1] != np.NINF:
                self.tasks[i][-1] = priority_ratings[i]
        # print("priority values based on total number of sub-tasks required: \n", subtask_number_priority)
        # print("priority values based on surplus of robots to complete sub-tasks per task: \n", availability_priority)
        # print("priority values based on robot distance value for compatible tasks: \n", distance_priority)
        # print("overall priority rating: \n", priority_ratings)

    def allocate_tasks_greedy(self):
        graph = self.graph
        for j in range(len(graph)):
            for i in range(len(graph[0])):
                if self.graph[j][i] == np.Inf:
                    graph[j][i] = np.NINF
        # print("transpose(graph) \n", np.transpose(graph))
        if any(np.transpose(self.robots)[-1] == np.inf):
            priority_task_index = np.argmax(np.transpose(self.tasks)[-1])
            # print("priority_task_index \n", priority_task_index)
            if any(np.transpose(graph)[priority_task_index] != np.NINF):
                best_match_robot_index = np.argmax(np.transpose(graph)[priority_task_index])
                # print("best_match_robot_index \n", best_match_robot_index)
                task_and_types = self.tasks[priority_task_index][self.subtask_index:-1]
                # print("task_and_types \n", task_and_types)
                robot_and_types = self.robots[best_match_robot_index][self.subtask_index:-1]
                # print("robot_and_types \n", robot_and_types)
                chosen_subtask_index = np.argmax(
                    robot_and_types * task_and_types * self.alpha) + self.subtask_index
                # print("chosen_subtask_index \n", chosen_subtask_index)
                self.update_robot_states(best_match_robot_index, priority_task_index)
                self.update_task_states(best_match_robot_index, priority_task_index, chosen_subtask_index)
            else:
                self.update_robot_states()
                self.update_task_states()
        else:
            self.update_robot_states()
            self.update_task_states()

    def update_robot_states(self, robot_index=None, task_index=None):
        # print(self.robots)
        if robot_index != None:
            if self.robots[robot_index][-1] == np.Inf:  # if the robot is idle, assign it to the task
                self.robots[robot_index][-1] = -task_index
        for i in range(len(self.robots)):
            if self.robots[i][-1] <= 0:  # for each robot that is assigned to a task:
                distance_to_task = self.tasks[int(-self.robots[i][-1])][1:3] - self.robots[i][1:3]  # check each robots distance from its assigned task
                direction_to_task = distance_to_task / math.sqrt(distance_to_task[0]**2 + distance_to_task[1]**2)  # calculate the direction from each robot to its assigned task
                self.robots[i][1:3] = direction_to_task * self.velocity + self.robots[i][1:3]  # update the position of each robot to its assigned task with velocity
                distance_to_task = self.tasks[int(-self.robots[i][-1])][1:3] - self.robots[i][1:3]
                if math.sqrt(distance_to_task[0]**2 + distance_to_task[1]**2) <= self.distance_tolerance:  # check if task was reached for each robot
                    self.robots[i][-1] = np.Inf
                    # print("robot reached task, heading to new task")

    def update_task_states(self, robot_index=None, task_index=None, chosen_subtask_index=None):
        if robot_index != None:
            self.tasks[task_index][chosen_subtask_index] -= 1
            self.subtasks_completed += 1
            # print(self.robots)
        for task in self.tasks:
            if task[-1] != np.NINF:
                if all(task[self.subtask_index:-1] == 0):
                    task[-1] = np.NINF
                    self.tasks_completed += 1


    def save(self):
        state = np.concatenate((self.robots, self.tasks))
        self.saved_states = np.concatenate((self.saved_states, state), axis=1)

    def run(self, alpha=1, beta=1, gamma=1 / 2, zeta=5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        # while any(np.transpose(self.tasks)[-1] != np.NINF):
        while self.t < self.t_a:
            if all(np.transpose(self.tasks)[-1] == np.NINF):
                break
        # while self. tasks_completed < 3:
        # while self.subtasks_completed < 10:
            self.make_graph()
            self.prioritize()
            self.allocate_tasks_greedy()
            if self.save_data:
                self.save()
            self.t += 1
        if self.tasks_completed == len(self.tasks):
            delta = 1
        else:
            delta = 0
        objfunc = ((self.t / self.t_a) * delta) + (2 - (self.tasks_completed / len(self.tasks))) * (1 - delta)
        if self.save_data:
            print("Task states: \n", self.tasks)
            print("Total time taken: \n", self.t)
            np.savetxt("optimized_save_state_file.csv", self.saved_states, fmt="%2.2f", delimiter=",")
        return objfunc
