MRTA.py works in three phases:
 - graph construction
	- based on energy and task requirements
 - task prioritization
	- based on three rules to be tuned with hyperparameters:
		- total number of sub-tasks required
		- surplus of robots to complete sub-tasks per task
		- robot distance value for compatible tasks
 - task allocation
	- pairs robots with task to complete one of its subtasks. 
	- currenlty uses greedy algorithm:
		- highest priority task completely allocated first, with highest corresponding robot graph value. 
		- once the task is completed, next highest priority task is allocated, and so on.
To optimize:
run "PSO_MRTA.py" or "GA_MRTA.py"
You can play around with the settings in the options line for each file, and specify the file the convergence history is saved to
*warning* "GA_MRTA.py" fails to save to a file properly

To get and save results from some optimized results:
In "run_MRTA.py", enter the variables into "MRTA.run()" and run the python file

After all tasks have been completed (episode completion, the code will generate a save file of all pertinent states of 
the robots and environment to a .txt file.

Save file format:
given:
 - n# tasks
 - m# robots
 - p# subtask types
 - episode time t

Output is t arrays, where each array is the state at time t (in seconds).
each array is [n+m, 7+p] and is of the format:

columns (left to right, robot/task format): 
task number/robot number|x-position|y-position|battery level/energy requirement|energy usage rate/unused|type 1 compatibility/type 1 requirement|type 2 compatibility/type 2 requirement|...etc.|robot state/task state
robot state can take integer values describing if it is on its way to a task, or infinity if it is idle.
task state can take a float value expressing its priority in th tasl priority queue (higher means high priority), or negative infinity if the task is complete.

rows (top to bottom):
rows 0:n are the different robots
rows n:n+m are the different tasks