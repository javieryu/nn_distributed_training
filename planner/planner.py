import matplotlib.pyplot as plt
import numpy as np
from queue import PriorityQueue
from itertools import count
import random

class PathPlanner:
	def __init__(self, grid):
		"""
		Constructor of the PathPlanner Class.
		:param grid: Numpy array that represents the
		occupancy map/grid. List should only contain 0's
		for open nodes and 1's for obstacles/walls.
		"""
		self.grid = grid
		self.goal = None

	def h(self, cell):
		"""
		heuristic function: L1 distance
		"""
		return np.sum(abs(cell - self.goal))

	def reconstruct_path(self, start, current_path, cameFrom):
		"""
		current is a numpy array where each row is a coordinate, with the end coordinate in last row
		cameFrom is a dictionary
		"""
		if np.size(current_path) == 2 :
			predecessor = cameFrom[tuple(current_path)]
		else :
			predecessor = cameFrom[tuple(current_path[0,:])]

		current_path = np.vstack((predecessor, current_path))
		if all(predecessor == start) :
			return current_path
		else :
			return self.reconstruct_path(start, current_path, cameFrom)

	def get_neighbors(self, current, row, col):
		neighbors = np.array([current + [1, 0], current + [-1, 0], 
						      current + [0, 1], current + [0, -1]])
		false_neighbors = []
		for i in range(np.shape(neighbors)[0]) :
			if not (neighbors[i,0] >= 0 and neighbors[i,0] <= row-1 and neighbors[i,1] >= 0 and neighbors[i,1] <= col-1 and self.grid[tuple(neighbors[i,:])] == 0) :
				false_neighbors = np.append(false_neighbors, i)

		return np.delete(neighbors, false_neighbors, 0)

	def a_star(self, start, goal):
		"""
		A* Planner method. Finds a plan from a starting node
		to a goal node if one exits.
		:param start: Initial node in an Occupancy map. [x, y].
		Type: numpy integer vector
		:param goal: Goal node in an Occupancy map. [x, y].
		Type: numpy integer vector
		:return: Found path or False if it fails.
		"""
		self.goal = goal
		row, col = np.shape(self.grid)
		tiebreaker = count() # needed for unique priorities in priority queue. Python is dumb

		# For cell n, gScore[n] is the cost of the cheapest path from start to n currently known.
		gScore = np.full((row,col), np.inf)
		gScore[tuple(start)] = 0

		# For node n, fScore[n] := gScore[n] + h(n). fScore[n] represents our current best guess as to
		# how short a path from start to finish can be if it goes through n.
		fScore = np.full((row,col), np.inf)
		fScore[tuple(start)] = gScore[tuple(start)] + self.h(start)

		# For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start
		# to n currently known.
		cameFrom = {}

		# The set of discovered nodes that may need to be (re-)expanded.
		# Initially, only the start node is known.
		openSet = PriorityQueue()
		openSet.put((fScore[tuple(start)], next(tiebreaker), start))

		while not openSet.empty() :
			score, number, current = openSet.get()
			if all(current == goal) :
				return self.reconstruct_path(start, current, cameFrom)

			neighbors = self.get_neighbors(current, row, col)
			for i in range(np.shape(neighbors)[0]) :
				neighbor = neighbors[i,:]
				tentative_gScore = gScore[tuple(current)] + 1
				if tentative_gScore < gScore[tuple(neighbor)] :
					# This path to neighbor is better than any previous one.
					cameFrom[tuple(neighbor)] = current
					gScore[tuple(neighbor)] = tentative_gScore
					fScore[tuple(neighbor)] = gScore[tuple(neighbor)] + self.h(neighbor)
					
					in_queue = False
					for item in openSet.queue :
						if all(neighbor == item[2]) :
							in_queue = True
							break

					if not in_queue :
						openSet.put((fScore[tuple(neighbor)], next(tiebreaker), neighbor))

		return False

	def waypoint_following(self, waypoints) :
		"""
		Given an array of waypoints, generate path through all of them using A*
		"""
		path = waypoints[0,:] # starting coordinate
		for i in range(np.shape(waypoints)[0]-1) :
			path = np.vstack((path, self.a_star(waypoints[i,:], waypoints[i+1,:])[1:,:]))
		return path

	def random_walk(self, start, length) :
		path = np.empty((length,2))
		path[0,:] = start
		for i in range(length-1) : 
			while True :
				a = random.choice(np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]))
				next_state = np.int_(path[i,:] + a)
				# print("Next State: ", next_state)
				if self.grid[tuple(next_state)] == 0 :
					break

			path[i+1,:] = next_state

		return path

