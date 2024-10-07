# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)
import heapq
import copy
import sys
def dis(start, objectives):
    #return min(abs(start[0] - obj[0]) + abs(start[1] - obj[1]) for obj in objectives)
    if isinstance(objectives[0], tuple):
        return abs(start[0] - objectives[0][0]) + abs(start[1] - objectives[0][1])
    return abs(start[0] - objectives[0]) + abs(start[1] - objectives[1])
def backward(neighbor, prev, st):
    count = 0
    curr = neighbor
    while curr != st:
        curr = prev[curr]
        count += 1
    return count
def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)
def new_maze(maze, start, end):
    new_maze = copy.deepcopy(maze)
    new_maze.setStart(start)
    new_maze.setObjectives([end])
    return new_maze
def backtrack(cur, prev_map):
    p = []
    while cur != None:
        p.append(cur[0])
        cur = prev_map[cur]
    p.reverse()
    return p

def food_to_eat(node, goals):
    ret = []
    for i in goals:
        if node != i:
            ret.append(i)
    return ret
def prim_heuristic(node, food_from_node, len_map, food):
    if len(food_from_node) == 0:
        return 0
    result = 0
    visited = [food.index(food_from_node[0])]
    unvisited = []
    for i in range(1, len(food_from_node)):
        unvisited.append(food.index(food_from_node[i]))
    while len(visited) != len(food_from_node):
        min_edges = []
        for cv in visited:
            min_len = sys.maxsize
            for index in unvisited:
                edge = (min(index, cv), max(index, cv))
                if len_map[edge] < min_len:
                    (min_len, min_n) = (len_map[edge], index)
            min_edges.append((min_len, min_n))
        edge = min(min_edges)
        unvisited.remove(edge[1])
        visited.append(edge[1])
        result += edge[0]
    to_one = []
    for food in food_from_node:
        to_one.append(dis(node, food))
    return result + min(to_one)
def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    queue = []
    prev = {}
    curr = maze.getStart()
    queue.append(curr)
    visited = set()
    visited.add(curr)
    while(len(queue)>0):
        curr = queue.pop(0)
        if maze.isObjective(curr[0], curr[1]):
            path = [curr]
            while curr != maze.getStart():
                curr = prev[curr]
                path.append(curr)
            return path[::-1]
        #result.append(curr)
        for neighbor in maze.getNeighbors(curr[0], curr[1]):
            if neighbor not in visited:
                queue.append(neighbor)
                prev[neighbor] = curr
                visited.add(neighbor)
    
def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    p_queue = []
    prev = {}
    st = maze.getStart()
    ob = maze.getObjectives()
    #curr = (dis(st, ob), st)
    #p_queue.put(curr)
    heapq.heappush(p_queue, (dis(st, ob), st))
    visited = set()
    visited.add(st)
    while(len(p_queue) > 0):
        #p = curr
        #curr = p_queue.get(1)
        curr = heapq.heappop(p_queue)[1]
        #visited.add(curr)
        #prev[curr] = p
        if maze.isObjective(curr[0], curr[1]):
            path = [curr]
            while curr != maze.getStart():
                curr = prev[curr]
                path.append(curr)
            return path[::-1]
                
        #result.append(curr)
        for neighbor in maze.getNeighbors(curr[0], curr[1]):
            if neighbor not in visited:
                prev[neighbor] = curr
                #p_queue.put(dis(neighbor, ob), neighbor)
                heapq.heappush(p_queue, (dis(neighbor, ob) + backward(neighbor, prev, st), neighbor))
                visited.add(neighbor)
    return []

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    return astar_multi(maze)

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    food = maze.getObjectives()
    paths = []
    for i in range(len(food)):
        for j in range(i+1, len(food)):
            paths.append((i, j))
    len_map = {}
    for path in paths:
        c_maze = new_maze(maze, food[path[0]], food[path[1]])
        len_map[path] = len(astar(c_maze)) - 1
    prev_map = {(start, tuple(food)):None}
    backward_map = {(start, tuple(food)):0}
    p_queue = []
    s_node = (prim_heuristic(start, tuple(food), len_map, food), (start, tuple(food)))
    heapq.heappush(p_queue, s_node)
    
    while p_queue:
        cur = heapq.heappop(p_queue)
        cur_pos = cur[1][0]
        if (len(cur[1][1]) == 0):
            return backtrack(cur[1], prev_map)
        neighbors = maze.getNeighbors(cur_pos[0], cur_pos[1])
        for n in neighbors:
            food_from_node = tuple(food_to_eat(n, cur[1][1]))
            b_node = (n, food_from_node)
            if b_node not in backward_map or backward_map[b_node] > backward_map[cur[1]] + 1:
                backward_map[b_node] = backward_map[cur[1]] + 1
                prev_map[b_node] = cur[1]
                new_node = (backward_map[b_node] + prim_heuristic(n, food_from_node, len_map, food), b_node)
                heapq.heappush(p_queue, new_node)

def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    #prev = {}
    path = []
    curr = maze.getStart()
    visited = set()
    def dfs(x, y):
        path.append((x, y))
        visited.add((x, y))
        for neighbor in maze.getNeighbors(x, y):
            if neighbor not in visited:
                #visited.add(neighbor)
                dfs(neighbor[0], neighbor[1])
                path.append((x, y))
    dfs(curr[0], curr[1])
    return path
                
