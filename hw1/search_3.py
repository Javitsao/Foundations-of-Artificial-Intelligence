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
    return min(abs(start[0] - obj[0]) + abs(start[1] - obj[1]) for obj in objectives)
def diss(start, objectives):
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
def getPath(cur, prev_map):
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
def mst_heuristic(node, goals, lmap, objectives):
    if len(goals) == 0:
        return 0
    result = 0
    cur_v = [objectives.index(goals[0])]
    vertices = []
    for i in range(1, len(goals)):
        vertices.append(objectives.index(goals[i]))
    while len(cur_v) != len(goals):
        min_paths = []
        for cv in cur_v:
            min_nv = sys.maxsize
            min_n = None
            for vert in vertices:
                if vert < cv:
                    edge = (vert, cv)
                else:
                    edge = (cv, vert)
                if lmap[edge] < min_nv:
                    min_nv = lmap[edge]
                    min_n = vert
            min_paths.append((min_nv, min_n))
        min_p = min(min_paths)
        vertices.remove(min_p[1])
        result += min_p[0]
        cur_v.append(min_p[1])
    l = []
    for x in goals:
        l.append(diss(node, x))
    return result + min(l)
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
        #p = curr
        curr = queue.pop(0)
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
    while(len(p_queue)>0):
        #p = curr
        #curr = p_queue.get(1)
        curr = heapq.heappop(p_queue)[1]
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

    # a map of the previous or parent nodes
    prev = {(start, tuple(food)):None}

    # the priority queue gets (f, distance to current node(g), (current node, remaing food))
    p_queue = []
    s_node = (mst_heuristic(start, tuple(food), len_map, food)+0, 0, (start, tuple(food)))
    heapq.heappush(p_queue, s_node)

    cur_node_dst_map = {s_node[2]:0}

    while p_queue:
        cur = heapq.heappop(p_queue)
        cur_pos = cur[2][0]
        if (len(cur[2][1]) == 0):
            return getPath(cur[2], prev)

        neighbors = maze.getNeighbors(cur_pos[0], cur_pos[1])
        for n in neighbors:
            food_from_node = tuple(food_to_eat(n, cur[2][1]))
            dst_node = (n, food_from_node)
            if dst_node in cur_node_dst_map and cur_node_dst_map[dst_node] <= cur_node_dst_map[cur[2]]+1:
                continue
            #update distance map of the node
            cur_node_dst_map[dst_node] = cur_node_dst_map[cur[2]]+1
            #update node's parent
            prev[dst_node] = cur[2]

            #update p_queue: this part is borrowed from the textbook Fig. 3.26
            old_f = cur[0]
            new_f = cur_node_dst_map[dst_node]+mst_heuristic(n, food_from_node, len_map, food)
            #new_f = max(old_f, new_f)

            new_node = (new_f, cur_node_dst_map[dst_node], dst_node)
            heapq.heappush(p_queue, new_node)


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []
