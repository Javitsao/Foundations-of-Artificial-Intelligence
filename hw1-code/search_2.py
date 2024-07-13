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

from maze import Maze
def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def get_dis(maze: Maze, start):
    from collections import deque
    n, m = maze.getDimensions()
    dis = [[-1 for i in range(m)] for j in range(n)]
    q = deque([start])
    dis[start[0]][start[1]] = 0
    while q:
        nx, ny = q.popleft()
        for (x, y) in maze.getNeighbors(nx, ny):
            if dis[x][y] == -1:
                dis[x][y] = dis[nx][ny] + 1
                q.append((x, y))
    return dis

def astar_mst(maze: Maze):
    import heapq
    foods = maze.getObjectives()
    k = len(foods)
    dis_from_foods = [get_dis(maze, food) for food in foods]
    
    class UnionFind:
        def __init__(self, n):
            self.lead = list(range(n))
            
        def find(self, x):
            if self.lead[x] == x: return x
            else: 
                self.lead[x] = self.find(self.lead[x])
                return self.lead[x]
        
        def same(self, x, y):
            return self.find(x) == self.find(y)

        def union(self, x, y):
            x = self.find(x)
            y = self.find(y)
            self.lead[x] = y

    def kruskal(edges):
        edges = sorted(edges, key=lambda x: x[2])
        dsu = UnionFind(k + 1)
        mst = 0
        for u, v, w in edges:
            if not dsu.same(u, v):
                mst += w
                dsu.union(u, v)
        return mst

    mst_dp = dict()
    class Node:
        def __init__(self, x, y, food_ids, g, parent=None):
            self.x = x
            self.y = y
            self.food_ids = food_ids
            self.g = g
            self.parent = parent
            self.h = self._get_h()

        @property
        def f(self):
            return self.g + self.h

        def goal(self):
            return self.food_ids == 0

        def check_eat(self):
            for i in range(k):
                if (self.x, self.y) == foods[i] and (self.food_ids >> i & 1):
                    self.food_ids ^= 1 << i
                    break
        
        def _get_h(self):
            self.check_eat()
            if self.food_ids == 0:
                return 0

            mst = mst_dp.get(self.food_ids)
            if not mst:
                edges = []
                for i in range(k):
                    if not (self.food_ids >> i & 1): continue
                    for j in range(k):
                        if not (self.food_ids >> j & 1): continue
                        w = dis_from_foods[i][foods[j][0]][foods[j][1]]
                        edges.append((i, j, w))
                mst = kruskal(edges)
                mst_dp[self.food_ids] = mst

            mi = 1000000000
            for i in range(k):
                if not (self.food_ids >> i & 1): continue
                mi = min(mi, dis_from_foods[i][self.x][self.y])
            mst += mi
            return mst

        def __eq__(self, other):
            return self.x == other.x \
                and self.y == other.y \
                and self.food_ids == other.food_ids
        
        def __lt__(self, other):
            return self.f < other.f
        
        def __hash__(self) -> int:
            return (self.x, self.y, self.food_ids).__hash__()
        
    start = maze.getStart()
    start = Node(x=start[0], y=start[1], food_ids=(1 << k) - 1, g=0)
    pq = []
    from collections import defaultdict
    vis = defaultdict(lambda: False)
    dis = defaultdict(lambda: 1000000000)
    
    heapq.heappush(pq, start)
    dis[start] = start.g
    while pq:
        now = heapq.heappop(pq)
        if vis[now]: continue
        vis[now] = True
        if now.goal():
            path = []
            while now:
                path.append((now.x, now.y))
                now = now.parent
            return path[::-1]
        
        for (x, y) in maze.getNeighbors(now.x, now.y):
            child = Node(x=x, y=y, food_ids=now.food_ids, g=now.g + 1, parent=now)
            if not vis[child] and child.g < dis[child]:
                dis[child] = child.g
                heapq.heappush(pq, child)

    return []

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return astar_mst(maze)

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return astar_mst(maze)

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    return astar_mst(maze)

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return astar_mst(maze)

def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    n, m = maze.getDimensions()
    vis = [[False for i in range(m)] for j in range(n)]
    path = []
    def dfs(x, y):
        path.append((x, y))
        vis[x][y] = True
        for (nx, ny) in maze.getNeighbors(x, y):
            if not vis[nx][ny]: 
                dfs(nx, ny)
                path.append((x, y))
    dfs(start[0], start[1])
    return path