import heapq

class Node():
    def __init__(self,x,y,g,h):
        self.x=x
        self.y=y
        self.g=g
        self.h=h
        self.f=g+h
        self.parent = None
    def __lt__(self,other):
        return self.f < other.f

def heuristic(start,goal):
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])
def a_star_algo(grid,start,goal):
    open_list = []
    closed_list = set()

    start_node = Node(start[0],start[1],0,heuristic(start,goal))
    heapq.heappush(open_list,start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        if (current_node.x,current_node.y) == goal:
            path = []
            while current_node:
                path.append((current_node.x,current_node.y))
                current_node = current_node.parent
            return path[::-1]
        
        closed_list.add((current_node.x,current_node.y))

        for dx,dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            x,y=current_node.x + dx,current_node.y+dy

            if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 0:
                if (x,y) in closed_list:
                    continue
                g_cost = current_node.g + 1
                h_cost = heuristic((x,y),goal)
                neighbour = Node(x,y,g_cost,h_cost)
                neighbour.parent = current_node

                heapq.heappush(open_list,neighbour)
    return None

grid = [
    [0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0,0)
goal = (4,4)

path = a_star_algo(grid,start,goal)

print("Path: ",path)