import heapq

class Node():
    def __init__(self,name,g,h):
        self.name = name
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = None
    def __lt__(self,other):
        return self.f<other.f

def a_star_algo(graph,heuristics,start,goal_node):
    open_list = []
    closed_list = set()

    start_node = Node(start,0,heuristics[start])
    heapq.heappush(open_list,start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.name == goal_node:
            path=[]
            while current_node:
                path.append(current_node.name)
                current_node = current_node.parent
            return path[::-1]            

        closed_list.add(current_node.name)

        for neighbour , cost in graph[current_node.name].items():
            if neighbour in closed_list:
                continue
            g_cost = current_node.g + cost
            h_cost = heuristics[neighbour]
            neighbour_node = Node(neighbour,g_cost,h_cost)
            neighbour_node.parent = current_node
            heapq.heappush(open_list,neighbour_node)
    return None        

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'D': 2},
    'C': {'A': 4, 'E': 5},
    'D': {'B': 2, 'E': 1},
    'E': {'C': 5, 'D': 1, 'F': 3, 'G': 2},
    'F': {'E': 3},
    'G': {'E': 2}
}

heuristics = {
    'A': 6, 'B': 5, 'C': 4,
    'D': 3, 'E': 2, 'F': 3,
    'G': 0
}

start_node = "A"
goal_node="G"

path = a_star_algo(graph,heuristics,start_node,goal_node)

print("Path: ",path)