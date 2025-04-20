import heapq

class Node:
    def __init__(self, x, y, g, h):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

def heuristic(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

def a_star_algo(capacity, start, goal):
    open_list = []
    closed_list = set()

    start_node = Node(start[0], start[1], 0, heuristic(start, goal))
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        if (current_node.x, current_node.y) == goal:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        closed_list.add((current_node.x, current_node.y))

        possible_moves = [
            (capacity[0], current_node.y),  # Fill jug 1
            (current_node.x, capacity[1]),  # Fill jug 2
            (0, current_node.y),            # Empty jug 1
            (current_node.x, 0),            # Empty jug 2
            # Pour jug 1 to jug 2
            (max(0, current_node.x - (capacity[1] - current_node.y)),
             min(capacity[1], current_node.y + current_node.x)),
            # Pour jug 2 to jug 1
            (min(capacity[0], current_node.x + current_node.y),
             max(0, current_node.y - (capacity[0] - current_node.x)))
        ]

        for move in possible_moves:
            x, y = move
            if (x, y) in closed_list:
                continue

            g_cost = current_node.g + 1
            h_cost = heuristic((x, y), goal)
            neighbor = Node(x, y, g_cost, h_cost)
            neighbor.parent = current_node

            heapq.heappush(open_list, neighbor)

    return None

capacity = (4, 3)
start = (0, 0)
goal = (2, 0)

path = a_star_algo(capacity, start, goal)
print("Path to reach the goal:", path)
