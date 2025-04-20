import heapq

goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

moves = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

def find_empty_tile(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def heuristic(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:
                goal_x, goal_y = divmod(state[i][j] - 1, 3)
                distance += abs(i - goal_x) + abs(j - goal_y)
    return distance

def get_neighbors(state):
    neighbors = []
    x, y = find_empty_tile(state)

    for move, (dx, dy) in moves.items():
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < 3 and 0 <= new_y < 3:
            new_state = [row[:] for row in state]
            new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
            neighbors.append((new_state, move))
    return neighbors

def a_star(initial_state):
    pq = []
    heapq.heappush(pq, (heuristic(initial_state), initial_state, []))
    visited = set()

    while pq:
        cost, state, path = heapq.heappop(pq)

        if state == goal_state:
            return path

        visited.add(tuple(map(tuple, state)))

        for new_state, move in get_neighbors(state):
            if tuple(map(tuple, new_state)) not in visited:
                heapq.heappush(pq, (cost + 1 + heuristic(new_state), new_state, path + [move]))

    return None

initial_state = [
    [1, 2, 3],
    [4, 0, 6],
    [7, 5, 8]
]

solution = a_star(initial_state)
print("Moves to solve:", solution)
