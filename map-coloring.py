def is_consistent(assignment, var, val, adjacency_list):
    for neighbor in adjacency_list[var]:
        if neighbor in assignment and assignment[neighbor] == val:
            return False
    return True

def select_unassigned_variable(variables, assignment):
    for var in variables:
        if var not in assignment:
            return var
    return None

def backtracking(assignment, variables, domains, adjacency_list):
    if len(assignment) == len(variables):
        return assignment  # Found a valid coloring

    var = select_unassigned_variable(variables, assignment)

    for val in domains[var]:
        if is_consistent(assignment, var, val, adjacency_list):
            assignment[var] = val  # Assign color
            result = backtracking(assignment, variables, domains, adjacency_list)
            if result:
                return result  # Return solution if found
            assignment.pop(var)  # Backtrack if no solution

    return None  # No solution found

if __name__ == "__main__":
    # Define regions (nodes in the graph)
    variables = ['WA', 'NT', 'SA', 'QLD', 'NSW', 'VIC', 'TAS']

    # Define possible colors for each region
    domains = {var: ['Red', 'Green', 'Blue'] for var in variables}

    # Define adjacency list (graph edges)
    adjacency_list = {
        'WA': ['NT', 'SA'],
        'NT': ['WA', 'SA', 'QLD'],
        'SA': ['WA', 'NT', 'QLD', 'NSW', 'VIC'],
        'QLD': ['NT', 'SA', 'NSW'],
        'NSW': ['SA', 'QLD', 'VIC'],
        'VIC': ['SA', 'NSW'],
        'TAS': []  # Tasmania is isolated
    }

    # Solve using backtracking
    assignment = {}
    solution = backtracking(assignment, variables, domains, adjacency_list)

    # Print result
    if solution:
        print("Solution:", solution)
    else:
        print("No Solution Found")
