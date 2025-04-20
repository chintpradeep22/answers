def is_consistent(assignment, row, col):
    for r, c in assignment.items():
        if c == col or abs(r - row) == abs(c - col):  # Same column or diagonal conflict
            return False
    return True


def select_unassigned_variable(n, assignment):
    for row in range(n):
        if row not in assignment:
            return row
    return None


def backtracking(assignment, n):
    if len(assignment) == n:
        return assignment  # Found a valid solution

    row = select_unassigned_variable(n, assignment)
    
    for col in range(n):
        if is_consistent(assignment, row, col):
            assignment[row] = col  # Place queen
            result = backtracking(assignment, n)
            if result:
                return result  # Return solution if found
            assignment.pop(row)  # Backtrack if no solution

    return None  # No solution found


if __name__ == "__main__":
    n = 8  # 8-Queens problem
    assignment = {}
    solution = backtracking(assignment, n)
    
    if solution:
        print("8-Queens Solution:", solution)
    else:
        print("No Solution Found")
