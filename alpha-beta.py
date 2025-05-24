def alpha_beta_pruning(node, depth, alpha, beta, maximizing, level=0):
    print(f"Depth: {depth}, Node: {node}, Max Player: {maximizing}, Alpha: {alpha}, Beta: {beta}") 
    if depth == 0: # Base case: leaf node
        return node
    
    if maximizing:
        value = float('-inf')
        for i, child in enumerate(node):
            value = max(value, alpha_beta_pruning(child, depth - 1, alpha, beta, False, level + 1))
            alpha = max(alpha, value)
            if alpha >= beta:
                print(f"Pruned at Level {level} (MAX): Remaining Nodes = {node[i+1:]} with α = {alpha} >= β = {beta}")
                break  # Beta cut-off
        return value
    else:
        value = float('inf')
        for i, child in enumerate(node):
            value = min(value, alpha_beta_pruning(child, depth - 1, alpha, beta, True, level + 1))
            beta = min(beta, value)
            if beta <= alpha:
                print(f"Pruned at Level {level} (MIN): Remaining Nodes = {node[i+1:]} with β = {beta} <= α = {alpha}")
                break  # Alpha cut-off
        return value

def build_tree(leaves):
    """Builds a binary tree from leaves for Minimax."""
    tree = leaves
    while len(tree) > 1:
        tree = [tree[i:i + 2] for i in range(0, len(tree), 2)]
    return tree[0]

# User Input
levels = 4
leaf_values = [3, 17, 2, 12, 15, 25, 30, 5]

tree = build_tree(leaf_values)
pruned_value = alpha_beta_pruning(tree, levels - 1, float('-inf'), float('inf'), True)
print(f"Optimal Minimax Value: {pruned_value}")
