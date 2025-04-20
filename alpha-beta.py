import math 
import sys 
def get_height(n): 
    return math.ceil(math.log2(n)) 
def minmax(height, depth, node_index, max_player, values, alpha, beta): 
    print(f"Depth: {depth}, Node: {node_index}, Max Player: {max_player}, Alpha: {alpha}, Beta: {beta}") 
    if depth == height: 
        return values[node_index]
    if max_player: 
        best_value = -sys.maxsize 
        for i in range(2): 
            val = minmax(height, depth + 1, node_index * 2 + i, False, values, alpha, beta) 
            best_value = max(best_value, val) 
            alpha = max(alpha, best_value) 
            if beta <= alpha: 
                print(f"Pruning at Node: {node_index} (Max Player), Alpha: {alpha}, Beta: {beta}") 
                break
        return best_value 
    else: 
        best_value = sys.maxsize 
        for i in range(2): 
            val = minmax(height, depth + 1, node_index * 2 + i, True, values, alpha, beta) 
            best_value = min(best_value, val) 
            beta = min(beta, best_value) 
            if beta <= alpha: 
                print(f"Pruning at Node: {node_index} (Min Player), Alpha: {alpha}, Beta: {beta}")
                break
        return best_value
values = [3, 17, 2, 12, 15, 25, 30, 5]
height = get_height(len(values)) 
result = minmax(height, 0, 0, True, values, -sys.maxsize, sys.maxsize) 
print("Result:", result)
