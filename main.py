import sys
import numpy


goal_state = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
total_expansions = 0


class Node:
    def __init__(self, tiles, heuristic_h: int, heuristic_g: int, explored: bool, parent):
        self.tiles = tiles
        self.heuristic_h = heuristic_h
        self.heuristic_g = heuristic_g
        self.explored = explored
        self.parent = parent
        self.children = []


def a_star(root, solution_nodes):
    global total_expansions
    visited = [root]
    pq = []
    expand(root, pq, visited)
    total_expansions += 1
    while total_expansions < 10000:
        pq.sort(reverse=False, key=sort_function)
        if pq:
            temp = pq.pop(0)
            while check_visited(visited, temp.tiles):
                temp = pq.pop(0)
            if goal_test(temp.tiles):
                make_solution(temp, solution_nodes)
                return True
            else:
                expand(temp, pq, visited)
                visited.append(temp)
                total_expansions += 1
        else:
            print("Failed")
            exit(1)
        # Test for goal state


def expand(root, pq, visited):
    # Gets location of the blank tile
    puzzle_state = root.tiles
    blank_loc = [0, 0]
    get_blank_loc(puzzle_state, blank_loc)
    new_g = root.heuristic_g + 1

    # The next block determines what the next possible states are
    # based on the current location of the blank tile
    if blank_loc[0] == 0:
        if blank_loc[1] == 0:
            make_state_2(pq, visited, root, puzzle_state, new_g,
                         0, 0, 0, 1, 1, 0)

        if blank_loc[1] == 1:
            make_state_3(pq, visited, root, puzzle_state, new_g,
                         0, 1, 0, 0, 0, 2, 1, 1)

        if blank_loc[1] == 2:
            make_state_2(pq, visited, root, puzzle_state, new_g,
                         0, 2, 0, 1, 1, 2)

    if blank_loc[0] == 1:
        if blank_loc[1] == 0:
            make_state_3(pq, visited, root, puzzle_state, new_g,
                         1, 0, 0, 0, 2, 0, 1, 1)

        if blank_loc[1] == 1:
            make_state_4(pq, visited, root, puzzle_state, new_g,
                         1, 1, 1, 2, 0, 1, 2, 1, 1, 0)

        if blank_loc[1] == 2:
            make_state_3(pq, visited, root, puzzle_state, new_g,
                         1, 2, 1, 1, 0, 2, 2, 2)

    if blank_loc[0] == 2:
        if blank_loc[1] == 0:
            make_state_2(pq, visited, root, puzzle_state, new_g,
                         2, 0, 2, 1, 1, 0)

        if blank_loc[1] == 1:
            make_state_3(pq, visited, root, puzzle_state, new_g,
                         2, 1, 2, 0, 1, 1, 2, 2)

        if blank_loc[1] == 2:
            make_state_2(pq, visited, root, puzzle_state, new_g,
                         2, 2, 2, 1, 1, 2)


def make_state_2(pq, visited, root, puzzle_state, new_g,
                 blank_y: int, blank_x: int,
                 swap1_y: int, swap1_x: int,
                 swap2_y: int, swap2_x: int):
    new_state1 = numpy.copy(puzzle_state)
    swap_tiles(new_state1, blank_y, blank_x, swap1_y, swap1_x)
    if even_parity(new_state1) and not check_visited(visited, new_state1):
        heuristic1 = get_heuristic(new_state1)
        new_node1 = Node(new_state1, heuristic1, new_g, False, root)
        pq.append(new_node1)
        #visited.append(new_node1)
        print("Adding:")
        print(new_state1)

    new_state2 = numpy.copy(puzzle_state)
    swap_tiles(new_state2, blank_y, blank_x, swap2_y, swap2_x)
    if even_parity(new_state2) and not check_visited(visited, new_state2):
        heuristic2 = get_heuristic(new_state2)
        new_node2 = Node(new_state2, heuristic2, new_g, False, root)
        pq.append(new_node2)
        #visited.append(new_node2)
        print("Adding:")
        print(new_state2)

    print()


def make_state_3(pq, visited, root, puzzle_state, new_g,
                 blank_y: int, blank_x: int,
                 swap1_y: int, swap1_x: int,
                 swap2_y: int, swap2_x: int,
                 swap3_y: int, swap3_x: int):
    new_state1 = numpy.copy(puzzle_state)
    swap_tiles(new_state1, blank_y, blank_x, swap1_y, swap1_x)
    if even_parity(new_state1) and not check_visited(visited, new_state1):
        heuristic1 = get_heuristic(new_state1)
        new_node1 = Node(new_state1, heuristic1, new_g, False, root)
        pq.append(new_node1)
        #visited.append(new_node1)
        print("Adding:")
        print(new_state1)

    new_state2 = numpy.copy(puzzle_state)
    swap_tiles(new_state2, blank_y, blank_x, swap2_y, swap2_x)
    if even_parity(new_state2) and not check_visited(visited, new_state2):
        heuristic2 = get_heuristic(new_state2)
        new_node2 = Node(new_state2, heuristic2, new_g, False, root)
        pq.append(new_node2)
        #visited.append(new_node2)
        print("Adding:")
        print(new_state2)

    new_state3 = numpy.copy(puzzle_state)
    swap_tiles(new_state3, blank_y, blank_x, swap3_y, swap3_x)
    if even_parity(new_state3) and not check_visited(visited, new_state3):
        heuristic2 = get_heuristic(new_state3)
        new_node3 = Node(new_state3, heuristic2, new_g, False, root)
        pq.append(new_node3)
        #visited.append(new_node3)
        print("Adding:")
        print(new_state3)

    print()


def make_state_4(pq, visited, root, puzzle_state, new_g,
                 blank_y: int, blank_x: int,
                 swap1_y: int, swap1_x: int,
                 swap2_y: int, swap2_x: int,
                 swap3_y: int, swap3_x: int,
                 swap4_y: int, swap4_x: int):
    new_state1 = numpy.copy(puzzle_state)
    swap_tiles(new_state1, blank_y, blank_x, swap1_y, swap1_x)
    if even_parity(new_state1) and not check_visited(visited, new_state1):
        heuristic1 = get_heuristic(new_state1)
        new_node1 = Node(new_state1, heuristic1, new_g, False, root)
        pq.append(new_node1)

    new_state2 = numpy.copy(puzzle_state)
    swap_tiles(new_state2, blank_y, blank_x, swap2_y, swap2_x)
    if even_parity(new_state2) and not check_visited(visited, new_state2):
        heuristic2 = get_heuristic(new_state2)
        new_node2 = Node(new_state2, heuristic2, new_g, False, root)
        pq.append(new_node2)
        #visited.append(new_node2)

    new_state3 = numpy.copy(puzzle_state)
    swap_tiles(new_state3, blank_y, blank_x, swap3_y, swap3_x)
    if even_parity(new_state3) and not check_visited(visited, new_state3):
        heuristic3 = get_heuristic(new_state3)
        new_node3 = Node(new_state3, heuristic3, new_g, False, root)
        pq.append(new_node3)
        #visited.append(new_node3)

    new_state4 = numpy.copy(puzzle_state)
    swap_tiles(new_state4, blank_y, blank_x, swap4_y, swap4_x)
    if even_parity(new_state4) and not check_visited(visited, new_state4):
        heuristic4 = get_heuristic(new_state4)
        new_node4 = Node(new_state4, heuristic4, new_g, False, root)
        pq.append(new_node4)
        #visited.append(new_node4)


# Checks for even parity, returns true if even
def even_parity(puzzle_state):
    counter = 0
    for i in range(0, 3):
        for j in range(0, 3):
            if puzzle_state[i][j] != 0:
                counter += check_inversions(puzzle_state, puzzle_state[i][j], i, j)
    if counter % 2 == 1:
        return False
    else:
        return True


# Checks to see if the new_state is a duplicate, returns True if it is
def check_visited(visited, new_state) -> bool:
    for i in visited:
        prev_state = i.tiles
        flag = True
        for j in range(0, 3):
            for k in range(0, 3):
                if new_state[j][k] != prev_state[j][k]:
                    flag = False
        if flag:
            return True
    return False


# Checks individual tile for inversions, returns number of inversions
# relative to the tile
def check_inversions(puzzle_state, val: int, i: int, j: int) -> int:
    counter = 0
    # Getting index of next puzzle tile
    if j == 2 and i == 2:   # At the end of the puzzle
        return 0
    elif j == 2:            # Go to next row
        j = 0
        i += 1
    else:                   # Go to next tile
        j += 1

    # Iterating through the remainder of the puzzle
    while i < 3:
        while j < 3:
            if puzzle_state[i][j] < val and puzzle_state[i][j] != 0:  # Inversion is found
                counter += 1
            j += 1
        i += 1
        j = 0
    return counter


# Gets Manhattan Distance for a given state
def get_manhattan(puzzle_state) -> int:
    counter = 0
    for i in range(0, 3):
        for j in range(0, 3):
            if puzzle_state[i][j] != 0:
                if puzzle_state[i][j] != goal_state[i][j]:
                    m = ((puzzle_state[i][j] - 1) // 3)
                    n = ((puzzle_state[i][j] - 1) % 3)
                    counter += abs(m-i) + abs(n-j)
    return counter


# Gets the number of out-of-place tiles for a given state
def get_tiles_oop(puzzle_state) -> int:
    counter = 0
    for i in range(0, 3):
        for j in range(0, 3):
            if puzzle_state[i][j] != 0 and puzzle_state[i][j] != goal_state[i][j]:
                counter += 1
    return counter


# Gets the location of a given state's blank tile
# For use in determining next possible states
def get_blank_loc(puzzle_state, blank_loc):
    for i in range(0, 3):
        for j in range(0, 3):
            if puzzle_state[i][j] == 0:
                blank_loc[0] = i
                blank_loc[1] = j
                break


# Builds the solution path from the goal state to each parent node
def make_solution(node, solution_nodes):
    while node:
        solution_nodes.append(node)
        node = node.parent


# Tests a given state against the goal state
# Returns False if not equivalent
def goal_test(puzzle_state) -> bool:
    for i in range(0, 3):
        for j in range(0, 3):
            if puzzle_state[i][j] != goal_state[i][j]:
                return False
    return True


# Sort function for ordering the priority queue
def sort_function(node):
    return node.heuristic_h + node.heuristic_g


# Function for formatting the printing of a single step in the solution route
def print_step(root, index):
    tiles = root.tiles
    c_tiles = root.children[index].tiles
    print("[", tiles[0][0], " ", tiles[0][1], " ", tiles[0][2], "] ---> [",
          c_tiles[0][0], " ", c_tiles[0][1], " ", c_tiles[0][2], "]")
    print("[", tiles[1][0], " ", tiles[1][1], " ", tiles[1][2], "] ---> [",
          c_tiles[1][0], " ", c_tiles[1][1], " ", c_tiles[1][2], "]")
    print("[", tiles[2][0], " ", tiles[2][1], " ", tiles[2][2], "] ---> [",
          c_tiles[2][0], " ", c_tiles[2][1], " ", c_tiles[2][2], "]")


# Function for formatting the printing of a puzzle state
def print_tiles(tiles):
    print("[", tiles[0][0], " ", tiles[0][1], " ", tiles[0][2], "]")
    print("[", tiles[1][0], " ", tiles[1][1], " ", tiles[1][2], "]")
    print("[", tiles[2][0], " ", tiles[2][1], " ", tiles[2][2], "]")
    print()


# Function for swapping the tiles in positions (y1, x1) and (y2, x2)
def swap_tiles(puzzle_state, y1, x1, y2, x2):
    puzzle_state[y1][x1], puzzle_state[y2][x2] = puzzle_state[y2][x2], puzzle_state[y1][x1]
    # temp = numpy.copy(puzzle_state[pos1[0]][pos1[1]])
    # puzzle_state[pos1[0]][pos1[1]] = numpy.copy(puzzle_state[pos2[0]][pos2[1]])
    # puzzle_state[pos2[0]][pos2[1]] = numpy.copy(temp)


# Calculates h(x) for a given puzzle state
def get_heuristic(puzzle_state) -> int:
    return get_tiles_oop(puzzle_state)  # get_manhattan(puzzle_state) + get_tiles_oop(puzzle_state)


if __name__ == '__main__':
    sys.setrecursionlimit(100000000)
    puzzle = numpy.array([[1, 2, 3], [5, 4, 7], [6, 0, 8]])

    if not even_parity(puzzle):
        print("Odd parity, no solution")
        exit(1)
    result = True
    for r in range(0, 3):
        for c in range(0, 3):
            if puzzle[r][c] != goal_state[r][c]:
                result = False
    if result:
        print("Already at goal state")
        exit(2)

    root_heuristic = get_tiles_oop(puzzle)  # get_manhattan(puzzle) + get_tiles_oop(puzzle)
    root_node = Node(puzzle, root_heuristic, 0, True, None)
    solution = []
    total_expansions = 0

    a_star(root_node, solution)
    solution = list(reversed(solution))
    solution_counter = 0
    print()
    for x in solution:
        print("=====", solution_counter, "=====")
        print_tiles(x.tiles)
        solution_counter += 1
    print(total_expansions, "total expansions")
