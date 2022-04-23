import numpy
import gc


goal_state = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])


class Node:
    def __init__(self, tiles, heuristic_h: int, heuristic_g: int, explored: bool, parent):
        self.tiles = tiles
        self.heuristic_h = heuristic_h
        self.heuristic_g = heuristic_g
        self.explored = explored
        self.parent = parent
        self.children = []


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


def get_tiles_oop(puzzle_state) -> int:
    counter = 0
    for i in range(0, 3):
        for j in range(0, 3):
            if puzzle_state[i][j] != i + j + 1:
                counter += 1
    return counter


def get_blank_loc(puzzle_state, blank_loc):
    for i in range(0, 3):
        for j in range(0, 3):
            if puzzle_state[i][j] == 0:
                blank_loc[0] = i
                blank_loc[1] = j
                break


def a_star(root, solution_nodes):
    root.explored = True
    puzzle_state = root.tiles

    # Test for goal state
    goal_test = True
    for i in range(0, 3):
        for j in range(0, 3):
            if puzzle[i][j] != goal_state[i][j]:
                goal_test = False
    if goal_test:
        solution_nodes.append(root)
        return True

    # gets location of blank tile
    # blank_loc = numpy.where(puzzle_state == 0)
    blank_loc = [0, 0]
    get_blank_loc(puzzle_state, blank_loc)

    # The next block determines what the next possible states are
    # based on the current location of the blank tile
    if blank_loc[0] == 0:
        if blank_loc[1] == 0:
            new_state1 = numpy.copy(puzzle_state)
            swap_tiles(new_state1, 0, 0, 0, 1)
            if even_parity(new_state1):
                heuristic1 = get_heuristic(new_state1)
                new_node1 = Node(new_state1, heuristic1, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node1)

            new_state2 = numpy.copy(puzzle_state)
            swap_tiles(new_state2, 0, 0, 1, 0)
            if even_parity(new_state2):
                heuristic2 = get_heuristic(new_state2)
                new_node2 = Node(new_state2, heuristic2, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node2)

        if blank_loc[1] == 1:
            new_state1 = numpy.copy(puzzle_state)
            swap_tiles(new_state1, 0, 1, 0, 0)
            if even_parity(new_state1):
                heuristic1 = get_heuristic(new_state1)
                new_node1 = Node(new_state1, heuristic1, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node1)

            new_state2 = numpy.copy(puzzle_state)
            swap_tiles(new_state2, 0, 1, 0, 2)
            if even_parity(new_state2):
                heuristic2 = get_heuristic(new_state2)
                new_node2 = Node(new_state2, heuristic2, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node2)

            new_state3 = numpy.copy(puzzle_state)
            swap_tiles(new_state3, 0, 1, 1, 1)
            if even_parity(new_state3):
                heuristic3 = get_heuristic(new_state3)
                new_node3 = Node(new_state3, heuristic3, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node3)

        if blank_loc[1] == 2:
            new_state1 = numpy.copy(puzzle_state)
            swap_tiles(new_state1, 0, 2, 0, 1)
            if even_parity(new_state1):
                heuristic1 = get_heuristic(new_state1)
                new_node1 = Node(new_state1, heuristic1, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node1)

            new_state2 = numpy.copy(puzzle_state)
            swap_tiles(new_state2, 0, 2, 1, 2)
            if even_parity(new_state2):
                heuristic2 = get_heuristic(new_state2)
                new_node2 = Node(new_state2, heuristic2, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node2)

    if blank_loc[0] == 1:
        if blank_loc[1] == 0:
            new_state1 = numpy.copy(puzzle_state)
            swap_tiles(new_state1, 1, 0, 0, 0)
            if even_parity(new_state1):
                heuristic1 = get_heuristic(new_state1)
                new_node1 = Node(new_state1, heuristic1, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node1)

            new_state2 = numpy.copy(puzzle_state)
            swap_tiles(new_state2, 1, 0, 2, 0)
            if even_parity(new_state2):
                heuristic2 = get_heuristic(new_state2)
                new_node2 = Node(new_state2, heuristic2, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node2)

            new_state3 = numpy.copy(puzzle_state)
            swap_tiles(new_state3, 1, 0, 1, 1)
            if even_parity(new_state3):
                heuristic3 = get_heuristic(new_state3)
                new_node3 = Node(new_state2, heuristic3, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node3)

        if blank_loc[1] == 1:
            new_state1 = numpy.copy(puzzle_state)
            swap_tiles(new_state1, 1, 1, 0, 1)
            if even_parity(new_state1):
                heuristic1 = get_heuristic(new_state1)
                new_node1 = Node(new_state1, heuristic1, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node1)

            new_state2 = numpy.copy(puzzle_state)
            swap_tiles(new_state2, 1, 1, 2, 1)
            if even_parity(new_state2):
                heuristic2 = get_heuristic(new_state2)
                new_node2 = Node(new_state2, heuristic2, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node2)

            new_state3 = numpy.copy(puzzle_state)
            swap_tiles(new_state3, 1, 1, 1, 0)
            if even_parity(new_state3):
                heuristic3 = get_heuristic(new_state3)
                new_node3 = Node(new_state3, heuristic3, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node3)

            new_state4 = numpy.copy(puzzle_state)
            swap_tiles(new_state4, 1, 1, 1, 2)
            if even_parity(new_state4):
                heuristic4 = get_heuristic(new_state4)
                new_node4 = Node(new_state4, heuristic4, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node4)

        if blank_loc[1] == 2:

            new_state1 = numpy.copy(puzzle_state)
            swap_tiles(new_state1, 1, 2, 1, 1)
            if even_parity(new_state1):
                heuristic1 = get_heuristic(new_state1)
                new_node1 = Node(new_state1, heuristic1, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node1)

            new_state2 = numpy.copy(puzzle_state)
            swap_tiles(new_state2, 1, 2, 0, 2)
            if even_parity(new_state2):
                heuristic2 = get_heuristic(new_state2)
                new_node2 = Node(new_state2, heuristic2, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node2)

            new_state3 = numpy.copy(puzzle_state)
            swap_tiles(new_state3, 1, 2, 2, 2)
            if even_parity(new_state3):
                heuristic3 = get_heuristic(new_state3)
                new_node3 = Node(new_state3, heuristic3, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node3)

    if blank_loc[0] == 2:
        if blank_loc[1] == 0:

            new_state1 = numpy.copy(puzzle_state)
            swap_tiles(new_state1, 2, 0, 2, 1)
            if even_parity(new_state1):
                heuristic1 = get_heuristic(new_state1)
                new_node1 = Node(new_state1, heuristic1, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node1)

            new_state2 = numpy.copy(puzzle_state)
            swap_tiles(new_state2, 2, 0, 1, 0)
            if even_parity(new_state2):
                heuristic2 = get_heuristic(new_state2)
                new_node2 = Node(new_state2, heuristic2, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node2)

        if blank_loc[1] == 1:
            new_state1 = numpy.copy(puzzle_state)
            swap_tiles(new_state1, 2, 1, 2, 0)
            if even_parity(new_state1):
                heuristic1 = get_heuristic(new_state1)
                new_node1 = Node(new_state1, heuristic1, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node1)

            new_state2 = numpy.copy(puzzle_state)
            swap_tiles(new_state2, 2, 1, 1, 1)
            if even_parity(new_state2):
                heuristic2 = get_heuristic(new_state2)
                new_node2 = Node(new_state2, heuristic2, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node2)

            new_state3 = numpy.copy(puzzle_state)
            swap_tiles(new_state3, 2, 1, 2, 2)
            if even_parity(new_state3):
                heuristic3 = get_heuristic(new_state3)
                new_node3 = Node(new_state3, heuristic3, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node3)

        if blank_loc[1] == 2:
            new_state1 = numpy.copy(puzzle_state)
            swap_tiles(new_state1, 2, 2, 2, 1)
            if even_parity(new_state1):
                heuristic1 = get_heuristic(new_state1)
                new_node1 = Node(new_state1, heuristic1, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node1)

            new_state2 = numpy.copy(puzzle_state)
            swap_tiles(new_state2, 2, 2, 1, 2)
            if even_parity(new_state2):
                heuristic2 = get_heuristic(new_state2)
                new_node2 = Node(new_state2, heuristic2, root.heuristic_g, False, puzzle_state)
                root.children.append(new_node2)

    # Calls a_star with the next best heuristic
    heuristic = 0
    index = 0
    length = len(root.children)
    for j in range(0, length):
        for i in range(0, len(root.children)):
            if root.children[i].heuristic_g + root.children[i].heuristic_h > heuristic:
                heuristic = root.children[i].heuristic_g + root.children[i].heuristic_h
                index = i
        if a_star(root.children[index], solution_nodes):
            solution_nodes.append(root)
            return True
        else:
            del root.children[index]
            gc.collect()
            heuristic = 0
            index = 0
    else:
        return False


def swap_tiles(puzzle_state, y1, x1, y2, x2):
    puzzle_state[y1][x1], puzzle_state[y2][x2] = puzzle_state[y2][x2], puzzle_state[y1][x1]
    # temp = numpy.copy(puzzle_state[pos1[0]][pos1[1]])
    # puzzle_state[pos1[0]][pos1[1]] = numpy.copy(puzzle_state[pos2[0]][pos2[1]])
    # puzzle_state[pos2[0]][pos2[1]] = numpy.copy(temp)


def get_heuristic(puzzle_state) -> int:
    return get_manhattan(puzzle_state) + get_tiles_oop(puzzle_state)


if __name__ == '__main__':
    puzzle = numpy.array([[1, 2, 3], [0, 4, 5], [6, 7, 8]])

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

    root_heuristic = get_manhattan(puzzle) + get_tiles_oop(puzzle)
    root_node = Node(puzzle, root_heuristic, 0, True, None)
    solution = None

    a_star(root_node, solution)
    print(solution)
