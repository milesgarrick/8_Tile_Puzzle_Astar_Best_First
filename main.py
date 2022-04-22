import sys
import treelib
import numpy


goal_state = numpy.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])


class Node:
    def __init__(self, tiles, heuristic, explored, parent):
        self.tiles = numpy.zeros((3, 3), int)
        self.heuristic = heuristic
        self.explored = explored
        self.parent = parent


def even_parity(puzzle_state):
    counter = 0
    for i in range (0, 3):
        for j in range (0, 3):
            if puzzle_state[i][j] != 0:
                counter += check_inversions(puzzle_state, puzzle_state[i][j], i, j)
    if counter % 2 == 1:
        return False
    else:
        return True


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

    #Iterating through the remainder of the puzzle
    while i < 3:
        while j < 3:
            if puzzle_state[i][j] < val:  # Inversion is found
                counter += 1
            j += 1
        i += 1
        j = 0
    return counter


if __name__ == '__main__':
    puzzle = numpy.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    tree = treelib.Tree()
    tree.create_node("root", data=puzzle)
    if not even_parity(puzzle):
        print("Odd parity, no solution")
        exit(1)
    result = True
    for i in range(0, 3):
        for j in range(0, 3):
            if puzzle[i][j] != goal_state[i][j]:
                result = False
    if result:
        print("Already at goal state")
        exit(2)


