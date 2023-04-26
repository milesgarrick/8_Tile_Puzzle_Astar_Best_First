# 8 Tile Puzzle Solver

Written by Miles Johnson

## Summary
This program uses the A\* algorithm and a best-first search to solve the 8 tile puzzle.

## How to run
Install dependencies and run with the following:

	python3 -m pip install -r requirements.txt
	python3 main.py

## Node class
This class holds the current puzzle state, heuristic values, an exploration status, a parent node, and an array of child nodes

## Understanding the readout
The program can execute either algorithm.  The solution path found is displayed, as well as the number of expansions done to find the solution.

It is currently set to run the Best First algorithm.  To use A\* instead, uncomment line 514 and comment out line 515 of main.py.
