# Sokoban solver

[![Sokoban solver](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/c/sokoban-solver-01-run-20261121)
[![Rank](https://make.fun/badge/sokoban-solver-01-run-20261121/hurulsufi12-art/sokoban-solver/rank.svg)](https://make.fun/c/sokoban-solver-01-run-20261121)
[![Reward](https://make.fun/badge/sokoban-solver-01-run-20261121/hurulsufi12-art/sokoban-solver/reward.svg)](https://make.fun/c/sokoban-solver-01-run-20261121)

Push-optimal Sokoban solver for the make.fun bounty "Build the best Sokoban
solver, 21 November 2026".

## Idea

A generated level is a one-square hallway with 1–4 rooms off it. A box pushed
out of a room can never be pushed back, so every box stays in its room and the
level's push-optimum is the sum of its rooms' optima. Each room (3–4 boxes in
a 6×6 slot) is solved push-optimally on its own with a breadth-first search
over pushes, pruned by dead squares (reverse pull-back from the goals) and
frozen 2×2 blocks; the player's walks between pushes are shortest paths. A*
with an admissible box-to-goal matching heuristic, then greedy best-first,
are fallbacks when a room's BFS exhausts its node budget, so a valid solution
is still written instead of an empty answer.

The solver is deterministic: fixed direction order, sorted boxes, budgets by
node count (never the clock), no randomness.

## Run

```sh
node main.ts
```

Reads `/input/<id>/level.txt` for `i001`–`i005` (or the directories named by
`MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR`) and writes
`/output/<id>/solution.txt` for each: the player's moves in `u/d/l/r`.

## Files

- `main.ts` — the whole solver (Node 22, standard library only)
- `README.md` — this file
- `LICENSE` — MIT
