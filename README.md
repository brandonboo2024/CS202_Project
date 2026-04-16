# C++ Final Solver

First C++ draft of the final submission solver.

## Algorithm Ported

- Serial SGS priority portfolio.
- Parallel SGS priority portfolio.
- Priority rules: bottom level, transitive successors, resource demand, critical-path slack, longest duration, shortest duration.
- Incumbent activity-list perturbation.
- Optional second activity-list perturb/decode pass on 20% of perturbation iterations.
- Single-pass forward-backward improvement.
- Deadline-controlled randomized loop.

Here, "parallel SGS" is an RCPSP schedule generation scheme, not multithreading. It advances through time and starts multiple eligible activities at the same time point when resource capacity allows.

## Build

Use any C++17 compiler:

```bash
g++ -O2 -std=c++17 -pthread solver.cpp -o solver
```

## Run

Final submission mode reads from stdin and prints exactly one line:

```bash
./solver < instance.SCH
```

Local benchmark mode accepts a path and optional seconds-per-instance budget:

```bash
./solver ../Project/sm_j20/PSP1.SCH 0.02
```

It also accepts an optional worker count for threaded restart experiments:

```bash
./solver ../Project/sm_j20/PSP1.SCH 0.02 4
```

The output is comma-separated start times for jobs `1..N`, or `-1` when provably infeasible.

