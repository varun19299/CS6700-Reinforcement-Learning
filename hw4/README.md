# README

Assignment 4, CS 6700.

## Grid World Problem

Run as :

```
usage: q1.py [-h] [--stages {10,25,-1}] [--terminal {99,3}] [--supress {0,1}]
             [--verbose {0,1}]

optional arguments:
  -h, --help           show this help message and exit
  --stages {10,25,-1}  Number of stages to solve for DP, -1 indicates till
                       convergence of reward.
  --terminal {99,3}    Terminal State in the grid world.
  --supress {0,1}      Whether to supress plots from showing up.
  --verbose {0,1}      Verbosity.
```

## Taxi Stand Problem

Run as:

```
usage: q2.py [-h] [--stages {10,20,30}] [--iter_type {value,policy,mpi,gauss}]

optional arguments:
  -h, --help            show this help message and exit
  --stages {10,20,30}
  --iter_type {value,policy,mpi,gauss}
```

## Report

See `hw4/report/contents.tex`. Submitted report may be generated from `hw4/report/overlay.tex`.