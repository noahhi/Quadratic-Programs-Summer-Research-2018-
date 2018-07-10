# A Computational Study of Solution Techniques for 0-1 Quadratic Programs.
Conducted at Dickinson College by Noah Hunt-Isaak with faculty advisor Richard Forrester during the summer of 2018.

## Files

* ### quadratics.py
Generates problem instances of various classes of 0-1 quadratic program (QKP, KQKP, UQP, etc..)
* ### cplex_solve.py, gurobi_solve.py, xpress_solve.py
Model a QP with desired linearization technique using CPLEX, Gurobi, or Xpress commercial solvers.
* ### run_trials.py
Run a set of trials
* ### read_data.py
Use after run_trials to read in dataframe, export to excel and graph  
