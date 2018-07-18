# **QKP** ~150
* ### mixed sign ~50,60 (ub 70 took 2000 seconds)

# **KQKP** ~50,55 (60 took 2000 seconds)
* ### highly density dependent (n=55, den=75 slow. n=60, den=25 fast)
* ### mixed sign - Solve much faster

# **QSAP**  n~4,5 m~15
* ### ub (6,18) takes ~3000 seconds

# **HSP** (n=40, den=75 took 500 seconds. (45,75) took 6,000)
* ### 100% dense take way longer

# **UQP** n~55, den-75

# glover bounds
#### tight/tighter consistently beat original
#### tight=tighter for (qsap)

# glover constraints
#### using all 4 (lhs) not worth it
#### sub2 beats sub1?!


# things to test
* low density, sparse problems (best with std?)
* default quadratic solvers (confirm this is worse than linearizations)
* compare solver speeds (is cplex better for some things vs gurobi for others?)
*

# Observations
* cplex seems to beat other solvers for large problems
* glover tight bounts, 1st or 2nd sub seems best
* glovers > ss >> std ~ elf
* ut > symmetric?
