# long-term-fair-mdps

## Dependenies
* numpy, mdptoolbox (all)
* cvxpy (model-based)


## Included files
* mdp.py has the constructor and a few functions for making sure we can work with groups that are a subset of the states rather than all of the states.
* model-based.py is the convex program from Wen et al. (2019) algorithm 1.  Currently runs into numerical issues with the convex solver, I think due to how small the space of feasible solutions is.  (Though it could very well be a bug too.)
* model-free.py is incomplete, but working towards Algorithm 2 from the above paper.
