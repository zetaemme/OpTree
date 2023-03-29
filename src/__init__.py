"""Main module for the OpSion project.

Classes:
    Dataset
    Separation

Functions:
    find_budget
    build_decision_tree
    cheapest_separation
    wolsey_greedy_heuristic
    submodular_function_1
"""
from joblib import Memory

memory = Memory("../cache", verbose=0)
