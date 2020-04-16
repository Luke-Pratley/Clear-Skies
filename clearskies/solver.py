import optimusprimal.linear_operators as linear_operators
import clearskies.core
import numpy as np
from enum import Enum


class algorithm(Enum):
    l1_constrained = 0
    l1_unconstrained = 1
    l2_unconstrained = 2


def solver(algo, image, sigma, weights, wav=["dirac, db1, db2, db3, db4"], levels=6, beta=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False}):
    psi = linear_operators.dictionary(wav, levels, image.shape)
    data = image * weights
    if algo == algorithm.l1_constrained:
        return clearskies.core.l1_constrained_solver(data, sigma, weights, psi, beta, options)
    if algo == algorithm.l1_unconstrained:
        return clearskies.core.l1_unconstrained_solver(data, sigma, weights, psi, beta, options)
    if algo == algorithm.l2_unconstrained:
        return clearskies.core.l2_unconstrained_solver(data, sigma, weights, psi, beta, options)
    raise ValueError("Algorithm not reconginized.")
