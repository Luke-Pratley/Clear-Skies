import optimusprimal.linear_operators as linear_operators
import clearskies.core
from enum import Enum
import logging


class algorithm(Enum):
    l1_constrained = 0
    l1_unconstrained = 1
    l2_unconstrained = 2


logger = logging.getLogger('Clear Skies')

def solver(algo, image, sigma, weights, wav=["dirac, db1, db2, db3, db4"], levels=6, beta=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, "positivity": False},  warm_start = None, axes = None):
    logger.info("Image shape %s", image.shape)
    logger.info("Using wavelets %s with %s levels", wav, levels)
    logger.info("Using an estimated noise level of %s (weighted image units, i.e. Jy/Beam)", sigma)
    psi = linear_operators.dictionary(wav, levels, image.shape, axes)
    data = image * weights
    starting_data = data * weights
    if(warm_start is not None):
        logger.info("Using warm start.")
        starting_data = warm_start
    if algo == algorithm.l1_constrained:
        logger.info("Denosing using constrained l1 regularization")
        return clearskies.core.l1_constrained_solver(data, starting_data, sigma, weights, psi, beta, options)
    if algo == algorithm.l1_unconstrained:
        logger.info("Denosing using unconstrained l1 regularization")
        return clearskies.core.l1_unconstrained_solver(data, starting_data, sigma, weights, psi, beta, options)
    if algo == algorithm.l2_unconstrained:
        logger.info("Denosing using unconstrained l2 regularization")
        return clearskies.core.l2_unconstrained_solver(data, starting_data, sigma, weights, psi, beta, options)
    raise ValueError("Algorithm not reconginized.")
