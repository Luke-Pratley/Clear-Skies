import optimusprimal.linear_operators as linear_operators
import clearskies.core
from enum import Enum
import logging
import numpy as np


class algorithm(Enum):
    l1_constrained = 0
    l1_unconstrained = 1
    l2_unconstrained = 2
    l1_poisson_constrained = 3
    l1_poisson_unconstrained = 4


logger = logging.getLogger('Clear Skies')


def solver(algo, image, sigma, weights, wav=["dirac"], levels=6, beta=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, "positivity": False},  warm_start=None, background=None, axes=None):
    """
    Chooses algorithm and solver
    """
    logger.info("Image shape %s", image.shape)
    logger.info("Using wavelets %s with %s levels", wav, levels)
    logger.info(
        "Using an estimated noise level of %s (weighted image units, i.e. Jy/Beam)", sigma)
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
    if algo == algorithm.l1_poisson_constrained:
        logger.info(
            "Denosing using constrained l1 regularization with poisson constraint")
        logger.info(
            "Ignoring Sigma since noise level is determined by the data")
        if(background is not None):
            logger.info("Using background.")
        else:
            background = image * 0
        return clearskies.core.l1_poissonian_constrained_solver(data, starting_data, len(np.ravel(image[image > 0])), weights, background, psi, beta, options)
    if algo == algorithm.l1_poisson_unconstrained:
        logger.info(
            "Denosing using unconstrained l1 regularization with poisson likelihood")
        logger.info(
            "Ignoring Sigma since noise level is determined by the data")
        if(background is not None):
            logger.info("Using background.")
        else:
            background = image * 0
        return clearskies.core.l1_poissonian_unconstrained_solver(data, starting_data, weights, background, psi, beta, options)
    raise ValueError("Algorithm not reconginized.")
