import optimusprimal.prox_operators as prox_operators
import optimusprimal.grad_operators as grad_operators
import optimusprimal.linear_operators as linear_operators
import optimusprimal.primal_dual as primal_dual
import numpy as np
import logging


def l1_constrained_solver(data, sigma, weights, psi, beta = 1e-3, options = {'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': False, 'real': False}):
    """
    Solve constrained l1 regularization problem
    """
    phi = linear_operators.diag_matrix_operator(weights)
    size = len(np.ravel(data))
    epsilon = np.sqrt(size + 2 * np.sqrt(2 * size)) * sigma
    p = prox_operators.l2_ball(epsilon, data, phi)
    p.beta = np.max(np.abs(weights))**2
    f = None
    if options['real'] == True:
        if options["positivity"] == True:
            f = prox_operators.positive_prox()
        else:
            f = prox_operators.reality_prox()
    h = prox_operators.l1_norm(np.max(np.abs(psi.dir_op(data))) * beta, psi)
    return primal_dual.FBPD(phi.adj_op(data), options, f, h, p, None)


def l1_unconstrained_solver(data, sigma, weights, psi, beta = 1e-3, options = {'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': False, 'real': False}):
    """
    Solve unconstrained l1 regularization problem
    """

    phi = linear_operators.diag_matrix_operator(weights)
    g = grad_operators.l2_norm(sigma, data, phi)
    g.beta = np.max(np.abs(weights))**2 / sigma**2
    h = prox_operators.l1_norm(beta, psi)
    f = None
    if options['real'] == True:
        if options["positivity"] == True:
            f = prox_operators.positive_prox()
        else:
            f = prox_operators.reality_prox()
    return primal_dual.FBPD(phi.adj_op(data), options, f, h, None, g)

def l2_unconstrained_solver(data, sigma, weights, psi, sigma_signal = 1e-3, options = {'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': True, 'real': False}):
    """
    Solve unconstrained l1 regularization problem
    """

    phi = linear_operators.diag_matrix_operator(weights)
    g = grad_operators.l2_norm(sigma, data, phi)
    g.beta = np.max(np.abs(weights))**2 / sigma**2
    h = prox_operators.l2_square_norm(sigma_signal, psi)
    f = None
    if options['real'] == True:
        if options["positivity"] == True:
            f = prox_operators.positive_prox()
        else:
            f = prox_operators.reality_prox()
    return primal_dual.FBPD(phi.adj_op(data), options, f, h, None, g)
