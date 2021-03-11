import optimusprimal.prox_operators as prox_operators
import optimusprimal.grad_operators as grad_operators
import optimusprimal.linear_operators as linear_operators
import optimusprimal.primal_dual as primal_dual
import numpy as np
import logging


def l1_constrained_solver(data, warm_start, sigma, weights, psi, beta=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': False, 'real': False}):
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
        f = prox_operators.real_prox()
    if options["positivity"] == True:
        f = prox_operators.positive_prox()
    h = prox_operators.l1_norm(np.max(np.abs(psi.dir_op(data))) * beta, psi)
    return primal_dual.FBPD(warm_start, options, None, f, h, p)


def l1_unconstrained_solver(data, warm_start, sigma, weights, psi, beta=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': False, 'real': False}):
    """
    Solve unconstrained l1 regularization problem
    """

    phi = linear_operators.diag_matrix_operator(weights)
    g = grad_operators.l2_norm(sigma, data, phi)
    g.beta = np.max(np.abs(weights))**2 / sigma**2
    if beta <= 0:
        h = None
    else:
        h = prox_operators.l1_norm(beta, psi)
    f = None
    if options['real'] == True:
        if options["positivity"] == True:
            f = prox_operators.positive_prox()
        else:
            f = prox_operators.real_prox()
    return primal_dual.FBPD(warm_start, options, g, f, h)


def l2_unconstrained_solver(data, warm_start, sigma, weights, psi, sigma_signal=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': True, 'real': False}):
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
    return primal_dual.FBPD(warm_start, options, g, f, h)


def l1_poissonian_constrained_solver(data, warm_start, sigma, weights, background, psi,
                                     beta=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': False, 'real': False}):
    """
    Solve constrained l1 regularization problem with poisson noise
    """
    phi = linear_operators.diag_matrix_operator(weights)
    size = len(np.ravel(data))
    epsilon = sigma
    p = prox_operators.poisson_loglike_ball(epsilon, data, background, 50, phi)
    p.beta = np.max(np.abs(weights))**2
    f = None
    if options['real'] == True:
        if options["positivity"] == True:
            f = prox_operators.positive_prox()
        else:
            raise Exception("Positivity required.")
    h = prox_operators.l1_norm(np.max(np.abs(psi.dir_op(data))) * beta, psi)
    return primal_dual.FBPD(warm_start, options, None, f, h, p)


def l1_poissonian_unconstrained_solver(data, warm_start, weights, background, psi,
                                       beta=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': False, 'real': False}):
    """
    Solve unconstrained l1 regularization problem with poisson noise
    """

    phi = linear_operators.diag_matrix_operator(weights)
    p = prox_operators.poisson_loglike(data, background, phi)
    p.beta = np.max(np.abs(weights))**2
    if beta <= 0:
        h = None
    else:
        h = prox_operators.l1_norm(beta, psi)
    if options['real'] == True:
        if options["positivity"] == True:
            f = prox_operators.positive_prox()
        else:
            raise Exception("Positivity required.")
    f = None
    return primal_dual.FBPD(warm_start, options, None, f, h, p)

def tv_constrained_solver(data, warm_start, sigma, weights, beta=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': False, 'real': False}):
    """
    Solve constrained tv regularization problem
    """
    phi = linear_operators.diag_matrix_operator(weights)
    size = len(np.ravel(data))
    epsilon = np.sqrt(size + 2 * np.sqrt(2 * size)) * sigma
    p = prox_operators.l2_ball(epsilon, data, phi)
    p.beta = np.max(np.abs(weights))**2
    f = None
    if options['real'] == True:
        f = prox_operators.real_prox()
    if options["positivity"] == True:
        f = prox_operators.positive_prox()
    def forward(x):
        if x.ndim == 2:
            out = np.zeros((2, x.shape[0] - 1, x.shape[1] - 1))
            out[0, :, :] = (x[:-1, :-1] - x[1:, :-1])/2
            out[1, :, :] = (x[:-1, :-1] - x[:-1, 1:])/2
            return out
        else:
            raise Exception("Sorry, only two dimensions for now.")
    def backward(x):
        if x.ndim == 3:
            out = np.zeros((x.shape[1] + 1, x.shape[2] + 1))
            out[:-1, :-1] +=  x[0, :, :]/2
            out[:-1, :-1] +=  x[1, :, :]/2
            out[1:, :-1] +=  -x[0, :, :]/2
            out[:-1, 1:] +=  -x[1, :, :]/2
            return out
        else:
            raise Exception("Sorry, only two dimensions for now.")
    psi = linear_operators.function_wrapper(forward, backward)
    h = prox_operators.l21_norm(np.sqrt(np.max(np.sum(np.abs(psi.dir_op(data))**2), axis=0)) * beta, 0, psi)
    return primal_dual.FBPD(warm_start, options, None, f, h, p)

def tv_unconstrained_solver(data, warm_start, sigma, weights, beta=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': False, 'real': False}):
    """
    Solve unconstrained l1 regularization problem
    """

    phi = linear_operators.diag_matrix_operator(weights)
    g = grad_operators.l2_norm(sigma, data, phi)
    g.beta = np.max(np.abs(weights))**2 / sigma**2
    if beta <= 0:
        h = None
    else:
        def forward(x):
            if x.ndim == 2:
                out = np.zeros((2, x.shape[0] - 1, x.shape[1] - 1))
                out[0, :, :] = (x[:-1, :-1] - x[1:, :-1])/2
                out[1, :, :] = (x[:-1, :-1] - x[:-1, 1:])/2
                return out
            else:
                raise Exception("Sorry, only two dimensions for now.")
        def backward(x):
            if x.ndim == 3:
                out = np.zeros((x.shape[1] + 1, x.shape[2] + 1))
                out[:-1, :-1] +=  x[0, :, :]/2
                out[:-1, :-1] +=  x[1, :, :]/2
                out[1:, :-1] +=  -x[0, :, :]/2
                out[:-1, 1:] +=  -x[1, :, :]/2
                return out
            else:
                raise Exception("Sorry, only two dimensions for now.")
        psi = linear_operators.function_wrapper(forward, backward)
        h = prox_operators.l21_norm(beta, 0, psi)
    f = None
    if options['real'] == True:
        if options["positivity"] == True:
            f = prox_operators.positive_prox()
        else:
            f = prox_operators.real_prox()
    return primal_dual.FBPD(warm_start, options, g, f, h)

def l1_constrained_separation_solver(data, warm_start, sigma, weights, psi1, psi2, gamma=1, beta=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': False, 'real': False}):
    """
    Solve constrained l1 regularization problem for signal separation
    """
    def m_forward(x):
        return (x[0, :, :] + x[1, :, :]) * weights
    def m_backward(x):
        out = np.zeros((2, x.shape[0], x.shape[1]), dtype=complex)
        out[0] = x * weights
        out[1] = x * weights
        return out

    phi = linear_operators.function_wrapper(m_forward, m_backward)
    size = len(np.ravel(data))
    epsilon = np.sqrt(size + 2 * np.sqrt(2 * size)) * sigma
    p = prox_operators.l2_ball(epsilon, data, phi)
    p.beta = np.max(np.abs(weights))**2 * 4
    f = None
    if options['real'] == True:
        f = prox_operators.real_prox()
    if options["positivity"] == True:
        f = prox_operators.positive_prox()
    def w1_forward(x):
        return psi1.dir_op(x[0, :, :])
    def w2_forward(x):
        return psi2.dir_op(x[1, :, :])
    def w1_backward(x):
        out = psi1.adj_op(x)
        buff = np.zeros((2, out.shape[0], out.shape[1]), dtype=complex)
        buff[0] = out
        return buff
    def w2_backward(x):
        out = psi2.adj_op(x)
        buff = np.zeros((2, out.shape[0], out.shape[1]), dtype=complex)
        buff[1] = out
        return buff

    psi1_wrapper = linear_operators.function_wrapper(w1_forward, w1_backward)
    psi2_wrapper = linear_operators.function_wrapper(w2_forward, w2_backward)

    h = prox_operators.l1_norm(np.max(np.abs(psi1.dir_op(data))) * beta, psi1_wrapper)
    h.beta = 1
    r = prox_operators.l1_norm(np.max(np.abs(psi2.dir_op(data))) * beta * gamma, psi2_wrapper)
    r.beta = 1
    out = np.zeros((2, warm_start.shape[0], warm_start.shape[1]), dtype=complex)
    out[0] = warm_start
    out[1] = warm_start
    return primal_dual.FBPD(out, options, None, f, h, p, r)

def tv_constrained_separation_solver(data, warm_start, sigma, weights, psi1, gamma=1, beta=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': False, 'real': False}):
    """
    Solve constrained l1 regularization problem for signal separation
    """
    def m_forward(x):
        return (x[0, :, :] + x[1, :, :]) * weights
    def m_backward(x):
        out = np.zeros((2, x.shape[0], x.shape[1]), dtype=complex)
        out[0] = x * weights
        out[1] = x * weights
        return out

    phi = linear_operators.function_wrapper(m_forward, m_backward)
    size = len(np.ravel(data))
    epsilon = np.sqrt(size + 2 * np.sqrt(2 * size)) * sigma
    p = prox_operators.l2_ball(epsilon, data, phi)
    p.beta = np.max(np.abs(weights))**2 * 4
    f = None
    if options['real'] == True:
        f = prox_operators.real_prox()
    if options["positivity"] == True:
        f = prox_operators.positive_prox()
    def forward(x):
        if x.ndim == 2:
            out = np.zeros((2, x.shape[0] - 1, x.shape[1] - 1))
            out[0, :, :] = (x[:-1, :-1] - x[1:, :-1])/2
            out[1, :, :] = (x[:-1, :-1] - x[:-1, 1:])/2
            return out
        else:
            raise Exception("Sorry, only two dimensions for now.")
    def backward(x):
        if x.ndim == 3:
            out = np.zeros((x.shape[1] + 1, x.shape[2] + 1))
            out[:-1, :-1] +=  x[0, :, :]/2
            out[:-1, :-1] +=  x[1, :, :]/2
            out[1:, :-1] +=  -x[0, :, :]/2
            out[:-1, 1:] +=  -x[1, :, :]/2
            return out
        else:
            raise Exception("Sorry, only two dimensions for now.")
    def w1_forward(x):
        return psi1.dir_op(x[0, :, :])
    def w2_forward(x):
        return forward(x[1, :, :])
    def w1_backward(x):
        out = psi1.adj_op(x)
        buff = np.zeros((2, out.shape[0], out.shape[1]), dtype=complex)
        buff[0] = out
        return buff
    def w2_backward(x):
        out = backward(x)
        buff = np.zeros((2, out.shape[0], out.shape[1]), dtype=complex)
        buff[1] = out
        return buff

    psi1_wrapper = linear_operators.function_wrapper(w1_forward, w1_backward)
    psi2_wrapper = linear_operators.function_wrapper(w2_forward, w2_backward)

    h = prox_operators.l1_norm(np.max(np.abs(psi1.dir_op(data))) * beta, psi1_wrapper)
    h.beta = 1
    r = prox_operators.l21_norm(np.sqrt(np.max(np.sum(np.abs(forward(data))**2,axis=0))) * beta * gamma, 0, psi2_wrapper)
    r.beta = 1
    out = np.zeros((2, warm_start.shape[0], warm_start.shape[1]), dtype=complex)
    out[0] = warm_start
    out[1] = warm_start
    return primal_dual.FBPD(out, options, None, f, h, p, r)
