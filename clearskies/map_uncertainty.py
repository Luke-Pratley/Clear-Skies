import numpy as np
import optimusprimal.linear_operators as linear_operators
import optimusprimal.map_uncertainty as map_uncertainty


def uncertainty_quantification(x_sol, data, sigma, weights, wav, levels, gamma, alpha, top, region_size = 16, iters = 10, tol = 1e-3):
    psi = linear_operators.dictionary(wav, levels, x_sol.shape)
    W = weights
    obj = lambda x: gamma * np.sum(np.abs(psi.dir_op(x))) + np.sum(np.abs(W * x - W * data)**2)/(2 * sigma**2)
    bound = obj(x_sol) + float(len(np.ravel(x_sol))) + np.sqrt(len(np.ravel(x_sol)) * 16. * np.log(3./alpha))
    return map_uncertainty.create_credible_region(x_sol, region_size, obj, bound, iters, tol, top)
