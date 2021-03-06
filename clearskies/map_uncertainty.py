import numpy as np
import optimusprimal.linear_operators as linear_operators
import optimusprimal.map_uncertainty as map_uncertainty


def uncertainty_quantification(x_sol, data, sigma, weights, wav, levels, gamma, options={'alpha': 0.99, "top": 1e3, "bottom": 0, "region_size": 16, "iters": 10, "tol": 1e-3}):
    psi = linear_operators.dictionary(wav, levels, x_sol.shape)
    W = weights
    obj = lambda x: gamma * np.sum(np.abs(psi.dir_op(x))) + np.sum(np.abs(W * x - W * data)**2)/(2 * sigma**2)
    bound = obj(x_sol) + float(len(np.ravel(x_sol))) + np.sqrt(len(np.ravel(x_sol)) * 16. * np.log(3./options['alpha']))
    return map_uncertainty.create_local_credible_interval(x_sol, options['region_size'], obj, bound, options['iters'], options['tol'], options['bottom'], options['top'])
