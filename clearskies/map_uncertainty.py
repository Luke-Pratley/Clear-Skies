import numpy as np
import optimusprimal.linear_operators as linear_operators
import optimusprimal.map_uncertainty as map_uncertainty


def uncertainty_quantification(x_sol, data, sigma, weights, wav, levels, gamma, options={'alpha': 0.99, "top": 1e3, "bottom": 0, "region_size": 16, "iters": 10, "tol": 1e-3}):
    psi = linear_operators.dictionary(wav, levels, x_sol.shape)
    W = weights
    obj = lambda data_sol, data_mask, wav_sol, wav_mask: gamma * np.sum(np.abs(wav_sol + wav_mask)) + np.sum(np.abs(W * (data_sol + data_mask) - W * data)**2)/(2 * sigma**2)
    bound = obj(x_sol, 0, psi.dir_op(x_sol), 0) + float(len(np.ravel(x_sol))) + np.sqrt(float(len(np.ravel(x_sol))) * 16. * np.log(3./options['alpha']))
    print(obj(x_sol, 0, psi.dir_op(x_sol), 0))
    print(np.sqrt(float(len(np.ravel(x_sol))) * 16. * np.log(3./options['alpha'])))
    phi = linear_operators.identity()
    return map_uncertainty.create_local_credible_interval_fast(x_sol, phi, psi, options['region_size'], obj, bound, options['iters'], options['tol'], options['bottom'], options['top'])
