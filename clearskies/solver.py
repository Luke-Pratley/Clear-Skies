from astropy.io import fits 
import optimusprimal.linear_operators as linear_operators
import clearskies
from enum import Enum

class algorithm(Enum):
    constrained = 1
    unconstrained = 2

def open_fits(image_file):
    hdu_list = fits.open(image_file)
    return hdu_list[0].data
def open_header(image_file):
    hdu_list = fits.open(image_file)
    return hdu_list[0].header

def solver(algo, image, sigma, weights, wav = ["dirac, db1, db2, db3, db4"], levels = 6, beta = 1e-3, options = {'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False}):
    psi = linear_operators.dictionary(wav, levels, image.shape)
    data = image * weights
    if algo == algorithm.constrained:
        return clearskies.core.constrained_solver(data,sigma, weights, psi, beta, options)
    else if algo == algorithm.unconstrained:
        return clearskies.core.unconstrained_solver(data,sigma, weights, psi, beta, options)
    else:
        raise ValueError("Algorithm not reconginized.")
