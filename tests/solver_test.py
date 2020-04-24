import sys
import numpy as np
from clearskies import solver
from clearskies import core
from astropy.io import fits
from optimusprimal import linear_operators


def open_fits(image_file):
    hdu_list = fits.open(image_file)
    return hdu_list[0].data


def open_header(image_file):
    hdu_list = fits.open(image_file)
    return hdu_list[0].header



def test_constrained():
    input_file = "data/lmc.fits"
    x_true = open_fits(input_file)
    
    options = {'tol': 1e-3, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'real': False, 'positivity': False}
    ISNR = 30.
    sigma = 10**(-ISNR/20.) * np.sqrt(np.sum(np.abs(x_true)**2)/np.sqrt(x_true.shape[0] * x_true.shape[1]))
    width, height = x_true.shape

    W = np.ones(x_true.shape)
    
    y = W * x_true + np.random.normal(0, sigma, x_true.shape)
    
    
    wav = ["dirac", "db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8"]
    levels = 4
    psi = linear_operators.dictionary(wav, levels, y.shape)
    data = y * W
    
    z, diag = solver.solver(solver.algorithm.l1_constrained, y, sigma, W, wav, levels, 1e-3, options)
    z_expected, diag_expected = core.l1_constrained_solver(data, sigma, W, psi, 1e-3, options)

    SNR = np.log10(np.sqrt(np.sum(np.abs(x_true)**2))/np.sqrt(np.sum(np.abs(x_true - z)**2))) * 20.
    assert(SNR > ISNR)
    size = z.shape[0] * z.shape[1]
    assert(np.linalg.norm(W * z - y) < np.sqrt(size + 2 * np.sqrt(size)) * sigma)
    assert(np.all(np.isclose(z, z_expected)))
