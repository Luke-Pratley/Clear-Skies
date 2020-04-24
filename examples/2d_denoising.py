import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../../Optimus-Primal')
import numpy as np
from clearskies import solver
from clearskies import map_uncertainty
from astropy.io import fits

def open_fits(image_file):
    hdu_list = fits.open(image_file)
    return hdu_list[0].data
def open_header(image_file):
    hdu_list = fits.open(image_file)
    return hdu_list[0].header

input_file = "../data/lmc.fits"

x_true = open_fits(input_file)
header = open_header(input_file)

options = {'tol': 1e-3, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': False, 'real': False}
ISNR = 20.
sigma = 10**(-ISNR/20.) * np.sqrt(np.sum(np.abs(x_true)**2)/(x_true.shape[0] * x_true.shape[1]))
width, height = x_true.shape 

W = np.ones(x_true.shape)

y = W * x_true + np.random.normal(0, sigma, x_true.shape)


wav = ["db1", "db2", "db3", "db4","db5", "db6", "db7", "db8"]
#wav = ["db8"]
#wav = ["dirac"]
levels = 4

z, diag = solver.solver(solver.algorithm.l1_unconstrained, y, sigma, W, wav, levels, 0.1, options)
SNR = np.log10(np.sqrt(np.sum(np.abs(x_true)**2))/np.sqrt(np.sum(np.abs(x_true - z)**2))) * 20.
print("Input SNR = ", ISNR)
print("Recovered SNR = ", SNR)
fits.writeto("outputs/2d_denoising_"+str(int(ISNR))+"_input.fits",np.real(y), header=header, overwrite=True)
fits.writeto("outputs/2d_denoising_"+str(int(ISNR))+"_output.fits",np.real(z), header=header, overwrite=True)
top = np.std(z) / 4 * 10
error_p, error_m, mean = map_uncertainty.uncertainty_quantification(np.real(z), y, sigma, W, wav, levels, 10., 0.99, top, 4, 1000, 1e-2) 

fits.writeto("outputs/2d_denoising_"+str(int(ISNR))+"_output_error.fits",np.real(error_p + error_m), header=header, overwrite=True)
fits.writeto("outputs/2d_denoising_"+str(int(ISNR))+"_output_mean.fits",np.real(mean), header=header, overwrite=True)
