import numpy as np


def anscombe(x):
    return 2 * np.sqrt(x + 3./8.)


def inverse_anscombe(z):
    return 1./4. * z**2 + 1./4. * np.sqrt(3.0/2.0) * z**(-1) - 11./8. * z**(-2.0) + 5./8. * np.sqrt(3./2.) * z**(-3.0) - 1./8.
