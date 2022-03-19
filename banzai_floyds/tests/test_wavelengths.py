import numpy as np
from matplotlib import pyplot as mp


def gaussian(x, mu, sig, str):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * str


def test_1d_metric():
    # make a random list of lines
    line_list_length = 10
    rng = np.random.default_rng()
    line_strs = rng.random(line_list_length)
    peak_cntrs = np.sort(rng.random(line_list_length))
    peak_angstroms = peak_cntrs * 7000 + 3000
    test_lines = [{"wavelength": peak_cntr, 'line_strength': line_strs[i]}
                  for i, peak_cntr in enumerate(peak_angstroms)]

    # simulate a spectrum with some fraction of those lines
    nx = 512
    data_1d = np.zeros(nx)
    peak_pixels = peak_cntrs * nx
    fwhm = 5

    # data_1d += np.random.poisson(500.0, size=data_1d.shape)
    data_1d += np.random.normal(100, scale=10, size=data_1d.shape)
    for i, peak in enumerate(peak_pixels):
        data_1d += gaussian(np.arange(512), peak, fwhm, line_strs[i] * 1200)

    print(data_1d)
    mp.plot(data_1d)
    mp.show()

    # Set the dispersion and some minor distortion
    # add noise to the spectrum

    # Cross correlate the spectrum
    # Find the linear part of the wavelength solution
    pass
