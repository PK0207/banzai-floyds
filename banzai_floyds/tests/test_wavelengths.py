from matplotlib import pyplot as mp
from banzai_floyds.wavelengths import gauss, linear_wavelength_solution
import numpy as np
from astropy.table import Table


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

    # simulate a spectrum
    nx = 512
    data_1d = np.zeros(nx)

    # Add known lines
    peak_pixels = peak_cntrs * nx
    fwhm = 1
    flux_scale = 1200
    for i, peak in enumerate(peak_pixels):
        data_1d += gaussian(np.arange(nx), peak, fwhm, line_strs[i] * flux_scale)
    mp.plot(data_1d)

    # add extra lines
    extre_lines_num = 3
    unused_peaks = np.sort(rng.random(extre_lines_num)) * nx
    unused_str = rng.random(extre_lines_num)
    for i, peak in enumerate(unused_peaks):
        data_1d += gaussian(np.arange(nx), peak, fwhm, unused_str[i] * flux_scale)

    # add continuum:
    data_1d += gaussian(np.arange(nx), nx // 2, nx // 10, flux_scale / 10)

    # Set the dispersion and some minor distortion

    # add noise to the spectrum
    noise_level = 30
    bias = 5
    data_1d += np.random.normal(bias, scale=noise_level, size=data_1d.shape)

    # Cross correlate the spectrum
    # Find the linear part of the wavelength solution

    mp.plot(data_1d)
    mp.show()
    pass


def test_linear_wavelength_solution():
    # make a random list of lines
    lines = Table({'wavelength': np.random.uniform(low=3500.0, high=5500.0, size=10),
                   'strength': np.random.uniform(low=0.0, high=1.0, size=10)})
    nx = 1001
    input_spectrum = np.zeros(nx)
    min_wavelength = 3200
    dispersion = 2.5
    # Why the coefficients in poly1d are in reverse order from numpy.polynomial.legendre is just beyond me
    input_wavelength_solution = np.poly1d((dispersion, min_wavelength))
    x_pixels = np.arange(nx)
    line_width = 3

    # simulate a spectrum with some fraction of those lines
    for line in lines:
        # And why roots is a property on poly1d objects and a method on numpy.polynomial.legendre. 🤦
        input_spectrum += line['strength'] * gauss(x_pixels, (input_wavelength_solution - line['wavelength']).roots,
                                                   line_width)

    linear_model = linear_wavelength_solution(input_spectrum, 0.01 * np.ones_like(input_spectrum), lines,
                                              dispersion, line_width, np.arange(4000, 5001))
    assert linear_model(0) == min_wavelength
