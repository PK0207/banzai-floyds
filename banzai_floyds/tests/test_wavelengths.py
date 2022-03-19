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

    # Set the dispersion and some minor distortion

    # add noise to the spectrum
    noise_level = 30
    bias = 5
    data_1d += np.random.normal(bias, scale=noise_level, size=data_1d.shape)

    # Cross correlate the spectrum
    # Find the linear part of the wavelength solution

    print(data_1d)
    mp.plot(data_1d)
    mp.show()
    pass
