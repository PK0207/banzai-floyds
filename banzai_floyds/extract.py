# from banzai_nres.utils import trace_utils
from banzai_floyds.orders import order_region
from numpy.polynomial.legendre import Legendre
import numpy as np


def unweighted_extract(data, error, orders, extraction_height=7):
    order_coefficients = orders.coeffs
    order_domains = orders.domains
    spectra = []
    spectra_errors = []
    # for coefficients, domain in zip(order_coefficients, order_domains):
    #     region = order_region(extraction_height, Legendre(coefficients, domain=domain), data.shape)
    #     # 2-D slice
    #     region_slice = trace_utils.get_trace_region(region)
    #     spectrum = data[region_slice].sum(axis=1)
    #     spectrum_error = np.sqrt((error[region_slice] * error[region_slice]).sum(axis=1))
    #     spectra.append(spectrum)
    #     spectra_errors.append(spectrum_error)
    return spectra, spectra_errors
