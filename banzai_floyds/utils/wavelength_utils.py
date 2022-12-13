from astropy.io import fits
from numpy.polynomial.legendre import Legendre
import numpy as np


class WavelengthSolution:
    def __init__(self, polymomials, line_widths, line_tilts):
        self._line_widths = line_widths
        self._polynomials = polymomials
        self._line_tilts = line_tilts

    def data(self, orders):
        model_wavelengths = np.zeros_like(orders).astype(float)
        # Recall that numpy arrays are indexed y,x
        x2d, y2d = np.meshgrid(np.arange(orders.shape[1]), np.arange(orders.shape[0]))

        order_ids = np.unique(orders)
        order_ids = order_ids[order_ids != 0]
        for i, order in enumerate(order_ids):
            tilted_x = x2d + np.tan(np.deg2rad(self._line_tilts[i])) * y2d
            model_wavelengths[orders == order] += self._polynomials[i](tilted_x[orders == order])
        return model_wavelengths

    def to_header(self):
        header = fits.Header()
        for i, (polynomial, width, tilt) in enumerate(zip(self._polynomials, self._line_widths, self._line_tilts)):
            header[f'LINWIDE{i + 1}'] = width
            header[f'LINTILT{i + 1}'] = tilt
            header[f'POLYORD{i + 1}'] = polynomial.degree()
            header[f'POLYDOM{i + 1}'] = str(list(polynomial.domain))
            for j, coef in enumerate(polynomial.coef):
                header[f'COEF{i + 1}_{j}'] = coef
        return header

    @property
    def coefficients(self):
        return [polynomial.coef for polynomial in self._polynomials]

    @property
    def line_widths(self):
        return self._line_widths

    @property
    def line_tilts(self):
        return self._line_tilts

    @classmethod
    def from_header(cls, header):
        orders = np.arange(1, len([x for x in header.keys() if 'POLYORD' in x]) + 1)
        line_widths = []
        line_tilts = []
        polynomials = []
        for order in orders:
            line_tilts.append(header[f'LINTILT{order}'])
            line_widths.append(header[f'LINWIDE{order}'])
            polynomials.append(Legendre((float(header[f'COEF{order}_{i}'])
                                         for i in range(int(header[f'POLYORD{order}']) + 1)),
                               domain=eval(header[f'POLYDOM{order}'])))
        return cls(polynomials, line_widths, line_tilts)


def tilt_coordinates(tilt_angle, x, y):
    r"""
    Find the x coordinate of a pixel as if it was along the order center to use for the wavelength solution

    Parameters
    ----------
    tilt_angle: float angle in degrees counterclockwise to tilt the lines
    x: x pixel coordinates
    y: y pixel coordinates
    center: function to calculate the order center as a function of x (in pixels)

    Returns
    -------
    tilted_coordinates: array of the same shape as x and y

    Notes
    -----
    This is effectively finding the x intercept, given a slope that is based on the tilt angle, and x0, y0 being a point
    on the line.
    \    |
     \   |
      \  |
       \ϴ|
        |ϴ\
        |  \
        |   \
        |    \

    x_tilt = -b / m
    b = (y0 - m x0)
    x_tilt = -(y0 - m x0) / m
    x_tilt = x0 - y0/m
    m = -cot(ϴ)
    x_tilt = x0 - y0 * tan(ϴ)
    """

    return x + y * np.tan(np.deg2rad(tilt_angle))
