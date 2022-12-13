import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai import context
from banzai.data import CCDData
from astropy.io import fits
from banzai_floyds.orders import Orders, order_region
from banzai_floyds.wavelengths import CalibrateWavelengths
from banzai_floyds.utils.wavelength_utils import tilt_coordinates
from banzai_floyds.frames import FLOYDSObservationFrame
from banzai_floyds.extract import Extractor

from banzai_floyds.tests.test_utils import plot_array


def generate_fake_science_frame():
    nx = 2048
    ny = 512
    order_height = 93
    order1 = Legendre((135.4,  81.8,  45.2, -11.4), domain=(0, 1700))
    order2 = Legendre((410, 17, 63, -12), domain=(475, 1975))
    data = np.zeros((ny, nx))
    errors = np.zeros_like(data)
    orders = Orders([order1, order2], (ny, nx), order_height)

    # # make a reasonable wavelength model
    wavelength_model1 = Legendre((7487.2, 2662.3, 20., -5., 1.), domain=(0, 1700))
    wavelength_model2 = Legendre((4573.5, 1294.6, 15.), domain=(475, 1975))
    line_widths = [CalibrateWavelengths.INITIAL_LINE_WIDTHS[i] for i in range(1, 3)]
    line_tilts = [CalibrateWavelengths.INITIAL_LINE_TILTS[i] for i in range(1, 3)]
    dispersions = [CalibrateWavelengths.INITIAL_DISPERSIONS[i] for i in range(1, 3)]
    flux_scale = 80000.0
    read_noise = 7.0

    # Calculate the tilted coordinates
    x2d, y2d = np.meshgrid(np.arange(nx), np.arange(ny))
    for order_center, wavelength_model, tilt, line_width, dispersion in \
            zip((order1, order2),
                (wavelength_model1, wavelength_model2),
                line_tilts,
                line_widths,
                dispersions):
        input_order_region = order_region(order_height, order_center, (ny, nx))
        tilted_x = tilt_coordinates(tilt, x2d[input_order_region],
                                    y2d[input_order_region] - order_center(x2d[input_order_region]))

        # Fill in both used and unused lines that have strengths, setting a reasonable signal to noise
        # lines = arc_lines.used_lines + arc_lines.unused_lines
        # for line in lines:
        #     if line['line_strength'] == 'nan':
        #         continue
        #     wavelengths = wavelength_model(tilted_x)
        #     line_sigma = fwhm_to_sigma(line_width)
        #     data[input_order_region] += line['line_strength'] * gauss(wavelengths,
        #                                                               line['wavelength'], line_sigma) * flux_scale
    # Add poisson noise
    errors += np.sqrt(data)
    data = np.random.poisson(data).astype(float)

    # Add read noise
    errors = np.sqrt(errors * errors + read_noise)
    data += np.random.normal(0.0, read_noise, size=(ny, nx))
    plot_array(data)
    # save the data, errors, and orders to a floyds frame
    frame = FLOYDSObservationFrame([CCDData(data, fits.Header({}), uncertainty=errors)], 'foo.fits')
    frame.orders = orders
    # return the test frame and the input wavelength solution
    return frame


def test_full_extraction_stage():
    np.random.seed(234132)
    input_context = context.Context({})
    frame = generate_fake_science_frame()
    # stage = Extractor(input_context)
    # frame = stage.do_stage(frame)
