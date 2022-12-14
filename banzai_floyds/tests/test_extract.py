import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai import context
from banzai.data import CCDData
from astropy.io import fits
from banzai_floyds.orders import Orders, order_region
from banzai_floyds.wavelengths import CalibrateWavelengths
from banzai_floyds.utils.wavelength_utils import WavelengthSolution
from banzai_floyds.frames import FLOYDSObservationFrame
from banzai_floyds.extract import Extractor

from banzai_floyds.tests.test_utils import plot_array, upload_fits


def generate_fake_science_frame():
    test_spectra = "/home/jchatelain/git/banzai-floyds/banzai_floyds/tests/data/sample_science/ogg2m001-en06-20221210-0013-e00.fits"
    frame = upload_fits(test_spectra)
    nx = frame.data.shape[1]
    ny = frame.data.shape[0]
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
    wavelengths = WavelengthSolution([wavelength_model1, wavelength_model2], line_widths, line_tilts)

    # save the data, errors, and orders to a floyds frame
    # frame = FLOYDSObservationFrame([CCDData(data, fits.Header({}), uncertainty=errors)], 'foo.fits')

    frame.orders = orders
    frame.wavelengths = wavelengths

    # return the test frame and the input wavelength solution
    return frame


def test_full_extraction_stage():
    np.random.seed(234132)
    input_context = context.Context({})
    frame = generate_fake_science_frame()
    # plot_array(frame.data)
    stage = Extractor(input_context)
    frame = stage.do_stage(frame)
