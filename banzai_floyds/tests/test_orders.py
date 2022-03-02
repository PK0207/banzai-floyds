from banzai_floyds.orders import estimate_order_centers, order_region, fit_order_curve, OrderSolver
import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai.tests.utils import FakeContext
from banzai_floyds.frames import FLOYDSObservationFrame
from banzai.data import CCDData
from astropy.io import fits


def test_blind_center_search():
    data = np.zeros((512, 101))
    error = np.ones((512, 101))
    # Put a trace center at 91 and 355
    order_centers = [91, 355]
    order_height = 85
    for center in order_centers:
        order_slice = order_region(order_height, Legendre([center], domain=(0, data.shape[1] - 1)),
                                   data.shape)
        data[order_slice] = 1000.0
        error[order_slice] = 100.0

    found_centers = estimate_order_centers(data, error, order_height)
    for i in order_centers:
        assert i in found_centers


def test_fit_orders():
    data = np.zeros((512, 501))
    error = np.ones((512, 501))
    order_center = 151
    input_center_params = [order_center, 10, 20]
    order_height = 85
    input_order_region = order_region(order_height, Legendre(input_center_params, domain=(0, data.shape[1] - 1)),
                                      data.shape)
    data[input_order_region] = 1000.0
    error[input_order_region] = 100.0

    # Give an initial guess of the center, but zero curvature to make sure we converge
    fitted_order_model = fit_order_curve(data, error, order_height, [order_center, 0, 0])
    fitted_order_region = order_region(order_height, fitted_order_model, data.shape)
    assert (input_order_region == fitted_order_region).all()


def test_order_solver_stage():
    np.random.seed(19231442)
    data = np.zeros((516, 503))
    error = np.ones((516, 503))
    order_centers = 145, 371
    input_center_params = [[order_centers[0], 15, 15], [order_centers[1], 17, 12]]
    order_height = 87
    read_noise = 5  # everything is gain = 1
    data += np.random.normal(0.0, scale=read_noise, size=data.shape)
    input_order_regions = [order_region(order_height, Legendre(params, domain=(0, data.shape[1] - 1)),
                                        data.shape) for params in input_center_params]

    for region in input_order_regions:
        data[region] += np.random.poisson(500.0, size=data.shape)[region]
    error = np.sqrt(read_noise ** 2.0 + np.abs(data))

    order_solver = OrderSolver(FakeContext())
    order_solver.ORDER_HEIGHT = order_height
    image = FLOYDSObservationFrame([CCDData(data=data, uncertainty=error, meta=fits.Header({}))], 'foo.fits')
    image = order_solver.do_stage(image)

    for i, input_region in enumerate(input_order_regions):
        # Make sure less than a hundred pixels deviate from the input. This corresponds to a one pixel offset for less
        # than 10% of the width of the order
        assert (input_region != (image.orders.data == (i + 1))).sum() < 100
