from banzai.stages import Stage
import numpy as np
from astropy.table import Table


def fit_profile(data, uncertainty, orders):
    # for each order
    # Pass a match filter (with correct s/n scaling) with a gaussian with a default width, in steps of 0.05 pixels
    # If the peak pixel of the match filter is > 2 times the median (or something like that) keep the point
    # fit a polynomial to the points that make the cut to get an estimate of the trace, use the match filter metric instead of chi^2
    # save the polynomial for the profile
    pass

def fit_background(data, uncertainty, profile_fits, orders, wavelength_bins):
    # For each order
    # for each wavelength_bin
    # fit a polynomial and a gaussian with a fixed position letting the width and height vary
    # use the line widths and profile centers to make a 2d profile map
    # Take the polynomial evaluated at every pixel in the order to save in the background file
    pass


def get_wavelength_bins(wavelengths):
    """
    Set the wavelength bins to be at the pixel edges along the center of the orders.
    """
    # TODO: in the long run we probably shouldn't bin at all and just do a full 2d sky fit (including all flux in the order, yikes)

    return [model(np.arange(min(model.domain) - 0.5, max(model.domain) + 1)) for model in wavelengths._polynomials]


def extract(data, uncertainty, background, weights, wavelengths, wavelength_bins):
    # Each pixel is the integral of the flux over the full area of the pixel
    # we want the average at the center of the pixel (where the wavelength is well defined)
    # Apparently if you integrate over a plane, the integral and the average are the same, so we can treat the pixel value as being the average
    # at the center of the pixel to first order
    
    results = {'flux': [], 'fluxerror': [], 'wavelength': [], 'binwidth': []}
    for i, lower_edge in enumerate(wavelength_bins[:-1]):
        results['wavelength'].append((wavelength_bins[i + 1] + lower_edge) / 2.0)
        results['binwidth'].append(wavelength_bins[i + 1] - lower_edge)

        pixels_to_bin = np.logical_and(wavelengths >= lower_edge, wavelengths < wavelength_bins[i + 1])
        # This should be equivilent to Horne 1986 optimal extraction
        flux = np.sum(weights[pixels_to_bin] * data[pixels_to_bin] * uncertainty[pixels_to_bin]**-2)
        flux_normalization = np.sum(weights[pixels_to_bin] * uncertainty[pixels_to_bin] ** -2)
        results['flux'].append(flux / flux_normalization)
        uncertainty = np.sqrt(np.sum(weights[pixels_to_bin] ** 2 * uncertainty[pixels_to_bin] ** -2))
        results['fluxerror'].append( uncertainty / flux_normalization)
    
    return Table(results)

def combine_wavelegnth_bins(wavelength_bins):
    """
    Combine wavelength bins, taking the small delta (higher resolution) bins
    """
    # Find the overlapping bins
    # Assume that the orders are basically contiguous and monotonically increasing
    wavelength_regions = [(min(order_bins), max(order_bins)) for order_bins in wavelength_bins]
    bin_sizes = [np.mean(order_bins[1:] - order_bins[:-1]) for order_bins in wavelength_bins]

    # Assume the smaller of the bin widths are from the blue order
    # We assume here we only have 2 orders and that one order does not fully encompass the other
    min_wavelength = min(np.array(wavelength_regions).ravel())
    blue_order_index = 0 if min_wavelength in wavelength_regions[0] else 1
    red_order_index = 0 if blue_order_index else 1

    overlap_end_index = np.min(np.argwhere(wavelength_bins[red_order_index] > max(wavelength_regions[blue_order_index])))
    # clean up the middle partial overlaps
    if wavelength_bins[red_order_index][overlap_end_index] - wavelength_bins[blue_order_index][-1] < bin_sizes[blue_order_index]:
        overlap_end_index += 1
    return np.hstack([wavelength_bins[blue_order_index], wavelength_bins[red_order_index][overlap_end_index:]])


class Extractor(Stage):
    def do_stage(self, image):
        image.wavelength_bins = get_wavelength_bins(image.wavelengths)
        image.profile, image.background = fit_profile_and_background(image.data, image.uncertainty, image.orders)
        image.extracted = extract(image.data, image.uncertainty, image.background, image.weights, image.wavelegnths, image.wavelength_bins)
        image.spectrum = combine_orders(image.extracted)
        return image
