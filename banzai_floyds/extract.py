from banzai.stages import Stage
import numpy as np
from astropy.table import Table, vstack
from banzai_floyds.matched_filter import maximize_match_filter
from numpy.polynomial.legendre import Legendre
from banzai_floyds.utils.fitting_utils import gauss, fwhm_to_sigma


def profile_gauss_fixed_width(params, x, sigma):
    center, background_level = params
    return gauss(x, center, sigma) + background_level


def background_fixed_profile_center(params, x, center):
    sigma, *coeffs = params
    background = Legendre(coef=coeffs, domain=(np.min(x), np.max(x)))
    return gauss(x, center, sigma) + background(x)

def bins_to_bin_edges(bins):
    bin_edges = bins['center'] - (bins['width'] / 2.0)
    bin_edges = np.append(bin_edges, bins['center'][-1] + (bins['width'][-1] / 2.0))
    return bin_edges

def bin_data(data, uncertainty, wavelengths, orders, wavelength_bins):
    binned_data = None
    x2d, y2d = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    for order_id, order_wavelengths in zip(orders.order_ids, wavelength_bins):
        in_order = orders.data == order_id
        # Throw away the data that is outside the first and last bins
        min_wavelength = wavelength_bins[order_id - 1][0]['center'] -  (wavelength_bins[order_id - 1][0]['width'] / 2.0)
        max_wavelength = wavelength_bins[order_id - 1][-1]['center'] + (wavelength_bins[order_id - 1][-1]['width'] / 2.0)

        in_order = np.logical_and(in_order, wavelengths.data > min_wavelength)
        in_order = np.logical_and(in_order, wavelengths.data < max_wavelength)

        y_order = y2d[in_order] - orders.center(x2d[in_order], order_id)
        data_table = Table({'data': data[in_order], 'uncertainty': uncertainty[in_order], 'wavelength': wavelengths.data[in_order], 
                            'x': x2d[in_order], 'y': y2d[in_order], 'y_order': y_order})
        bin_number = np.digitize(data_table['wavelength'], bins_to_bin_edges(order_wavelengths))
        data_table['wavelength_bin'] = wavelength_bins[order_id - 1]['center'][bin_number - 1]
        data_table['order'] = order_id
        if binned_data is None:
            binned_data = data_table
        else:
            binned_data = vstack([binned_data, data_table])
    return binned_data.group_by(('order', 'wavelength_bin'))


def fit_profile(data, profile_width=4):
    trace_centers = []
    # for each order
    trace_points = Table({'wavelength': [], 'center': [], 'order': []})
    for data_to_fit in data.groups:
        # Pass a match filter (with correct s/n scaling) with a gaussian with a default width
        best_fit_center, _ = maximize_match_filter((data_to_fit['y_order'][np.argmax(data_to_fit['data'])], 0.05), data_to_fit['data'],
                                                    data_to_fit['uncertainty'], profile_gauss_fixed_width, data_to_fit['y_order'],
                                                    args=(fwhm_to_sigma(profile_width),))
        # If the peak pixel of the match filter is > 2 times the median (or something like that) keep the point
        peak = np.argmin(np.abs(data_to_fit['y_order'] - best_fit_center))
        if data_to_fit['data'][peak] / data_to_fit['uncertainty'][peak] > 2.0 * np.median(np.abs(data_to_fit['data'] / data_to_fit['uncertainty'])):
            trace_points = vstack([trace_points, Table({'wavelength': [data_to_fit['wavelength_bin'][0]], 'center': [best_fit_center], 'order': [data_to_fit['order'][0]]})])

    # save the polynomial for the profile
    trace_centers = [Legendre.fit(order_data['wavelength'], order_data['center'], deg=5) 
                     for order_data in trace_points.group_by('order').groups]
    return trace_centers


def fit_background(data, profile_fits, poly_order=4, default_width=4):
    # In principle, this should be some big 2d fit where we fit the profile center, the profile width,
    #   and the background in one go
    profile_width = {'wavelength': [], 'width': [], 'order': []}
    background_fit = Table({'x': [], 'y': [], 'background': []})
    for data_to_fit in data.groups:
        profile = profile_fits[data_to_fit['order'][0] - 1]
        wavelength_bin = data_to_fit['wavelength_bin'][0]
        order_id = data_to_fit['order'][0]

        # Pass a match filter (with correct s/n scaling) with a gaussian with a default width
        initial_coeffs = np.zeros(poly_order + 1)
        initial_coeffs[0] = np.median(data_to_fit['data']) / np.max(data_to_fit['data'])
        # Normalize to the peak of the gaussian
        initial_coeffs[0] /= np.sqrt(2.0 * np.pi) * fwhm_to_sigma(default_width)
        initial_guess = fwhm_to_sigma(default_width),*initial_coeffs
        best_fit_sigma, *best_fit_coeffs = maximize_match_filter(initial_guess, data_to_fit['data'], data_to_fit['uncertainty'],
                                                                 background_fixed_profile_center, data_to_fit['y_order'],
                                                                 args=(profile(wavelength_bin),))
        # If the peak of the profile is 2 > than the peak of the background, keep the profile width
        peak = np.argmin(np.abs(data_to_fit['y']))
        if (data_to_fit['data'][peak] / data_to_fit['uncertainty'][peak] > 2.0) * np.median(np.abs(data_to_fit['data'] / data_to_fit['uncertainty'])):
            profile_width['wavelength'].append(wavelength_bin)
            profile_width['width'].append(best_fit_sigma)
            profile_width['order'].append(order_id)
        # The match filter is insensitive to the normalization, so we do a simply chi^2 fit for the normalization
        # minimize sum(d - norm * poly)^2 / sig^2)
        # norm = sum(d / sig^2) / sum(poly / sig^2)
        normalization = np.sum(data_to_fit['data'] / (data_to_fit['uncertainty'] ** 2.0))
        normalization /= np.sum(background_fixed_profile_center((best_fit_sigma, *best_fit_coeffs), data_to_fit['y_order'], profile(wavelength_bin)) * data_to_fit['uncertainty'] ** -2.0)
        background_polynomial = Legendre(coef=np.array(best_fit_coeffs) * normalization, domain=(np.min(data_to_fit['y_order']), np.max(data_to_fit['y_order'])))
        background_fit = vstack([background_fit, Table({'x': data_to_fit['x'], 'y': data_to_fit['y'], 'background': background_polynomial(data_to_fit['y_order'])})])

    profile_widths = [Legendre.fit(order_data['wavelength'], order_data['width'], deg=5) for order_data in Table(profile_width).group_by('order').groups]
    return background_fit, profile_widths



def get_wavelength_bins(wavelengths):
    """
    Set the wavelength bins to be at the pixel edges along the center of the orders.
    """
    # TODO: in the long run we probably shouldn't bin at all and just do a full 2d sky fit
    #   (including all flux in the order, yikes)
    all_bin_edges = [model(np.arange(min(model.domain) - 0.5,
                                     max(model.domain) + 1)) for model in wavelengths._polynomials]
    return [Table({'center': (bin_edges[1:] + bin_edges[:-1]) / 2.0,
                   'width': bin_edges[1:] - bin_edges[:-1]}) for bin_edges in all_bin_edges]


def extract(data, uncertainty, background, weights, wavelengths, wavelength_bins):
    # Each pixel is the integral of the flux over the full area of the pixel.
    # We want the average at the center of the pixel (where the wavelength is well-defined).
    # Apparently if you integrate over a pixel, the integral and the average are the same,
    #   so we can treat the pixel value as being the average at the center of the pixel to first order.

    results = {'flux': [], 'fluxerror': [], 'wavelength': [], 'binwidth': []}
    for i, lower_edge in enumerate(wavelength_bins[:-1]):
        results['wavelength'].append((wavelength_bins[i + 1] + lower_edge) / 2.0)
        results['binwidth'].append(wavelength_bins[i + 1] - lower_edge)

        pixels_to_bin = np.logical_and(wavelengths >= lower_edge, wavelengths < wavelength_bins[i + 1])
        # This should be equivalent to Horne 1986 optimal extraction
        flux = np.sum(weights[pixels_to_bin] * (data[pixels_to_bin] - background[pixels_to_bin]) * uncertainty[pixels_to_bin]**-2)
        flux_normalization = np.sum(weights[pixels_to_bin]**2 * uncertainty[pixels_to_bin] ** -2)
        results['flux'].append(flux / flux_normalization)
        uncertainty = np.sqrt(np.sum(weights[pixels_to_bin]))
        results['fluxerror'].append(uncertainty / flux_normalization)
    
    return Table(results)


def combine_wavelegnth_bins(wavelength_bins):
    """
    Combine wavelength bins, taking the small delta (higher resolution) bins
    """
    # Find the overlapping bins
    # Assume that the orders are basically contiguous and monotonically increasing
    wavelength_regions = [(min(order_bins['center']), max(order_bins['center'])) for order_bins in wavelength_bins]
    bin_sizes = [np.mean(order_bins['width']) for order_bins in wavelength_bins]

    # Assume the smaller of the bin widths are from the blue order
    # We assume here we only have 2 orders and that one order does not fully encompass the other
    min_wavelength = min(np.array(wavelength_regions).ravel())
    blue_order_index = 0 if min_wavelength in wavelength_regions[0]['center'] else 1
    red_order_index = 0 if blue_order_index else 1

    overlap_end_index = np.min(np.argwhere(wavelength_bins[red_order_index]['center'] >
                                           np.max(wavelength_regions[blue_order_index]['center'])))
    # clean up the middle partial overlaps
    middle_bin_upper = wavelength_bins[red_order_index]['center'][overlap_end_index + 1]
    middle_bin_upper -= wavelength_bins[red_order_index]['width'][overlap_end_index] / 2.0
    middle_bin_lower = wavelength_bins[blue_order_index]['center'][-1] + wavelength_bins[blue_order_index]['width'] / 2.0
    middle_bin_center = (middle_bin_upper + middle_bin_lower) / 2.0
    middle_bin_width = middle_bin_upper - middle_bin_lower
    overlap_end_index += 1
    new_bins = {'center': np.hstack([wavelength_bins[blue_order_index]['center'], [middle_bin_center],
                                     wavelength_bins[red_order_index][overlap_end_index:]['center']]),
                'width': np.hstack([wavelength_bins[blue_order_index]['center'],
                                    [middle_bin_width],
                                    wavelength_bins[red_order_index][overlap_end_index:]['center']])}
    return Table(new_bins)


class Extractor(Stage):
    def do_stage(self, image):
        image.wavelength_bins = get_wavelength_bins(image.wavelengths)
        image.binned_data = bin_data(image.data, image.uncertainty, image.wavelengths, image.orders, image.wavelength_bins)
        profile_centers = fit_profile(image.binned_data)
        background, profile_widths = fit_background(image.binned_data, profile_centers)
        image.background = background
        image.profiles = profile_centers, profile_widths
        extracted = []
        for i in range(len(image.orders.centers)):
            in_order = image.orders.data == i + 1
            extracted.append(extract(image.data[in_order], image.uncertainty[in_order], image.background[in_order],
                                     image.weights[in_order], image.wavelengths[in_order], image.wavelength_bins[i]))
            extracted[i]['order'] = i + 1
        image.extracted = vstack(extracted)
        # TODO: Stitching together the orders is going to require flux calibration and probably 
        # a scaling due to aperture corrections

        return image
