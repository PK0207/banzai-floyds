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


def fit_profile(data, uncertainty, wavelengths, orders, wavelength_bins, profile_width=4):
    x2d, y2d = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

    trace_centers = []
    # for each order
    order_ids = orders.order_ids
    for order_id, order_wavelengths in zip(order_ids, wavelength_bins):
        in_order = orders.data == order_id
        y = y2d[in_order] - orders.center(x2d[in_order], order_id)
        data_table = Table({'data': data[in_order], 'uncertainty': uncertainty[in_order],
                            'wavelength': wavelengths.data[in_order], 'x': x2d[in_order], 'y': y})
        bin_edges = order_wavelengths['center'] - (order_wavelengths['width'] / 2.0)
        bin_edges = np.append(bin_edges, order_wavelengths['center'][-1] + (order_wavelengths['width'][-1] / 2.0))

        data_table['wavelength_bin'] = np.digitize(data_table['wavelength'], bin_edges)
        data_table = data_table.group_by('wavelength_bin')
        trace_points = {'wavelength': [], 'center': []}
        for wavelength, data_to_fit in zip(order_wavelengths, data_table.groups):
            # Pass a match filter (with correct s/n scaling) with a gaussian with a default width
            best_fit_center, _ = maximize_match_filter((data_to_fit['y'][np.argmax(data_to_fit['data'])], 0.05),
                                                       data_to_fit['data'], data_to_fit['uncertainty'],
                                                       profile_gauss_fixed_width, data_to_fit['y'],
                                                       args=(fwhm_to_sigma(profile_width),))
            # If the peak pixel of the match filter is > 2 times the median (or something like that) keep the point
            peak = np.argmin(np.abs(data_to_fit['y'] - best_fit_center))
            if data_to_fit['data'][peak] / data_to_fit['uncertainty'][peak] > 2.0 * \
                    np.median(np.abs(data_to_fit['data'] / data_to_fit['uncertainty'])):
                trace_points['wavelength'].append(wavelength['center'])
                trace_points['center'].append(best_fit_center)
        # fit a polynomial to the points that make the cut to get an estimate of the trace, use the match filter
        # metric instead of chi^2
        # save the polynomial for the profile
        trace_centers.append(Legendre.fit(trace_points['wavelength'], trace_points['center'], deg=5))
    return trace_centers


def fit_background(data, uncertainty, wavelengths, profile_fits, orders, wavelength_bins, poly_order=4,
                   default_width=4):
    # In principle, this should be some big 2d fit where we fit the profile center, the profile width,
    #   and the background in one go
    x2d, y2d = np.meshgrid(np.arange(data.shape[1])), np.arange(data.shape[0])

    background_fits = []
    profile_widths = []
    # for each order
    for order, order_wavlengths, profile in zip(orders, wavelength_bins, profile_fits):
        in_order = order.data == order.value
        y = (y2d - order.center(x2d))[in_order]
        profile_width = {'wavelength': [], 'width': []}
        background_fit = {'wavelength': [], 'coeffs': []}
        for wavelength_bin in order_wavlengths:
            # We should probably cache this calculation?
            wavelength_inds = np.logical_and(wavelengths[in_order] <= (wavelength_bin['center']
                                                                       + wavelength_bin['width'] / 2.0),
                                             wavelengths[in_order] >= (wavelength_bin['center']
                                                                       - wavelength_bin['width'] / 2.0))
            data_to_fit = data[in_order][wavelength_inds]
            error_to_fit = uncertainty[in_order][wavelength_inds]
            y_to_fit = y[wavelength_inds]

            # Pass a match filter (with correct s/n scaling) with a gaussian with a default width
            initial_guess = fwhm_to_sigma(default_width), *np.zeros(poly_order + 1)
            best_fit_sigma, *best_fit_coeffs = maximize_match_filter(initial_guess, data_to_fit, error_to_fit,
                                                                     background_fixed_profile_center, y_to_fit,
                                                                     args=(profile(wavelength_bin['center'])))
            # If the peak of the profile is 2 > than the peak of the background, keep the profile width
            peak = np.argmin(np.abs(y_to_fit - best_fit_sigma))
            if data_to_fit[peak] / error_to_fit[peak] > 2.0 * np.median(np.abs(data_to_fit / error_to_fit)):
                profile_width['wavelength'].append(wavelength_bin['center'])
                profile_width['width'].append(best_fit_sigma)
            background_fit['wavelength'].append(wavelength_bin['center'])
            # The match filter is insensitive to the normalization, so we do a simply chi^2 fit for the normalization
            # minimize sum(d - norm * poly)^2 / sig^2)
            # norm = sum(d / sig^2) / sum(poly / sig^2)
            normalization = np.sum(data_to_fit / error_to_fit / error_to_fit)
            normalization /= np.sum(background_fixed_profile_center((best_fit_sigma, *best_fit_coeffs), y_to_fit,
                                                                    profile(wavelength_bin['center']))
                                    / error_to_fit / error_to_fit)
            background_fit['coeffs'].append(best_fit_coeffs * normalization)
        background_fits.append(background_fit)
        # fit a polynomial to the points that make the cut to get an estimate of the trace, use the match filter metric
        #   instead of chi^2
        # save the polynomial for the profile
        profile_widths.append(Legendre.fit(profile_widths['wavelengths'], profile_widths['width'], deg=5))
    return background_fits, profile_widths


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
    # Apparently if you integrate over a plane, the integral and the average are the same,
    #   so we can treat the pixel value as being the average at the center of the pixel to first order.

    results = {'flux': [], 'fluxerror': [], 'wavelength': [], 'binwidth': []}
    for i, lower_edge in enumerate(wavelength_bins[:-1]):
        results['wavelength'].append((wavelength_bins[i + 1] + lower_edge) / 2.0)
        results['binwidth'].append(wavelength_bins[i + 1] - lower_edge)

        pixels_to_bin = np.logical_and(wavelengths >= lower_edge, wavelengths < wavelength_bins[i + 1])
        # This should be equivalent to Horne 1986 optimal extraction
        flux = np.sum(weights[pixels_to_bin] * data[pixels_to_bin] * uncertainty[pixels_to_bin]**-2)
        flux_normalization = np.sum(weights[pixels_to_bin] * uncertainty[pixels_to_bin] ** -2)
        results['flux'].append((flux - background) / flux_normalization)
        uncertainty = np.sqrt(np.sum(weights[pixels_to_bin] ** 2 * uncertainty[pixels_to_bin] ** -2))
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
        profile_centers = fit_profile(image.data, image.uncertainty, image.wavelengths, image.orders,
                                      image.wavelength_bins)
        background, profile_widths = fit_background(image.data, image.uncertainty, image.wavelengths, profile_centers,
                                                    image.orders, image.wavelength_bins)
        image.background = background
        image.profiles = profile_centers, profile_widths
        extracted = []
        for i in range(len(image.orders.centers)):
            in_order = image.orders.data == i + 1
            extracted.append(extract(image.data[in_order], image.uncertainty[in_order], image.background[in_order],
                                     image.weights[in_order], image.wavelengths[in_order], image.wavelength_bins[i]))
            extracted[i]['order'] = i + 1
        image.extracted = vstack(extracted)
        image.spectrum = extract(image.data, image.uncertainty, image.background, image.weights, image.wavelengths,
                                 combine_wavelegnth_bins(image.wavelength_bins))
        return image
