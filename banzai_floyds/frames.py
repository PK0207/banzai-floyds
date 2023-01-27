from banzai.lco import LCOObservationFrame, LCOFrameFactory, LCOCalibrationFrame
from typing import Optional
from banzai.frames import ObservationFrame
from banzai.data import DataProduct, HeaderOnly, ArrayData
from banzai_floyds.orders import Orders
from banzai_floyds.utils.wavelength_utils import WavelengthSolution
import numpy as np
from banzai_floyds.utils.fitting_utils import gauss
import os
from astropy.io import fits


class FLOYDSObservationFrame(LCOObservationFrame):
    def __init__(self, hdu_list: list, file_path: str, frame_id: int = None, hdu_order: list = None):
        self.orders = None
        self.wavelengths = None
        self._profile_fits = None
        self._background_fits = None
        self.wavelegnth_bins = None
        self.binned_data = None
        LCOObservationFrame.__init__(self, hdu_list, file_path, frame_id=frame_id, hdu_order=hdu_order)

    def get_1d_and_2d_spectra_products(self, runtime_context):
        filename_1d = self.get_output_filename(runtime_context).replace('.fits', '-1d.fits')
        frame_1d = LCOObservationFrame([HeaderOnly(self.meta.copy()), self['SPECTRUM1D']],
                                       os.path.join(self.get_output_directory(runtime_context), filename_1d))
        fits_1d = frame_1d.to_fits(runtime_context)
        fits_1d['SPECTRUM1D'].name = 'SPECTRUM'
        filename_2d = filename_1d.replace('-1d.fits', '-2d.fits')

        fits_1d[0].header['L1ID2D'] = filename_2d
        output_product_1d = DataProduct.from_fits(fits_1d, filename_1d, self.get_output_directory(runtime_context))

        # TODO consider saving the background coeffs or the profile coeffs?
        frame_2d = LCOObservationFrame([hdu for hdu in self._hdus if hdu.name not in ['SPECTRUM1D', 'CCF']],
                                       os.path.join(self.get_output_directory(runtime_context), filename_2d))
        fits_2d = frame_2d.to_fits(runtime_context)
        fits_2d[0].header['L1ID1D'] = filename_1d
        output_product_2d = DataProduct.from_fits(fits_2d, filename_2d, self.get_output_directory(runtime_context))
        return output_product_1d, output_product_2d

    def get_output_data_products(self, runtime_context):
        if self.obstype != 'SPECTRUM':
            return super().get_output_data_products(runtime_context)
        else:
            return self.get_1d_and_2d_spectra_products()

    @property
    def profile(self):
        return self._hdus['PROFILE'].data

    @profile.setter
    def profile(self, value):
        self._profile_fits = value
        profile_centers, profile_widths = value
        profile_data = np.zeros(self.orders.shape)
        x2d, y2d = np.meshgrid(np.arange(profile_data.shape[1])), np.arange(profile_data.shape[0])

        for order, order_wavelengths, profile_center, profile_width in zip(self.orders, self.wavelength_bins,
                                                                           profile_centers, profile_widths):
            # TODO: This needs to refactored into a function that is used in multiple places
            in_order = order.data == order.value
            y = (y2d - order.center(x2d))[in_order]
            for wavelength_bin in order_wavelengths:
                center = profile_center(wavelength_bin.center)
                sigma = profile_width(wavelength_bin.center)
                # We should probably cache this calculation?
<<<<<<< HEAD
                wavelength_inds = np.logical_and(self.wavelengths[in_order] <= (wavelength_bin.center + wavelength_bin.width / 2.0), 
                                                 self.wavelengths[in_order] >= (wavelength_bin.center - wavelength_bin.width / 2.0))
                # TODO: Make sure this is normalzied correctly
=======
                wavelength_inds = np.logical_and(self.wavelengths[in_order] <=
                                                 (wavelength_bin.center + wavelength_bin.width / 2.0),
                                                 self.wavelengths[in_order] >=
                                                 (wavelength_bin.center - wavelength_bin.width / 2.0))
>>>>>>> 178a9c05a739b920a173ef928332dc6c8ff73736
                profile_data[in_order][wavelength_inds] = gauss(y[wavelength_inds], center, sigma)
        self._hdus['PROFILE'] = ArrayData(profile_data, meta=fits.Header({}))

    @property
    def background(self):
        return self._hdus['BACKGROUND'].data()

    @profile.setter
    def background(self, value):
        self._background_fits = value
        background_data = np.zeros(self.data.shape)
        x2d, y2d = np.meshgrid(np.arange(background_data.shape[1])), np.arange(background_data.shape[0])

        for order, order_wavelengths, background_fit in zip(self.orders, self.wavelength_bins, self._background_fits):
            # TODO: This needs to refactored into a function that is used in multiple places
            in_order = order.data == order.value
            y = (y2d - order.center(x2d))[in_order]
            for wavelength_bin, background_model in zip(order_wavelengths, background_fit):
                # We should probably cache this calculation?
                wavelength_inds = np.logical_and(self.wavelengths[in_order] <=
                                                 (wavelength_bin.center + wavelength_bin.width / 2.0),
                                                 self.wavelengths[in_order] >=
                                                 (wavelength_bin.center - wavelength_bin.width / 2.0))
                background_data[in_order][wavelength_inds] = background_model(y[wavelength_inds])
        self._hdus['BACKGROUND'] = ArrayData(background_data, meta=fits.Header({}))


class FLOYDSCalibrationFrame(LCOCalibrationFrame, FLOYDSObservationFrame):
    def __init__(self, hdu_list: list, file_path: str, frame_id: int = None, grouping_criteria: list = None,
                 hdu_order: list = None):
        LCOCalibrationFrame.__init__(self, hdu_list, file_path,  grouping_criteria=grouping_criteria)
        FLOYDSObservationFrame.__init__(self, hdu_list, file_path, frame_id=frame_id, hdu_order=hdu_order)
        self.wavelengths = None

    def write(self, runtime_context):
        LCOCalibrationFrame.write(self, runtime_context)


class FLOYDSFrameFactory(LCOFrameFactory):
    @property
    def observation_frame_class(self):
        return FLOYDSObservationFrame

    @property
    def calibration_frame_class(self):
        return FLOYDSCalibrationFrame

    @staticmethod
    def is_empty_coordinate(coordinate):
        return 'nan' in str(coordinate).lower() or 'n/a' in str(coordinate).lower()

    def open(self, path, runtime_context) -> Optional[ObservationFrame]:
        image = super().open(path, runtime_context)

        # Set a default BIASSEC and TRIMSEC if they are unknown
        if image.meta.get('BIASSEC', 'UNKNOWN').lower() in ['unknown', 'n/a']:
            image.meta['BIASSEC'] = '[2049:2079,1:512]'
        if image.meta.get('TRIMSEC', 'UNKNOWN').lower() in ['unknown', 'n/a']:
            image.meta['TRIMSEC'] = '[1:2048,1:512]'
        # Load the orders if they exist
        if 'ORDER_COEFFS' in image:
            polynomial_order = image['ORDER_COEFFS'].meta['POLYORD']
            coeffs = [np.array([row[f'c{i}'] for i in range(polynomial_order + 1)])
                      for row in image['ORDER_COEFFS'].data]
            domains = [(row['domainmin'], row['domainmax']) for row in image['ORDER_COEFFS'].data]
            models = [np.polynomial.legendre.Legendre(coeff_set, domain=domain)
                      for coeff_set, domain in zip(coeffs, domains)]
            image.orders = Orders(models, image.data.shape, image['ORDER_COEFFS'].meta['HEIGHT'])
        if 'WAVELENGTHS' in image:
            image.wavelengths = WavelengthSolution.from_header(image['WAVELENGTHS'].meta)
        return image
