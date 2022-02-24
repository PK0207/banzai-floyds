from banzai.lco import LCOObservationFrame, LCOFrameFactory, LCOCalibrationFrame
from typing import Optional
from banzai.frames import ObservationFrame
from banzai_floyds.orders import Orders
import numpy as np


class FLOYDSObservationFrame(LCOObservationFrame):
    def __init__(self, hdu_list: list, file_path: str, frame_id: int = None, hdu_order: list = None):
        self.orders = None
        LCOObservationFrame.__init__(self, hdu_list, file_path, frame_id=frame_id, hdu_order=hdu_order)


class FLOYDSCalibrationFrame(LCOCalibrationFrame, FLOYDSObservationFrame):
    def __init__(self, hdu_list: list, file_path: str, frame_id: int = None, grouping_criteria: list = None,
                 hdu_order: list = None):
        LCOCalibrationFrame.__init__(self, hdu_list, file_path,  grouping_criteria=grouping_criteria)
        FLOYDSObservationFrame.__init__(self, hdu_list, file_path, frame_id=frame_id, hdu_order=hdu_order)

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
            models = [np.polynomial.legendre.Legendre(coeff_set, domain=(0, image.data.shape[1] - 1))
                      for coeff_set in coeffs]
            image.orders = Orders(models, image.data.shape, image['ORDER_COEFFS'].meta['WIDTH'])
        return image
