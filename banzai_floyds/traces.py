from banzai.calibrations import CalibrationUser
from banzai.stages import Stage


class InitTraces(CalibrationUser):
    @property
    def calibration_type(self):
        pass

    def apply_master_calibration(self, image, master_calibration_image):
        # If a previous trace exists, load it.
        # Otherwise try a blind solve
        # Take a vertical slice down about the middle of the chip
        # Find the two biggest peaks in summing the signal to noise
        # This is effectively a match filter with a top hat kernel
        # Initialize the guess to be 90 pixels high at the center of the match filter peak
        # Do an initial fit to get the curvature of the slit
        pass


class RefineTraces(Stage):
    def do_stage(self, image):
        # Using the supplied initial guess
        # Use an edge detection filter (Canny?) to get the slit edge nearest our original slit edge guess
        # Average the slit width across the high signal to noise part of the flat
        # Using that, maximize the signal to noise inside the slit region by fitting a legendre polynomial
        # Note that we always evaluate the signal to noise in integer pixel regions
        return image
