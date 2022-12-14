import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval
from astropy.io import fits
from banzai_floyds.frames import FLOYDSObservationFrame


def plot_array(data, overlays=None):
    if len(data) == 2:
        plt.plot(data[0], data[1])
    elif len(data.shape) > 1:
        z_interval = ZScaleInterval().get_limits(data)
        plt.imshow(data, cmap='gray', vmin=z_interval[0], vmax=z_interval[1])
        # plt.imshow(data, cmap='gray', vmin=0, vmax=1)
    else:
        plt.plot(data)
    if overlays:
        for overlay in overlays:
            plt.plot(overlay[0], overlay[1], color="green")
    plt.show()


def make_fits(data, header=None, filename="test.fits"):
    hdu = fits.PrimaryHDU(data, header)
    hdu.writeto(filename)


def upload_fits(spectra):
    """Requires unpacked FLOYDS fits file"""
    try:
        hdul = fits.open(spectra)
    except FileNotFoundError:
        print("Cannot find file {}".format(spectra))
        return None

    hdul[0].meta = hdul[0].header
    hdul[0].uncertainty = np.zeros_like(hdul[0].data)
    frame = FLOYDSObservationFrame(hdul, spectra)
    return frame
