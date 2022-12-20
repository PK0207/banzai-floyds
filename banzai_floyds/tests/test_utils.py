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
