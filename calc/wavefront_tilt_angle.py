import numpy as np
import calc



def wavefront_tilt_angle(tilt_coef, data_shape, sample_pitch, wavelength):
    """
    Calculates the amount of tilt that np.exp(tilt_coef * 2j * np.pi * np.linspace(0, 1, data_shape[1])) phase shift would produce.
    """
    wavelength_in_pixels = wavelength / sample_pitch[1] # Calculates how many pixels fit in one wavelength

    x_size = data_shape[1] * sample_pitch[1] # data length in x direction
    max_phase_shift = tilt_coef * wavelength_in_pixels * sample_pitch[0] # phase shift length in z direction
    angle = np.arctan(max_phase_shift / x_size) 

    return angle