import numpy as np
import calc


def center_tilted_output(shifted_output_field, tilt_coef, data_shape, sample_pitch, wavelength, return_shift_idx =False):
    """
    Shifts back the output field focal spot back to the center after the input wavefront was tilted
    """
    angle = calc.wavefront_tilt_angle(tilt_coef, data_shape, sample_pitch, wavelength) # angle of tilt
    output_shift = int(data_shape[0] * np.tan(angle)) # output shift in pixels
    centered_output = np.roll(shifted_output_field, -output_shift) # cyclic shift of the output field, so that the focus is at the center

    if return_shift_idx == False:
    	return centered_output
    else:
    	return centered_output, output_shift