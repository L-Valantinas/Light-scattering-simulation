import numpy as np
import calc
from main import propagate

def angular_memory_effect_analysis(n: np.ndarray, k_0: np.float, sample_pitch, input_field: np.ndarray):
    """
    """
    max_tilt_coef = 20
    tilt_coef_range = np.arange(-max_tilt_coef, max_tilt_coef).ravel()
    output_storage = np.zeros([tilt_coef_range.size, n.shape[1]], dtype = np.complex)

    for idx, coefficient in enumerate(tilt_coef_range):

        phase_shift = np.exp(coefficient * 2j * np.pi * np.linspace(-0.5, 0.5, n.shape[1])) # tilt phaseshift
        tilted_field = input_field * phase_shift # tilting the input field
        output_field = propagate(n, k_0, sample_pitch, tilted_field) # propagating the tilted field

        output_storage[idx, :] = output_field.ravel()

    return output_storage