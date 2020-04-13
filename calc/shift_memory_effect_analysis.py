import numpy as np
import matplotlib.pyplot as plt
import calc


def shift_memory_effect_analysis(max_shift, input_field: np.ndarray, TM: np.ndarray, sample_pitch,  plot_std = True):
    """
    Shifting multiple outputs and computing their std
    """
    shift_range = np.arange(max_shift)
    shifted_outputs = np.zeros([max_shift, input_field.size])

    #If the transmission matrix is given, the outputs are comupted with it, instead of BPM
    for idx in shift_range:
        shifted_input = np.roll(input_field, idx).ravel() # shifts the input wavefront with cyclic boudary conditions
        shifted_outputs[idx, :] = np.roll((np.abs(TM @ shifted_input)**2).ravel(), -idx) # saves the intensity of the output. Shifts it back to the center
    shifted_outputs = shifted_outputs / np.linalg.norm(shifted_outputs, 1) #normalising all outputs


    #Comparing the outputs with a standard deviation method
    subtracted_outputs = np.zeros(shifted_outputs.shape)
    for idx in shift_range:
        subtracted_outputs[idx] = shifted_outputs[idx] - shifted_outputs[0] # subtracting the outputs with the reference
    std_of_outputs = np.std(subtracted_outputs, axis = 1) # Computing the standard deviation of subtracted outputs
    if plot_std:
        fig, axs = plt.subplots()
        axs.plot(shift_range * sample_pitch[1] * 1e6, std_of_outputs)
        axs.set_title('Shift effect standard deviation')
        axs.set(xlabel = '$\mu m$')
        plt.plot(block = False)


    return shifted_outputs