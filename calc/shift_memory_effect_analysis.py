import numpy as np
import matplotlib.pyplot as plt
import calc
from calc.fitting import logistic_curve
from scipy.optimize import curve_fit

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
        img_1, = axs.plot(shift_range * sample_pitch[1] * 1e6, std_of_outputs, color = 'black')
        img_2 = axs.scatter(shift_range * sample_pitch[1] * 1e6, std_of_outputs, s = 40, color = 'crimson', edgecolors = 'black')
        axs.set_title('Shift memory effect decay')
        axs.set(xlabel = '$\delta x, \ \mu m$', ylabel = 'Standard deviation, p.d.u.')

        #Fitting a logistic curve onto the results
        popt, pcov = curve_fit(logistic_curve, shift_range * sample_pitch[1] * 1e6, std_of_outputs)
        smooth_range = np.linspace(-0.1, max_shift * sample_pitch[1] * 1e6, 1000)
        fitted_std = logistic_curve(smooth_range, *popt)
        img_fit, = axs.plot(smooth_range, fitted_std, color = 'blue')

        axs.text(0.65 * (max_shift * sample_pitch[1] * 1e6), 0.2 * std_of_outputs.max(), 'Fitting parameters: \nL = {:.3g} \nx_0 = {:.3g} \nk = {:.3g}'.format(*popt))

        plt.legend([(img_2, img_1), img_fit], ['STD($\delta x $)', '$L / [1 + \exp(-k (\delta x - x_0)) ]$'])
        plt.plot(block = False)


    return shifted_outputs