import numpy as np
import matplotlib.pyplot as plt
import calc
from calc.fitting import logistic_curve
from scipy.optimize import curve_fit


def angular_memory_effect_analysis(tilt_coef_range, input_field, TM, data_shape, sample_pitch, wavelength, plot_std = True):
    """
    Tilting multiple outputs and computing their std
    """
    zernike_rho = np.linspace(-0.5, 0.5, data_shape[1])


    centered_outputs = np.zeros([tilt_coef_range.size, data_shape[1]]) # For storage of outputs that where tilted and then artificially shifted back to the center
    shift_idxs = np.zeros(tilt_coef_range.size) # Will store information on how much outputs needed to be shifted to be centered
    for idx, coef in enumerate(tilt_coef_range):
        output_intensity = np.abs( TM @ ( input_field * np.exp(2j * np.pi * coef * zernike_rho) ).ravel() )**2 # propagation via TM, only saving intensity
        centered_outputs[idx, :], shift_idxs[idx] = calc.center_tilted_output(output_intensity, coef, data_shape, sample_pitch, wavelength, return_shift_idx = True)

    centered_outputs = centered_outputs / np.linalg.norm(centered_outputs, 1) # Normalising all outputs


    input_field = input_field.ravel()
    reference_output = np.abs( TM @ input_field )**2 # untilted output field, which we will use for std
    reference_output = reference_output / np.linalg.norm(reference_output)

    # fig, axs = plt.subplots()
    # axs.plot(centered_outputs[-1,:])
    # plt.show()

    #Comparing the outputs with a standard deviation method
    subtracted_outputs = np.zeros(centered_outputs.shape)
    for idx in range(centered_outputs[:,0].size):
        subtracted_outputs[idx, :] = centered_outputs[idx, :] - centered_outputs[0, :] # subtracting the outputs with the reference

    std_of_outputs = np.std(subtracted_outputs, axis = 1) # Computing the standard deviation of subtracted outputs
    if plot_std:
        fig, axs = plt.subplots()
        img_1, = axs.plot(shift_idxs * sample_pitch[1] * 1e6, std_of_outputs, color = 'black')
        img_2 = axs.scatter(shift_idxs * sample_pitch[1] * 1e6, std_of_outputs, s = 40, color = 'crimson', edgecolors = 'black')
        axs.set_title('Angular memory effect decay [focused beam]')
        axs.set(xlabel = '$\delta x, \ \mu m$', ylabel = 'Standard deviation, p.d.u.')

        #Fitting a logistic curve onto the results
        popt, pcov = curve_fit(logistic_curve, shift_idxs * sample_pitch[1] * 1e6, std_of_outputs)
        smooth_range = np.linspace(-0.1, shift_idxs[-1] * sample_pitch[1] * 1e6, 1000)
        fitted_std = logistic_curve(smooth_range, *popt)
        img_fit, = axs.plot(smooth_range, fitted_std, color = 'blue')

        axs.text(0.65 * (shift_idxs[-1] * sample_pitch[1] * 1e6), 0.2 * std_of_outputs.max(), 'Fitting parameters: \nL = {:.3g} \nx_0 = {:.3g} \nk = {:.3g}'.format(*popt))

        plt.legend([(img_2, img_1), img_fit], ['STD($\delta x $)', '$L / [1 + \exp(-k (\delta x - x_0)) ]$'])
        plt.plot(block = False)

    return centered_outputs