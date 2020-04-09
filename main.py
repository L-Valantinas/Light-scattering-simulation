# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 18:03:58 2020

@author: lauva
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as F
from scipy.optimize import curve_fit
import scipy as sci

import calc
import utils.display as disp
import utils.array
from utils.cache import disk




def propagate(n: np.ndarray, k_0: np.float, sample_pitch, input_field: np.ndarray, return_internal_fields=False):
    """
    Calculates wave propagation using the beam propagation method.

    Returns an array with the output or internal field. This array is either the size of the input_field, or
    a stack of such arrays of the same dimensions as the refractive index, n.
    """
    z_range, x_range = utils.array.calc_ranges(n.shape, sample_pitch)
    kx_range = 2*np.pi * utils.array.calc_frequency_ranges(x_range)
    kz_for_every_kx = np.sqrt(np.maximum(0.0, k_0**2 - kx_range**2))
    propagator_ft = (kx_range**2 < k_0**2) * np.exp(1j * kz_for_every_kx * sample_pitch[0])

    # Make a copy now, and make sure it has the right number of dimensions
    field = np.array(input_field)
    while field.ndim < n.ndim:
        field = field[np.newaxis, :]

    if return_internal_fields:
        result = np.zeros(n.shape, dtype=np.complex)

    heterogeneous = True # For this variable to ever become False absorbing walls need to be turned off.
    for z_idx in range(n.shape[0]):
        # Propagate plane waves in Fourier space unless we just passed a heterogeneous layer
        if heterogeneous:
            field_ft =F.fft(field)
        else:
            field_ft *= phase_factor
        # propagate field
        field_ft *= propagator_ft  # Fourier multiplication == convolution in real space

        # Check if the current layer is free space propagation or not
        heterogeneous = np.any(np.abs(n[z_idx, :] - n[z_idx, 0]) > 1e-6)
        

        if heterogeneous or return_internal_fields:
            field =F.ifft(field_ft)  # return to real space

        if heterogeneous:
            field *= np.exp(1j * k_0 * (n[z_idx, :] - 1.0) * sample_pitch[0])  # refract
        else:
            phase_factor = np.exp(1j * k_0 * (n[z_idx, 0] - 1.0) * sample_pitch[0])  # delay
            field *= phase_factor

        if return_internal_fields:
            result[z_idx, :] = field

    if not return_internal_fields:
        result =F.ifft(field_ft)

    return result


@disk.cache
def T_matrix_measurement(n: np.ndarray, k_0: np.float, sample_pitch):
    Matrix_shape = np.size(n[1])
    Transmission_matrix=np.zeros([Matrix_shape,Matrix_shape], dtype=np.complex)#

    for pointsource_position in range(Matrix_shape):
        input_field = np.zeros(Matrix_shape, dtype = np.complex)
        input_field[pointsource_position]=1 #Setting pointsource position
        output_field = propagate(n, k_0, sample_pitch, input_field) # Do the beam propagation (Save only the last row of the grid)
        #Recording the Transmission matrix by pieces (converting A_z into a column vector)
        Transmission_matrix[:, pointsource_position] = output_field.ravel()

    return Transmission_matrix


def Matrix_pseudo_inversion(Matrix: np.ndarray, singular_value_minimum = 0.1, plot_singular_values = False):
    #Singular Value Decomposition
    U, S, Vh = np.linalg.svd(Matrix)
    V = Vh.T.conj()
    #Plots the Singular Values
    if plot_singular_values:
        plt.plot(np.arange(S.size),S)
        plt.show(block = False)
    #Inverse singular value matrix
    S_inv, singular_value_minimum = 1/S, 1/singular_value_minimum
    S_inv[S_inv > singular_value_minimum] = 0 #Remove terms that are above a certain value, since they don't contribute much to the output, but dominate the input (Only using eigenvectors colse to open channels)
    return V @ np.diag(S_inv) @ U.T.conj()#using an svd property to get the inverse matrix


@disk.cache
def angular_memory_effect_analysis(tilt_step: np.float, max_tilt_coef: np.float, n: np.ndarray, k_0: np.float, sample_pitch, input_field: np.ndarray):
    """
    """
    tilt_coef_range = np.arange(-max_tilt_coef, max_tilt_coef, tilt_step).ravel() # The range of Zernike polynomial coefficients
    all_output_fields = np.zeros([tilt_coef_range.size, n.shape[1]]) # holds the output fields for each tilt value
    for idx, coefficient in enumerate(tilt_coef_range):
        phase_shift = np.exp(coefficient * 2j * np.pi * np.linspace(-0.5, 0.5, n.shape[1])) # tilt phaseshift
        tilted_field = input_field * phase_shift # tilting the input field
        output_field = propagate(n, k_0, sample_pitch, tilted_field) # propagating the tilted field

        #taking intensity and normalising
        output_field = np.abs(output_field)**2
        output_field /= output_field.max()

        all_output_fields[idx, :] = output_field.ravel() # saving the results
    return all_output_fields



def main():
# =============================================================================
    
    #Defining grid properties
    
    data_shape = [256, 256]  # Number of grid points in z and x  [pixels]
    
    #wavelength in meters
    wavelength = 500e-9

    # Pixel size
    sample_pitch = np.array((1.0, 1.0)) * wavelength/4
    
    #Data size is selected having in mind pixel number per wavelength, to minimise aliasing
    data_size = [data_shape[axis] * sample_pitch[axis] for axis in range(2)] #labda/4 per step # box size in z and x  [units of meters]. It has to be symmetric for refractive index circle to work properly
    
    #Extending x_axis for absorbing walls effect
    x_shape_multiplier = 2
    #Setting bounds of the accountable x_range
    x_bounds = [int(data_shape[1]*(-0.5+0.5*x_shape_multiplier)),int(data_shape[1]*(0.5+0.5*x_shape_multiplier))]
    #Extended grid for absorbing walls
    data_shape[1] *= x_shape_multiplier
    data_size[1] *= x_shape_multiplier
    
    
    #defining position grid
    z_range = (np.arange(data_shape[0])-np.floor(data_shape[0]/2))*sample_pitch[0]
    x_range = (np.arange(data_shape[1])-np.floor(data_shape[1]/2))*sample_pitch[1]

    
# =============================================================================
    # Material properties
    rng = np.random.RandomState()
    rng.seed(0)

    #refractive index
    n_limits = np.array((1, 1.5)) # refractive index magnitude limits in the material
    n=np.ones(data_shape, dtype=np.complex) #The grid of refractive indices in each pixel
    print('[Calculating the scattering properties of the material]')

    #CIRCLE AT THE CENTER
    #The refractive index of a circle at the center
    turn_on_center_sphere = False
    if turn_on_center_sphere:
        refractive_index_of_circle = n_limits[1] - 1
        center_circle_radius = 8e-6
        center_coordinates = [int(i/2) for i in data_shape]
        n += calc.generate_circle(refractive_index_of_circle, center_circle_radius,
                               center_coordinates, data_shape, data_size)

    
    
    
    #RANDOMLY PLACED CIRCLES OF REFRACTIVE INDEX IN THE GRID
    turn_on_random_spheres = False
    if turn_on_random_spheres:
        number_of_circles = 20
        max_circle_radius = 5e-6
        refractive_index_of_random_circles= n_limits[1] - 1
        random_circle_radii = np.ndarray.flatten(max_circle_radius * rng.rand(1, number_of_circles))
        random_circle_z_coordinates = rng.choice(np.arange(0, data_shape[0]), number_of_circles)
        random_circle_x_coordinates = rng.choice(np.arange(x_bounds[0], x_bounds[1]), number_of_circles)
        
        for Nr, random_circle_radius in enumerate(random_circle_radii):
            n += calc.generate_circle(refractive_index_of_random_circles, random_circle_radius,
                                 [random_circle_z_coordinates[Nr], random_circle_x_coordinates[Nr]],
                                 data_shape, data_size)
    
    
    
    
    #RANDOM SCATTERING LAYER OF DEFINED LENGTH
    turn_on_layer = True
    if turn_on_layer:
        layer_size_z = 10e-6
        refractive_index_deviation_range= n_limits - 1#The the refractive index deviation range
        layer = calc.scattering_layer(layer_size_z, data_shape, data_size, refractive_index_deviation_range, offset = 0)
        n += layer[0]
        
    
        
    #ABSORBTION AT THE EDGES  
    #Linearly increasing absorbtion
    max_extinction_coef = 0.1j #1e-10j
    
    #Setting up grid
    d_grid_extinction = np.abs(np.arange(data_shape[1])-data_shape[1]/2)
    #setting up linearly increasing coefficients
    d_grid_extinction = d_grid_extinction/(data_shape[1]/2/max_extinction_coef)-max_extinction_coef/x_shape_multiplier
    d_grid_extinction *= x_shape_multiplier/(x_shape_multiplier-1) #scaling to max_extinction_coef
    #setting the middle zone coefficients to 0
    d_grid_extinction[x_bounds[0]:x_bounds[1]] = 0
    
    n += d_grid_extinction

    # n[n > n_limits[1]] = n_limits[1] # reducing the refractive index to the designated maximum value, where it is exceeded
    
# =============================================================================
    # T_matrix calculation

    k_0 = 2 * np.pi / wavelength

    print('[Calculating the transmission matrix]')
    #Calculating the transmission matrix for the whole x range, including the absorbing walls
    Transmission_matrix = T_matrix_measurement(n, k_0, sample_pitch)

    
    print('[Calculating the inverse of the transmission matrix]')
    #Pseudo inverse T-matrix
    Transmission_matrix_inverse = Matrix_pseudo_inversion(Transmission_matrix, 1e-3, plot_singular_values = False)

# =============================================================================        
    #Specific wavefront propagation BPM simulation
     
    #Setting which determines, whether the beam propagation will be done or not
    do_the_beam_propagation = True
    do_the_memory_effect_analysis = False
    
    if do_the_beam_propagation:
        #Defining source and its position
        # A_z = np.zeros([data_shape[1]], dtype = np.complex)
        # A_z[int(data_shape[1]/2-5):int(data_shape[1]/2+5)]=1
        # A_z = A_z.astype(np.complex)
        
        
        #Defining a Gaussian source
        # sigma = wavelength / 2
        # target_field = np.exp(-0.5*x_range**2/sigma**2)
        # target_field = target_field.astype(np.complex)
        target_field = np.zeros(data_shape[1]).ravel()
        target_field[(int(data_shape[1]/2)-5):(int(data_shape[1]/2)+5)] = 1

        phase_shift = np.exp(0 * 2j * np.pi * np.linspace(-0.5, 0.5, data_shape[1])) # phaseshift

        print('[Calculating the corrected input wavefront]')
        #Using the inverse transmission matrix to invert the input wavefront
        inverted_input_field = (Transmission_matrix_inverse @ target_field.flatten())[np.newaxis, :] * phase_shift
        inverted_input_field /= np.linalg.norm(inverted_input_field.ravel())


        print('[Propagating the corrected wavefront]')
        focused_field = propagate(n, k_0, sample_pitch, inverted_input_field, True)
        output_field = focused_field[-1,:]





        if do_the_memory_effect_analysis:

            print('[Analysing the angular optical memory effect]')



            #Comparing outputs

            ### Will need to calculate max range first


            memory_effect = angular_memory_effect_analysis(0.4, 80, n, k_0, sample_pitch, inverted_input_field)
            
            number_of_outputs = memory_effect[:, 0].size
            element_range = np.arange(-80, 80, 0.4)
            x_mean_range = np.zeros(memory_effect[:,0].shape)
            for idx in range(number_of_outputs):
                #measuring the I weighted mean of displacement for a specific output
                tot_I = np.sum(memory_effect[idx])# sum of irradiances
                x_mean =  np.sum( memory_effect[idx] / tot_I * x_range *1e6)
                x_mean_range[idx] = x_mean

            popt, pcov = curve_fit(calc.third_order_polynomial, element_range, x_mean_range)
            displacement_fit = calc.third_order_polynomial(element_range, *popt)
            
            #Calculating the critical points
            displacement_fit_derivative = np.abs(np.gradient(displacement_fit))
            x_center_idx = int(number_of_outputs / 2 - 1) # the center of x range in index notation
            critical_point_idxs = list(displacement_fit_derivative[:x_center_idx]).index(displacement_fit_derivative[:x_center_idx].min()), x_center_idx + list(displacement_fit_derivative[x_center_idx:]).index(displacement_fit_derivative[x_center_idx:].min())
            critical_points = np.array([ [element_range[idx], displacement_fit[idx]] for idx in critical_point_idxs ] )


            #Calculating the standard deviation
            outputs_std = np.std(memory_effect, 1)
            

            fig, axs_mem = plt.subplots(1, 2)
            axs_mem[0].plot(element_range, x_mean_range)
            axs_mem[0].plot(element_range, displacement_fit)
            axs_mem[0].scatter(critical_points[:, 0], critical_points[:, 1], color = 'black')
            axs_mem[0].set(xlabel = '$Z_1^{-1}$ coefficient', ylabel = '$ \langle x \\rangle$, $ \mu m $ ')
            axs_mem[0].set_title('Average displacement')

            axs_mem[1].plot(element_range, outputs_std)
            axs_mem[1].set_title('Standard deviation of output fields')
            axs_mem[1].set(xlabel = '$Z_1^{-1}$ coefficient', ylabel = '$ \sigma$, $ \mu m $ ') 
                      
            plt.show(block = False)

            
    
# =============================================================================        
    # DISPLAY    
    



    
    
    if do_the_beam_propagation:
        fig1, axs = plt.subplots(2,2)

        ranges_wo_absorbtion = [z_range,x_range/x_shape_multiplier]
        ranges_w_absorbtion = [z_range,x_range]
        extent_partial = disp.ranges2extent(*ranges_wo_absorbtion) * 1e6    
        extent_full = disp.ranges2extent(*ranges_w_absorbtion) * 1e6
        #Plots the phase map of the wavefront in area outside the absorbing walls
        axs[0,0].imshow(disp.complex2rgb(focused_field[:,x_bounds[0]:x_bounds[1]], 2), extent = extent_partial)
        axs[0,0].set(xlabel = '$\mu$m', ylabel = '$\mu$m')
        axs[0,0].set_title('$E_{field}$ phase map')

        #Plots the light intensity in the area outside the absorbing walls
        I = np.abs(focused_field**2)
        img = axs[1,0].imshow(I[:,x_bounds[0]:x_bounds[1]], cmap = 'seismic', extent = extent_partial)#, extent = utils.ranges2extent(*ranges) * 1e6)
        disp.colorbar(img)
        axs[1,0].set(xlabel = '$\mu$m', ylabel = '$\mu$m')
        axs[1,0].set_title('$E_{field}$ intensity map')


        axs[1,1].plot(x_range*1e6, np.abs(output_field)**2)
        axs[1,1].set(xlabel ='$\mu$m')
        axs[1,1].set_title('Output field')

        axs[0,1].imshow(disp.complex2rgb(n - 1), extent = extent_full)
        axs[0,1].set(xlabel = '$\mu$m', ylabel = '$\mu$m')
        axs[0,1].set_title('$\Delta n$')

        plt.subplots_adjust(hspace = 0.4, wspace = 0.2)
        plt.show(block = False)


        planewave = np.ones(data_shape[1]).flatten()
        planewave_propagation = propagate(n, k_0, sample_pitch, planewave, True)
        fig, axs = plt.subplots()
        axs.imshow(disp.complex2rgb(planewave_propagation), extent = extent_full)
        axs.set(xlabel = '$\mu$m', ylabel = '$\mu$m')
        plt.show(block = False)
        
    
    
    #Presentation graphs
# =============================================================================
    turn_on_presentation_graphs = True
    if turn_on_layer and turn_on_presentation_graphs:
        sigma = wavelength / 2
        target_field = np.exp(-0.5*x_range**2/sigma**2)
        target_field = target_field.astype(np.complex)
        disp.scattering_presentation(n, k_0, sample_pitch, target_field, Transmission_matrix_inverse)
# =============================================================================
    return Transmission_matrix,n
    

    


if __name__ == "__main__":
    output,refractive=main()
    plt.show(block = True)
    