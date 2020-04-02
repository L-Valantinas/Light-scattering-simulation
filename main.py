# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 18:03:58 2020

@author: lauva
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as F

import utils.calc
import utils.display as disp
import utils.array
from utils.cache import disk




def propagate(n: np.ndarray, k_0: np.float, sample_pitch, input_field: np.ndarray, return_internal_fields=False):
    """
    Calculates wave propagation using the beam propagation method.

    :param n: The complex refractive index distribution
    :param sample_pitch: The sample pitch
    :param k_0: The wavenumber at refractive index 1.
    :param input_field: The complex input field.
    :param return_internal_fields: Boolean to indicate whether internal fields should be returned or not. (default: False)
    :return: Returns an array with the output or internal field. This array is either the size of the input_field, or
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

    heterogeneous = True
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
def T_matrix_measurement(n, k_0, sample_pitch):
    
    Matrix_shape = np.size(n[1])
    Transmission_matrix=np.zeros([Matrix_shape,Matrix_shape], dtype=np.complex)#

    for pointsource_position in range(Matrix_shape):
        input_field = np.zeros(Matrix_shape, dtype = np.complex)
        input_field[pointsource_position]=1 #Setting pointsource position
        output_field = propagate(n, k_0, sample_pitch, input_field) # Do the beam propagation (Save only the last row of the grid)
        #Recording the Transmission matrix by pieces (converting A_z into a column vector)
        Transmission_matrix[:, pointsource_position] = output_field.ravel()

    return Transmission_matrix


def Matrix_pseudo_inversion(Matrix, singular_value_minimum = 0.1, plot_singular_values = False):
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
    #The grid of refractive indices in each pixel
    n=np.ones(data_shape, dtype=np.complex)

    n_0, n_max = [1, 1.5] # refractive index limits in the material
    
    print('[Calculating the scattering properties of the material]')
    #CIRCLE AT THE CENTER
    #The refractive index of a circle at the center
    turn_on_center_sphere = False
    if turn_on_center_sphere:
        refractive_index_of_circle = 0
        center_circle_radius = 2e-6
        center_coordinates = [int(i/2) for i in data_shape]
        n += utils.calc.generate_circle(refractive_index_of_circle, center_circle_radius,
                               center_coordinates, data_shape, data_size)

    
    
    
    #RANDOMLY PLACED CIRCLES OF REFRACTIVE INDEX IN THE GRID
    turn_on_random_spheres = False
    if turn_on_random_spheres:
        number_of_circles = 20
        max_circle_radius = 5e-6
        refractive_index_of_random_circles=0.5
        random_circle_radii = np.ndarray.flatten(max_circle_radius * rng.rand(1, number_of_circles))
        random_circle_z_coordinates = rng.choice(np.arange(0, data_shape[0]), number_of_circles)
        random_circle_x_coordinates = rng.choice(np.arange(x_bounds[0], x_bounds[1]), number_of_circles)
        
        for Nr, random_circle_radius in enumerate(random_circle_radii):
            n += utils.calc.generate_circle(refractive_index_of_random_circles, random_circle_radius,
                                 [random_circle_z_coordinates[Nr], random_circle_x_coordinates[Nr]],
                                 data_shape, data_size)
    
    
    
    
    #RANDOM SCATTERING LAYER OF DEFINED LENGTH
    turn_on_layer = True
    if turn_on_layer:
        layer_size_z = 10e-6
        refractive_index_deviation_range=[0,0.5]#The the refractive index deviation range
        layer = utils.calc.scattering_layer(layer_size_z, data_shape, data_size, refractive_index_deviation_range, offset = 0)
        n += layer[0]
        
    
        
    #ABSORBTION AT THE EDGES  
    #Linearly increasing absorbtion
    max_extinction_coef = 0.1j
    
    #Setting up grid
    d_grid_extinction = np.abs(np.arange(data_shape[1])-data_shape[1]/2)
    #setting up linearly increasing coefficients
    d_grid_extinction = d_grid_extinction/(data_shape[1]/2/max_extinction_coef)-max_extinction_coef/x_shape_multiplier
    d_grid_extinction *= x_shape_multiplier/(x_shape_multiplier-1) #scaling to max_extinction_coef
    #setting the middle zone coefficients to 0
    d_grid_extinction[x_bounds[0]:x_bounds[1]] = 0
    
    n += d_grid_extinction

    n[n > n_max] = n_max # reducing the refractive index to the designated maximum value, where it is exceeded
    
# =============================================================================
    # T_matrix calculation

    k_0 = 2 * np.pi / wavelength

    print('[Calculating the transmission matrix]')
    #Calculating the transmission matrix for the whole x range, including the absorbing walls
    Transmission_matrix = T_matrix_measurement(n, k_0, sample_pitch)

    
    print('[Calculating the inverse of the transmission matrix]')
    #Pseudo inverse T-matrix
    Transmission_matrix_inverse = Matrix_pseudo_inversion(Transmission_matrix, 0.1, plot_singular_values = False)

# =============================================================================        
    #Specific wavefront propagation BPM simulation
     
    #Setting which determines, whether the beam propagation will be done or not
    do_the_beam_propagation = True
    
    if do_the_beam_propagation:
        #Defining source and its position
        # A_z = np.zeros([data_shape[1]], dtype = np.complex)
        # A_z[int(data_shape[1]/2-5):int(data_shape[1]/2+5)]=1
        # A_z = A_z.astype(np.complex)
        
        
        #Defining a Gaussian source
        sigma = wavelength / 2
        target_field = np.exp(-0.5*x_range**2/sigma**2)
        target_field = target_field.astype(np.complex)

        phase_shift = np.exp(0 * 2j * np.pi * np.linspace(-0.5, 0.5, data_shape[1])) # phaseshift

        print('[Calculating the corrected input wavefront]')
        #Using the inverse transmission matrix to invert the input wavefront
        inverted_input_field = (Transmission_matrix_inverse @ target_field.flatten())[np.newaxis, :] * phase_shift
        inverted_input_field /= np.linalg.norm(inverted_input_field.ravel())


        print('[Propagating the corrected wavefront]')
        focused_field = propagate(n, k_0, sample_pitch, inverted_input_field, True)
        output_field = focused_field[-1,:]

        # for i in np.arange(-2, 3, 1)*1:
        #     inverted_input_field = inverted_input_field_original[:, np.clip(i+ np.arange(data_shape[1]), 0, data_shape[1]-1)]
        #     inverted_input_field /= np.linalg.norm(inverted_input_field.flatten())
        #     #Do the BPM        
        #     focused_field = propagate(n, k_0, sample_pitch, inverted_input_field, True)
        #     output_field = focused_field[-1,:]

        #     plt.plot(x_range * 1e6, np.abs(output_field)**2)

        # plt.show(block = False)
    
    
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

        #Plots the light intensity in the area outside the absorbing walls
        I = np.abs(focused_field**2)
        img = axs[1,0].imshow(I[:,x_bounds[0]:x_bounds[1]], cmap = 'seismic', extent = extent_partial)#, extent = utils.ranges2extent(*ranges) * 1e6)
        disp.colorbar(img)
        axs[1,0].set(xlabel = '$\mu$m', ylabel = '$\mu$m')


        axs[1,1].plot(x_range*1e6, np.abs(output_field)**2)
        axs[1,1].set(xlabel ='$\mu$m')

        axs[0,1].imshow(disp.complex2rgb(n - 1), extent = extent_full)

        plt.show(block = False)
        
        
        
    
    
#    output_field_fourier = F.fftshift(F.fft(output_field))
#    plt.plot(np.abs(output_field_fourier)**2)
#    plt.show()
    
    
    #Interactive plot (use for presentations)
# =============================================================================
#    A_z[:] = 0
#    A_z[int(data_shape[1]/2-5):int(data_shape[1]/2+5)]=1
#    A_z = A_z.astype(np.complex)
#    A_scattered = simulate_propagation(A_z, bpm_arguments)
#    
#    ranges=[z_range,x_range/x_shape_multiplier]
#    
#    fig, (ax1, ax2) = plt.subplots(ncols = 2)
#         
#    I=np.abs(A_scattered**2)
#    I = I/np.max(I)
#    
#    img1 = ax1.imshow(I[:,x_bounds[0]:x_bounds[1]], cmap = 'seismic', extent = utils.ranges2extent(*ranges) * 1e6)
#    colorbar(img1)
#    
#    if turn_on_layer:
#        layer_grid = np.zeros(n[:,x_bounds[0]:x_bounds[1]].shape)
#        layer_grid[layer[1],:] = 1
#        layer_grid = np.ma.masked_where(layer_grid<1, layer_grid)
#        ax1.imshow(layer_grid, alpha = 0.5, cmap = 'RdYlGn',extent = utils.ranges2extent(*ranges) * 1e6)
#    ax1.set(xlabel = '$\mu$m', ylabel = '$\mu$m')
#    ax1.set_title('(a)')
#
#
#  
#    
#    I=np.abs(focused_field**2)
#    I = I/np.max(I)
#    
#    img2 = ax2.imshow(I[:,x_bounds[0]:x_bounds[1]], cmap = 'seismic', extent = utils.ranges2extent(*ranges) * 1e6)
#    colorbar(img2)
#    
#    if turn_on_layer:
#        layer_grid = np.zeros(n[:,x_bounds[0]:x_bounds[1]].shape)
#        layer_grid[layer[1],:] = 1
#        layer_grid = np.ma.masked_where(layer_grid<1, layer_grid)
#        ax2.imshow(layer_grid, alpha = 0.5, cmap = 'RdYlGn',extent = utils.ranges2extent(*ranges) * 1e6)
#    ax2.set(ylabel = '$\mu$m', xlabel = '$\mu$m')
#    ax2.set_title('(b)')
#    
#    plt.tight_layout(pad = 0.3)
#    plt.show()
# =============================================================================
    return Transmission_matrix,n
    

    


if __name__ == "__main__":
    output,refractive=main()
    plt.show(block = True)
    