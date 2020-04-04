import calc
import numpy as np

def scattering_layer(layer_size_z, data_shape, data_size, refractive_index_deviation_range = [0,0.5], offset = 0):
    """
    Returns a random deviation in refractive index in a layer
    """

    #Refractive index shift grid
    d_grid = np.zeros(data_shape, np.complex)
    
    #Layer size in z coordinates
    layer_size_pixels = int(layer_size_z/data_size[0]*data_shape[0])
    #Layer_position
    layer_bounds_z = [int((data_shape[0]+k*layer_size_pixels)/2 + offset)-1 for k in [-1,1]]


    #Randomised variables
    rng=np.random.RandomState()
    rng.seed(0)
    #Generating random refractive indices
    d_grid[layer_bounds_z[0]:layer_bounds_z[1],:] = refractive_index_deviation_range[0] + rng.rand(layer_size_pixels, data_shape[1]) * np.diff(refractive_index_deviation_range)
    
    #returns the difference of the refractive index from the background and the coordinates of the layer
    return d_grid, layer_bounds_z