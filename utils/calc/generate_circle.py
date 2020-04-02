import utils.calc
import numpy as np


def radius(row,column):
    r=(abs(row**2)+abs(column**2))**0.5
    return r

def generate_circle(refractive_index, circle_radius, position, data_shape, data_size):
    #Settings
    #The grid of deviations of refractive indices from the normal
    d_grid=np.zeros(data_shape, dtype=np.complex)
    circle_radius_in_pixels=circle_radius/data_size[0]*data_shape[0]#circle radius in pixels
    #A set of grid indices that are within the set radius
    rows=[]
    columns=[]
    #Retrieving the set of indices
    for x_index in range(data_shape[1]):
        for z_index in range(data_shape[0]):
            #Note that only the absolute value matters for the radius func
            x,z = x_index - (position[1]+1), z_index - (position[0]+1)
            if radius(z,x)<circle_radius_in_pixels:
                columns.append(x_index)
                rows.append(z_index)
    #The value by which the refractive index differs
    d_grid[rows,columns]=refractive_index
    return d_grid