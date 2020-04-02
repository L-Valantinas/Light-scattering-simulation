import numpy as np
import matplotlib.pyplot as plt
import utils.display
from main import propagate

def scattering_presentation(n, k_0, sample_pitch, target_field, TM):
    
    ranges = [(np.arange(n.shape[i])-np.floor(n.shape[i]/2))*sample_pitch[i] for i in range(2)]
    extent = utils.display.ranges2extent(*ranges) * 1e6
    n_real = np.real(n) #real part of the refractive index

    scattered_field = propagate(n, k_0, sample_pitch, target_field, return_internal_fields = True)
    I=np.abs(scattered_field**2)
    I = I/np.max(I)

    fig, (ax1, ax2) = plt.subplots(ncols = 2)
    img1 = ax1.imshow(I, cmap = 'seismic', extent = extent)
    utils.display.colorbar(img1)
    
    ax1.imshow(np.ma.masked_where(n_real<1.01, n_real), alpha = 0.5, cmap = 'magma', extent = extent) # plots the refractive index on top
    ax1.set(xlabel = '$\mu$m', ylabel = '$\mu$m')
    ax1.set_title('(a)')

    input_field = (TM @ target_field.ravel())[np.newaxis, :]
    focused_field = propagate(n, k_0, sample_pitch, input_field, return_internal_fields = True)
    I=np.abs(focused_field**2)
    I = I/np.max(I)

    img2 = ax2.imshow(I, cmap = 'seismic', extent = extent)
    utils.display.colorbar(img2)
    
    ax2.imshow(np.ma.masked_where(n_real<1.01, n_real), alpha = 0.5, cmap = 'magma', extent = extent)
    ax2.set(ylabel = '$\mu$m', xlabel = '$\mu$m')
    ax2.set_title('(b)')

    plt.show()