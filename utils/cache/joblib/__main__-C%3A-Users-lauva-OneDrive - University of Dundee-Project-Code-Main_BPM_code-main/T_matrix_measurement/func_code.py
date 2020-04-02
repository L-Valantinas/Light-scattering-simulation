# first line: 138
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
