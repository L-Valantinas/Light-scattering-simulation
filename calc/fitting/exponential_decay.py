import numpy as np

def exponential_decay(t, N_0, scaling_coef):
	return (N_0 * np.exp(-scaling_coef * t))