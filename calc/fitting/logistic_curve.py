import numpy as np

def logistic_curve(x, L, x_0, k):
	return L / ( 1 + np.exp( -k * (x - x_0))  )