import calc
import numpy as np
from utils.cache import disk



@disk.cache
def random_circles(number: np.int, max_radius: np.float, n_circle: np.float, n: np.ndarray, data_size, x_bounds):
	"""
	Generates radomly placed circles in the optical system
	"""
	rng = np.random.RandomState()
	rng.seed(5)

	data_shape = n.shape
	n_circle += -1

	random_circle_radii = np.ndarray.flatten(max_radius * rng.rand(1, number))
	random_circle_z_coordinates = rng.choice(np.arange(0, data_shape[0]), number)
	random_circle_x_coordinates = rng.choice(np.arange(x_bounds[0], x_bounds[1]), number)

	for idx, random_circle_radius in enumerate(random_circle_radii):
		n += calc.generate_circle(n_circle, random_circle_radius,
		                     [random_circle_z_coordinates[idx], random_circle_x_coordinates[idx]],
		                     data_shape, data_size)
	return n