import numpy as np

# Configuration for the number of runs and size of data
N = 20
n = int(1e4)

# Configuration for true theta
true_theta = np.array([0, 3, -9, 4, -9, 15, 0, -7, 1, 0])  # Set bias=True
# true_theta = np.array([-2., -1., 0., 1., 2.])            # Set bias=False
# true_theta = np.array([1, 1, 1, 1, 1])                   # Set bias=False

# Whether or not a bias term is included
bias_setting = True

nu_list = [0.45, 0.5, 0.66, 0.75, 1.0, 1.05]
gamma_list = [0.45, 0.5, 0.66, 0.75, 1.0, 1.05]
e = 1.0
