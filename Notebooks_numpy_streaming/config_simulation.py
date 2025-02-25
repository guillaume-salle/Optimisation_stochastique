import numpy as np
from functools import partial

import algorithms_numpy as algo

from objective_functions_numpy.streaming import (
    LinearRegression,
    LogisticRegression,
    GeometricMedian,
    SphericalDistribution,
    pMeans,
)
from simulation import Simulation
from datasets_numpy import (
    generate_logistic_regression,
    generate_linear_regression,
    generate_geometric_median,
    generate_spherical_distribution,
    generate_p_means,
    covtype,
)


# Configuration for the number of runs and size of data
N = 10
n = int(1e4)
dim = 50

# Configuration for true parameter
# Value from the article:
# true_param = np.array([0.0, 3.0, -9.0, 4.0, -9.0, 15.0, 0.0, -7.0, 1.0, 0.0])
# true_param = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])  # Slides, bias=False
# true_param = torch.tensor([1, 1, 1, 1, 1])  # Poly, bias=False
true_param = np.random.randn(dim)
true_param[0] = 1.0  # Bias term

alpha_list = [0.45, 0.5, 0.66, 0.75, 1.0, 1.05]
gamma_list = [0.45, 0.5, 0.66, 0.75, 1.0, 1.05]
r_values = [1, 5]

# Linear regression
bias_setting = True
simulation_linear_regression = partial(
    Simulation,
    obj_function=LinearRegression(bias=bias_setting),
    true_param=true_param,
    generate_dataset=partial(
        generate_linear_regression, bias=bias_setting, toeplitz=False, diag=False
    ),
    r_values=r_values,
)

# Linear regression Toeplitz
bias_setting = True
simulation_linear_regression_toeplitz = partial(
    Simulation,
    obj_function=LinearRegression(bias=bias_setting),
    true_param=true_param,
    generate_dataset=partial(
        generate_linear_regression, bias=bias_setting, toeplitz=True, diag=False
    ),
    r_values=r_values,
)

# Linear regression Toeplitz + diag
bias_setting = True
simulation_linear_regression_toeplitz_diag = partial(
    Simulation,
    obj_function=LinearRegression(bias=bias_setting),
    true_param=true_param,
    generate_dataset=partial(
        generate_linear_regression, bias=bias_setting, toeplitz=True, diag=True
    ),
    r_values=r_values,
)

# Logistic regression
bias_setting = True
simulation_logistic_regression = partial(
    Simulation,
    obj_function=LogisticRegression(bias=bias_setting),
    true_param=true_param,
    generate_dataset=partial(generate_logistic_regression, bias=bias_setting),
    r_values=r_values,
)

# Logistic regression Toeplitz
bias_setting = True
simulation_logistic_regression_toeplitz = partial(
    Simulation,
    obj_function=LogisticRegression(bias=bias_setting),
    true_param=true_param,
    generate_dataset=partial(generate_logistic_regression, bias=bias_setting),
    r_values=r_values,
)

# Logistic regression Toeplitz + diag
bias_setting = True
simulation_logistic_regression_toeplitz_diag = partial(
    Simulation,
    obj_function=LogisticRegression(bias=bias_setting),
    true_param=true_param,
    generate_dataset=partial(generate_logistic_regression, bias=bias_setting),
    r_values=r_values,
)

# Geometric median
true_param_geometric_median = np.zeros(10)
simulation_geometric_median = partial(
    Simulation,
    obj_function=GeometricMedian(),
    true_param=true_param_geometric_median,
    generate_dataset=generate_geometric_median,
    r_values=r_values,
)

# Spherical distribution
mu = np.zeros(3)
r = 2.0
delta = 0.2
true_param_spherical_distribution = np.append(mu, r)
simulation_spherical_distribution = partial(
    Simulation,
    obj_function=SphericalDistribution(),
    true_param=true_param_spherical_distribution,
    generate_dataset=partial(generate_spherical_distribution, delta=delta),
    r_values=[0.5, 1],
)


# p-means
p = 1.5
true_param_p_means = np.zeros(dim)
simulation_p_means = partial(
    Simulation,
    obj_function=pMeans(p=p),
    true_param=true_param_p_means,
    generate_dataset=generate_p_means,
    r_values=r_values,
)

# Covtype
train_covtype, test_covtype, name = covtype()
eval_covtype = partial(
    Simulation,
    obj_function=LogisticRegression(bias=True),
    true_param=None,
    generate_dataset=None,
    dataset=train_covtype,
    test_dataset=test_covtype,
    dataset_name=name,
    initial_param=np.zeros(54 + 1),
    r_values=None,
)
