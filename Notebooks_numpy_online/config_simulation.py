import numpy as np
from functools import partial

import algorithms_numpy as algo

from objective_functions_numpy_online import (
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
N = 20
n = int(1e4)
batch_size_power = 0
batch_size_power_list = [0]

# Configuration for true parameter
# Value from the article:
true_param = np.array([0.0, 3.0, -9.0, 4.0, -9.0, 15.0, 0.0, -7.0, 1.0, 0.0])
# true_param = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])  # Slides, bias=False
# true_param = torch.tensor([1, 1, 1, 1, 1])  # Poly, bias=False

# Whether or not a bias term is included
bias_setting = True

alpha_list = [0.45, 0.5, 0.66, 0.75, 1.0, 1.05]
gamma_list = [0.45, 0.5, 0.66, 0.75, 1.0, 1.05]
e_values = [1, 2]

# Linear regression
simulation_linear_regression = partial(
    Simulation,
    objective_function=LinearRegression(bias=bias_setting),
    batch_size_power=batch_size_power,
    batch_size_power_list=batch_size_power_list,
    true_param=true_param,
    generate_dataset=partial(generate_linear_regression, bias=bias_setting),
    e_values=e_values,
)

# Logistic regression
simulation_logistic_regression = partial(
    Simulation,
    objective_function=LogisticRegression(bias=bias_setting),
    batch_size_power=batch_size_power,
    batch_size_power_list=batch_size_power_list,
    true_param=true_param,
    generate_dataset=partial(generate_logistic_regression, bias=bias_setting),
    e_values=e_values,
)

# Geometric median
true_param_geometric_median = np.zeros(10)
simulation_geometric_median = partial(
    Simulation,
    objective_function=GeometricMedian(),
    batch_size_power=batch_size_power,
    batch_size_power_list=batch_size_power_list,
    true_param=true_param_geometric_median,
    generate_dataset=generate_geometric_median,
    e_values=e_values,
)

# Spherical distribution
mu = np.zeros(3)
r = 2.0
delta = 0.2
true_param_spherical_distribution = np.append(mu, r)
simulation_spherical_distribution = partial(
    Simulation,
    objective_function=SphericalDistribution(),
    batch_size_power=batch_size_power,
    batch_size_power_list=batch_size_power_list,
    true_param=true_param_spherical_distribution,
    generate_dataset=partial(generate_spherical_distribution, delta=delta),
    e_values=[0.5, 1],
)


# p-means
d = 40
p = 1.5
true_param_p_means = np.zeros(d)
simulation_p_means = partial(
    Simulation,
    objective_function=pMeans(p=p),
    batch_size_power=batch_size_power,
    batch_size_power_list=batch_size_power_list,
    true_param=true_param_p_means,
    generate_dataset=generate_p_means,
    e_values=e_values,
)

# Covtype
train_covtype, test_covtype, name = covtype()
eval_covtype = partial(
    Simulation,
    objective_function=LogisticRegression(bias=True),
    batch_size_power=batch_size_power,
    batch_size_power_list=batch_size_power_list,
    true_param=None,
    generate_dataset=None,
    dataset=train_covtype,
    test_dataset=test_covtype,
    dataset_name=name,
    initial_param=np.zeros(54 + 1),
    e_values=None,
)
