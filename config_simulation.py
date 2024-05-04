import numpy as np
from functools import partial
from objective_functions import (
    LinearRegression,
    LogisticRegression,
    GeometricMedian,
    SphericalDistribution,
    pMeans,
)
from simulation import Simulation
from datasets import (
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

# Configuration for true theta
true_theta = np.array([0, 3, -9, 4, -9, 15, 0, -7, 1, 0])  # Set bias=True
# true_theta = np.array([-2., -1., 0., 1., 2.])            # Set bias=False
# true_theta = np.array([1, 1, 1, 1, 1])                   # Set bias=False

# Whether or not a bias term is included
bias_setting = True

nu_list = [0.45, 0.5, 0.66, 0.75, 1.0, 1.05]
gamma_list = [0.45, 0.5, 0.66, 0.75, 1.0, 1.05]
e_values = [1, 2]

simulation_linear_regression = partial(
    Simulation,
    g=LinearRegression(bias=bias_setting),
    e_values=e_values,
    true_theta=true_theta,
    generate_dataset=partial(generate_linear_regression, bias=bias_setting),
)

simulation_logistic_regression = partial(
    Simulation,
    g=LogisticRegression(bias=bias_setting),
    e_values=e_values,
    true_theta=true_theta,
    generate_dataset=partial(generate_logistic_regression, bias=bias_setting),
)

true_theta_geometric_median = np.zeros(10)
simulation_geometric_median = partial(
    Simulation,
    g=GeometricMedian(),
    e_values=e_values,
    true_theta=true_theta_geometric_median,
    generate_dataset=partial(generate_geometric_median, cov="article"),
)

mu = np.zeros(3)
r = 2.0
delta = 0.2
true_theta_spherical_distribution = np.append(mu, r)
simulation_spherical_distribution = partial(
    Simulation,
    g=SphericalDistribution(),
    e_values=[0.5, 1],
    true_theta=true_theta_spherical_distribution,
    generate_dataset=partial(generate_spherical_distribution, delta=delta),
)

d = 40
p = 1.5
true_theta_p_means = np.zeros(d)
simulation_p_means = partial(
    Simulation,
    g=pMeans(p=p),
    e_values=e_values,
    true_theta=true_theta_p_means,
    generate_dataset=partial(generate_p_means, cov="article"),
)

train_covtype, test_covtype = covtype()
eval_covtype = partial(
    Simulation,
    g=LogisticRegression(bias=True),
    true_theta=None,
    e_values=None,
    generate_dataset=None,
    dataset=train_covtype,
    test_dataset=test_covtype,
    initial_theta=np.zeros(54 + 1),
)
