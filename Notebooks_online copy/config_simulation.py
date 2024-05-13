import torch
from functools import partial
from objective_functions import (
    LinearRegression,
    LogisticRegression,
    GeometricMedian,
    SphericalDistribution,
    pMeans,
)
from simulation import Simulation
from experiment_datasets import (
    generate_logistic_regression,
    generate_linear_regression,
    generate_geometric_median,
    generate_spherical_distribution,
    generate_p_means,
    covtype,
)

# usna streaming: essayer c_nu = d**0.5, d**2/3 et d  TODO


# Configuration for the number of runs and size of data
N = 10
n = int(1e4)

batch_size = 1  # Online setting

# Configuration for true theta
true_theta = torch.tensor(
    [0.0, 3.0, -9.0, 4.0, -9.0, 15.0, 0.0, -7.0, 1.0, 0.0]
)  # Article, set bias=True
# true_theta = torch.tensor([-2., -1., 0., 1., 2.])            # Slides, bias=False
# true_theta = torch.tensor([1, 1, 1, 1, 1])                   # Poly, bias=False

# Whether or not a bias term is included
bias_setting = True

nu_list = [0.45, 0.5, 0.66, 0.75, 1.0, 1.05]
gamma_list = [0.45, 0.5, 0.66, 0.75, 1.0, 1.05]
e_values = [1, 2]

simulation_linear_regression = partial(
    Simulation,
    g=LinearRegression(bias=bias_setting),
    true_theta=true_theta,
    generate_dataset=partial(generate_linear_regression, bias=bias_setting),
    e_values=e_values,
)

simulation_logistic_regression = partial(
    Simulation,
    g=LogisticRegression(bias=bias_setting),
    true_theta=true_theta,
    generate_dataset=partial(generate_logistic_regression, bias=bias_setting),
    e_values=e_values,
)

true_theta_geometric_median = torch.zeros(10)
simulation_geometric_median = partial(
    Simulation,
    g=GeometricMedian(),
    true_theta=true_theta_geometric_median,
    generate_dataset=partial(generate_geometric_median, cov="article"),
    e_values=e_values,
)

mu = torch.zeros(3)
r = 2.0
delta = 0.2
true_theta_spherical_distribution = torch.cat((mu, torch.tensor([r])))
simulation_spherical_distribution = partial(
    Simulation,
    g=SphericalDistribution(),
    true_theta=true_theta_spherical_distribution,
    generate_dataset=partial(generate_spherical_distribution, delta=delta),
    e_values=[0.5, 1],
)

d = 40
p = 1.5
true_theta_p_means = torch.zeros(d)
simulation_p_means = partial(
    Simulation,
    g=pMeans(p=p),
    true_theta=true_theta_p_means,
    generate_dataset=partial(generate_p_means, cov="article"),
    e_values=e_values,
)

train_covtype, test_covtype, name = covtype()
eval_covtype = partial(
    Simulation,
    g=LogisticRegression(bias=True),
    true_theta=None,
    generate_dataset=None,
    dataset=train_covtype,
    test_dataset=test_covtype,
    dataset_name=name,
    initial_theta=torch.zeros(54 + 1),
    e_values=None,
)

simulations = [
    simulation_linear_regression,
    simulation_logistic_regression,
    simulation_geometric_median,
    simulation_spherical_distribution,
    simulation_p_means,
]

simulations_with_riccati = [
    simulation_linear_regression,
    simulation_logistic_regression,
    simulation_geometric_median,
]

evaluations = [eval_covtype]
