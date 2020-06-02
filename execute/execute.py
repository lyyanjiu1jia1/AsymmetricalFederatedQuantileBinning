import numpy as np

from algorithm import utils
from algorithm.algorithms import CentralizedQuantile, FederatedQuantile

# original feature
feature = utils.load_data(0)

# feature_for_centralized only reserves values at even indices
feature_for_centralized = np.array([])
for i in range(feature.size):
    if i % 2 == 0:
        feature_for_centralized = np.append(feature_for_centralized, feature[i])

# runner setup
bin_num = 10
epsilon = 1e-8

# centralized runner
centralized_quantile = CentralizedQuantile(bin_num)
centralized_quantile.run(feature_for_centralized)
centralized_quantile.print()


# federated runner
federated_quantile = FederatedQuantile(bin_num, epsilon)
federated_quantile.run(feature)
federated_quantile.print()
