import numpy as np
from matplotlib import pyplot as plt


class Quantile(object):
    def __init__(self, bin_num):
        self.bin_num = bin_num
        self.split_points = np.array([])

    def run(self, feature):
        pass

    def print(self):
        print("split points = {}".format(self.split_points))

    def show(self):
        t_plot = [i for i in range(self.split_points.size)]
        plt.plot(t_plot, self.split_points, 'b')
        plt.show()


class CentralizedQuantile(Quantile):
    def run(self, feature):
        feature = np.sort(feature)
        bin_density = feature.size // self.bin_num

        for i in range(1, self.bin_num):
            self.split_points = np.append(self.split_points, feature[i * bin_density])


class FederatedQuantile(Quantile):
    def __init__(self, bin_num, epsilon):
        super(FederatedQuantile, self).__init__(bin_num)
        self.epsilon = epsilon

    def run(self, feature):
        feature_cache = self.parse_feature(feature)
        bin_density = sum(feature_cache.values()) // self.bin_num

        fixed_max_val = max(feature_cache.keys())
        split_point = min(feature_cache.keys())

        for i in range(1, self.bin_num):
            min_val = split_point
            max_val = fixed_max_val

            while max_val - min_val > self.epsilon:
                mid_val = (max_val + min_val) / 2

                # count the number of ones in (min_val, mid_val)
                ones_count = 0
                for k, v in feature_cache.items():
                    if split_point <= k <= mid_val:
                        ones_count += v

                # judge
                if ones_count <= bin_density:
                    min_val = mid_val
                else:
                    max_val = mid_val

            # conclude
            split_point = (max_val + min_val) / 2

            # record
            self.split_points = np.append(self.split_points, split_point)

    @staticmethod
    def parse_feature(feature):
        cache = {}
        for i in range(feature.size):
            if i % 2 == 0:
                if feature[i] in cache.keys():
                    cache[feature[i]] += 1
                else:
                    cache[feature[i]] = 1
            else:
                if feature[i] not in cache.keys():
                    cache[feature[i]] = 0
        return cache
