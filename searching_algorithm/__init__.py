from searching_algorithm.simulated_annealing import SA_scaling_algorithm
from searching_algorithm.greedy import greedy_scaling_algorithm
def getSearchingAlgorithm(name,**kwargs):

    scaling_algorithms = {
        # 'default': default_scaling_algorithm,
        'greedy': greedy_scaling_algorithm,
        'SA': SA_scaling_algorithm
    }

    return scaling_algorithms[name](**kwargs)