import numpy as np

import utils
import algorithm


single_alpha_comparison = [
  algorithm.Settings(algorithm.ExpectedSarsa, {"alpha": 0.9}),
  algorithm.Settings(algorithm.VQ, {"alpha": 0.2}),
  algorithm.Settings(algorithm.QLearning, {"alpha": 0.9}),
  algorithm.Settings(algorithm.SarsaAlg, {"alpha": 0.9})
]

algorithms_list = [
  algorithm.ExpectedSarsa,
  algorithm.VQ,
  algorithm.QLearning,
  algorithm.SarsaAlg
]

alpha_list = utils.float_range(start=0.1, stop=1.0, step_size=0.05)

# algorithm.Settings(algorithm.VQ, {"alpha": 0.5}),
# algorithm.Settings(algorithm.VQ, {"alpha": 0.1}),
# algorithm.Settings(algorithm.QLearning, {"alpha": 0.5}),
# algorithm.Settings(algorithm.QLearning, {"alpha": 0.1}),
# algorithm.Settings(algorithm.Sarsa, {"alpha": 0.5}),
# algorithm.Settings(algorithm.Sarsa, {"alpha": 0.1})

# algorithm.Settings(algorithm.VQ, {"alpha_variable": True}),
