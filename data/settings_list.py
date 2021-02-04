import algorithm


SETTINGS_LIST = [
  algorithm.Settings(algorithm.ExpectedSarsa, {"alpha": 0.9}),
  algorithm.Settings(algorithm.VQ, {"alpha": 0.2}),
  algorithm.Settings(algorithm.QLearning, {"alpha": 0.9}),
  algorithm.Settings(algorithm.Sarsa, {"alpha": 0.9})
]

# algorithm.Settings(algorithm.VQ, {"alpha": 0.5}),
# algorithm.Settings(algorithm.VQ, {"alpha": 0.1}),
# algorithm.Settings(algorithm.QLearning, {"alpha": 0.5}),
# algorithm.Settings(algorithm.QLearning, {"alpha": 0.1}),
# algorithm.Settings(algorithm.Sarsa, {"alpha": 0.5}),
# algorithm.Settings(algorithm.Sarsa, {"alpha": 0.1})

# algorithm.Settings(algorithm.VQ, {"alpha_variable": True}),
