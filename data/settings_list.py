# import constants
import algorithm


SETTINGS_LIST = [
  algorithm.Settings(algorithm.Sarsa, {"alpha": 0.5}),
  algorithm.Settings(algorithm.Sarsa, {"alpha": 0.1})
]
