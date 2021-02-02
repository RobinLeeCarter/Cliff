import constants
from algorithm import settings
import algorithm


SETTINGS_LIST = [
  settings.Settings(algorithm.Sarsa, {"alpha": constants.ALPHA})
]

# SETTINGS_DICT = {key: value for key, value in enumerate(SETTINGS_LIST)}
