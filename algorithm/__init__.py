# abstract
from algorithm.abstract import Episodic
from algorithm.abstract import EpisodicOnline
from algorithm.abstract import EpisodicMonteCarlo

# concrete control
from algorithm.control import ExpectedSarsa
from algorithm.control import QLearning
from algorithm.control import Sarsa
from algorithm.control import VQ

# concrete policy evaluation
from algorithm.policy_evaluation import ConstantAlphaMC
from algorithm.policy_evaluation import TD0

# factor to generate an algorithm from a Settings
from algorithm.factory import Factory
