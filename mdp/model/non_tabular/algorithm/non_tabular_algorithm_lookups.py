from __future__ import annotations
from typing import Type

from mdp.model.non_tabular.algorithm.non_tabular_algorithm import NonTabularAlgorithm
from mdp.model.non_tabular.algorithm.episodic.episodic_sarsa import EpisodicSarsa
from mdp import common


def get_algorithm_lookup() -> dict[common.AlgorithmType, Type[NonTabularAlgorithm]]:
    a = common.AlgorithmType
    return {
        a.NON_TABULAR_EPISODIC_SARSA: EpisodicSarsa
    }


def get_name_lookup() -> dict[common.AlgorithmType, str]:
    a = common.AlgorithmType
    return {
        a.NON_TABULAR_EPISODIC_SARSA: "Episodic Sarsa"
    }

