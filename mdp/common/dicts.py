from __future__ import annotations

from mdp.common.enums import PolicyType, ParallelContextType

policy_name: dict[PolicyType, str] = {
    PolicyType.TABULAR_DETERMINISTIC: 'Deterministic',
    PolicyType.TABULAR_E_GREEDY: 'ε-greedy',
    PolicyType.TABULAR_NONE: 'No policy',
    PolicyType.TABULAR_RANDOM: 'Random'
}

parallel_context_str: dict[ParallelContextType, str] = {
    ParallelContextType.FORK: 'fork',
    ParallelContextType.SPAWN: 'spawn',
    ParallelContextType.FORK_SERVER: 'forkserver'
}
