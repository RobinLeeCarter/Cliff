from __future__ import annotations

from mdp.common.enums import PolicyType, ParallelContextType

policy_name: dict[PolicyType, str] = {
    PolicyType.DETERMINISTIC: 'Deterministic',
    PolicyType.E_GREEDY: 'ε-greedy',
    PolicyType.NONE: 'No policy',
    PolicyType.RANDOM: 'Random'
}

parallel_context_str: dict[ParallelContextType, str] = {
    ParallelContextType.FORK_GLOBAL: 'fork',
    ParallelContextType.FORK_PICKLE: 'fork',
    ParallelContextType.SPAWN: 'spawn',
    ParallelContextType.FORK_SERVER: 'forkserver'
}
