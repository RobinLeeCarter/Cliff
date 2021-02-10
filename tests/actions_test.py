from __future__ import annotations

import environments

cliff = environments.Cliff()
print(cliff.actions_shape)

random_walk = environments.RandomWalk()
print(random_walk.actions_shape)

print(cliff.actions_shape)
