from mdp.scenarios.mountain_car.model.state import State
from mdp.scenarios.mountain_car.model.action import Action

s = State(is_terminal=False, position=0.1, velocity=0.3)
a = Action(acceleration=0.0)

print(s.values)
print(a.values)

# for k, v in s.__dataclass_fields__.items():
#     print(k, v)

# for field in dataclasses.fields(s):
#     print(field)


