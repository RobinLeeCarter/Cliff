from dataclasses import dataclass
from typing import Optional


@dataclass
class Parent:
    name: str
    age: int
    ugly: bool = False

    def print_name(self):
        print(self.name)

    def print_age(self):
        print(self.age)

    def print_id(self):
        print(f'The Name is {self.name} and {self.name} is {self.age} year old')


@dataclass
class Child(Parent):
    school: Optional[str] = None
    ugly: bool = True

    def __post_init__(self):
        assert self.school is not None

    def print_school(self):
        print(f'School is {self.school}')


jack = Parent('jack snr', 32, ugly=True)
jack_son = Child('jack jnr', 12, school="", ugly=True)

jack.print_id()
jack_son.print_id()
jack_son.print_school()
