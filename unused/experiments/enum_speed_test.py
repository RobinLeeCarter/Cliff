from __future__ import annotations
import timeit
import time
import enum


class Square(enum.IntEnum):
    NORMAL = 0
    CLIFF = 1
    START = 2
    END = 3


class NewSquare:
    NORMAL = 0
    CLIFF = 1
    START = 2
    END = 3

    # square_list = [
    #     NORMAL,
    #     CLIFF,
    #     START,
    #     END
    # ]


def square_if(i: int) -> Square:
    if i == 0:
        return Square.NORMAL
    elif i == 1:
        return Square.CLIFF
    elif i == 2:
        return Square.START
    else:
        return Square.END


square_dict = {
    0: Square.NORMAL,
    1: Square.CLIFF,
    2: Square.START,
    3: Square.END
}

square_list = [
    Square.NORMAL,
    Square.CLIFF,
    Square.START,
    Square.END
]


x: int = 2

# enum
y = Square.END
my_bool = (x == Square.CLIFF)
val = Square(x)
print(y, my_bool, val)

# class
y1 = NewSquare.END
my_bool2 = (x == NewSquare.CLIFF)
val_class = x

val2 = square_if(x)
print(val2)

val3 = square_dict[x]
print(val3)

val4 = square_list[x]
print(val4)


def get_time_ns(stmt: str) -> float:
    iterations = 1_000_000
    total_times = timeit.repeat(setup=SETUP_CODE, stmt=stmt, timer=time.process_time_ns, number=iterations)
    single_time_: float = min(total_times) / iterations
    return single_time_


SETUP_CODE = '''
import timeit
import time
import enum


class Square(enum.IntEnum):
    NORMAL = 0
    CLIFF = 1
    START = 2
    END = 3


class NewSquare:
    NORMAL = 0
    CLIFF = 1
    START = 2
    END = 3

    square_list = [
        NORMAL,
        CLIFF,
        START,
        END
    ]


def square_if(i: int) -> Square:
    if i == 0:
        return Square.NORMAL
    elif i == 1:
        return Square.CLIFF
    elif i == 2:
        return Square.START
    else:
        return Square.END


square_dict = {
    0: Square.NORMAL,
    1: Square.CLIFF,
    2: Square.START,
    3: Square.END
}

square_list = [
    Square.NORMAL,
    Square.CLIFF,
    Square.START,
    Square.END
]

x: int = 2
'''

enum_instance = '''
y = Square.END
'''

enum_compare = '''
my_bool = (x == Square.CLIFF)
'''

enum_generate = '''
val = Square(x)
'''

class_instance = '''
y1 = NewSquare.END
'''

class_compare = '''
my_bool2 = (x == NewSquare.CLIFF)
'''

class_generate = '''
val_class = x
'''

if_generate = '''
val2 = square_if(x)
'''

dict_generate = '''
val3 = square_dict[x]
'''

list_generate = '''
val4 = square_list[x]
'''


single_time = get_time_ns(enum_instance)
print(f"enum_instance  : {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(enum_compare)
print(f"enum_compare   : {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(enum_generate)
print(f"enum_generate  : {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

print()

single_time = get_time_ns(class_instance)
print(f"class_instance : {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(class_compare)
print(f"class_compare  : {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(class_generate)
print(f"class_generate : {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

print()

single_time = get_time_ns(if_generate)
print(f"if_generate    : {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(dict_generate)
print(f"dict_generate  : {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(list_generate)
print(f"list_generate  : {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

# numpy_time = single_time
#
# single_time = get_time_ns(rng_code)
# print(f"rng_code   : {single_time:.0f} ns")
# print(f"{10**9/single_time:.0f} per second")
#
# single_time = get_time_ns(numba_code)
# print(f"numba_code: {single_time:.0f} ns")
# print(f"{10**9/single_time:.0f} per second")
#
# print(f"numba_code better ratio : {numpy_time/single_time:.1f}")
