class First(object):
    def __init__(self):
        print("first")


class Second(First):
    def __init__(self):
        print("pre-second")
        super().__init__()
        print("post-second")


class Third(First):
    def __init__(self):
        print("pre-third")
        super().__init__()
        print("post-third")


class Fourth(Second, Third):
    def __init__(self):
        print("pre-forth")
        super().__init__()
        print("post-forth")


test = Fourth()
