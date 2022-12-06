class Covariance:
    R"""
    Base covariance function.
    """

    def __init__(self):
        pass

    def __call__(self, x, y):
        return self.k(x, y)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(self, other)

    def __pow__(self, other):
        return Pow(self, other)


class Add(Covariance):
    R"""
    Supports adding covariance functions with + operator.
    """

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "(" + repr(self.left) + " + " + repr(self.right) + ")"

    def __call__(self, x, y):
        if callable(self.right):
            return self.left(x, y) + self.right(x, y)
        print(type(self.right))
        return self.left(x, y) + self.right


class Mul(Covariance):
    R"""
    Supports multiplying covariance functions with * operator.
    """

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "(" + repr(self.left) + " * " + repr(self.right) + ")"

    def __call__(self, x, y):
        if callable(self.right):
            return self.left(x, y) * self.right(x, y)
        return self.left(x, y) * self.right


class Pow(Covariance):
    R"""
    Supports taking a covariance function to a power with ** operator.
    """

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "(" + repr(self.left) + " ** " + repr(self.right) + ")"

    def __call__(self, x, y):
        return self.left(x, y) ** self.right
