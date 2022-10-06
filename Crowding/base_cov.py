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
        return self.__repl__()

    def __repl__(self):
        return (
            "(" + self.left.__repl__() + " + " + self.right.__repl__() + ")"
        )

    def __call__(self, X):
        if isinstance(self.right, Covariance):
            return self.left(X) + self.right(X)
        return self.left(X) + self.right


class Mul(Covariance):
    R"""
    Supports multiplying covariance functions with * operator.
    """

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return self.__repl__()

    def __repl__(self):
        return (
            "(" + self.left.__repl__() + " * " + self.right.__repl__() + ")"
        )

    def __call__(self, X):
        if isinstance(self.right, Covariance):
            return self.left(X) * self.right(X)
        return self.left(X) * self.right


class Pow(Covariance):
    R"""
    Supports taking a covariance function to a power with ** operator.
    """

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return self.__repl__()

    def __repl__(self):
        return (
            "(" + self.left.__repl__() + " ** " + self.right.__repl__() + ")"
        )

    def __call__(self, X):
        return self.left(X) ** self.right
