import math


class Value:
    def __init__(self, data, children=()):
        self.data = data
        self.grad = 0
        self._previous = set(children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # +
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    # *
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    # -var
    def __neg__(self):
        return self * -1

    # -
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    # **
    def __pow__(self, other):
        assert isinstance(other, (float, int)), "unsupported power value"

        out = Value(self.data ** other, (self, ))

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    # /
    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    # e ** x
    def exp(self):
        out = Value(math.exp(self.data), (self, ))

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        v = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(v, (self, ))

        def _backward():
            self.grad += (1 - v ** 2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        x = self.data
        out = Value((x + abs(x)) / 2, (self, ))

        def _backward():
            self.grad += (1 if x > 0 else 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        sorted_list = []
        visited = set()

        def topo(v):
            if v in visited:
                return

            visited.add(v)
            for child in v._previous:
                topo(child)
            sorted_list.append(v)
        topo(self)

        self.grad = 1  # base case (local derivative = 1)
        for v in reversed(sorted_list):
            v._backward()
