from jax.config import config
config.update("jax_enable_x64", True)
from jax.numpy import repeat, newaxis, tensordot, sqrt, exp, square, eye, float64
from jax.numpy import sum as arraysum
from base_cov import Covariance
from util import distance


class Matern32(Covariance):
    def __init__(self, ls=1.0):
        super().__init__()
        self.ls = ls

    def k(self, x, y):
        R"""
        :math:`(1 + \frac{\sqrt{3}||x-y||}{l}) \cdot e^{-\frac{\sqrt{3}||x-y||}{l}}`
        """
        r = sqrt(3.0) * distance(x, y) / self.ls
        similarity = (r + 1) * exp(-r)
        return similarity


class Matern52(Covariance):
    def __init__(self, ls=1.0):
        super().__init__()
        self.ls = ls

    def k(self, x, y):
        R"""
        :math:`(1 + \frac{\sqrt{5}||x-y||}{l} + \frac{5||x-y||^2}{3l^2})
        \cdot e^{-\frac{\sqrt{5}||x-y||}{l}}`
        """
        r = sqrt(5.0) * distance(x, y) / self.ls
        similarity = (r + square(r)/3 + 1) * exp(-r)
        return similarity


class ExpQuad(Covariance):
    def __init__(self, ls=1.0):
        super().__init__()
        self.ls = ls

    def k(self, x, y):
        R"""
        :math:`e^{-\frac{||x-y||^2}{2 l^2}}`
        """
        r = distance(x, y) / self.ls
        similarity = exp(-square(r) / 2)
        return similarity


class Exponential(Covariance):
    def __init__(self, ls=1.0):
        super().__init__()
        self.ls = ls

    def k(self, x, y):
        R"""
        :math:`e^{-\frac{||x-y||}{2l}}`
        """
        r = distance(x, y) / self.ls
        similarity = exp(-r / 2)
        return similarity


class RatQuad(Covariance):
    def __init__(self, alpha, ls=1.0):
        super().__init__()
        self.ls = ls
        self.alpha = alpha

    def k(self, x, y):
        R"""
        :math:`(1 + \frac{||x-y||^2}{2 \alpha l^2})^{-\alpha l}`
        """
        r = distance(x, y) / self.ls
        similarity = (square(r) / (2 * self.alpha) + 1) ** -self.alpha
        return similarity
