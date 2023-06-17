from jax.numpy import sqrt, exp, square
from .base_cov import Covariance
from .util import distance, select_active_dims


class Matern32(Covariance):
    R"""
    Implementation of the Matern-3/2 kernel function, a member of the Matern
    family of kernels.

    The Matern-3/2 kernel function is defined as:

    .. math::

        (1 + \frac{\sqrt{3}||x-y||}{l}) \cdot e^{-\frac{\sqrt{3}||x-y||}{l}}

    where `x` and `y` are input vectors and `l` is the length-scale.

    This class can be used as a function curry, meaning it can be called
    like a function on two inputs `x` and `y`.

    Parameters
    ----------
    ls : float, optional
        The length-scale parameter, which controls the width of the kernel.
        Larger values result in wider kernels, and smaller values in narrower kernels.
        Default is 1.0.

    active_dims : array-like, slice or scalar, optional
        The indices of the active dimensions. If specified, the kernel function
        will only be computed over these dimensions. Default is None, which means
        all dimensions are active.
    """

    def __init__(self, ls=1.0, active_dims=None):
        super().__init__()
        self.ls = ls
        self.active_dims = active_dims

    def k(self, x, y):
        R"""
        Compute the Matern-3/2 kernel function between inputs `x` and `y`.

        The kernel function is computed over the active dimensions, specified
        by the `active_dims` parameter during initialization.

        Parameters
        ----------
        x : array-like
            First input array.
        y : array-like
            Second input array.

        Returns
        -------
        similarity : float
            The computed kernel function.
        """
        x = select_active_dims(x, self.active_dims)
        y = select_active_dims(y, self.active_dims)
        r = sqrt(3.0) * distance(x, y) / self.ls
        similarity = (r + 1) * exp(-r)
        return similarity


class Matern52(Covariance):
    R"""
    Implementation of the Matern-5/2 kernel function, a member of the Matern
    family of kernels.

    The Matern-5/2 kernel function is defined as:

    .. math::

        (1 + \frac{\sqrt{5}||x-y||}{l} + \frac{5||x-y||^2}{3l^2})
        \cdot e^{-\frac{\sqrt{5}||x-y||}{l}}

    where `x` and `y` are input vectors and `l` is the length-scale.

    This class can be used as a function curry, meaning it can be called
    like a function on two inputs `x` and `y`.

    Parameters
    ----------
    ls : float, optional
        The length-scale parameter, which controls the width of the kernel.
        Larger values result in wider kernels, and smaller values in narrower kernels.
        Default is 1.0.

    active_dims : array-like, slice or scalar, optional
        The indices of the active dimensions. If specified, the kernel function
        will only be computed over these dimensions. Default is None, which means
        all dimensions are active.
    """

    def __init__(self, ls=1.0, active_dims=None):
        super().__init__()
        self.ls = ls
        self.active_dims = active_dims

    def k(self, x, y):
        R"""
        Compute the Matern-5/2 kernel function between inputs `x` and `y`.

        The kernel function is computed over the active dimensions, specified
        by the `active_dims` parameter during initialization.

        Parameters
        ----------
        x : array-like
            First input array.
        y : array-like
            Second input array.

        Returns
        -------
        similarity : float
            The computed kernel function.
        """
        x = select_active_dims(x, self.active_dims)
        y = select_active_dims(y, self.active_dims)
        r = sqrt(5.0) * distance(x, y) / self.ls
        similarity = (r + square(r) / 3 + 1) * exp(-r)
        return similarity


class ExpQuad(Covariance):
    R"""
    Exponentiated Quadratic kernel, also known as the squared exponential or the Gaussian kernel.

    The kernel is defined as:

    .. math::

        e^{-\frac{||x-y||^2}{2 l^2}}

    This class can be used as a function curry, meaning it can be called
    like a function on two inputs `x` and `y`.

    Parameters
    ----------
    ls : float, optional
        The length-scale parameter, which controls the width of the kernel.
        Larger values result in wider kernels, and smaller values in narrower kernels.
        Default is 1.0.

    active_dims : array-like, slice or scalar, optional
        The indices of the active dimensions. If specified, the kernel function
        will only be computed over these dimensions. Default is None, which means
        all dimensions are active.
    """

    def __init__(self, ls=1.0, active_dims=None):
        super().__init__()
        self.ls = ls
        self.active_dims = active_dims

    def k(self, x, y):
        R"""
        Compute the Exponentiated Quadratic kernel function between inputs `x` and `y`.

        The kernel function is computed over the active dimensions, specified
        by the `active_dims` parameter during initialization.

        Parameters
        ----------
        x : array-like
            First input array.
        y : array-like
            Second input array.

        Returns
        -------
        similarity : float
            The computed kernel function.
        """
        x = select_active_dims(x, self.active_dims)
        y = select_active_dims(y, self.active_dims)
        r = distance(x, y) / self.ls
        similarity = exp(-square(r) / 2)
        return similarity


class Exponential(Covariance):
    R"""
    Exponential kernel.

    The kernel is defined as:

    .. math::

        e^{-\frac{||x-y||}{2l}}

    This class can be used as a function curry, meaning it can be called
    like a function on two inputs `x` and `y`.

    Parameters
    ----------
    ls : float, optional
        The length-scale parameter, which controls the width of the kernel.
        Larger values result in wider kernels, and smaller values in narrower kernels.
        Default is 1.0.

    active_dims : array-like, slice or scalar, optional
        The indices of the active dimensions. If specified, the kernel function
        will only be computed over these dimensions. Default is None, which means
        all dimensions are active.
    """

    def __init__(self, ls=1.0, active_dims=None):
        super().__init__()
        self.ls = ls
        self.active_dims = active_dims

    def k(self, x, y):
        R"""
        Compute the Exponential kernel function between inputs `x` and `y`.

        The kernel function is computed over the active dimensions, specified
        by the `active_dims` parameter during initialization.

        Parameters
        ----------
        x : array-like
            First input array.
        y : array-like
            Second input array.

        Returns
        -------
        similarity : float
            The computed kernel function.
        """
        x = select_active_dims(x, self.active_dims)
        y = select_active_dims(y, self.active_dims)
        r = distance(x, y) / self.ls
        similarity = exp(-r / 2)
        return similarity


class RatQuad(Covariance):
    R"""
    Rational Quadratic kernel.

    The kernel is defined as:

    .. math::

        (1 + \frac{||x-y||^2}{2 \alpha l^2})^{-\alpha l}

    This class can be used as a function curry, meaning it can be called
    like a function on two inputs `x` and `y`.

    Parameters
    ----------
    ls : float, optional
        The length-scale parameter, which controls the width of the kernel.
        Larger values result in wider kernels, and smaller values in narrower kernels.
        Default is 1.0.

    alpha : float
        The alpha parameter of the Rational Quadratic kernel.

    active_dims : array-like, slice or scalar, optional
        The indices of the active dimensions. If specified, the kernel function
        will only be computed over these dimensions. Default is None, which means
        all dimensions are active.
    """

    def __init__(self, alpha, ls=1.0, active_dims=None):
        super().__init__()
        self.ls = ls
        self.alpha = alpha
        self.active_dims = active_dims

    def k(self, x, y):
        R"""
        Compute the Rational Quadratic kernel function between inputs `x` and `y`.

        The kernel function is computed over the active dimensions, specified
        by the `active_dims` parameter during initialization.

        Parameters
        ----------
        x : array-like
            First input array.
        y : array-like
            Second input array.

        Returns
        -------
        similarity : float
            The computed kernel function.
        """
        x = select_active_dims(x, self.active_dims)
        y = select_active_dims(y, self.active_dims)
        r = distance(x, y) / self.ls
        similarity = (square(r) / (2 * self.alpha) + 1) ** -self.alpha
        return similarity
