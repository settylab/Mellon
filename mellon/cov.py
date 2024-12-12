from jax.numpy import sqrt, exp, square, einsum, repeat
from .base_cov import Covariance
from .util import distance, select_active_dims, expand_to_inactive, distance_grad


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
        """
        Generate a function to compute the gradient of the Matern-3/2 kernel function.

        This method returns a callable that, when given an array `y`, computes the gradient
        of the Matern-3/2 kernel function with respect to `y`, considering `x` as the fixed
        input. The computation is restricted to the active dimensions specified in the
        covariance function instance.

        Parameters
        ----------
        x : array-like
            The fixed input array used as the first argument in the Matern-3/2 kernel.
            Its shape should be compatible with the active dimensions of the kernel.

        Returns
        -------
        Callable
            A function that takes an array `y` as input and returns the gradient of the
            Matern-3/2 kernel function with respect to `y`, evaluated at the pair `(x, y)`.
            The gradient is computed only over the active dimensions.
        """
        x = select_active_dims(x, self.active_dims)
        y = select_active_dims(y, self.active_dims)
        r = sqrt(3.0) * distance(x, y) / self.ls
        similarity = (r + 1) * exp(-r)
        return similarity

    def k_grad(self, x):
        """
        Produce a function that computes the gradient of the Matern-3/2 kernel function
        with the left argument set to x with respect to y for the active_dims.

        Parameters
        ----------
        x : array-like
            First input array.

        Returns
        -------
        k_grad : callable
            Function that computes the gradient of the Matern-3/2 kernel function.
        """
        x_shape = x.shape
        active_dims = self.active_dims
        x = select_active_dims(x, active_dims)
        dist_grad = distance_grad(x)
        factor = sqrt(3.0) / self.ls

        def k_grad(y):
            y_shape = y.shape
            y = select_active_dims(y, active_dims)
            dist, grad = dist_grad(y)
            r = -factor * dist[..., None]
            dr = factor * grad
            similarity_grad = r * dr * exp(r)
            target_shape = x_shape[:-1] + y_shape
            full_grad = expand_to_inactive(similarity_grad, target_shape, active_dims)
            return full_grad

        return k_grad


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

    def k_grad(self, x):
        """
        Generate a function to compute the gradient of the Matern-5/2 kernel function.

        This method returns a callable that, when given an array `y`, computes the gradient
        of the Matern-5/2 kernel function with respect to `y`, considering `x` as the fixed
        input. The computation is restricted to the active dimensions specified in the
        covariance function instance.

        Parameters
        ----------
        x : array-like
            The fixed input array used as the first argument in the Matern-5/2 kernel.
            Its shape should be compatible with the active dimensions of the kernel.

        Returns
        -------
        Callable
            A function that takes an array `y` as input and returns the gradient of the
            Matern-5/2 kernel function with respect to `y`, evaluated at the pair `(x, y)`.
            The gradient is computed only over the active dimensions.
        """
        x_shape = x.shape
        active_dims = self.active_dims
        x = select_active_dims(x, active_dims)
        dist_grad = distance_grad(x)
        factor = sqrt(5.0) / self.ls

        def k_grad(y):
            y_shape = y.shape
            y = select_active_dims(y, active_dims)
            dist, grad = dist_grad(y)
            r = factor * dist[..., None]
            dr = factor * grad
            similarity_grad = -1 / 3 * exp(-r) * r * (r + 1) * dr
            target_shape = x_shape[:-1] + y_shape
            full_grad = expand_to_inactive(similarity_grad, target_shape, active_dims)
            return full_grad

        return k_grad


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

    def k_grad(self, x):
        """
        Generate a function to compute the gradient of the Exponentiated Quadratic kernel function.

        This method returns a callable that, when given an array `y`, computes the gradient
        of the Exponentiated Quadratic kernel function with respect to `y`, considering `x` as the fixed
        input. The computation is restricted to the active dimensions specified in the
        covariance function instance.

        Parameters
        ----------
        x : array-like
            The fixed input array used as the first argument in the Exponentiated Quadratic kernel.
            Its shape should be compatible with the active dimensions of the kernel.

        Returns
        -------
        Callable
            A function that takes an array `y` as input and returns the gradient of the
            Exponentiated Quadratic kernel function with respect to `y`, evaluated at the pair `(x, y)`.
            The gradient is computed only over the active dimensions.
        """
        x_shape = x.shape
        active_dims = self.active_dims
        x = select_active_dims(x, active_dims)
        dist_grad = distance_grad(x)

        def k_grad(y):
            y_shape = y.shape
            y = select_active_dims(y, active_dims)
            dist, grad = dist_grad(y)
            r = dist[..., None] / self.ls
            dr = grad / self.ls
            similarity_grad = -r * dr * exp(-square(r) / 2)
            target_shape = x_shape[:-1] + y_shape
            full_grad = expand_to_inactive(similarity_grad, target_shape, active_dims)
            return full_grad

        return k_grad


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

    def k_grad(self, x):
        """
        Generate a function to compute the gradient of the Rational Quadratic kernel function.

        This method returns a callable that, when given an array `y`, computes the gradient
        of the Rational Quadratic kernel function with respect to `y`, considering `x` as the fixed
        input. The computation is restricted to the active dimensions specified in the
        covariance function instance.

        Parameters
        ----------
        x : array-like
            The fixed input array used as the first argument in the Rational Quadratic kernel.
            Its shape should be compatible with the active dimensions of the kernel.

        Returns
        -------
        Callable
            A function that takes an array `y` as input and returns the gradient of the
            Rational Quadratic kernel function with respect to `y`, evaluated at the pair `(x, y)`.
            The gradient is computed only over the active dimensions.
        """
        x_shape = x.shape
        active_dims = self.active_dims
        x = select_active_dims(x, active_dims)
        dist_grad = distance_grad(x)

        def k_grad(y):
            y_shape = y.shape
            y = select_active_dims(y, active_dims)
            dist, grad = dist_grad(y)
            r = dist[..., None] / self.ls
            dr = grad / self.ls
            similarity_grad = -1 / 2 * dr * exp(-r / 2)
            target_shape = x_shape[:-1] + y_shape
            full_grad = expand_to_inactive(similarity_grad, target_shape, active_dims)
            return full_grad

        return k_grad


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
        The alpha parameter of the Rational Quadratic kernel. Default is 1.0.

    active_dims : array-like, slice or scalar, optional
        The indices of the active dimensions. If specified, the kernel function
        will only be computed over these dimensions. Default is None, which means
        all dimensions are active.
    """

    def __init__(self, alpha=1.0, ls=1.0, active_dims=None):
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

    def k_grad(self, x):
        """
        Generate a function to compute the gradient of the Matern-3/2 kernel function.

        This method returns a callable that, when given an array `y`, computes the gradient
        of the Matern-3/2 kernel function with respect to `y`, considering `x` as the fixed
        input. The computation is restricted to the active dimensions specified in the
        covariance function instance.

        Parameters
        ----------
        x : array-like
            The fixed input array used as the first argument in the Matern-3/2 kernel.
            Its shape should be compatible with the active dimensions of the kernel.

        Returns
        -------
        Callable
            A function that takes an array `y` as input and returns the gradient of the
            Matern-3/2 kernel function with respect to `y`, evaluated at the pair `(x, y)`.
            The gradient is computed only over the active dimensions.
        """
        x_shape = x.shape
        active_dims = self.active_dims
        x = select_active_dims(x, active_dims)
        dist_grad = distance_grad(x)

        def k_grad(y):
            y_shape = y.shape
            y = select_active_dims(y, active_dims)
            dist, grad = dist_grad(y)
            r = dist[..., None] / self.ls
            dr = grad / self.ls
            similarity_grad = (
                -r * dr * (square(r) / (2 * self.alpha) + 1) ** (-self.alpha - 1)
            )
            target_shape = x_shape[:-1] + y_shape
            full_grad = expand_to_inactive(similarity_grad, target_shape, active_dims)
            return full_grad

        return k_grad


class Linear(Covariance):
    R"""
    Implementation of the Linear kernel.

    The Linear kernel function is defined as:

    .. math::

        \frac{x \cdot y}{l}

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
        """
        Compute the Linear kernel between inputs `x` and `y`.

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

        similarity = einsum("ij,kj->ik", x, y) / self.ls

        return similarity

    def k_grad(self, x):
        """
        Generate a function to compute the gradient of the Linear kernel function.

        This method returns a callable that, when given an array `y`, computes the gradient
        of the Linear kernel function with respect to `y`, considering `x` as the fixed
        input. The computation is restricted to the active dimensions specified in the
        covariance function instance.

        Parameters
        ----------
        x : array-like
            The fixed input array used as the first argument in the Linear kernel.
            Its shape should be compatible with the active dimensions of the kernel.

        Returns
        -------
        Callable
            A function that takes an array `y` as input and returns the gradient of the
            Linear kernel function with respect to `y`, evaluated at the pair `(x, y)`.
            The gradient is computed only over the active dimensions.
        """
        x_shape = x.shape
        active_dims = self.active_dims
        x = select_active_dims(x, active_dims)

        def k_grad(y):
            y_shape = y.shape
            y = select_active_dims(y, active_dims)
            similarity_grad = repeat(x[:, None, :], y.shape[0], axis=1) / self.ls
            target_shape = x_shape[:-1] + y_shape
            full_grad = expand_to_inactive(similarity_grad, target_shape, active_dims)
            return full_grad

        return k_grad
