import os
import numpy as np
import matplotlib.pyplot as plt

class Uniform:
    def __init__(self, n_dims):

        self.n_dims = n_dims

    def grad_log_p(self, x):
        x = np.asarray(x)
        assert (x.ndim == 1 and x.size == self.n_dims) or (x.ndim == 2 and x.shape[1] == self.n_dims), 'wrong size'

        return np.zeros_like(x)

class BoxUniform(Uniform):
    def __init__(self,lower,upper):
        lower = np.asanyarray(lower,dtype=float)
        upper = np.asanyarray(upper,dtype=float)
        assert lower.ndim == 1 and upper.ndim == 1 and lower.size == upper.size, 'wrong sizes'
        assert np.all(lower < upper), 'invalid upper and lower limits'

        Uniform.__init__(self, lower.size)

        self.lower = lower
        self.upper = upper
        self.volume = np.prod(upper - lower)

    def eval(self, x, ii=None, log=True):
        x = np.asarray(x)

        if x.ndim == 1:
            return self.eval(x[np.newaxis, :], ii, log)[0]

        if ii is None:

            in_box = np.logical_and(self.lower <= x, x <= self.upper)
            in_box = np.logical_and.reduce(in_box, axis=1)

            if log:
                prob = -float('inf') * np.ones(in_box.size, dtype=float)
                prob[in_box] = -np.log(self.volume)

            else:
                prob = np.zeros(in_box.size, dtype=float)
                prob[in_box] = 1.0 / self.volume

            return prob

        else:
            assert len(ii) > 0, 'list of indices can''t be empty'
            marginal = BoxUniform(self.lower[ii], self.upper[ii])
            return marginal.eval(x, None, log)

    def gen(self, n_samples=None, rng=np.random):
        one_sample = n_samples is None
        u = rng.rand(1 if one_sample else n_samples, self.n_dims)
        x = (self.upper - self.lower) * u + self.lower

        return x[0] if one_sample else x

class Prior(BoxUniform):
    def __init__(self):
        lower = np.full(4, -5.0)
        upper = np.full(4, +2.0)
        BoxUniform.__init__(self, lower, upper)
