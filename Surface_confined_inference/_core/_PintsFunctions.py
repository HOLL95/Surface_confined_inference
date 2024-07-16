import pints
import Surface_confined_inference as sci
import numpy as np
import matplotlib.pyplot as plt
class FourierGaussianLogLikelihood(pints.ProblemLogLikelihood):
    def __init__(self, problem):
        super().__init__(problem)

        # Get number of times, number of 
        self.filter_kwargs={
            "Fourier_window":problem._model._internal_options.Fourier_window,
            "top_hat_width":problem._model._internal_options.top_hat_width, 
            "Fourier_function":problem._model._internal_options.Fourier_function,
            "Fourier_harmonics":problem._model._internal_options.Fourier_harmonics,
        }
        self._FTvalues=sci.top_hat_filter(self._times, self._values, **self.filter_kwargs)
        self._nt = len(self._FTvalues)
        self._no = problem.n_outputs()
        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + self._no

        # Pre-calculate parts
        self._logn = 0.5 * self._nt * np.log(2 * np.pi)
        
    
    def __call__(self, x):
        sigma = np.asarray(x[-self._no:])
        if any(sigma <= 0):
            return -np.inf
        error = self._FTvalues - sci.top_hat_filter(self._times, self._problem.evaluate(x[:-self._no]), **self.filter_kwargs)
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.sum(error**2, axis=0) / (2 * sigma**2))