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
       
        if problem._model._internal_options.transient_removal!=0:
            self.time_idx=np.where(self._times>problem._model._internal_options.transient_removal)
            self._FTvalues=sci.top_hat_filter(self._times[self.time_idx], self._values[self.time_idx], **self.filter_kwargs)
            self.truncate=True
        else:
            self._FTvalues=sci.top_hat_filter(self._times, self._values, **self.filter_kwargs)
            self.truncate=False
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
        if self.truncate==True:
            sim_vals=sci.top_hat_filter(self._times[self.time_idx], self._problem.evaluate(x[:-self._no])[self.time_idx], **self.filter_kwargs)
        else:
            sim_vals=sci.top_hat_filter(self._times, self._problem.evaluate(x[:-self._no]), **self.filter_kwargs)

        error = self._FTvalues - sim_vals
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.sum(error**2, axis=0) / (2 * sigma**2))
class GaussianTruncatedLogLikelihood(pints.ProblemLogLikelihood):
    
    def __init__(self, problem):
        super().__init__(problem)

        if problem._model._internal_options.transient_removal!=0:
            self.time_idx=np.where(self._times>problem._model._internal_options.transient_removal)
            self._values=self._values[self.time_idx]
            self.truncate=True
        else:
            self.truncate=False
        self._nt = self._times[len(self.time_idx)]
        self._no = problem.n_outputs()

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + self._no

        # Pre-calculate parts
        self._logn = 0.5 * self._nt * np.log(2 * np.pi)

    def __call__(self, x):
        sigma = np.asarray(x[-self._no:])
        if any(sigma <= 0):
            return -np.inf
        if self.truncate==True:
            sim_vals=self._problem.evaluate(x[:-self._no])[self.time_idx]
        else:
            sim_vals=self._problem.evaluate(x[:-self._no])
        plt.plot(self._times[self.time_idx], self._values)
        plt.plot(self._times[self.time_idx], sim_vals)
        error = self._values -sim_vals
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.sum(error**2, axis=0) / (2 * sigma**2))
