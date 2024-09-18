import numpy as np
import Surface_confined_inference as sci
import os
class RunSingleExperimentMCMC(sci.SingleExperiment):
    def __init__(self, experiment_type, experiment_parameters, **kwargs):
        super().__init__(experiment_type, experiment_parameters, **kwargs)
     @sci._utils.temporary_options(normalise_parameters=False)
    def run(self, time_data, current_data,**kwargs):
        if "paralell" in kwargs:
            raise KeyError("You've mis-spelled paraLLeL")
        if "method" not in kwargs:
            kwargs["method"]=pints.CMAES
        if "runs" not in kwargs:
            kwargs["runs"]=1
        if "samples" not in kwargs:
            kwargs["samples"]=10000
        if "parallel" not in kwargs:
            kwargs["parallel"]=True
        if "Fourier_filter" not in kwargs:
            kwargs["Fourier_filter"]=False
        if "num_chains" not in kwargs:
            kwargs["num_chains"]=3
        if "CMAES_results_dir" not in kwargs:
            kwargs["CMAES_results_dir"]=False
        else:
            os.path.isdir(kwargs["CMAES_results_dir"])
            if kwargs["CMAES_results_dir"][-1]!="/":
                kwargs["CMAES_results_dir"]=wargs["CMAES_results_dir"]+"/"
            kwargs["starting_point"]=kwargs["CMAES_results_dir"]+"full_table.txt"
        if "starting_point" not in kwargs and kwargs["CMAES_results_dir"]==False:
            raise KeyError("MCMC requires a starting point")
        if "save_to_directory" not in kwargs:
            kwargs["save_to_directory"]=False
        if "dimensional" not in kwargs:
            kwargs["dimensional"]=True
        if kwargs["dimensional"]==True:
            time_data=self.nondim_t(time_data)
            current_data=self.nondim_i(current_data)
        problem=pints.SingleOutputProblem(self, time_data, current_data)
        if kwargs["Fourier_filter"]==True:
            log_Likelihood=sci.FourierGaussianLogLikelihood(problem)
            
        else:
            log_Likelihood=sci.GaussianTruncatedLogLikelihood(problem)
        init_error=log_Likelihood(kwargs["starting_point"])

        prior=pints.UniformLogPrior(
                                    [self._internal_memory["boundaries"][x][0] for x in self._optim_list]+[0.1*init_error],
                                    [self._internal_memory["boundaries"][x][1] for x in self._optim_list]+[10*init_error]
                                
                                    )
        log_posterior = pints.LogPosterior(log_Likelihood, log_prior)
        xs = [
            np.array(kwargs["starting_point"]), 
            for x in range(0, kwargs["num_chains"])
        ]
      
        mcmc=pints.MCMCController(log_posterior, kwargs["num_chains"],  method=pints.HaarioBardenetACMC)
        mcmc.set_max_iterations(kwargs["samples"])
        chains = mcmc.run()
        return chains