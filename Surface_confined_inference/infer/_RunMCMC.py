import numpy as np
import Surface_confined_inference as sci
import os
import copy
import  multiprocessing as mp
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
        if "num_cpu" not in kwargs:
            kwargs["num_cpu"]=len(os.sched_getaffinity(0))
        self.num_cpu=kwargs["num_cpu"]
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
            np.array(kwargs["starting_point"])
            for x in range(0, kwargs["num_chains"])
        ]
      
        mcmc=pints.MCMCController(log_posterior, kwargs["num_chains"],  method=pints.HaarioBardenetACMC)
        mcmc.set_max_iterations(kwargs["samples"])
        chains = mcmc.run()
        
        return chains
    def dispersion_simulator(self, solver, sim_params, times):
        """
        Args:
            Solver (function): function that takes time (list) and simualtion_params (dictionary) arguemnts and returns a current
            times (list): list of time points for the solver
            sim_params (dict): dictionary of simulation parameters, with the same keys as _optim_list
        Returns:
            list: list of current values at times

        Initially passes `sim_params` to the `nondimensionalise` function so all parameters are correctly represented in `_internal_memory["simulation_dict"]`,
        then caluclates bin values and associated weights by calling `disp_class.generic_dispersion`
        This returns a list of parameter values of size b^n (n dispersed parameters, b bins) and a list of weights, which is b^n tuples each of length n.
        For each weight, the `simulation_dict` is updated with the correct parameters, nondimensionalised and the resulting dictionary passed to the solver
        The resulting current is multiplied by the product of the appropriate weight tuple, and added to the dispersed current
        This dispersed current is then returned

        """
        self.num_cpu=mp.cpu_count()
        nd_dict = self.nondimensionalise(sim_params)
        disp_params, self._values, self._weights = self._disp_class.generic_dispersion(
            self._internal_memory["simulation_dict"], self._internal_memory["GH_values"]
        )
        self.solver=solver
        time_series = np.zeros(len(times))
        dictionaries=[copy.deepcopy(self._internal_memory["simulation_dict"]) for x in range(0, len(self._weights))]
        weights=[np.prod(x) for x in range(0, len(self._weights))]
        for i in range(0, len(self._weights)):
            for j in range(0, len(disp_params)):
                dictionaries[i][disp_params[j]] = self._values[i][j]
        
        iterable = [
            tuple([solver, params, times, weight]) 
            for params, weight in zip(dictionaries, weights)
        ]
        
        with mp.Pool(processes=self.num_cpu) as pool:
            results=pool.starmap(self.individual_sims, iterable)
        np_results=np.array(results)
        return np.add(np_results, axis=0)

    def individual_sims(self, solver, nd_dict, times, weight):
        return weight*np.array(solver(times, nd_dict))[0, :]