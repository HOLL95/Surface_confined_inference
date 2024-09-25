import numpy as np
import Surface_confined_inference as sci
import os
import copy
import  multiprocessing as mp
import matplotlib.pyplot as plt
import pints
import time
import SurfaceODESolver as sos
class RunSingleExperimentMCMC(sci.SingleExperiment):
    def __init__(self, experiment_type, experiment_parameters, **kwargs):
        
        super().__init__(experiment_type, experiment_parameters, **kwargs)
        self.k0_values=np.zeros(1000)
        self.i=0
    @sci._utils.temporary_options(normalise_parameters=False)
    def run(self, time_data, current_data,**kwargs):
        if "runs" not in kwargs:
            kwargs["runs"]=1
        if "samples" not in kwargs:
            kwargs["samples"]=10000
       
        if "Fourier_filter" not in kwargs:
            kwargs["Fourier_filter"]=False
        if "num_chains" not in kwargs:
            kwargs["num_chains"]=3
        if "CMAES_results_dir" not in kwargs:
            kwargs["CMAES_results_dir"]=False
        else:
            os.path.isdir(kwargs["CMAES_results_dir"])
            if kwargs["CMAES_results_dir"][-1]!="/":
                kwargs["CMAES_results_dir"]=kwargs["CMAES_results_dir"]+"/"
            kwargs["starting_point"]=sci._utils.read_param_table(kwargs["CMAES_results_dir"]+"Rounded_table.txt")[0]
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
        if "transformation" not in kwargs:
            kwargs["transformation"]="identity"
        if "sigma0" not in kwargs:
            kwargs["sigma0"]=0.075
        self.num_cpu=kwargs["num_cpu"]
        problem=pints.SingleOutputProblem(self, time_data, current_data)
        if kwargs["Fourier_filter"]==True:
            log_Likelihood=sci.FourierGaussianLogLikelihood(problem)
        else:
            log_Likelihood=sci.GaussianTruncatedLogLikelihood(problem)
       
        if "starting_point" in kwargs and kwargs["CMAES_results_dir"]==False:
            if len(kwargs["starting_point"])!=problem.n_parameters()+problem.n_outputs():
                kwargs["starting_point"]+=[sci._utils.RMSE(current_data, log_Likelihood._problem.evaluate(kwargs["starting_point"]))]
        kwargs["starting_point"][-1]*=100000
        error=kwargs["starting_point"][-1]
        lower=[self._internal_memory["boundaries"][x][0] for x in self._optim_list]+[0.1*error]
        upper=[self._internal_memory["boundaries"][x][1] for x in self._optim_list]+[10*error]

        log_prior=pints.UniformLogPrior(
                                       lower,
                                       upper 
                                    )
        log_posterior = pints.LogPosterior(log_Likelihood, log_prior)
        xs = [
            np.array(kwargs["starting_point"])
            for x in range(0, kwargs["num_chains"])
        ]
        if kwargs["transformation"]=="identity":
            transform=pints.IdentityTransformation(len(xs[0]))
        elif kwargs["transformation"]=="log":
            transform=pints.LogTransformation(len(xs[0]))
        elif isinstance(kwargs["transformation"], dict):
            transforms=[]   
            for i in range(0, len(self._optim_list)):
                if self._optim_list[i] in kwargs["transformation"]["log"]:
                    transforms.append(pints.LogTransformation(1))
                else:
                    transforms.append(pints.IdentityTransformation(1))
            transforms.append(pints.LogTransformation(problem.n_outputs()))
            transform=pints.ComposedTransformation(*transforms)
        init_sigma=[0 for x in range(0, log_posterior.n_parameters())]
        for i in range(0, len(xs[0])):
            init_sigma[i]=kwargs["sigma0"]*(upper[i]-lower[i])

        print(init_sigma)
        mcmc=pints.MCMCController(log_posterior, kwargs["num_chains"],  xs,transformation=transform,sigma0=np.array(init_sigma))
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
        if self._internal_options.experiment_type in ["FTACV", "DCV", "PSV"]:
            para_func=individual_ode_sims
        nd_dict = self.nondimensionalise(sim_params)
        
        disp_params, self._values, self._weights = self._disp_class.generic_dispersion(
            self._internal_memory["simulation_dict"], self._internal_memory["GH_values"]
        )
        print([sim_params[x] for x in self._optim_list])
        time_series = np.zeros(len(times))
        dictionaries=[copy.deepcopy(self._internal_memory["simulation_dict"]) for x in range(0, len(self._weights))]
        weights=[np.prod(x) for x in self._weights]
        for i in range(0, len(self._weights)):
            for j in range(0, len(disp_params)):
                self._internal_memory["simulation_dict"][disp_params[j]] = self._values[i][j]

            dictionaries[i]=self.nondimensionalise(sim_params)   

        iterable = [
            tuple([params, times, weight]) 
            for params, weight in zip(dictionaries, weights)
        ]
        
        with mp.Pool(processes=self.num_cpu) as pool:
            results=pool.starmap(para_func, iterable)
        np_results=np.array(results)
        return np.sum(np_results, axis=0)

def individual_ode_sims(nd_dict, times, weight):
    return weight*np.array(sos.ODEsimulate(times, nd_dict))[0, :]
def individual_sw_sims(nd_dict, times, weight):
    return weight*np.array(sos.SW_current(times, nd_dict))
