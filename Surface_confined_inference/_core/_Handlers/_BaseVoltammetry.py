import Surface_confined_inference as sci
import SurfaceODESolver as sos
import itertools
import copy 
import re
import numpy as np
import multiprocessing as mp
from typing import Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from ._ParameterHandler import ParameterHandler
import matplotlib.pyplot as plt
#context manager
#options
#parameter handling
#dispersion handling
    


def _parallel_ode_simulation(nd_dict, times, weight):
    """
    Execute ODE simulation for total current (Faradaic + capacitive) in parallel processing.
    
    Args:
        nd_dict (dict): Non-dimensionalized parameters dictionary for simulation
        times (array-like): Time points for simulation
        weight (float): Weight factor to multiply the current by (for dispersion calculations)
    
    Returns:
        numpy.ndarray: Weighted current array from index 0 (total current)
    """
    return weight * np.array(sos.ODEsimulate(times, nd_dict))[0, :]
def _parallel_faradaic_simulation(nd_dict, times, weight):
    return weight * np.array(sos.ODEsimulate(times, nd_dict))[2, :]
def _parallel_sw_simulation(nd_dict, times, weight):
    return weight * np.array(sos.SW_current(times, nd_dict))
def _funcswitch(experiment_type, Faradaic_only):
    """
     Select appropriate parallel simulation function based on experiment type and current type.
    """
    if experiment_type=="SquareWave":
        return _parallel_sw_simulation
    elif Faradaic_only is True:
        return _parallel_faradaic_simulation
    else:
        return _parallel_ode_simulation

@dataclass(frozen=True)
class ParameterInterface:
    """
    Immutable interface for parameter handling and non-dimensionalization. Provides link to data from ParameterHandler.

    Attributes:
        disp_function (Callable): Function for handling parameter dispersion, member function of sci.Dispersion
        sim_dict (dict): Dictionary of simulation parameters with values that will not be changed as part of the optimisation loop
        func_dict (dict): Dictionary mapping parameter names to non-dimensionalisation functions
        optim_list (list): List of parameters varied as part of optimisation loop
        gh_values (list): Gauss-Hermite quadrature values for dispersion

    Methods:
        - nondimensionalise(sim_params)
    """
    disp_function: Callable[[dict, list], tuple]
    sim_dict: dict = field(default_factory=dict)
    func_dict: dict = field(default_factory=dict)
    optim_list: list = field(default_factory=list)
    gh_values: list = field(default_factory=list)
   
    def nondimensionalise(self, sim_params):
        """
        Convert simulation parameters to non-dimensional form.
        Args:
            sim_params (dict): Dictionary of parameter values with same keys as optim_list (i.e. parameters changed as part of the optimisation loop)
        Returns:
            dict: Dictionary of appropriately non-dimensionalized parameters to be passed to a `simulate()` method. 
        Behavior:
            - Updates mutable copy of sim_dict with optimisation parmaeteers
            - Applies non-dimensionalization functions from func_dict to each parameter
            - Returns complete non-dimensionalized parameter dictionary
        """
        mutable_dict=copy.deepcopy(self.sim_dict)
        nd_dict = {}
        for key in sim_params.keys():
            mutable_dict[key] = sim_params[key]
        for key in self.sim_dict.keys():
            nd_dict[key] = self.func_dict[key](mutable_dict[key])
        return nd_dict

    

class BaseHandler(ABC):
    """
    Abstract base class for experiment handlers.
    
    Attributes:
        options: SingleExperiment options, provided from _Voltammetry
        param_interface (ParameterInterface): immutable class to take data and methods from ParameterHandler
    
    Methods:
        - simulate(sim_params, times)
        - dispersion_simulator(sim_params, times)
        - get_voltage(times, input_parameters, validation_parameters, **kwargs)
        - calculate_times(params, **kwargs) [abstract]
        - _get_parallel_function() [abstract]
    """
    def __init__(self, options, param_interface):
        self.options=options
        self.param_interface=param_interface
    def simulate(self, sim_params, times):
        """
        Base simulation method common to all child experiment handlers, to be called using super(). 
        Dispersion simulation is handled seperately. 
        Args:
            sim_params (dict): Dictionary of simulation parameters
            times (array-like): Time points for simulation [ASSUMES NON-DIMENSIONAL STATE]
        
        Returns:
            dict: Non-dimensionalised parameter dictionary
        """
        nd_dict = self.param_interface.nondimensionalise(sim_params)
        return nd_dict
    def dispersion_simulator(self, sim_params, times):
        """
        Simulate a dispersed parameter set using numericla quadrature. 
        
        Args:
            sim_params (dict): Dictionary of simulation parameters with same keys as optim_list
            times (list): List of time points for the solver
        
        Returns:
            numpy.ndarray: Weighted sum of currents from all dispersed parameter combinations
        
        Behavior:
            - Parameters associated with a distribution keyword (validated using ParameterHandler) are used to generate a list (parameters^bins) of values and appropriate weights
            - Each parameter/weight combination is used to generate a non-dimensionlised dictionary 
            - Simulation method is called using `_get_parallel_function()` from the child class to get around multiprocessing function pickling restrictions.
            - Executes simulations in parallel (if parallel_cpu > 1) or sequentially. 
            - Returns weighted sum of all simulation results
        """
      
        mutable_dict=copy.deepcopy(self.param_interface.sim_dict)
        for key in sim_params.keys():
            mutable_dict[key]=sim_params[key]
        disp_params, values, weights = self.param_interface.disp_function(
            mutable_dict, self.param_interface.gh_values
        )

        dictionaries = []
        weight_products = []
        
        for i in range(len(weights)):
            for j, param in enumerate(disp_params):
                sim_params[param] = values[i][j]
            nd_dict_dispersed = self.param_interface.nondimensionalise(sim_params)
            dictionaries.append(nd_dict_dispersed)
            weight_products.append(np.prod(weights[i]))
        
        parallel_func = self._get_parallel_function()
        iterable = [(nd_dict, times, weight) for nd_dict, weight in zip(dictionaries, weight_products)]
        if self.options.parallel_cpu>1:
            pool = None
            try:
                pool = mp.Pool(processes=self.options.parallel_cpu)
                results = pool.starmap(parallel_func, iterable)
                np_results = np.array(results)
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()
        else:
            np_results=np.array([parallel_func(*x) for x in iterable])
        return np.sum(np_results, axis=0)
    def get_voltage(self, times, input_parameters, validation_parameters):
        """
        Base class for individual technique voltage calculations to be called by child using super() 
        Args:
            times (list): List of timepoints - responsibility of the user to make sure the dimensional format is correct
            input_parameters (dict): Dictionary of input parameters - as above
            validation_parameters (dict): To make sure the required parameters are all present. 
        
        Returns:
            dict: Validated and processed input parameters
        
        Exceptions:
            - Exception: From sci.check_input_dict if required parameters missing
        """ 
        sci.check_input_dict(
            input_parameters, copy.deepcopy(validation_parameters), optional_arguments=[]
        )
        input_parameters=copy.deepcopy(input_parameters)
        input_parameters = ParameterHandler.validate_input_parameters(input_parameters, self.options.experiment_type)
        return input_parameters
    @abstractmethod
    def calculate_times(self, params, **kwargs):
        """
        Base class to be overridden. 
        """
        raise NotImplementedError
    @abstractmethod
    def _get_parallel_function(self):
        """
        Base class to be overridden. 
        """
        return
      
class ContinuousHandler(BaseHandler):
    """
    Handler for continuous voltammetric experiments (FTACV, PSV, DCV).
    
    Inherits from BaseHandler.
    
    Methods:
        - simulate(simulation_params, times)
        - get_voltage(times, input_parameters, validation_parameters, **kwargs)
        - _get_parallel_function()
        - calculate_times(params, **kwargs)
    """
    def __init__(self, options, param_interface):
        super().__init__(options, param_interface)
    def simulate(self, simulation_params, times):
        """
        Execute simulation for continuous voltammetric experiments.
        
        Args:
            simulation_params (dict): Dictionary of simulation parameters. Nondimensionalised using super()
            times (array-like): Time points for simulation
        
        Returns:
            numpy.ndarray: Current values at specified times
        
        Behavior:
            - Uses dispersion_simulator if dispersion is enabled
            - Otherwise performs single ODE simulation
            - Returns total current (index 0) or Faradaic current (index 2) based on options
        """
        if self.options.dispersion==True:
            current = self.dispersion_simulator(simulation_params, times)
        else:
            nd_dict=super().simulate(simulation_params, times)
            if self.options.Faradaic_only==True:
                idx=2
            else:
                idx=0
            current=np.array(sos.ODEsimulate(times, nd_dict))[idx,:]
        return current
    def get_voltage(self, times, input_parameters, validation_parameters):
        """
        Calculate voltage waveform for continuous experiments.
    
        Args:
            times (array-like): Time points for voltage calculation
            input_parameters (dict): Dictionary of input parameters
            validation_parameters (dict): Dictionary for parameter validation
            **kwargs: Additional keyword arguments
        
        Returns:
            numpy.ndarray: Voltage values at specified times
        """
        inputs=super().get_voltage(times, input_parameters, validation_parameters)
        inputs["omega"] *= 2 * np.pi
        return sos.potential(times, inputs)
    def _get_parallel_function(self):
        return _funcswitch(self.options.experiment_type, self.options.Faradaic_only)
    def calculate_times(self, params, **kwargs):
        if "sampling_factor" not in kwargs:
            sampling_factor=200
        else:
            sampling_factor=kwargs["sampling_factor"]
        if self.options.experiment_type == "FTACV" or self.options.experiment_type == "DCV":
            end_time = 2 * abs(params["E_start"] - params["E_reverse"]) / params["v"]
        elif self.options.experiment_type == "PSV":
            if "PSV_num_peaks" not in kwargs:
                kwargs["PSV_num_peaks"]=50
            end_time = kwargs["PSV_num_peaks"]/ params["omega"]
        if self.options.experiment_type == "DCV":
            dt = 1 / (sampling_factor * params["v"])
        elif self.options.experiment_type == "FTACV" or self.options.experiment_type == "PSV":
            dt = 1 / (sampling_factor * params["omega"])
        times = np.arange(0, end_time, dt)
        return times
class SquareWaveHandler(BaseHandler):
    def __init__(self, options, param_interface):
        super().__init__(options, param_interface)
        self.SW_sampling(param_interface.sim_dict)
    def get_voltage(self, times, input_parameters, validation_parameters, **kwargs):
        inputs=super().get_voltage(times, input_parameters, validation_parameters, **kwargs)
        voltages=np.zeros(len(times))
        for i in times:
            i=int(i)
            voltages[i-1]=sos.SW_potential(i,inputs["sampling_factor"],inputs["scan_increment"],inputs["SW_amplitude"],inputs["E_start"],inputs["v"])
        voltages[-1]=voltages[-2]
        return voltages
    def calculate_times(self,params, **kwargs):
        end_time=(abs(params["delta_E"]/params["scan_increment"])*params["sampling_factor"])
        dt=1
        return  np.arange(0, end_time, dt)
    def _get_parallel_function(self):
        return _parallel_sw_simulation
    def SW_sampling(self,parameters, **kwargs):
        self.SW_params={}
        sampling_factor=parameters["sampling_factor"]
        self.SW_params["end"]=int(abs(parameters['delta_E']//parameters['scan_increment']))
        p=np.array(range(0, self.SW_params["end"]))
        self.SW_params["b_idx"]=((sampling_factor*p)+(sampling_factor/2))-1
        self.SW_params["f_idx"]=(p*sampling_factor)-1
        Es=parameters["E_start"]#-parameters["E_0"]
        self.SW_params["E_p"]=(Es+parameters["v"]*(p*parameters['scan_increment']))
        self.SW_params["sim_times"]=self.calculate_times(parameters)

    def SW_peak_extractor(self, current, **kwargs):
        if "mean" not in kwargs:
            kwargs["mean"]=0
        if "window_length" not in kwargs:
            kwargs["window_length"]=1
        j=np.array(range(1, self.SW_params["end"]*self._internal_memory["input_parameters"]["sampling_factor"]))
        if kwargs["mean"]==0:
            forwards=np.zeros(len(self.SW_params["f_idx"]))
            backwards=np.zeros(len(self.SW_params["b_idx"]))
            forwards=np.array([current[x] for x in self.SW_params["f_idx"]])
            backwards=np.array([current[int(x)] for x in self.SW_params["b_idx"]])
        else:
            raise NotImplementedError
            indexes=[self.SW_params["f_idx"], self.SW_params["b_idx"]]
            sampled_currents=[np.zeros(len(self.SW_params["f_idx"])), np.zeros(len(self.SW_params["b_idx"]))]
            colours=["red", "green"]
            mean_idx=copy.deepcopy(sampled_currents)
            for i in range(0, len(self.SW_params["f_idx"])):
                for j in range(0, len(sampled_currents)):
                    x=indexes[j][i]
                    data=self.rolling_window(current[int(x-kwargs["mean"]-1):int(x-1)], kwargs["window_length"])
                    #plt.scatter(range(int(x-kwargs["mean"]-1),int(x-1)), data, color=colours[j])
                    #mean_idx[j][i]=np.mean(range(int(x-kwargs["mean"]-1),int(x-1)))
                    sampled_currents[j][i]=np.mean(data)

            forwards=np.zeros(len(self.SW_params["f_idx"]))
            backwards=np.zeros(len(self.SW_params["b_idx"]))
            forwards=np.array([current[x-1] for x in self.SW_params["f_idx"]])
            backwards=np.array([current[int(x)-1] for x in self.SW_params["b_idx"]])
        return forwards, backwards, backwards-forwards, self.SW_params["E_p"]
    def simulate(self, parameters, times):
        if self.options.dispersion==True:   
            current = self.dispersion_simulator(sim_params, times)
        else:
            nd_dict=super().simulate(parameters, times)
            times=self.SW_params["sim_times"]
            current = sos.SW_current(times, nd_dict)
        sw_dict={"total":current}
        sw_dict["forwards"], sw_dict["backwards"], sw_dict["net"], E_p=self.SW_peak_extractor(current)
        if self.options.square_wave_return!="total":
            polynomial_cap=nd_dict["Cdl"]*np.ones(len(E_p))
            keys=["CdlE1", "CdlE2", "CdlE3"]
            for i in range(0, len(keys)):
                cdl=keys[i]
                if nd_dict[cdl]!=0:
                    ep_power=np.power(E_p, i+1)
                    polynomial_cap=np.add(polynomial_cap, nd_dict[cdl]*ep_power)
            current=np.add(polynomial_cap, sw_dict[self.options.square_wave_return])
        return current



class ExperimentHandler:
    @staticmethod
    def create(options, param_interface):
        if options.experiment_type=="SquareWave":
            return SquareWaveHandler(options, param_interface)
        else:
            return ContinuousHandler(options,param_interface)

