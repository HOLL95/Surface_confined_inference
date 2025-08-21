import SurfaceODESolver as sos
import Surface_confined_inference as sci
from Surface_confined_inference._utils import RMSE
from ._Handlers._ParameterHandler import ParameterHandler
from ._Handlers._BaseVoltammetry import ExperimentHandler, ParameterInterface
from pathlib import Path
import pints
import collections.abc
import numbers
from warnings import warn
import itertools
import numpy as np
import copy
import re
import time
import math
import matplotlib.pyplot as plt
import json
class SingleExperiment(sci.BaseExperiment,sci.OptionsAwareMixin):
    _change_options=["experiment_type", "GH_quadrature", "dispersion_bins", "phase_only", "square_wave_return", "input_params"]
    def __init__(self, experiment_type, experiment_parameters, options_handler=None, **kwargs):
        """
        A class for performing electrochemical experiments with parameter estimation capabilities. SingleExperiment is an "orchestrator" class
        It initialises several classes to do with parameter handling (ParameterHandler), dispersion (Dispersion) and simulation (ExperimentHandler). 

        The intended flow is:
        Initialise SingleExperiment -> initialise SingleExperimentOptions
        Set optim_list, fixed_parameters -> initialise ParameterHandler, Dispersion, ExperimentHandler, NDclass
        Each subsequent change to optim_list, fixed_parameters or an optin in _change_options should force a re-initialisation of the handlers
        ExperimentHandler is then availiable for simulation. 
        
        Inherits from BaseExperiment and OptionsAwareMixin to provide class loading and option setting capabilities respectively. 

        Args:
            experiment_type (str): Type of experiment ("FTACV", "PSV", "DCV", "SquareWave")
            experiment_parameters (dict): Dictionary of experiment input parameters
            options_handler (class, optional): Custom options handler class for loading child classes. 
            **kwargs: Additional keyword arguments for experiment options
        
        Member Methods:
            __init__, dim_i, dim_e, dim_t, nondim_i, nondim_e, nondim_t, n_parameters, 
            n_outputs, redimensionalise, optim_list (property), boundaries (property), 
            fixed_parameters (property), _intitialise_phandler, experiment_top_hat, 
            change_normalisation_group, get_voltage, calculate_times, simulate, save_class, 
            __setattr__
        """
        kwargs["experiment_type"] = experiment_type
        kwargs["input_params"]=experiment_parameters
        self._input_parameters=experiment_parameters
        if options_handler is None:
            options_handler=None
        self._options_handler=options_handler
        self._options_class = sci.SingleExperimentOptions(options_handler=options_handler,**kwargs)
        self._internal_options=self._options_class._experiment_options
        self._phandler=ParameterHandler(parameters=[], boundaries={}, options=self._internal_options, fixed_parameters={})
        self._warning_raised=False
        self.experiment_type=self._internal_options.experiment_type

    """
    Various helper functions
    """
    def dim_i(self, current):return np.multiply(current, self._ND_class.c_I0)
    def dim_e(self, potential):return np.multiply(potential, self._ND_class.c_E0)
    def dim_t(self, time):return np.multiply(time, self._ND_class.c_T0)
    def nondim_i(self, current):return np.divide(current, self._ND_class.c_I0)
    def nondim_e(self, potential):return np.divide(potential, self._ND_class.c_E0)
    def nondim_t(self, time):return np.divide(time, self._ND_class.c_T0)
    def n_parameters( self,):return len(self.optim_list)
    def n_outputs(self,):return 1

    @property
    def optim_list(self):
        return self._phandler.context.optim_list

    @optim_list.setter
    def optim_list(self, parameters):
        """
        Set the list of parameters to be optimized.
        
        Args:
            parameters (list): List of parameter names to be set when calling simulate or test_vals functions
        
        Raises:
            Exception: If upper and lower boundaries for parameters are not present and options.problem == "inverse"
            Exception: If dispersion parameters have been configured incorrectly
        """
        self._intitialise_phandler(parameters=parameters)
    @property
    def boundaries(self):
        return self._phandler.context.boundaries
    @boundaries.setter
    def boundaries(self, boundaries):
        """
        Set parameter boundaries for optimisation.
        Args:
            boundaries (dict): Upper and lower boundaries in format {"parameter": [lower, upper]}
        Raises:
            TypeError: If boundaries is not a dictionary
            ValueError: If any lower bound is greater than or equal to upper bound
        """
        if isinstance(boundaries, dict) is False:
            return TypeError("boundaries need to be of type dict")
        for key in boundaries.keys():
            if (boundaries[key][0]>= boundaries[key][1]):
                raise ValueError(
                    f'{key}: {boundaries[key][0]} is greater than or equal to {boundaries[key][1]}'
                )
        else:
            self._intitialise_phandler(boundaries=boundaries)
    @property
    def fixed_parameters(self):
        return self._phandler.context.fixed_parameters

    @fixed_parameters.setter
    def fixed_parameters(self, parameter_values):
        """
        Args:
            parameter_values (dict): Dictionary of values that will remain constant with each simulate call
    
        Raises:
            TypeError: If parameter_values is not a dictionary
        """
        if isinstance(parameter_values, dict) is False:
            return TypeError("Argument needs to be a dictionary")
        else:
            
            self._intitialise_phandler(fixed_parameters=parameter_values)
    def _intitialise_phandler(self, **kwargs):
        """
        Initialize the parameter handler with current settings.
        
        Args:
            **kwargs: Keyword arguments including parameters, boundaries, fixed_parameters, options. 
            If the kwarg is not provided, the value already in the instance will be used. 
        
        Behavior:
            Creates and configures the ParameterHandler instance with current parameter settings.
            Sets up dispersion handling, non-dimensionalization, and parameter interface.
            Initializes experiment handler for the specific experiment type.
        
        Raises:
            Various: Depending on parameter validation results
        """
        default={
            "parameters":self.optim_list,
            "boundaries":self.boundaries,
            "fixed_parameters":self.fixed_parameters,
            "options":self._internal_options
        }
        for key in default.keys():
            if key not in kwargs:
                kwargs[key]=default[key]
        self._phandler=ParameterHandler(**kwargs)
        self._param_context=self._phandler.context
        self._disp_context=self._phandler._disp_context
        self._ND_class=sci.NDParams(self._internal_options.experiment_type, self._internal_options.input_params)
        self._internal_options.dispersion=self._disp_context.dispersion
        self._ND_class.construct_function_dict(self._param_context.sim_dict, self._internal_options.experiment_type)
        if self._internal_options.dispersion==True:
            self._disp_class=sci.Dispersion(
                                            self._internal_options.dispersion_bins,
                                            self._disp_context.dispersion_parameters,
                                            self._disp_context.dispersion_distributions,
                                            self.optim_list,
                                            self.fixed_parameters,
                                        )
            functor=self._disp_class.generic_dispersion
        else:
            functor=lambda d, l: (d, l)

        self._param_interface=ParameterInterface(func_dict=self._ND_class.function_dict,
                                                sim_dict=self._param_context.sim_dict,
                                                disp_function=functor, 
                                                gh_values=self._disp_context.GH_values,
                                                optim_list=self.optim_list
                                                )
        self._ExperimentHandler=ExperimentHandler.create(self._internal_options, self._param_interface)
    def experiment_top_hat(self, times, current, **kwargs):
        """
        Apply Fourier top-hat filtering to experimental data using options set in SingleExpermentOptions, or those set by **kwargs
        
        Args:
            times (list): List of timepoints with constant dt
            current (list): List of current values at times
            **kwargs: Additional Fourier filtering options
        
        Returns:
            list: Filtered Fourier transform values based on specified harmonics
        """
        Fourier_options=["Fourier_window", "Fourier_function", "top_hat_width", "Fourier_harmonics"]
        for key in Fourier_options:
            if key not in kwargs:
                kwargs[key]=getattr(self, key)
        return sci.top_hat_filter(times, current, **kwargs)
    def change_normalisation_group(self, parameters, mode): 
        """
        interface to ParameterHandler
        """
        return self._phandler.change_normalisation_group(parameters, mode)
    def get_voltage(self, times, **kwargs):
        """
        Iterface to ExperimentHandler
        """
        if "input_parameters" not in kwargs:
            params=self._input_parameters
        else:
            params=kwargs["input_parameters"]
            kwargs.pop("input_parameters")
        return self._ExperimentHandler.get_voltage(times, params, self._input_parameters,**kwargs)
    def calculate_times(self, **kwargs):
        """
        Iterface to ExperimentHandler
        """
        if "input_parameters" not in kwargs:
            kwargs["input_parameters"]=self._input_parameters
        times=self._ExperimentHandler.calculate_times(kwargs["input_parameters"], **kwargs)
        if "dimensional" not in kwargs:
            return times
        elif kwargs["dimensional"]==False and self._internal_options.experiment_type!="SquareWave":
            return self.nondim_t(times)
        
    def simulate(self, parameters, times):
        """
        Simulate the electrochemical experiment with given parameters.
        
        Args:
            parameters (list): List of parameter values (same length as optim_list)
            times (list): List of time points for simulation
        
        Raises:
            ValueError: If optim_list is not set
            ValueError: If parameters and optim_list have different lengths
            ValueError: If more information is needed to configure Disperion
        
        Returns:
            list: Current values at the specified time points
        """
        if self.optim_list is None:
            raise ValueError("optim_list variable needs to be set, even if it is an empty list")
        if len(parameters) != len(self.optim_list):
            raise ValueError(f"Parameters and optim_list need to be the same length, currently parameters={len(parameters)} and optim_list={len(self.optim_list)}")
        if self._disp_context.dispersion_warning!="":
            raise ValueError(self._disp_context.dispersion_warning)
        if self._internal_options.normalise_parameters == True:
            sim_params = dict(zip(self.optim_list,self._phandler.change_normalisation_group(parameters, "un_norm")))
        else:
            sim_params = dict(zip(self.optim_list, parameters))
        return self._ExperimentHandler.simulate(sim_params, times)
        
    def save_class(self,path):
        """
        Save the experiment class configuration to a JSON file.
        
        Args:
            path (str): File path for saving (automatically adds .json extension if missing)
        
        Behavior:
            Serializes the experiment configuration including options, parameters,
            boundaries, and optimization settings to a JSON file for later reconstruction.
        
        Returns:
            None: Saves to file
        """
        save_dict={"Options":self._internal_options.as_dict(),}
        if self._options_handler is not None:
            save_dict["Options_handler"]={"name":self._options_handler.__name__, "module":self._options_handler.__module__}
        else:
            save_dict["Options_handler"]=self._options_handler
        save_dict["class"]={"name":self.__class__.__name__, "module":self.__class__.__module__}
            
        option_keys=self._internal_options.get_option_names()

        if "tr" in self._input_parameters:
            self._input_parameters.pop("tr")
        save_dict["Experiment_parameters"]=self._input_parameters
        save_dict["optim_list"]=self.optim_list
        save_dict["fixed_parameters"]=self.fixed_parameters
        save_dict["boundaries"]=self.boundaries
        if path[-5:]!=".json":
            path+=".json"
        with open(path, "w") as f:
            json.dump(save_dict, f)
    def __setattr__(self, name, value):
        """
        Custom attribute setter with validation for experiment options.
        
        Behavior:
            Validates certain critical attributes (like input_params and experiment_type)
            and triggers re-initialization of parameter handler when change_options are modified.
        Raises:
            ValueError: If trying to change experiment_type of existing class
            Various: From input parameter validation
        """
        if name=="input_params":
            value=ParameterHandler.validate_input_parameters(value, self._internal_options.experiment_type)
        if name=="experiment_type" and hasattr(self._internal_options, "experiment_type"):
            if value!=self._internal_options.experiment_type:
                raise ValueError("Cannot switch experiment of existing class ({0}), please create a new one".format(self._internal_options.experiment_type))
        super().__setattr__(name, value)
        if name in self._change_options:
            self._intitialise_phandler(options=self._internal_options)
        

    
   
    
   
    