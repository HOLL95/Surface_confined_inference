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
        Initialize a SingleExperiment object, for use with a pints.ForwardProblem interface.
        Args:
            experiment_type (str): Type of experiment (FTACV, PSV, DCV, SquareWave).
            experiment_parameters (dict): Dictionary of experiment input parameters.
        Initialises:
            class: NDClass, responsible for non-dimensionalising parameters in a manner distinct to each technique
            class: OptionsDecorator, responsible for storing different simulation options, and verifying that the provided option is of the correct type, or from the list of accepted options
            variable: _internal_memory, single dictionary responsible for passing class data between functions
            variable: essential_parameters, the parameter keys neccessary for simulation
            variable: _optim_list
        Raises:
            Exception: If the experiment string is not of accepted types
            Exception: The required experimental input parameters for each technique are not found
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
    def dim_i(self, current):return np.multiply(current, self._ND_class.c_I0)
    def dim_e(self, potential):return np.multiply(potential, self._ND_class.c_E0)
    def dim_t(self, time):return np.multiply(time, self._ND_class.c_T0)
    def nondim_i(self, current):return np.divide(current, self._ND_class.c_I0)
    def nondim_e(self, potential):return np.divide(potential, self._ND_class.c_E0)
    def nondim_t(self, time):return np.divide(time, self._ND_class.c_T0)
    def n_parameters( self,):return len(self.optim_list)
    def n_outputs(self,):return 1
    def redimensionalise(self, nondim_dict):
        return self._ND_class.redimensionalise(nondim_dict)

    @property
    def optim_list(self):
        """
        Returns:
            list: internal variable `_optim_list`
        """
        return self._phandler.context.optim_list

    @optim_list.setter
    def optim_list(self, parameters):
        """
        Args:
            parameters (list): list of parameter names to be set when calling the simulate or test_vals functions

        Checks the provided parameters. Will throw an error if
        1) Upper and lower boundaries for the parameter are not present in the `_internal_memory` variable,
        2) Dispersion parameters have been configured incorrectly
        """
        self._intitialise_phandler(parameters=parameters)
    @property
    def boundaries(self):
        """
        Returns:
            dict: internal variable _internal_memory["boundries"]
        """
        return self._phandler.context.boundaries
    @boundaries.setter
    def boundaries(self, boundaries):
        """
        Args:
            boundaries (dict): Upper and lower boundaries of the form "parameter":[Lower, Upper]
        Modifies:
            _internal_memory["boundaries"]
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
            parameter_values (dict): Dictionary of values that will remain the same with each call to simulate
        Modifies:
            _internal_memory["fixed_parameters"]
        """
        if isinstance(parameter_values, dict) is False:
            return TypeError("Argument needs to be a dictionary")
        else:
            
            self._intitialise_phandler(fixed_parameters=parameter_values)
    def _intitialise_phandler(self, **kwargs):
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
        if self._disp_context.dispersion_warning !="":
            if self._warning_raised==False:
                print(ParameterHandler.create_warning(self._disp_context.dispersion_warning))
                self._warning_raised=True
        if self._warning_raised==True:
            if self._disp_context.dispersion_warning=="":
                print(ParameterHandler.create_warning("Parameters set correctly"))
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
        Fourier_options=["Fourier_window", "Fourier_function", "top_hat_width", "Fourier_harmonics"]
        for key in Fourier_options:
            if key not in kwargs:
                kwargs[key]=getattr(self, key)
        return sci.top_hat_filter(times, current, **kwargs)
    def change_normalisation_group(self, parameters, mode): return self._phandler.change_normalisation_group(parameters, mode)
    def get_voltage(self, times, **kwargs):
        if "input_parameters" not in kwargs:
            params=self._input_parameters
        else:
            params=kwargs["input_parameters"]
            kwargs.pop("input_parameters")
        return self._ExperimentHandler.get_voltage(times, params, self._input_parameters,**kwargs)
    def calculate_times(self, **kwargs):
        if "input_parameters" not in kwargs:
            kwargs["input_parameters"]=self._input_parameters
        times=self._ExperimentHandler.calculate_times(kwargs["input_parameters"], **kwargs)
        if "dimensional" not in kwargs:
            return times
        elif kwargs["dimensional"]==False and self._internal_options.experiment_type!="SquareWave":
            return self.nondim_t(times)
        
    def simulate(self, parameters, times):
        """
        Args:
            times (list): list of times to pass to the solver
            parameters (list): list of parameters to be passed to the solver - needs to be the same length as _optim_list, will throw an error if not. The variable `_optim_list` needs to be intiialised
            before this function is called to allow for various santity checks to take place.
        Returns:
            list: A list of current values at the times passed to the `simulate` function
        Raises:
            Exception: If `_optim_list` and the arg `parameters` are not of the same length
            Exception: if the variable `_optim_list` has not been set then no checking of the parameters has occured, and so an error is thrown.

        """
        if self.optim_list is None:
            raise ValueError("optim_list variable needs to be set, even if it is an empty list")
        if len(parameters) != len(self.optim_list):
            raise ValueError(f"Parameters and optim_list need to be the same length, currently parameters={len(parameters)} and optim_list={len(self.optim_list)}")
        if self._internal_options.normalise_parameters == True:
            sim_params = dict(zip(self.optim_list,self.cls.change_normalisation_group(parameters, "un_norm")))
        else:
            sim_params = dict(zip(self.optim_list, parameters))
        return self._ExperimentHandler.simulate(sim_params, times)
        
    def save_class(self,path):
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
        if name=="input_params":
            value=ParameterHandler.validate_input_parameters(value, self._internal_options.experiment_type)
        if name=="experiment_type" and hasattr(self._internal_options, "experiment_type"):
            if value!=self._internal_options.experiment_type:
                raise ValueError("Cannot switch experiment of existing class ({0}), please create a new one".format(self._internal_options.experiment_type))
        super().__setattr__(name, value)
        if name in self._change_options:
            self._intitialise_phandler(options=self._internal_options)
        

    
   
    
   
    