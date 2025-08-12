import SurfaceODESolver as sos
import Surface_confined_inference as sci
from Surface_confined_inference._utils import RMSE
from ._Handler._ParameterHandler import ParameterHandler
from ._Handler._BaseVoltammetry import ExperimentHandler, SolverContext
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
     _change_options=["experiment_type", "GH_quadrature", "dispersion_bins", "phase_only", "square_wave_return"]
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
        for key in extra_args.keys():
            accepted_arguments[key] += extra_args[key]
        kwargs["experiment_type"] = experiment_type
        self._input_parameters=experiment_parameters
        if options_handler is None:
            options_handler=None
        self._options_handler=options_handler
        self._options_class = sci.SingleExperimentOptions(options_handler=options_handler,**kwargs)
        self._phandler=None
        self._internal_options=self._options_class._experiment_options
        self.experiment_type=self._internal_options.experiment_type
        self._phandler=ParameterHandler([], {}, self._internal_options, {})
        experiment_parameters=self._phandler.validate_input_parameters(experiment_parameters)
        self._NDclass = sci.NDParams(
            self._internal_options.experiment_type, experiment_parameters
        )

        if experiment_type!="SquareWave":
            self._essential_parameters = [
                "E0",
                "k0",
                "Cdl",
                "gamma",
                "CdlE1",
                "CdlE2",
                "CdlE3",
                "alpha",
                "Ru",
                "phase",
            ]
        else:
            self._essential_parameters=[
                "E0",
                "k0",
                "gamma",
                "alpha",
                "Cdl",
                "CdlE1",
                "CdlE2",
                "CdlE3",
                "phase"
            ]
    def dim_i(self, current):return np.multiply(current, self._NDclass.c_I0)
    def dim_e(self, potential):return np.multiply(potential, self._NDclass.c_E0)
    def dim_t(self, time):return np.multiply(time, self._NDclass.c_T0)
    def nondim_i(self, current):return np.divide(current, self._NDclass.c_I0)
    def nondim_e(self, potential):return np.divide(potential, self._NDclass.c_E0)
    def nondim_t(self, time):return np.divide(time, self._NDclass.c_T0)
    def n_parameters( self,):return len(self.optim_list)
    def n_outputs(self,):return 1
    def redimensionalise(self, nondim_dict):
        return self._NDclass.redimensionalise(nondim_dict)

    @property
    def optim_list(self):
        """
        Returns:
            list: internal variable `_optim_list`
        """
        return self._phandler._optim_list

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
        return self._phandler.boundaries
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
        return self._phandler.fixed_parameters

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
            "parameters":self._phandler.optim_list,
            "boundaries":self._phandler.boundaries,
            "fixed_parameters":self._phandler.fixed_parameters,
            "options":self._internal_options
        }
        for key in default.keys():
            if key not in kwargs:
                kwargs[key]=default[key]
        self._phandler=ParameterHandler(**kwargs)
        self._internal_options.dispersion=self._phandler.options.dispersion
        self._ND_class.construct_function_dict(self._phandler.sim_dict, self._internal_options.experiment_type)
        self._SolverContext=SolverContext(self._ND_class.function_dict,
                                                self._phandler.sim_dict
                                                self._phandler.disp_class, 
                                                self._phandler.GH_list,
                                                self.optim_list
                                                )
        self._ExperimentHandler=ExperimentHandler(self._internal_options, self._SolverContext)
    def experiment_top_hat(self, times, current, **kwargs):
        Fourier_options=["Fourier_window", "Fourier_function", "top_hat_width", "Fourier_harmonics"]
        for key in Fourier_options:
            if key not in kwargs:
                kwargs[key]=getattr(self, key)
        return sci.top_hat_filter(times, current, **kwargs)
    def change_normalisation_group(self, parameters, mode): return self._phandler.change_normalisation_group(parameters, moder)
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
        if self.options.normalise_parameters == True:
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

        
        save_dict["Experiment_parameters"]=self._internal_memory["input_parameters"]
        save_dict["optim_list"]=self.optim_list
        save_dict["fixed_parameters"]=self._internal_memory["fixed_parameters"]
        save_dict["boundaries"]=self._internal_memory["boundaries"]
        if path[-5:]!=".json":
            path+=".json"
        with open(path, "w") as f:
            json.dump(save_dict, f)
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in self._change_options:
            self._intitialise_phandler(options=self._internal_options)

    
   
    
   
    