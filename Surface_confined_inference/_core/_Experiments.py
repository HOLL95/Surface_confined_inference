import SurfaceODESolver as sos
import Surface_confined_inference as sci
from Surface_confined_inference._utils import RMSE
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

class SingleExperiment:
    def __init__(self, experiment_type, experiment_parameters, **kwargs):
        """
        Initialize a SingleExperiment object, for use with a pints.ForwardProblem interface.

        Args:
            experiment_type (str): Type of experiment (FTACV, PSV, DCV, SquareWave).
            experiment_parameters (dict): Dictionary of experiment input parameters.
            **kwargs: Additional keyword arguments for options to be passed to Options class. Each of these options will also be set as an attribute of this class for convenience

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

        all_experiments = ["FTACV", "PSV", "DCV", "SquareWave"]
        accepted_arguments = {
            key: [ "area", "Temp", "N_elec", "Surface_coverage"]
            for i in range(0, len(all_experiments))
            for key in all_experiments
        }
        extra_args = {
            "FTACV": ["E_start", "E_reverse","v", "omega", "phase", "delta_E"],
            "PSV": ["Edc","omega", "phase", "delta_E", "num_peaks"],
            "DCV": ["E_start", "E_reverse","v"],
            "SquareWave": [
                "E_start", "E_reverse",
                "scan_increment",
                "sampling_factor",
                "delta_E",
                "v",
                "SW_amplitude",
            ],
        }

        for key in extra_args.keys():
            accepted_arguments[key] += extra_args[key]
        if experiment_type not in all_experiments:
            raise Exception(
                "'{0}' not in list of experiments. Simulated experiments are \n{1}".format(
                    experiment_type, ", ".join(all_experiments)
                )
            )
        self._internal_memory = {
            "input_parameters": experiment_parameters,
            "boundaries": {},
            "fixed_parameters": {},
        }
        kwargs["experiment_type"] = experiment_type
        self._internal_options = OptionsDecorator(**kwargs)
        [
            setattr(self, key, getattr(self._internal_options, key))
            for key in Options().accepted_arguments
        ]

        sci.check_input_dict(
            experiment_parameters,
            accepted_arguments[self._internal_options.experiment_type],
        )
        self._NDclass = sci.NDParams(
            self._internal_options.experiment_type, experiment_parameters
        )
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
        self._optim_list = None

    def calculate_times(self, **kwargs):
        """
        Calculates the time of an experiment e.g. for sythetnic data. Each experiment has an end time, and a timestep.
        FTV/DCV: Defined by the voltage window (V) and scan rate (V s^-1), doubled as you go up and down
        PSV: Defined by the number of oscillations as part of the initialisation of the Experiment class
        SquareWave: Defined per square wave "oscillation"

        Args:
            sampling_factor (int, default=200): defines the timestep dt. For the experiments involving a sinusoid (PSV, FTACV, SquareWave), this is taken to mean
            X points per oscillation. For DCV, it is taken to mean X points per second.
            Dimensional: controls whether the time is returned in dimensional or non-dimensional form
        Returns:
            np.ndarray: Array of times
        Raises:
            NotImplementedError: Squarewave currently not implemented
        """
        params = self._internal_memory["input_parameters"]
        if "sampling_factor" not in kwargs:
            sampling_factor=200
        else:
            sampling_factor=kwargs["sampling_factor"]
        if "dimensional" not in kwargs:
            dimensional=False
        else:
            dimensional=kwargs["dimensional"]
        if (
            self._internal_options.experiment_type == "FTACV"
            or self._internal_options.experiment_type == "DCV"
        ):
            end_time = 2 * abs(params["E_start"] - params["E_reverse"]) / params["v"]
        elif self._internal_options.experiment_type == "PSV":
            end_time = params["num_peaks"] / params["omega"]
        elif self._internal_options.experiment_type == "SquareWave":
            raise NotImplementedError()
        if self._internal_options.experiment_type == "DCV":
            dt = 1 / sampling_factor
        elif (
            self._internal_options.experiment_type == "FTACV"
            or self._internal_options.experiment_type == "PSV"
        ):
            dt = 1 / (sampling_factor * params["omega"])
        elif self._internal_options.experiment_type == "SquareWave":
            raise NotImplementedError()

        times = np.arange(0, end_time, dt)

        if dimensional == False:
            times = self.nondim_t(times)

        return times

    def dim_i(self, current):
        """
        Convenience function to return dimensional current
        Args:
            current (number):list of non-dimensional current values
        Returns:
            np.ndarray: list of dimensional current values
        """
        return np.multiply(current, self._NDclass.c_I0)

    def dim_e(self, potential):
        """
        Convenience function to return dimensional potential
        Args:
            potential (number):list of non-dimensional potential values
        Returns:
            np.ndarray: list of dimensional potential values
        """
        return np.multiply(potential, self._NDclass.c_E0)

    def dim_t(self, time):
        """
        Convenience function to return dimensional time
        Args:
            time (number):list of non-dimensional time values
        Returns:
            np.ndarray: list of dimensional time values
        """

        return np.multiply(time, self._NDclass.c_T0)

    def nondim_i(self, current):
        """
        Convenience function to return non-dimensional current
        Args:
            current (number):list of dimensional current values
        Returns:
            np.ndarray: list of non-dimensional current values
        """
        return np.divide(current, self._NDclass.c_I0)

    def nondim_e(self, potential):
        """
        Convenience function to return non-dimensional potential
        Args:
            potential (number):list of dimensional potential values
        Returns:
            np.ndarray: list of non-dimensional potential values
        """
        return np.divide(potential, self._NDclass.c_E0)

    def nondim_t(self, time):
        """
        Convenience function to return non-dimensional time
        Args:
            time (number):list of dimensional time values
        Returns:
            np.ndarray: list of non-dimensional time values
        """
        return np.divide(time, self._NDclass.c_T0)

    def n_parameters(
        self,
    ):
        """
        Required for pints to return the dimension of the optimisation problem

        Returns:
            int: length of the `_optim_list` variable
        """
        return len(self._optim_list)
    def n_outputs(self,):
        return 1

    def get_voltage(self, times, **kwargs):
        """
        Args:
            times (list): list of timepoints
            input_parameters (dict, optional): dictionary of input parameters
            dimensional (bool, optional): whether or not the times are in dimensional format or not
        Returns:
            list: list of potential values at the provided time points. By default, these values will be in dimensional format (V)
        """
        if "input_parameters" not in kwargs:
            input_parameters=None
        else:
            input_parameters=kwargs["input_parameters"]
        if "dimensional" not in kwargs:
            kwargs["dimensional"]=False
        if input_parameters == None:
            input_parameters = self._internal_memory["input_parameters"]
        else:
            sci.check_input_dict(
                input_parameters, list(self._internal_memory["input_parameters"].keys())
            )
        input_parameters=copy.deepcopy(input_parameters)
        input_parameters = self.validate_input_parameters(input_parameters)
        if "tr" in input_parameters:
            if kwargs["dimensional"]==True:
                input_parameters["tr"]*=self._NDclass.c_T0
        input_parameters["omega"] *= 2 * np.pi
        return sos.potential(times, input_parameters)

    def dispersion_checking(self, optim_list):
        """

        Function called when defining `_optim_list` variable, to check for the presence of dispersed parameters.
        Args:
            optim_list (list): list of str variables
        Initialises:
            class: Dispersion, responsible for veryifying that the necessary dispersion parameters are present
        Modifies:
            Options - dispersion, sets True or False
        Raises:
            Exception: if the correct distribution parameters (for example mean and standard deviation) are not provided in `_optim_list` or `fixed_parameters`

        Scans through all parameter names (i.e. those defined in optim_list and fixed_parameters) to check for the associated dispersion parameter
        Dispersion is defined by setting parameters `{parameter}_{dispersion_flag}`. For example, if E0 has a normal distribution of values, we need `E0_mean` and `E0_std` variables
        If a `{dispersion_flag}` is found in `all_parameters` (which is the combination of `_optim_list` and `fixed_parameters`) we create a new `{variable}:list` pair in `disp_param_dict`
        This means the other `{dispersion_flag}` associated with that parameter will be added to the list associated with that `{variable}`
        Once all parameters have been scanned through, we check to see which distribution each `{variable}:list` pair is associated with, which is achieved through set comparison
        The `Dispersion` class is then initialised using the following information:
        The number of bins for each dipseresed parameter (set in `_internal_options`)
        The distribution for each parameter
        The `{variables}` with which each distribution is associated

        The required information is also initiliased for Gauss-Hermite quadratture, if required. If no normal distribution is found, then the GH_quadrature information is set to None

        """
        all_parameters = list(
            set(optim_list + list(self._internal_memory["fixed_parameters"].keys()))
        )
        disp_flags = [
            ["mean", "std"],
            ["logmean", "logscale"],
            ["lower", "upper"],
            ["mean", "std", "skew"],
            ["logupper", "loglower"],
        ]  # Set not name must be unique
        all_disp_flags = list(set(itertools.chain(*disp_flags)))
        disp_check = [
            any(flag in param for flag in all_disp_flags) for param in all_parameters
        ]
        if any(disp_check):
            self._internal_options.dispersion = True

            distribution_names = [
                "normal",
                "lognormal",
                "uniform",
                "skewed_normal",
                "log_uniform",
            ]
            distribution_dict = dict(
                zip(distribution_names, [set(x) for x in disp_flags])
            )
            disp_param_dict = {}
            for i in range(0, len(all_parameters)):
                for j in range(0, len(all_disp_flags)):
                    if all_disp_flags[j] in all_parameters[i]:
                        try:
                            m = re.search(
                                ".+?(?=_" + all_disp_flags[j] + ")", all_parameters[i]
                            )
                            param = m.group(0)
                            if param in disp_param_dict:
                                disp_param_dict[param].append(all_disp_flags[j])
                            else:
                                disp_param_dict[param] = [all_disp_flags[j]]
                        except:
                            continue
            dispersion_parameters = list(disp_param_dict.keys())
            dispersion_distributions = []
            distribution_keys = list(distribution_dict.keys())
            for param in dispersion_parameters:
                param_set = set(disp_param_dict[param])
                found_distribution = False
                difference = [
                    len(distribution_dict[key] - param_set) for key in distribution_keys
                ]
                min_diff = min(difference)
                best_guess = distribution_keys[difference.index(min_diff)]
                if min_diff != 0:
                    raise Exception(
                        f"Dispersed parameter {param}, assuming the distribution is {best_guess} requires {[f'{param}_{x}' for x in list(distribution_dict[best_guess])]} but only {[f'{param}_{x}' for x in list(param_set)]} found"
                    )
                else:
                    dispersion_distributions.append(best_guess)
            orig_GH_value = self._internal_options.GH_quadrature
            if "normal" not in dispersion_distributions:
                self._internal_options.GH_quadrature = False
            if self._internal_options.GH_quadrature == True:
                self._internal_memory["GH_values"] = self.GH_setup(
                    dispersion_distributions
                )
            else:
                self._internal_memory["GH_values"] = None
            self._internal_options.GH_quadrature = orig_GH_value
            self._disp_class = sci.Dispersion(
                self._internal_options.dispersion_bins,
                dispersion_parameters,
                dispersion_distributions,
                all_parameters,
                self._internal_memory["fixed_parameters"],
            )
        else:
            self._internal_options.dispersion = False

    @property
    def optim_list(self):
        """
        Returns:
            list: internal variable `_optim_list`
        """
        return self._optim_list

    @optim_list.setter
    def optim_list(self, parameters):
        """
        Args:
            parameters (list): list of parameter names to be set when calling the simulate or test_vals functions

        Checks the provided parameters. Will throw an error if
        1) Upper and lower boundaries for the parameter are not present in the `_internal_memory` variable,
        2) Dispersion parameters have been configured incorrectly
        """
        missing_parameters = []
        for i in range(0, len(parameters)):
            if parameters[i] not in self._internal_memory["boundaries"]:
                missing_parameters.append(parameters[i])
        if len(missing_parameters) > 0:
            raise Exception(
                "Need to define boundaries for:\n %s" % ", ".join(missing_parameters)
            )
        self.dispersion_checking(parameters)
        if "cap_phase" not in parameters:
            self._internal_options.phase_only = True
        self.simulation_dict_construction(parameters)
        self._optim_list = parameters
        """if self.simulation_options["method"]=="square_wave":
            if "SWV_constant" in optim_list:
                self.simulation_options["SWV_polynomial_capacitance"]=True
        if "Upper_lambda" in optim_list:
            self.simulation_options["Marcus_kinetics"]=True
        if self.simulation_options["Marcus_kinetics"]==True:
            if "alpha" in optim_list:
                raise Exception("Currently Marcus kinetics are symmetric, so Alpha in meaningless")"""

    @property
    def boundaries(self):
        """
        Returns:
            dict: internal variable _internal_memory["boundries"]
        """
        return self._internal_memory["boundaries"]

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

        else:
            self._internal_memory["boundaries"] = boundaries
        for key in self._internal_memory["boundaries"].keys():
            if (
                self._internal_memory["boundaries"][key][0]
                >= self._internal_memory["boundaries"][key][1]
            ):
                raise Exception(
                    f'{key}: {self._internal_memory["boundaries"][key][0]} is greater than or equal to {self._internal_memory["boundaries"][key][1]}'
                )

    def validate_input_parameters(self, inputs):
        """
        Args:
            inputs (dict): Dictionary of parameters
        Returns:
            dict : modified dictionary of input parameters according to the method assigned to the class
        """
        if self._internal_options.experiment_type == "DCV":
            inputs["tr"] = self.nondim_t(
                abs(inputs["E_reverse"] - inputs["E_start"]) / inputs["v"]
            )
            inputs["omega"] = 0
            inputs["delta_E"] = 0
        elif self._internal_options.experiment_type == "FTACV":
            inputs["tr"] =  self.nondim_t(
                abs(inputs["E_reverse"] - inputs["E_start"]) / inputs["v"]
            )
        elif self._internal_options.experiment_type == "PSV":
            inputs["E_reverse"]=inputs["Edc"]
            inputs["tr"] = -1
            inputs["v"] = 0
        return inputs

    def simulation_dict_construction(self, parameters):
        """
        Args:
            parameters (list): passed when we set _optim_list, a list of str objects referring to parameter name
        Initialises/modifies:
            _internal_memory["simualtion_dict"]
        Raises:
            Exception: If parameters set in `essential_parameters` are not found in either `_optim_list` or `fixed_parameters`
        1) All parameters in `_optim_list` are set to `None`, as they will be changed on each simulation step
        2) All parameter values in `fixed_parameters` are passed to `simulation_dict`
        3) Checks to see if the parameters in _essential_parameters are present in `simulation_dict`
            a) If dispersion is present, then the parameter will be set as part of the `dispersion_simulator` function, so it is set to 0
            b) Else, if the missing parameter is one of the nonlinear capacitance parameters these are set to 0 for convenience
            c) Else an error is thrown
        4) Input parameters are fixed for each technique as appropriate using `validate_input_parameters`
        5) Each parameter is then assigned its own nondimensionalisation function using `NDclass`

        """
        if "simulation_dict" in self._internal_memory:
            simulation_dict = self._internal_memory["simulation_dict"]
        else:
            simulation_dict = copy.deepcopy(self._internal_memory["input_parameters"])

        missing_parameters = []
        all_parameters = list(
            set(parameters + list(self._internal_memory["fixed_parameters"].keys()))
        )
        for key in all_parameters:
            if key in parameters:
                simulation_dict[key] = None
                if key in self._internal_memory["fixed_parameters"]:
                    warn(
                        "%s in both fixed_parameters and optimisation_list, the fixed parameter will be ignored!"
                        % key
                    )
            elif key in self._internal_memory["fixed_parameters"]:
                simulation_dict[key] = self._internal_memory["fixed_parameters"][key]
        for key in self._essential_parameters:
            if key not in simulation_dict:
                if self._internal_options.dispersion == True:
                    if key in self._disp_class.dispersion_parameters:
                        simulation_dict[key] = 0
                        continue
                if "CdlE" in key:
                    simulation_dict[key] = 0
                else:
                    missing_parameters.append(key)
        if len(missing_parameters) > 0:
            raise Exception(
                "The following parameters either need to be set in optim_list, or set at a value using the fixed_parameters variable\n{0}".format(
                    ", ".join(missing_parameters)
                )
            )

        elif self._internal_options.kinetics != "Marcus":
            simulation_dict["Marcus_flag"] = 0
        simulation_dict = self.validate_input_parameters(simulation_dict)

        self._internal_memory["simulation_dict"] = simulation_dict

        self._NDclass.construct_function_dict(self._internal_memory["simulation_dict"])

    @property
    def fixed_parameters(self):
        return self._internal_memory["fixed_parameters"]

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
            self._internal_memory["fixed_parameters"] = parameter_values

    def GH_setup(self,dispersion_distributions):
        """
        Args:
            dispersion_distributions: list - list of distributions involved in dispersion
        Returns:
            dict: dictionary of nodes, weights and normal weights required for calculating Gauss-Hermite quadrature
        We assume here that for n>1 normally dispersed parameters then the order of the integral
        will be the same for all
        """
        try:
            disp_idx = dispersion_distributions.index("normal")
        except:
            raise KeyError("No normal distributions for GH quadrature")
        nodes = self._internal_options.dispersion_bins[disp_idx]
        labels = ["nodes", "weights", "normal_weights"]
        nodes, weights = np.polynomial.hermite.hermgauss(nodes)
        normal_weights = np.multiply(1 / math.sqrt(math.pi), weights)
        return dict(zip(labels, [nodes, weights, normal_weights]))

    def normalise(self, norm, boundaries):
        """
        Args:
            norm (number): value to be normalised
            boundaries (list): upper and lower bounds, in the format [lower, upper]
        Returns:
            number: Value normalised to value between 0 and 1 relative to the provided boundaries
        """
        return (norm - boundaries[0]) / (boundaries[1] - boundaries[0])

    def un_normalise(self, norm, boundaries):
        """
        Args:
            norm (number): value to be un-normalised
            boundaries (list): upper and lower bounds, in the format [lower, upper] used in the original normalisation
        Returns:
            number: Normalised value is returned to its original value, given the same boundaries used in the initial normalisation
        """
        return (norm * (boundaries[1] - boundaries[0])) + boundaries[0]

    def change_normalisation_group(self, parameters, method):
        """
        Args:
            parameters (list): list of numbers
            method (str): flag indicating if the value is to be normalised or un-normalised
        Returns:
            list: list of appropriately transformed values

        Note - this convenience method assumes the values are the parameters associated with the current parameters in optim_list
        """
        normed_params = copy.deepcopy(parameters)
        if method == "un_norm":
            for i in range(0, len(parameters)):
                normed_params[i] = self.un_normalise(
                    normed_params[i],
                    [
                        self._internal_memory["boundaries"][self._optim_list[i]][0],
                        self._internal_memory["boundaries"][self._optim_list[i]][1],
                    ],
                )
        elif method == "norm":
            for i in range(0, len(parameters)):
                normed_params[i] = self.normalise(
                    normed_params[i],
                    [
                        self._internal_memory["boundaries"][self._optim_list[i]][0],
                        self._internal_memory["boundaries"][self._optim_list[i]][1],
                    ],
                )
        return normed_params

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
        nd_dict = self.nondimensionalise(sim_params)
        disp_params, self._values, self._weights = self._disp_class.generic_dispersion(
            self._internal_memory["simulation_dict"], self._internal_memory["GH_values"]
        )
        time_series = np.zeros(len(times))
        self._disp_test = []
        for i in range(0, len(self._weights)):
            for j in range(0, len(disp_params)):
                self._internal_memory["simulation_dict"][disp_params[j]] = self._values[i][j]
            nd_dict = self.nondimensionalise(sim_params)
            time_series_current = np.array(solver(times, nd_dict))[0, :]
            if self._internal_options.dispersion_test == True:
                self._disp_test.append(time_series_current)
            time_series = np.add(
                time_series, np.multiply(time_series_current, np.prod(self._weights[i]))
            )
        return time_series

    def nondimensionalise(self, sim_params):
        """
        Args:
            sim_params (dict): dictionary of parameter values with the same keys as _optim_list
        Modifies:
            _internal_memory["simulation_dict"] with the values from sim_params
        Returns:
            dict: Dictionary of appropriately non-dimensionalised parameters
        """
        nd_dict = {}
        for key in self._optim_list:
            self._internal_memory["simulation_dict"][key] = sim_params[key]
        for key in self._internal_memory["simulation_dict"].keys():
            nd_dict[key] = self._NDclass.function_dict[key](
                self._internal_memory["simulation_dict"][key]
            )
        if self._internal_options.phase_only == True:
            self._internal_memory["simulation_dict"]["cap_phase"] = (
                self._internal_memory["simulation_dict"]["phase"]
            )
            nd_dict["cap_phase"] = self._internal_memory["simulation_dict"]["phase"]
        return nd_dict
   
    @sci._utils.temporary_options()
    def FTsimulate(self,parameters, times, **kwargs):
        Fourier_options=["Fourier_window", "Fourier_function", "top_hat_width", "Fourier_harmonics"]
        for key in Fourier_options:
            if key not in kwargs:
                kwargs[key]=getattr(self, key)
        self.optim_list=self._optim_list
        current=self.simulate(parameters, times)
        return sci.top_hat_filter(times, current, **kwargs)
    @sci._utils.temporary_options(normalise_parameters=False)
    def Dimensionalsimulate(self, parameters, times):
        self.optim_list=self._optim_list
        return self.simulate(parameters, times)
        
    def save_class(self,path):

        dict_class=vars(self)
        save_dict={"Options":{}}
        option_keys=Options().accepted_arguments.keys()
        for key in dict_class:
            if key in option_keys and key!="experiment_type":
                save_dict["Options"][key]=dict_class[key]
        save_dict["experiment_type"]=dict_class["experiment_type"]
        save_dict["Experiment_parameters"]=self._internal_memory["input_parameters"]
        save_dict["optim_list"]=self._optim_list
        save_dict["fixed_parameters"]=self._internal_memory["fixed_parameters"]
        save_dict["boundaries"]=self._internal_memory["boundaries"]
        if path[-5:]!=".json":
            path+=".json"
        with open(path, "w") as f:
            json.dump(save_dict, f)
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
        if self._optim_list is None:
            raise Exception(
                "optim_list variable needs to be set, even if it is an empty list"
            )
        if len(parameters) != len(self._optim_list):
            raise Exception(
                f"Parameters and optim_list need to be the same length, currently parameters={len(parameters)} and optim_list={len(self._optim_list)}"
            )
        if self._internal_options.normalise_parameters == True:
            sim_params = dict(
                zip(
                    self._optim_list,
                    self.change_normalisation_group(parameters, "un_norm"),
                )
            )

        else:
            sim_params = dict(zip(self._optim_list, parameters))
        if self._internal_options.experiment_type != "SquareWave":
            solver = sos.ODEsimulate
        if self._internal_options.dispersion == True:
            current = self.dispersion_simulator(solver,  sim_params, times)
        else:
            nd_dict = self.nondimensionalise(sim_params)
            current = np.array(solver(times, nd_dict))[0, :]


        return current
    @sci._utils.temporary_options(normalise_parameters=True)
    def Current_optimisation(self, time_data, current_data,**kwargs):
        if "tolerance" not in kwargs:
            kwargs["tolerance"]=1e-6
        if "method" not in kwargs:
            kwargs["method"]=pints.CMAES
        if "runs" not in kwargs:
            kwargs["runs"]=1
        if "unchanged_iterations" not in kwargs:
            kwargs["unchanged_iterations"]=200
        if "parallel" not in kwargs:
            kwargs["parallel"]=True
        if "dimensional" not in kwargs:
            kwargs["dimensional"]=True
        if "Fourier_filter" not in kwargs:
            kwargs["Fourier_filter"]=False
        if "sigma0" not in kwargs:
            kwargs["sigma0"]=0.075
        if "starting_point" not in kwargs:
            kwargs["starting_point"]="random"
        if "save_to_directory" not in kwargs:
            kwargs["save_to_directory"]=False
        elif kwargs["save_to_directory"]==False:
            if "save_csv" in kwargs:
                print("Warning - not saving run")
        elif isinstance(kwargs["save_to_directory"], str) is False:
            raise TypeError("If you want to save a run you need to provide a string name for the directory")
        else:
            if "save_csv" not in kwargs:
                kwargs["save_csv"]=False
        if kwargs["dimensional"]==True:
            time_data=self.nondim_t(time_data)
            current_data=self.nondim_i(current_data)
        problem=pints.SingleOutputProblem(self, time_data, current_data)
        if kwargs["Fourier_filter"]==True:
            log_Likelihood=sci.FourierGaussianLogLikelihood(problem)
            error=1000
        else:
            log_Likelihood=pints.GaussianLogLikelihood(problem)
            error=100
        boundaries=pints.RectangularBoundaries(
                                                np.zeros(log_Likelihood.n_parameters()), 
                                                np.append(
                                                    np.ones(len(self._optim_list)), 
                                                    [error for x in range(0, log_Likelihood._no)]
                                                )
                                            )
        if kwargs["starting_point"]=="random":
            x0=np.random.random(log_Likelihood.n_parameters())
        else:
            x0=kwargs["starting_point"]
        if kwargs["sigma0"] is not list:
            sigma0=[kwargs["sigma0"] for x in range(0,log_Likelihood.n_parameters())]
        else:
            sigma0=kwargs["sigma0"]
        best_score=-1e20
        scores=np.zeros(kwargs["runs"])
        parameters=np.zeros((kwargs["runs"], log_Likelihood.n_parameters()))
        noises=np.zeros(kwargs["runs"])
        currents=np.zeros((kwargs["runs"], len(time_data)))
        for i in range(0, kwargs["runs"]):
            optimiser=pints.OptimisationController(log_Likelihood, x0, sigma0=sigma0, boundaries=boundaries, method=kwargs["method"])
            optimiser.set_max_unchanged_iterations(iterations=kwargs["unchanged_iterations"], threshold=kwargs["tolerance"])
            optimiser.set_parallel(kwargs["parallel"])
            
            xbest, fbest=optimiser.run()
            scores[i]=fbest
            dim_params=list(self.change_normalisation_group(xbest[:-log_Likelihood._no], "un_norm"))
            parameters[i,:-log_Likelihood._no]=dim_params
            parameters[i, -log_Likelihood._no:]=xbest[-log_Likelihood._no:]
            if kwargs["save_to_directory"] is not False:
                currents[i,:]=self.dim_i(self.simulate(xbest[:-log_Likelihood._no], time_data))
                if i==0:
                    save_times=self.dim_t(time_data)
                    Path(kwargs["save_to_directory"]).mkdir(parents=True, exist_ok=True)
                    voltage=self.get_voltage(save_times, dimensional=True)
                    if self._internal_options.experiment_type=="FTACV":
                        DC_params=copy.deepcopy(self._internal_memory["input_parameters"])
                        DC_params["delta_E"]=0
                        DC_voltage=self.get_voltage(save_times, dimensional=True, input_parameters=DC_params)
                    else:
                        DC_voltage=None
        sorted_idx=np.flip(np.argsort(scores))
        sorted_params=[list(parameters[x,:]) for x in sorted_idx]
        if kwargs["save_to_directory"] is not False:
            parameters=np.array(sorted_params)
            sci.plot.save_results(save_times, 
                                    voltage, 
                                    self.dim_i(current_data), 
                                    currents, 
                                    kwargs["save_to_directory"], 
                                    self._internal_options.experiment_type, 
                                    self._internal_memory["boundaries"],
                                    save_csv=kwargs["save_csv"],
                                    optim_list=self._optim_list, 
                                    fixed_parameters=self.fixed_parameters,
                                    score=np.flip(sorted(scores)), 
                                    parameters=parameters,
                                    DC_voltage=DC_voltage
                                    )
        
                
        if kwargs["runs"]==1:
            return dim_params+list(xbest[-log_Likelihood._no:])
        else:
            return list(sorted_params)

    def __setattr__(self, name, value):
        """
        Args:
            name (str): The name of the attribute to set.
            value (Any): The value to assign to the attribute.

        If the attribute name is 'OptionsDecorator', it sets the attribute using the superclass's __setattr__ method.

        If the attribute name is in the list of accepted arguments from the Options class, it sets the attribute on the instance's internal options as well as the instance itself.
        This allows for the checking of the value of the option to take place

        For all other attribute names, it defaults to the superclass's __setattr__ method.
        """
        if name in ["OptionsDecorator"]:
            super().__setattr__(name, value)
        elif name in Options().accepted_arguments:
            setattr(self._internal_options, name, value)
            super().__setattr__(name, value)
        else:
            if name not in Options().other_attributes:
                print("Warning: {0} is not in the list of accepted options and will not change the behaviour of the simulations".format(name))
            super().__setattr__(name, value)


class Options:
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Arbitrary keyword arguments representing option names and values. Only accepted arguments defined in accepted_arguments are allowed.
        """
        self.accepted_arguments = {
            "experiment_type": {"type": str, "default": None},
            "GH_quadrature": {"type": bool, "default": True},
            "phase_only": {"type": bool, "default": True},
            "normalise_parameters": {"type": bool, "default": False},
            "kinetics": {
                "args": ["ButlerVolmer", "Marcus", "Nernst"],
                "default": "ButlerVolmer",
            },
            "dispersion": {"type": bool, "default": False},
            "dispersion_bins": {"type": collections.abc.Sequence, "default": []},
            "Fourier_fitting":{"type":bool, "default":False},
            "Fourier_function": {
                "args": ["composite", "abs", "real", "imaginary", "inverse"],
                "default": "abs",
            },
            "Fourier_harmonics": {
                "type": collections.abc.Sequence,
                "default": list(range(0, 10)),
            },
            "Fourier_window": {"args": ["hanning", False], "default": "hanning"},
            "top_hat_width": {"type": numbers.Number, "default": 0.5},
            "dispersion_test": {"type": bool, "default": False},
        }
        self.other_attributes=["_internal_memory", "_internal_options", "_NDclass", "_essential_parameters", "_optim_list", "boundaries", "fixed_parameters", "optim_list", "_disp_class", "_weights", "_disp_test", "_values"]
        if len(kwargs) == 0:
            self.options_dict = self.accepted_arguments
        else:
            self.options_dict = {}
            for kwarg in kwargs:
                if kwarg not in self.accepted_arguments:
                    raise Exception(f"{kwarg} not an accepted option")
            for key in self.accepted_arguments:
                if key in kwargs:
                    self.options_dict[key] = kwargs[key]
                else:
                    self.options_dict[key] = self.accepted_arguments[key]["default"]

    def checker(self, name, value, defaults):
        """
        Args:
            name (str): The name of the option.
            value (Any): The value to check.
            defaults (dict): The dictionary containing type or allowed arguments information for the option.
        """
        if "type" in defaults:
            if isinstance(defaults["type"], list) is not True:
                type_list = [defaults["type"]]
            else:
                type_list = defaults["type"]
            type_error = True
            for current_type in type_list:

                if isinstance(value, current_type) is True:
                    type_error = False
            if type_error == True:
                raise TypeError("{0} must be of type".format(name), defaults["type"])
        elif "args" in defaults:
            if value not in defaults["args"]:

                raise Exception(
                    "Value '{0}' not part of the following allowed arguments:\n{1}".format(
                        value, ", ".join(defaults["args"])
                    )
                )


class OptionsDecorator:
    def __init__(self, **kwargs):
        """
        Args:
           **kwargs: Arbitrary keyword arguments representing option names and values.
        """
        self.options = Options(**kwargs)

    def __getattr__(self, name):
        """

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            Any: The value of the requested attribute.

        Raises:
            AttributeError: If the attribute does not exist in the options dictionary.
        """
        if name in self.options.options_dict:
            return self.options.options_dict[name]
        raise AttributeError(f"Options has no attribute '{name}'")

    def __setattr__(self, name, value):
        """
        Args:
            name (str): The name of the attribute to set.
            value (Any): The value to assign to the attribute.

        Raises:
            AttributeError: If the attribute does not exist in the options dictionary.
        """
        if name in [
            "options"
        ]:  # To handle the initialization of the 'options' attribute
            super().__setattr__(name, value)
        elif name in self.options.options_dict:
            self.options.checker(name, value, self.options.accepted_arguments[name])
            self.options.options_dict[name] = value
        else:
            raise AttributeError(f"Options has no attribute '{name}'")



