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
class SingleExperiment(sci.BaseExperiment,sci.OptionsAwareMixin):
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

        all_experiments = ["FTACV", "PSV", "DCV", "SquareWave"]
        accepted_arguments = {
            key: [ "area", "Temp", "N_elec", "Surface_coverage"]
            for i in range(0, len(all_experiments))
            for key in all_experiments
        }
        extra_args = {
            "FTACV": ["E_start", "E_reverse","v", "omega", "phase", "delta_E"],
            "PSV": ["Edc","omega", "phase", "delta_E"],
            "DCV": ["E_start", "E_reverse","v"],
            "SquareWave": [
                "omega",
                "E_start", 
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
        if options_handler is None:
            options_handler=None
        self._options_handler=options_handler
        self._options_class = sci.SingleExperimentOptions(options_handler=options_handler,**kwargs)

        self._internal_options=self._options_class._experiment_options
        self.experiment_type=self._internal_options.experiment_type
        sci.check_input_dict(
            experiment_parameters,
            accepted_arguments[self._internal_options.experiment_type],
            optional_arguments=["phase_flag"]
        )
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
            self.SW_sampling()
        self._optim_list = []

    def calculate_times(self, **kwargs):
        """
        Calculates the time of an experiment e.g. for sythetnic data. Each experiment has an end time, and a timestep.
        FTV/DCV: Defined by the voltage window (V) and scan rate (V s^-1), doubled as you go up and down
        PSV: Defined by the number of oscillations as part of the initialisation of the Experiment class
        SquareWave: Defined per square wave "oscillation"

        Args:
            sampling_factor (int, default=200): defines the timestep dt. For the experiments involving a sinusoid (PSV, FTACV, SquareWave), this is taken to mean
            X points per oscillation. For DCV, it is taken to mean X points per second.
            input_parameters (dict, default=self._internal_memory["input_parameters"]): Defines the parameters used to calculate the time
            Dimensional: controls whether the time is returned in dimensional or non-dimensional form
        Returns:
            np.ndarray: Array of times
        Raises:
            NotImplementedError: Squarewave currently not implemented
        """
        if "input_parameters" not in kwargs:
            params = self._internal_memory["input_parameters"]
        else:
            params=kwargs["input_parameters"]
       
        if self._internal_options.experiment_type == "SquareWave":
            
            end_time=(abs(params["delta_E"]/params["scan_increment"])*params["sampling_factor"])
            dt=1
            return  np.arange(0, end_time, dt)
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
            if "PSV_num_peaks" not in kwargs:
                kwargs["PSV_num_peaks"]=50
            end_time = kwargs["PSV_num_peaks"]/ params["omega"]
        if self._internal_options.experiment_type == "SquareWave":
            
            end_time=(abs(params["delta_E"]/params["scan_increment"])*params["sampling_factor"])
            dt=1
        if self._internal_options.experiment_type == "DCV":
            dt = 1 / (sampling_factor * params["v"])
        elif (
            self._internal_options.experiment_type == "FTACV"
            or self._internal_options.experiment_type == "PSV"
        ):
            dt = 1 / (sampling_factor * params["omega"])
       
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
    def SW_sampling(self,**kwargs):
        #TODO implement setter property if someone changes delta_E part way through
        self._internal_memory["SW_params"]={}
        if "parameters" not in kwargs:
            parameters=self._internal_memory["input_parameters"]
        else:
            parameters=kwargs["parameters"]
        sampling_factor=parameters["sampling_factor"]
        self._internal_memory["SW_params"]["end"]=int(abs(parameters['delta_E']//parameters['scan_increment']))

        p=np.array(range(0, self._internal_memory["SW_params"]["end"]))
        self._internal_memory["SW_params"]["b_idx"]=((sampling_factor*p)+(sampling_factor/2))-1
        self._internal_memory["SW_params"]["f_idx"]=(p*sampling_factor)-1
        Es=parameters["E_start"]#-parameters["E_0"]
        self._internal_memory["SW_params"]["E_p"]=(Es+parameters["v"]*(p*parameters['scan_increment']))
        self._internal_memory["SW_params"]["sim_times"]=self.calculate_times()
    
    def SW_peak_extractor(self, current, **kwargs):
        if "mean" not in kwargs:
            kwargs["mean"]=0
        if "window_length" not in kwargs:
            kwargs["window_length"]=1
       
        j=np.array(range(1, self._internal_memory["SW_params"]["end"]*self._internal_memory["input_parameters"]["sampling_factor"]))

        if kwargs["mean"]==0:
            forwards=np.zeros(len(self._internal_memory["SW_params"]["f_idx"]))
            backwards=np.zeros(len(self._internal_memory["SW_params"]["b_idx"]))
            forwards=np.array([current[x] for x in self._internal_memory["SW_params"]["f_idx"]])
            backwards=np.array([current[int(x)] for x in self._internal_memory["SW_params"]["b_idx"]])
        else:
            raise NotImplementedError
            indexes=[self._internal_memory["SW_params"]["f_idx"], self._internal_memory["SW_params"]["b_idx"]]
            sampled_currents=[np.zeros(len(self._internal_memory["SW_params"]["f_idx"])), np.zeros(len(self._internal_memory["SW_params"]["b_idx"]))]
            colours=["red", "green"]
            mean_idx=copy.deepcopy(sampled_currents)
            for i in range(0, len(self._internal_memory["SW_params"]["f_idx"])):
                for j in range(0, len(sampled_currents)):
                    x=indexes[j][i]
                    data=self.rolling_window(current[int(x-kwargs["mean"]-1):int(x-1)], kwargs["window_length"])
                    #plt.scatter(range(int(x-kwargs["mean"]-1),int(x-1)), data, color=colours[j])
                    #mean_idx[j][i]=np.mean(range(int(x-kwargs["mean"]-1),int(x-1)))
                    sampled_currents[j][i]=np.mean(data)

            forwards=np.zeros(len(self._internal_memory["SW_params"]["f_idx"]))
            backwards=np.zeros(len(self._internal_memory["SW_params"]["b_idx"]))
            forwards=np.array([current[x-1] for x in self._internal_memory["SW_params"]["f_idx"]])
            backwards=np.array([current[int(x)-1] for x in self._internal_memory["SW_params"]["b_idx"]])
        return forwards, backwards, backwards-forwards, self._internal_memory["SW_params"]["E_p"]


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
        
        
        input_parameters["phase_flag"]=0
        checking_dict=copy.deepcopy(self._internal_memory["input_parameters"])
        checking_dict["phase_flag"]=input_parameters["phase_flag"]
        sci.check_input_dict(
            input_parameters, checking_dict, optional_arguments=[]
        )
        input_parameters=copy.deepcopy(input_parameters)
        input_parameters = self.validate_input_parameters(input_parameters)
        if self._internal_options.experiment_type!="SquareWave":
            if "tr" in input_parameters:
                if kwargs["dimensional"]==True:
                    input_parameters["tr"]*=self._NDclass.c_T0
            input_parameters["omega"] *= 2 * np.pi
            return sos.potential(times, input_parameters)
        else:
           
            voltages=np.zeros(len(times))

            for i in times:
                
                i=int(i)
                voltages[i-1]=sos.SW_potential(i,input_parameters["sampling_factor"],input_parameters["scan_increment"],input_parameters["SW_amplitude"],input_parameters["E_start"],input_parameters["v"])
            voltages[-1]=voltages[-2]
            return voltages
    def redimensionalise(self, nondim_dict):
        return self._NDclass.redimensionalise(nondim_dict)
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
            if len(dispersion_distributions)!=len(self._internal_options.dispersion_bins):
                raise ValueError("Need one bin for each of {0}, currently only have {1}".format(dispersion_distributions, self._internal_options.dispersion_bins))
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
            if self._internal_options.problem=="inverse":
                raise Exception(
                    "Need to define boundaries for:\n %s" % ", ".join(missing_parameters)
                )
            
                
        self.dispersion_checking(parameters)
        if "cap_phase" not in parameters and ("cap_phase" not in self._internal_memory["fixed_parameters"].keys()):
            self._internal_options.phase_only = True
        else:
            self._internal_options.phase_only = False
        if "simulation_dict" in self._internal_memory:
            del self._internal_memory["simulation_dict"]
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
            inputs["phase"]= 0
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
            _internal_memory["simulation_dict"]
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
                elif self._internal_options.experiment_type in ["DCV", "SquareWave"] and "phase" in key:
                    simulation_dict[key]=0
                else:
                    missing_parameters.append(key)
        if len(missing_parameters) > 0:
            raise Exception(
                "The following parameters either need to be set in optim_list, or set at a value using the fixed_parameters variable\n{0}".format(
                    ", ".join(missing_parameters)
                )
            )
        options_and_requirements={
            "phase_function":{"option_value":"constant", 
                    "actions":["param_check", "simulation_options"], 
                    "params":{"sinusoidal":["phase_delta_E", "phase_omega", "phase_phase"], "constant":["phase"]}, 
                    "options":{"values":{"constant":0, "sinusoidal":1}, "flag":"phase_flag"}},
            "kinetics":{"option_value":self._internal_options.kinetics,
                        "actions":["simulation_options"],
                        "options":{"values":{"Marcus":1, "ButlerVolmer":0}, "flag":"Marcus_flag"}}
        }
        if "theta" not in simulation_dict:
            simulation_dict["theta"]=0
        for key in options_and_requirements.keys():
            sub_dict=options_and_requirements[key]
            option_value=sub_dict["option_value"]
            for action in sub_dict["actions"]:
                if action=="simulation_options":
                    sim_dict_key=sub_dict["options"]["flag"]
                    sim_dict_value=sub_dict["options"]["values"][option_value]
                    simulation_dict[sim_dict_key]=sim_dict_value
                elif action =="params":
                    extra_required_params=sub_dict["params"][option_value]
                    for param in extra_required_params:
                        if param not in all_parameters:
                            raise Exception("Because of option {0} being set to {1}, the following parameters need to be in either optim_list or fixed_parameters: {2}\n. Currently missing {3}".format(key, option_value, extra_required_params, param))
                        else:
                            if param in parameters:
                                simulation_dict[param]=None
                            else:
                                simulation_dict[param]=self._internal_memory["fixed_parameters"][param]
                        


        simulation_dict = self.validate_input_parameters(simulation_dict)

        self._internal_memory["simulation_dict"] = simulation_dict
        if self._internal_options.experiment_type=="SquareWave":    
            self._NDclass.construct_function_dict_SW(self._internal_memory["simulation_dict"])
        else:
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
                normed_params[i] = sci.un_normalise(
                    normed_params[i],
                    [
                        self._internal_memory["boundaries"][self._optim_list[i]][0],
                        self._internal_memory["boundaries"][self._optim_list[i]][1],
                    ],
                )
        elif method == "norm":
            for i in range(0, len(parameters)):
                normed_params[i] = sci.normalise(
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
            time_series_current = solver(times, nd_dict)
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
        
        self.optim_list=self._optim_list
        current=self.simulate(parameters, times)
        return self.experiment_top_hat(times, current, **kwargs)
    def experiment_top_hat(self, times, current, **kwargs):
        Fourier_options=["Fourier_window", "Fourier_function", "top_hat_width", "Fourier_harmonics"]
        for key in Fourier_options:
            if key not in kwargs:
                kwargs[key]=getattr(self, key)
        return sci.top_hat_filter(times, current, **kwargs)

    #@sci._utils.temporary_options(normalise_parameters=False)
    def Dimensionalsimulate(self, parameters, times):
        self.optim_list=self._optim_list
        return self.simulate(parameters, self.nondim_t(times))
        
    def save_class(self,path):
        save_dict={"Options":self._internal_options.as_dict(),}
        if self._options_handler is not None:
            save_dict["Options_handler"]={"name":self._options_handler.__name__, "module":self._options_handler.__module__}
        else:
            save_dict["Options_handler"]=self._options_handler
        save_dict["class"]={"name":self.__class__.__name__, "module":self.__class__.__module__}
            
        option_keys=self._internal_options.get_option_names()

        
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
            solverclass=SolverWrapper(sos.ODEsimulate)
            if self._internal_options.Faradaic_only==False:
                solver=solverclass.ode_current_wrapper
            else:
                solver=solverclass.farad_current_wrapper
            t=times
        elif self._internal_options.experiment_type == "SquareWave":
            solverclass=SolverWrapper(sos.SW_current)
            solver=solverclass.swv_current_wrapper
            t=self._internal_memory["SW_params"]["sim_times"]
        if self._internal_options.dispersion == True:
            nd_dict = self.nondimensionalise(sim_params)
            
            current = self.dispersion_simulator(solver,  sim_params, t)
        else:
            nd_dict = self.nondimensionalise(sim_params)
            current = solver(t, nd_dict)
        if self._internal_options.experiment_type == "SquareWave":
            sw_dict={"total":current}
            sw_dict["forwards"], sw_dict["backwards"], sw_dict["net"], E_p=self.SW_peak_extractor(current)
            if self._internal_options.square_wave_return!="total":
                polynomial_cap=nd_dict["Cdl"]*np.ones(len(E_p))
                keys=["CdlE1", "CdlE2", "CdlE3"]
                
                for i in range(0, len(keys)):
                    cdl=keys[i]
                    if nd_dict[cdl]!=0:
                        ep_power=np.power(E_p, i+1)
                        polynomial_cap=np.add(polynomial_cap, nd_dict[cdl]*ep_power)
                current=np.add(polynomial_cap, sw_dict[self._internal_options.square_wave_return])
        #print(nd_dict["k0"])
        return current

    
    def parameter_array_simulate(self, sorted_dimensional_parameter_array, dimensional_times, **kwargs):
        times=dimensional_times
        if "contains_noise" not in kwargs:
            kwargs["contains_noise"]=True
        if "return_dimensional" not in kwargs:
            kwargs["return_dimensional"]=True
        if kwargs["return_dimensional"]==True:
            func=self.dim_i
        else:
            func=lambda x:x
        sdpa=np.array(sorted_dimensional_parameter_array)
        currents=np.zeros((len(sdpa), len(times)))
        for i in range(0, len(sdpa)):
            if kwargs["contains_noise"]==True:
                params=sdpa[i,:-1]
            else:
                params=sdpa[i,:]
            currents[i,:]=func(self.Dimensionalsimulate(params, times))
        if self._internal_options.experiment_type=="FTACV":
            DC_params=copy.deepcopy(self._internal_memory["input_parameters"])
            DC_params["delta_E"]=0
            DC_voltage=self.get_voltage(times, dimensional=True, input_parameters=DC_params)
        else:
            DC_voltage=None
        return {"Current_array":currents, "DC_voltage":DC_voltage}
    @sci._utils.temporary_options(normalise_parameters=True)

    def Current_optimisation(self, time_data, current_data,**kwargs):

        if "paralell" in kwargs:
            raise KeyError("You've mis-spelled paraLLeL")
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
        if "Fourier_fitting" not in kwargs:
            kwargs["Fourier_fitting"]=False
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
        if kwargs["Fourier_fitting"]==True:
            log_Likelihood=sci.FourierGaussianLogLikelihood(problem)
            error=5000
        else:
            log_Likelihood=sci.GaussianTruncatedLogLikelihood(problem)
            error=1000
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
            between_0_and_1=[(x>0 and x<1) for x in kwargs["starting_point"]]
            if all(between_0_and_1)==True:

                x0=kwargs["starting_point"]
            else:
                x0=self.change_normalisation_group(kwargs["starting_point"], "norm")
            diff=log_Likelihood.n_parameters()-len(x0)
            if diff>0:
                x0+=[error/5]*diff
                
        if kwargs["sigma0"] is not list:
            sigma0=[kwargs["sigma0"] for x in range(0,log_Likelihood.n_parameters())]
        else:
            sigma0=kwargs["sigma0"]
        best_score=-1e20
        scores=np.zeros(kwargs["runs"])
        parameters=np.zeros((kwargs["runs"], log_Likelihood.n_parameters()))
        noises=np.zeros(kwargs["runs"])
        for i in range(0, kwargs["runs"]):
            optimiser=pints.OptimisationController(log_Likelihood, x0, sigma0=sigma0, boundaries=boundaries, method=kwargs["method"])
            optimiser.set_max_unchanged_iterations(iterations=kwargs["unchanged_iterations"], threshold=kwargs["tolerance"])
            optimiser.set_parallel(kwargs["parallel"])
            xbest, fbest=optimiser.run()
            scores[i]=fbest
            dim_params=list(self.change_normalisation_group(xbest[:-log_Likelihood._no], "un_norm"))
            parameters[i,:-log_Likelihood._no]=dim_params
            parameters[i, -log_Likelihood._no:]=fbest#xbest[-log_Likelihood._no:]
            if kwargs["save_to_directory"] is not False:
                if i==0:
                    save_times=self.dim_t(time_data)
                    Path(kwargs["save_to_directory"]).mkdir(parents=True, exist_ok=True)
                    voltage=self.get_voltage(save_times, dimensional=True)
        sorted_idx=np.flip(np.argsort(parameters[:,-1]))
        sorted_params=[list(parameters[x,:]) for x in sorted_idx]
        if kwargs["save_to_directory"] is not False:
            parameters=np.array(sorted_params)
            sim_dict=self.parameter_array_simulate(parameters,save_times, contains_noise=True)
            sci.plot.save_results(save_times, 
                                    voltage, 
                                    self.dim_i(current_data), 
                                    sim_dict["Current_array"],
                                    kwargs["save_to_directory"], 
                                    self._internal_options.experiment_type, 
                                    self._internal_memory["boundaries"],
                                    save_csv=kwargs["save_csv"],
                                    optim_list=self._optim_list, 
                                    fixed_parameters=self.fixed_parameters,
                                    score=parameters[:,-1],
                                    parameters=parameters,
                                    DC_voltage=sim_dict["DC_voltage"]
                                    )
        
                
        if kwargs["runs"]==1:
            return dim_params+list(xbest[-log_Likelihood._no:])
        else:
            return list(sorted_params)

class SolverWrapper:
    def __init__(self, solver):
        self.solver=solver

    def ode_current_wrapper(self,times, params):
        return np.array(self.solver(times, params))[0,:]
    def farad_current_wrapper(self, times, params):
        return np.array(self.solver(times, params))[2,:]
    def swv_current_wrapper(self,times, params):
        return np.array(self.solver(times, params))
