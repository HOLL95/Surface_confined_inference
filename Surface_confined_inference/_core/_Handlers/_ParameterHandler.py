import Surface_confined_inference as sci
import copy
import re
import itertools
from dataclasses import dataclass, field
import os
@dataclass(frozen=True)
class DispersionContext:
    dispersion: bool
    dispersion_warning: str =""
    dispersion_parameters: list = field(default_factory=list)  
    dispersion_distributions: list = field(default_factory=list)  
    all_parameters: list = field(default_factory=list)  
    GH_values: list = field(default_factory=list)

@dataclass(frozen=True)
class ParameterContext:
    fixed_parameters: dict
    boundaries: dict
    optim_list: list 
    sim_dict: dict
def dispersion_checking( all_parameters, GH_quadrature, bins):
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
            dispersion = True

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
            if len(dispersion_distributions)!=len(bins):
                raise ValueError("Need one bin for each of {0}, currently only have {1}".format(dispersion_distributions, bins))

            normal=[x for x in range(0, len(dispersion_distributions)) if dispersion_distributions[x]=="normal"]
            GH_values=[]
            if GH_quadrature == True:
                if len(normal)>0:
                    
                    GH_values=[sci._utils.GH_setup(bins[x]) if x in normal else None for x in range(0, len(dispersion_distributions))]


            _disp_context =DispersionContext(
                dispersion_parameters=dispersion_parameters,
                dispersion_distributions=dispersion_distributions,
                all_parameters=all_parameters,
                dispersion=True,
                GH_values=GH_values
            )
            
            return _disp_context
        else:
            _disp_context=DispersionContext(dispersion=False)
            return _disp_context
def validate_boundaries(parameters, boundaries):
    missing_parameters = []

    for i in range(0, len(parameters)):
        if parameters[i] not in boundaries:
            missing_parameters.append(parameters[i])
    if len(missing_parameters) > 0:
            raise ValueError(
                "Need to define boundaries for:\n %s" % ", ".join(missing_parameters)
            )
def simulation_dict_construction(parameters, fixed_parameters, essential_parameters, dispersion_parameters, dispersion, simulation_dict, options):
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
        missing_parameters = []
        all_parameters = list(
            set(parameters + list(fixed_parameters.keys()))
        )
        for key in all_parameters:
            if key in parameters:
                simulation_dict[key] = None
                if key in fixed_parameters:
                    warn(
                        "%s in both fixed_parameters and optimisation_list, the fixed parameter will be ignored!"
                        % key
                    )
            elif key in fixed_parameters:
                simulation_dict[key] = fixed_parameters[key]
        for key in essential_parameters:
            if key not in simulation_dict:
                if dispersion == True:
                    if key in dispersion_parameters:
                        simulation_dict[key] = 0
                        continue
                if "CdlE" in key:
                    simulation_dict[key] = 0
                elif options.experiment_type in ["DCV", "SquareWave"] and "phase" in key:
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
            "kinetics":{"option_value":options.kinetics,
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
                                simulation_dict[param]=fixed_parameters[param]
                        

        if hasattr(options, "phase_only") and options.phase_only==True:
            simulation_dict["cap_phase"]=simulation_dict["phase"]
            simulation_dict["phase"]=simulation_dict["cap_phase"]
        simulation_dict = ParameterHandler.validate_input_parameters(simulation_dict, options.experiment_type)
        return simulation_dict
class ParameterHandler:
    def __init__(self, **kwargs):
        self.options=kwargs["options"]
        self.fixed_parameters=kwargs["fixed_parameters"]
        self.boundaries=kwargs["boundaries"]
        if self.options.experiment_type!="SquareWave":
            self._essential_parameters = ["E0","k0","Cdl","gamma","CdlE1","CdlE2","CdlE3","alpha","Ru","phase"]
        else:
            self._essential_parameters=["E0","k0","gamma","alpha","Cdl","CdlE1","CdlE2","CdlE3",]
        parameters=kwargs["parameters"]
        self._optim_list=parameters
        if self.options.problem=="inverse":
            validate_boundaries(self._optim_list, self.boundaries)
        
        num_params=len(parameters)       
        num_fixed=len(list(self.fixed_parameters.values()))
        sim_dict={}
        all_parameters= list(set(self._optim_list + list(self.fixed_parameters.keys())))
        if num_params+num_fixed==0:
            self._disp_context=DispersionContext(dispersion=False)
        elif num_params==0 or num_fixed==0:
            if num_params==0:
                elem="optim_list"
            elif num_fixed==0:
                elem="fixed_parameters"
            try: 
                self._disp_context=dispersion_checking(all_parameters, self.options.GH_quadrature, self.options.dispersion_bins)
                sim_dict=sim_dict=simulation_dict_construction(parameters, 
                                            self.fixed_parameters, 
                                            self._essential_parameters, 
                                            self._disp_context.dispersion_parameters, 
                                            self._disp_context.dispersion, 
                                            copy.deepcopy(self.options.input_params), 
                                            self.options)
            except Exception as e:
                
                
                warning_str=f"Warning: {elem} not set, resulting in: \n {e} \n Simulations will not behave as expected"
                self._disp_context=DispersionContext(dispersion=False, dispersion_warning=warning_str)
                
        else:   
            self._disp_context=dispersion_checking(all_parameters, self.options.GH_quadrature, self.options.dispersion_bins)
            sim_dict=simulation_dict_construction(parameters, 
                                            self.fixed_parameters, 
                                            self._essential_parameters, 
                                            self._disp_context.dispersion_parameters, 
                                            self._disp_context.dispersion, 
                                            copy.deepcopy(self.options.input_params), 
                                            self.options)
        self.context=ParameterContext(
            fixed_parameters=kwargs["fixed_parameters"],
            boundaries=kwargs["boundaries"],
            optim_list=kwargs["parameters"],
            sim_dict=sim_dict
        ) 
    @staticmethod
    def create_warning(warning_str):
        term=os.get_terminal_size()[0]
        return "#"*term+"\n"+warning_str+"\n"+"#"*(term-1)
   
    def change_normalisation_group(self, parameters, method):
        """
        Args:
            parameters (list): list of numbers
            method (str): flag indicating if the value is to be normalised or un-normalised
        Returns:
            list: list of appropriately transformed values
        """
        for key in self._optim_list:
            if key not in self.boundaries:
                raise KeyError("{0} not in bounadries {1}".format(key, self.boundaries.keys()))
        normed_params = copy.deepcopy(parameters)
        if method == "un_norm":
            for i in range(0, len(parameters)):
                normed_params[i] = sci.un_normalise(
                    normed_params[i],
                    [
                        self.boundaries[self._optim_list[i]][0],
                        self.boundaries[self._optim_list[i]][1],
                    ],
                )
        elif method == "norm":
            for i in range(0, len(parameters)):
                normed_params[i] = sci.normalise(
                    normed_params[i],
                    [
                        self.boundaries[self._optim_list[i]][0],
                        self.boundaries[self._optim_list[i]][1],
                    ],
                )
        return normed_params
    @staticmethod
    def validate_input_parameters(inputs, exp_type):
        """
        Args:
            inputs (dict): Dictionary of parameters
        Returns:
            dict : modified dictionary of input parameters according to the method assigned to the class
        """
        if exp_type == "DCV":
            inputs["tr"] = abs(inputs["E_reverse"] - inputs["E_start"]) / inputs["v"]
            
            inputs["omega"] = 0
            inputs["delta_E"] = 0
            inputs["phase"]= 0
        elif exp_type == "FTACV":
            inputs["tr"] =  abs(inputs["E_reverse"] - inputs["E_start"]) / inputs["v"]
            
        elif exp_type == "PSV":
            inputs["E_reverse"]=inputs["Edc"]
            inputs["tr"] = -1
            inputs["v"] = 0
        return inputs