import Surface_confined_inference as sci
import copy
import re
import itertools
from dataclasses import dataclass, field
import os
from numbers import Number
from warnings import warn
@dataclass(frozen=True)
class DispersionContext:
    """
    Immutable dataclass storing dispersion-related context information, to be used by both an orchestrator and simulattor class
    
    Attributes:
        dispersion (bool): Whether dispersion is enabled. Used to set disperion option in orchestrator class
        dispersion_warning (str): Warning message for dispersion issues (default: ""). Currently not used
        dispersion_parameters (list): List of parameters associated with a dispersion distribution (default: empty list)
        dispersion_distributions (list): List of distribution types for respective dispersed parameters (default: empty list)
        all_parameters (list): All parameter names including dispersed ones (default: empty list)
        GH_values (list): Gauss-Hermite quadrature values for normal distributions (default: empty list)
    """
    dispersion: bool
    dispersion_warning: str =""
    dispersion_parameters: list = field(default_factory=list)  
    dispersion_distributions: list = field(default_factory=list)  
    all_parameters: list = field(default_factory=list)  
    GH_values: list = field(default_factory=list)

@dataclass(frozen=True)
class ParameterContext:
    """
    Immutable dataclass storing information to be used by orchestrator and simulator classes
    
    Attributes:
        fixed_parameters (dict): Dictionary of fixed parameter values
        boundaries (dict): Dictionary of parameter boundaries to be used for minmax transformation
        optim_list (list): List of optimisation parameters (i.e. to be changed as part of a loop)
        sim_dict (dict): Simulation dictionary with all parameter values to be passed to C++ ODE simulator code, in dimensional form. 
    """
    fixed_parameters: dict
    boundaries: dict
    optim_list: list 
    sim_dict: dict
def dispersion_checking( all_parameters, GH_quadrature, bins):
        """
        Check for dispersion parameters and validate distribution requirements.
        
        Args:
            all_parameters (list): List of all parameter names (optim_list + fixed_parameters)
            GH_quadrature (bool): Whether to use Gauss-Hermite quadrature
            bins (list): Number of bins for each dispersed parameter
        
        Returns:
            DispersionContext: Context object containing dispersion information
        
        Raises:
            Exception: If required distribution parameters are missing
            ValueError: If number of bins doesn't match number of dispersed parameters
        
        Behavior:
            - Scans parameter names for dispersion flags (_mean, _std, etc.)
            - Groups parameters by distribution type (normal, lognormal, uniform, etc.)
            - Validates that all required parameters for each distribution are present
            - Sets up Gauss-Hermite quadrature values for normal distributions
            - Returns DispersionContext with dispersion configuration
        
        Distribution Types Supported:
            - normal: requires _mean, _std
            - lognormal: requires _logmean, _logscale  
            - uniform: requires _lower, _upper
            - skewed_normal: requires _mean, _std, _skew
            - log_uniform: requires _logupper, _loglower
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
    """
    Validate that all optimisation parameters have defined boundaries.
    
    Args:
        parameters (list): List of parameter names requiring boundaries
        boundaries (dict): Dictionary of parameter boundaries
    
    Raises:
        ValueError: If any parameters are missing from boundaries
    """
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
        Construct the simulation dictionary. As part of a simulation loop, the simulation parameter keys will be updated, and then everything non-dimensionalised. 
        
        Args:
            parameters (list): Optimisation parameters (set to None in dict) to be varied. 
            fixed_parameters (dict): Fixed parameter values to be kept constant
            essential_parameters (list): Required parameters for simulation
            dispersion_parameters (list): Parameters with dispersion
            dispersion (bool): Whether dispersion is enabled
            simulation_dict (dict): Base simulation dictionary to modify
            options: Options object with experiment configuration
        
        Returns:
            dict: Complete simulation dictionary with all parameters
        
        Raises:
            Exception: If essential parameters are missing and can't be defaulted
        
        Behavior:
            - Sets optimization parameters to None (filled during simulation)
            - Copies fixed parameter values to simulation dict
            - Validates all essential parameters are present
            - Sets defaults for missing optional parameters (CdlE*, phase, theta)
            - Applies experiment-specific options (kinetics flags, etc.)
            - Validates input parameters for experiment type
            - Handles special cases for phase_only mode
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
        if options.model=="square_scheme":
            simulation_dict["theta"]=1
            simulation_dict["model"]=1
        else:
            simulation_dict["model"]=0
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
        simulation_dict = ParameterHandler.validate_input_parameters(simulation_dict, options.experiment_type)
        return simulation_dict
class ParameterHandler:
    """
    Initialize the ParameterHandler with experiment parameters and options.
    
    Args:
        **kwargs: Keyword arguments containing:
            - options: Options object with experiment configuration
            - fixed_parameters (dict): Fixed parameter values
            - boundaries (dict): Parameter boundaries for optimization
            - parameters (list): List of optimization parameters
    
    Raises:
        ValueError: If boundaries are missing for optimization parameters
        Exception: If essential parameters are missing from optim_list or fixed_parameters
    
    Behavior:
        - Validates parameter boundaries if problem type is "inverse"
        - Checks for dispersion parameters and validates distributions
        - Constructs simulation dictionary with all required parameters
        - Sets up contexts for parameters and dispersion
    """
    def __init__(self, **kwargs):
        self.options=kwargs["options"]
        self.fixed_parameters=kwargs["fixed_parameters"]
        self.boundaries=kwargs["boundaries"]
        if self.options.model=="square_scheme":
            roman_switch={**{x:x*"i" for x in range(1, 4)}, **{x:"{0}v{1}".format(max(5-x, 0)*"i",max(x-5, 0)*"i") for x in range(4, 7)}}
            essential_p=[]
            for i in range(1, 7):
                essential_p+=[f"{x}_{i}" for x in ["E0", "k0","alpha"]]
                essential_p+=[f"{x}_{roman_switch[i]}" for x in ["kp", "kd"]]
        else:
            essential_p=["E0","k0","alpha"]
        
        if self.options.experiment_type!="SquareWave":
            self._essential_parameters = essential_p+["gamma","Cdl", "CdlE1","CdlE2","CdlE3", "Ru","phase"]
        else:
            self._essential_parameters=essential_p+["gamma","Cdl", "CdlE1","CdlE2","CdlE3"]
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
        """
        Create a formatted warning message with terminal-width borders.
        """
        term=os.get_terminal_size()[0]
        return "#"*term+"\n"+warning_str+"\n"+"#"*(term-1)
   
    def change_normalisation_group(self, parameters, method):
        """
        Transform parameters between normalized and un-normalized values. Required parameters to be in same order as in _optim_list. 
        
        Args:
            parameters (list): List of parameter values
            method (str): Either "norm" (normalize) or "un_norm" (denormalize)
        
        Returns:
            list: List of transformed parameter values
        
        Raises:
            KeyError: If parameter not found in boundaries
        
        Behavior:
            - Normalizes values to [0,1] range using boundaries if method="norm"
            - Denormalizes from [0,1] to actual range if method="un_norm"
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
        Validate and modify input parameters based on experiment type.
        
        Args:
            inputs (dict): Dictionary of input parameters
            exp_type (str): Experiment type ("DCV", "FTACV", "PSV", "SquareWave")
        
        Returns:
            dict: Modified dictionary with experiment-specific parameters
        
        Behavior:
            - Calculates experiment-specific parameters (tr, omega, etc.)
            - Modifies inputs in-place and returns modified copy
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