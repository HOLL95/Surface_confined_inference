import SurfaceODESolver as sos
from dataclasses import dataclass
import Surface_confined_inference as sci
import collections.abc
import numbers
from warnings import warn
import itertools
import numpy as np
import copy
import time
class SingleExperiment():
    def __init__(self,experiment_type, experiment_parameters, **kwargs):
        all_experiments=["FTACV", "PSV", "DCV", "SquareWave"] 
        accepted_arguments={key:["E_start","E_reverse", "area", "Temp", "N_elec", "Surface_coverage"] for i in range(0, len(all_experiments)) for key in all_experiments}
        extra_args={"FTACV":["v", "omega", "phase", "delta_E"],
                    "PSV":["omega", "phase", "delta_E", "num_peaks"],
                    "DCV":["v"],
                    "SquareWave":["scan_increment","sampling_factor", "delta_E","v", "SW_amplitude"]
                    }
        for key in extra_args.keys():
            accepted_arguments[key]+=extra_args[key]
        if experiment_type not in all_experiments:
            raise ValueError("\'{0}\' not in list of experiments. Simulated experiments are \n{1}".format(experiment_type, ("\n").join(all_experiments)))
        self._internal_memory={"input_parameters":experiment_parameters, "boundaries":{}, "fixed_parameters":{}}
        kwargs["experiment_type"]=experiment_type
        self._internal_options=OptionsDecorator(**kwargs)
        
        
        [setattr(self, key, getattr(self._internal_options, key))for key in Options().accepted_arguments]
        
        
        sci.check_input_dict(experiment_parameters, accepted_arguments[self._internal_options.experiment_type])
        self.NDclass=sci.NDParams(self._internal_options.experiment_type, experiment_parameters)
        self.essential_parameters=["E0", "k0", "Cdl", "gamma", "CdlE1", "CdlE2", "CdlE3", "alpha", "Ru", "phase"]
    def calculate_times(self, sampling_factor=200, dimensional=False):
        params=self._internal_memory["input_parameters"]
        if self._internal_options.experiment_type=="FTACV" or self._internal_options.experiment_type=="DCV":
            end_time=2*abs(params["E_start"]-params["E_reverse"])/params["v"]
        elif self._internal_options.experiment_type=="PSV":
            end_time=params["num_peaks"]/params["omega"]
        elif self._internal_options.experiment_type=="SquareWave":
            raise NotImplementedError()
        if self._internal_options.experiment_type=="DCV":
            dt=1/sampling_factor
        elif self._internal_options.experiment_type=="FTACV" or self._internal_options.experiment_type=="PSV":
            dt=1/(sampling_factor*params["omega"])
        elif self._internal_options.experiment_type=="SquareWave":
            raise NotImplementedError()
        times=np.arange(0, end_time, dt)
        if dimensional==False:
            times=self.nondim_t(times)
        return times
    def dim_i(self, current):
        return np.multiply(current, self.NDclass.c_I0)
    def dim_e(self, potential):
        return np.multiply(potential, self.NDclass.c_E0)
    def dim_t(self, time):
        return np.multiply(time, self.NDclass.c_T0)       
    def nondim_i(self, current):
        return np.divide(current, self.NDclass.c_I0)
    def nondim_e(self, potential):
        return np.divide(potential, self.NDclass.c_E0)
    def nondim_t(self, time):
        return np.divide(time, self.NDclass.c_T0)   
        
    def n_parameters(self,):
        return len(self._optim_list)
    
    def dispersion_checking(self, optim_list):
        
        disp_check_flags=["mean", "scale", "upper", "logupper"]#Must be unique for each distribution!
        disp_check=[[y in x for y in disp_check_flags] for x in optim_list]
        if True in [True in x for x in disp_check]:
            self._internal_options.dispersion=True
            if self._internal_options.GH_quadrature==True:
                GH_values=self.GH_setup()
            else:
                GH_values=None
            disp_flags=[["mean", "std"], ["shape","scale"], ["lower","upper"], ["mean","std", "skew"], ["logupper", "loglower"]]#Set not name must be unique
            all_disp_flags = list(set(itertools.chain(*list2d)))
            distribution_names=["normal", "lognormal", "uniform", "skewed_normal","log_uniform"]
            distribution_dict=dict(zip(distribution_names, disp_flags))
            disp_param_dict={}
            for i in range(0, len(optim_list)):
                for j in range(0, len(all_disp_flags)):
                    if all_disp_flags[j] in optim_list[i]:
                        try:
                            m=re.search('.+?(?=_'+all_disp_flags[j]+')', optim_list[i])
                            param=m.group(0)
                            if param in disp_param_dict:
                                disp_param_dict[param].append(all_disp_flags[j])
                            else:
                                disp_param_dict[param]=[all_disp_flags[j]]
                        except:
                            print(optim_list[i], all_disp_flags[j])
                            continue
            dispersion_parameters=list(disp_param_dict.keys())
            dispersion_distributions=[]
            self.fix_parameters({param:0 for param in dispersion_parameters})
            for param in dispersion_parameters:
                param_set=set(disp_param_dict[param])

                for key in distribution_dict.keys():

                    if set(distribution_dict[key])==param_set:
                        dispersion_distributions.append(key)
            
            """self.disp_class=dispersion(self.simulation_options, optim_list, GH_values)"""
        else:
            self._internal_options.dispersion=False
    @property
    def optim_list(self):
        return self._optim_list
    @optim_list.setter
    def optim_list(self, parameters):
        missing_parameters=[]
        for i in range(0, len(parameters)):
            if parameters[i] not in self._internal_memory["boundaries"]:
                missing_parameters.append(parameters[i])
        if len(missing_parameters)>0:        
            print("Need to define boundaries for:\n %s" % ("\n").join(missing_parameters))
            raise ValueError("Missing parameters")
        self.dispersion_checking(parameters)
        if "cap_phase" not in parameters:
            self._internal_options.phase_only=True
        self.simulation_dict_construction(parameters)
        self._optim_list=parameters
        """if self.simulation_options["method"]=="square_wave":
            if "SWV_constant" in optim_list:
                self.simulation_options["SWV_polynomial_capacitance"]=True
        if "Upper_lambda" in optim_list:
            self.simulation_options["Marcus_kinetics"]=True
        if self.simulation_options["Marcus_kinetics"]==True:
            if "alpha" in optim_list:
                raise ValueError("Currently Marcus kinetics are symmetric, so Alpha in meaningless")"""
    @property
    def boundaries(self):
        return self._internal_memory["boundaries"]
    @boundaries.setter
    def boundaries(self, boundaries):
        if isinstance(boundaries, dict) is False:
            return TypeError("boundaries need to be of type dict")
        else:
            self._internal_memory["boundaries"]=boundaries
    def simulation_dict_construction(self, parameters):
        if "simulation_dict" in self._internal_memory:
            simulation_dict=self._internal_memory["simulation_dict"]
        else:
            simulation_dict=copy.deepcopy(self._internal_memory["input_parameters"])
        for parameter in parameters:
            simulation_dict[parameter]=None
        missing_parameters=[]
        for key in self.essential_parameters:
            
            if key not in simulation_dict:
                if key not in self._internal_memory["fixed_parameters"]:
                    if "CdlE" in key:
                        simulation_dict[key]=0
                    else:
                        missing_parameters.append(key)
                else:
                    simulation_dict[key]=self._internal_memory["fixed_parameters"][key]
            elif key in simulation_dict and key in self._internal_memory["fixed_parameters"]:
                warn("%s in both fixed_parameters and optimisation_list, the fixed parameter will be ignored!" % key)
        if len(missing_parameters)>0:
            print("The following parameters either need to be set in optim_list, or set at a value using the fix_parameters function\n%s" % ("\n").join(missing_parameters))
            raise KeyError("Missing parameters")
        if self._internal_options.kinetics=="ButlerVolmer":
            simulation_dict["Marcus_flag"]=0
        elif self._internal_options.kinetics=="Marcus":
            simulation_dict["Marcus_flag"]=1
      
        if self._internal_options.experiment_type=="DCV" :
            simulation_dict["tr"]=self.nondim_t(abs(simulation_dict["E_reverse"]-simulation_dict["E_start"])/simulation_dict["v"])
            simulation_dict["omega"]=0
            simulation_dict["delta_E"]=0
        elif self._internal_options.experiment_type=="FTACV":
            simulation_dict["tr"]=  tr=self.nondim_t(abs(simulation_dict["E_reverse"]-simulation_dict["E_start"])/simulation_dict["v"])
        elif self._internal_options.experiment_type=="PSV":
            simulation_dict["tr"]=-10
            simulation_dict["v"]=0
        self._internal_memory["simulation_dict"]=simulation_dict
        

        self.NDclass.construct_function_dict(self._internal_memory["simulation_dict"])
    @property 
    def fixed_parameters(self):
        return self._internal_memory["fixed_parameters"]
    @fixed_parameters.setter
    def fixed_parameters(self, parameter_values):
        if isinstance(parameter_values, dict) is False:
            return TypeError("Argument needs to be a dictionary")
        else:
            self._internal_memory["fixed_parameters"]=parameter_values
    def GH_setup(self, dispersion_distributions, ):
        """
        We assume here that for n>1 normally dispersed parameters then the order of the integral
        will be the same for both 
        """
        try:
            disp_idx=dispersion_distributions.index("normal")
        except:
            raise KeyError("No normal distributions for GH quadrature")
        nodes=self._internal_options.dispersion_bins[disp_idx]
        labels=["nodes", "weights", "normal_weights"]
        nodes, weights=np.polynomial.hermite.hermgauss(nodes)
        normal_weights=np.multiply(1/math.sqrt(math.pi), weights)
        return dict(zip(labels, [nodes, weights, normal_weights]))
    def normalise(self, norm, boundaries):
        return  (norm-boundaries[0])/(boundaries[1]-boundaries[0])
    def un_normalise(self, norm, boundaries):
        return (norm*(boundaries[1]-boundaries[0]))+boundaries[0]
    def change_normalisation_group(self, parameters, method):
        normed_params=copy.deepcopy(parameters)
        
        if method=="un_norm":
            for i in range(0,len(parameters)):
                normed_params[i]=self.un_normalise(normed_params[i], [self._internal_memory["boundaries"][self._optim_list[i]][0],self._internal_memory["boundaries"][self._optim_list[i]][1]])
        elif method=="norm":
            for i in range(0,len(parameters)):
                normed_params[i]=self.normalise(normed_params[i], [self._internal_memory["boundaries"][self._optim_list[i]][0],self._internal_memory["boundaries"][self._optim_list[i]][1]])
        return normed_params
    def simulate(self, times, parameters):
        if self._internal_options.normalise==True:
            sim_params=dict(zip(self._optim_list, self.change_normalisation_group(parameters, "un_norm")))

        else:
            sim_params=dict(zip(self._optim_list, parameters))
        nd_dict={}
        
        start=time.time()
        for key in self._optim_list:
            self._internal_memory["simulation_dict"][key]=sim_params[key]    
        for key in self._internal_memory["simulation_dict"].keys():
            nd_dict[key]=self.NDclass.function_dict[key](self._internal_memory["simulation_dict"][key])
        if self._internal_options.phase_only==True:
            self._internal_memory["simulation_dict"]["cap_phase"]=self._internal_memory["simulation_dict"]["phase"]
            nd_dict["cap_phase"]=self._internal_memory["simulation_dict"]["phase"]
        if self._internal_options.experiment_type!="SquareWave":
            solver=sos.ODEsimulate
        start=time.time()
        current=np.array(solver(times, nd_dict))[0,:]
        
        return current
    def __setattr__(self, name, value):
        if name in ['OptionsDecorator']:  
            super().__setattr__(name, value)
        elif name in Options().accepted_arguments:
            setattr(self._internal_options, name, value)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)
        




class Options:
    def __init__(self,**kwargs):
        
        self.accepted_arguments={
            "experiment_type":{"type":str, "default":None},
            "GH_quadtrature":{"type":bool, "default":False},
            "phase_only":{"type":bool, "default":True},
            "normalise":{"type":bool, "default":False},
            "kinetics":{"args":["ButlerVolmer", "Marcus", "Nernst"], "default":"ButlerVolmer"},
            "dispersion":{"type":bool, "default":False},
            "dispersion_bins":{"type":collections.abc.Sequence, "default":[]},
            "transient_removal":{"type":[bool, numbers.Number], "default":False},
            "Fourier_filtering":{"type":bool, "default":False},
            "Fourier_function":{"args":["composite", "abs", "real", "imaginary"], "default":"composite"},
            "Fourier_harmonics":{"type":collections.abc.Sequence, "default":list(range(0, 10))},

        }
        if len(kwargs)==0:
            self.options_dict=self.accepted_arguments
        else:
            self.options_dict={}
            for kwarg in kwargs:
                if kwarg not in self.accepted_arguments:
                    raise ValueError(f"{kwarg} not an accepted option")
            for key in self.accepted_arguments:
                if key in kwargs:
                    self.options_dict[key]=kwargs[key]
                else:
                    self.options_dict[key]=self.accepted_arguments[key]["default"]
    def checker(self, name, value, defaults):
        if "type" in defaults:
            if isinstance(defaults["type"], list) is not True: 
                type_list=[defaults["type"]]
            else:
                type_list=defaults["type"]
            type_error=True
            for current_type in type_list:
               
                if isinstance(value, current_type) is True:
                    type_error=False
            if type_error==True:
                raise TypeError("{0} must be of type".format(name), defaults["type"])
        elif "args" in defaults:
            if value not in defaults["args"]:
                
                raise ValueError("Value '{0}' not part of the following allowed arguments:\n{1}".format(value, ("\n").join(defaults["args"])) )
    

class OptionsDecorator:
    def __init__(self, **kwargs):
        
        self.options = Options(**kwargs)

    def __getattr__(self, name):
        if name in self.options.options_dict:
            return self.options.options_dict[name]
        raise AttributeError(f"'OptionsDecorator' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in ['options']:  # To handle the initialization of the 'options' attribute
            super().__setattr__(name, value)
        elif name in self.options.options_dict:
            self.options.checker(name, value, self.options.accepted_arguments[name])
            self.options.options_dict[name] = value
        else:
            raise AttributeError(f"'OptionsDecorator' object has no attribute '{name}'")