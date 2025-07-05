import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from scipy.interpolate import CubicSpline
import itertools
import Surface_confined_inference as sci
from pathlib import Path
from ._InitialiseExperiment import InitialiseMultiExperiment, validate_input_dict
from ._FileReader import _process_data
from ._Grouping import initialise_grouping
from ._ParameterManager import ParameterManager
from pathlib import Path
import json
from ._utils import recursive_list_cast
from numbers import Number
from ._BaseMultiExperiment import BaseMultiExperiment
from .SyntheticFuncs import create_times
from ._Plotting import PlotManager
class MultiExperiment(sci.BaseMultiExperiment, sci.OptionsAwareMixin):
    _manual_options=["class_keys", "classes", "input_params", "group_to_conditions", "group_to_class", "group_to_parameters"]
    _allowed_experiments=["FTACV","PSV","DCV","SWV","SquareWave","Trumpet"]
    def __init__(self, input_params, **kwargs):
        self._internal_options = sci.MultiExperimentOptions(**kwargs)
        self.input_params=input_params
        self.file_list=self._internal_options.file_list
        
        self.class_keys=list(self.classes.keys())
        self._all_parameters=set()
        self._all_harmonics=set()
        for key in self.class_keys:
            self._all_parameters=self._all_parameters.union(self.classes[key]["class"].optim_list)
            if self.classes[key]["class"].experiment_type in ["FTACV","PSV"]:
                self._all_harmonics=self._all_harmonics.union(set(self.classes[key]["class"].Fourier_harmonics))
            else:
                if self._internal_options.SWV_e0_shift==True:
                    if "SquareWave" in self.classes[key]["class"].experiment_type:
                        if "anodic" not in key and "cathodic" not in key:
                            raise ValueError("If SWV_e0_shift is set to True, then all SWV experiments must be identified as anodic or cathodic, not {0}".format(key))
        self._all_harmonics=list(self._all_harmonics)
        self._all_parameters=list(self._all_parameters)
        
            
    @property
    def input_params(self):
        return self._input_params
    @input_params.setter
    def input_params(self, parameters):
        for key in parameters.keys():
            if key not in self._allowed_experiments:
                raise ValueError("Experiment {0} not enabled for MultiExperiments (only {1})".format(key, self._allowed_experiments))
            validate_input_dict(parameters[key], [key])
        initialised_classes=InitialiseMultiExperiment(parameters, 
                                                    common=self._internal_options.common, 
                                                    boundaries=self._internal_options.boundaries,
                                                    normalise=self._internal_options.normalise,
                                                    sim_class=sci.ParallelSimulator)
        self.classes=initialised_classes.classes
        self.class_keys=initialised_classes.all_keys
        self._input_params=parameters
    @property
    def file_list(self):
        return self._internal_options.file_list
    @file_list.setter
    def file_list(self, file_list):
        self._internal_options.file_list=file_list
        self.classes=_process_data(file_list, self.classes, self.class_keys)
    @property
    def group_list(self):
        if len(self._internal_options._group_list)==0:
            raise ValueError("Grouping of experiments into objectives hasn't been defined, please do so!")
        return self._internal_options._group_list
    @group_list.setter
    def group_list(self, group_list):
        self._internal_options._group_list=group_list
        self.group_to_conditions, self.group_to_class=initialise_grouping(group_list, self.classes)
        self._grouping_keys=list(self.group_to_conditions.keys())
        self._manager=ParameterManager(self._all_parameters, self.grouping_keys, self.classes, self._internal_options.SWV_e0_shift, self.group_to_class)
        self.group_to_parameters,_=self._manager.initialise_simulation_parameters(self._internal_options.seperated_parameters)
        if self._internal_options.synthetic==True:
            self.classes=create_times(self.classes, self.class_keys)
        self._plot_manager=PlotManager(self)
    @property
    def grouping_keys(self):
        if len(self._internal_options._group_list)==0:
            raise ValueError("Grouping of experiments into objectives hasn't been defined, please do so!")
        return self._grouping_keys
    @property
    def seperated_parameters(self):
        return self._internal_options._seperated_parameters
    @seperated_parameters.setter
    def seperated_parameters(self, seperation_dict):
        self.group_list
        self._internal_options.seperated_parameters=seperation_dict
        self.group_to_parameters, self._all_parameters=self._manager.initialise_simulation_parameters(seperation_dict)
    
    def evaluate(self, parameters):
        simulation_params_dict=self._manager.parse_input(parameters)
        simulation_values_dict={}
        for classkey in self.class_keys:
            cls=self.classes[classkey]["class"]
            sim_params=simulation_params_dict[classkey]
            simulation_values_dict[classkey]=cls.simulate(sim_params, self.classes[classkey]["times"])
        return simulation_values_dict
    def optimise_simple_score(self, parameters):
        simulation_values=self.evaluate(parameters)
        return self.simple_score(simulation_values)
    def simple_score(self, simulation_values_dict):
        score_dict={}
        for groupkey in self.grouping_keys:
            current_score=0
            for classkey in self.group_to_class[groupkey]:
                cls=self.classes[classkey]["class"]
                if "type:ft" not in groupkey:
                    classscore=sci._utils.RMSE(simulation_values_dict[classkey], self.classes[classkey]["data"])
                else:
                    classscore=sci._utils.RMSE(cls.experiment_top_hat(self.classes[classkey]["times"],simulation_values_dict[classkey]), self.classes[classkey]["FT"])
                if "scaling" in self.group_to_conditions[groupkey]:
                    classscore=self.scale(classscore, groupkey, classkey)
                current_score+=classscore
            score_dict[groupkey]=current_score
        return score_dict
    def results_table(self, parameters, mode="table"):
        self._manager.results_table(parameters, self.class_keys, mode)
    def scale(self, value, groupkey, classkey):
        value=copy.deepcopy(value)
        cls=self.classes[classkey]["class"]
        if "divide" in self.group_to_conditions[groupkey]["scaling"]:
            for param in self.group_to_conditions[groupkey]["scaling"]["divide"]:
                value/=cls._internal_memory["input_parameters"][param]
        if "multiply" in self.group_to_conditions[groupkey]["scaling"]:
            for param in self.group_to_conditions[groupkey]["scaling"]["multiply"]:
                value*=cls._internal_memory["input_parameters"][param]  
        return value
    def save_class(self, dir_path="saved", **kwargs):
        if "include_data" not in kwargs:
            kwargs["include_data"]=True
        indv_class_path=os.path.join(dir_path, "individual_classes",)
        for element in ["classes", "data"]:
            Path(os.path.join(indv_class_path, element)).mkdir(parents=True, exist_ok=True)
        for classkey in self.class_keys:
            try:
             self.classes[classkey]["class"].save_class(os.path.join(indv_class_path, "classes",classkey))
            except:
             print(classkey)
             raise
            Path(os.path.join(indv_class_path, "data", classkey)).mkdir(parents=True, exist_ok=True)
            data_dict={}
            data_path=os.path.join(indv_class_path, "data", classkey)
            if kwargs["include_data"]==True:
                for attr in ["zero_point", "zero_sim", "FT", "zero_point_ft", "data", "times"]:
                    if attr in self.classes[classkey]:
                        with open(os.path.join(data_path, "{0}.txt".format(attr)), "w") as f:
                            if isinstance(self.classes[classkey][attr], Number):
                                f.write(str(self.classes[classkey][attr]))
                            else:
                                np.savetxt(f, self.classes[classkey][attr])
            if "Zero_params" in self.classes[classkey]:
                with open(os.path.join(data_path, "Zero_params.json"), "w") as f:
                    json.dump(self.classes[classkey]["Zero_params"], f)
            if len(data_dict)!=0:    
                data_path = os.path.join(indv_class_path,"data", "{0}-data.json".format(classkey))
                with open(data_path, "w") as f:
                    json.dump(data_dict, f)
        multi_dict=self._internal_options.as_dict()
        multi_dict["include_data"]=kwargs["include_data"]
        multi_dict={key:recursive_list_cast(multi_dict[key]) for key in multi_dict.keys()}
        with open(os.path.join(dir_path, "multi_options.json"),"w") as f:
            json.dump(multi_dict, f)
    def check_grouping(self,):
        self._plot_manager.plot_results([], savename=None, show_legend=True)
    def results(self, **kwargs):
        defaults={"show_legend":True,
                "savename":None,
                "target_key":[None],
                "sim_plot_options":"simple",
                "axes":None}
        for key in defaults.keys():
            if key not in kwargs:
                kwargs[key]=defaults[key]
        self._plot_manager.results(**kwargs)
    def plot_2d_pareto(self, **kwargs):
        self._plot_manager.plot_2d_pareto(**kwargs)
    def pareto_parameter_plot(self, **kwargs):
        self._plot_manager.pareto_parameter_plot(**kwargs)
            
   #TODO Need to make the individual class options, optimisation lists immutable
    
        


