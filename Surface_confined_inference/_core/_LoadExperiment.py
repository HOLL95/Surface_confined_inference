import numpy as np
import json
import Surface_confined_inference as sci
class LoadSingleExperiment(sci.SingleExperiment):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            initilialisation_dict=json.load(f)
        
        super().__init__(initilialisation_dict["experiment_type"], 
                        initilialisation_dict["Experiment_parameters"],
                        **initilialisation_dict["Options"])
        self.fixed_parameters=initilialisation_dict["fixed_parameters"]
        self.boundaries=initilialisation_dict["boundaries"]
        self.optim_list=initilialisation_dict["optim_list"]
class ChangeTechnique(sci.SingleExperiment):
    def __init__(self, json_path, new_experiment,input_parameters):
        with open(json_path, "r") as f:
            initilialisation_dict=json.load(f)
        for key in initilialisation_dict["Experiment_parameters"]:
            if key not in input_parameters:
                input_parameters[key]=initilialisation_dict["Experiment_parameters"][key]
        if initilialisation_dict["experiment_type"]==new_experiment:
            raise ValueError("New experiment is the same as the old experiment - raising error")
        super().__init__(new_experiment, 
                        input_parameters,
                        **initilialisation_dict["Options"])
        self.fixed_parameters=initilialisation_dict["fixed_parameters"]
        self.boundaries=initilialisation_dict["boundaries"]
        self.optim_list=initilialisation_dict["optim_list"]
        