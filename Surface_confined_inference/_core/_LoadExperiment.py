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
        