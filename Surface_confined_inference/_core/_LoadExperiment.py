import numpy as np
import json
import Surface_confined_inference as sci
class LoadSingleExperiment:
    def __init__(self, json_path, **kwargs):
        with open(json_path, "r") as f:
            initilialisation_dict=json.load(f)
        allowed_args=["base","slurm","mcmc"]

        classes=dict(zip(allowed_args, 
                        [sci.SingleExperiment, sci.SingleSlurmSetup,sci.RunSingleExperimentMCMC]))
        if "class_type" not in kwargs:
            kwargs["class_type"]="base"
        elif kwargs["class_type"] not in allowed_args:
            raise KeyError("class_type needs to be one of {0}, not {1}".format(allowed_args, kwargs["class_type"]))
 
        self.experiment=classes[kwargs["class_type"]](initilialisation_dict["experiment_type"], 
                        initilialisation_dict["Experiment_parameters"],
                        **initilialisation_dict["Options"])
        self.experiment.fixed_parameters=initilialisation_dict["fixed_parameters"]
        self.experiment.boundaries=initilialisation_dict["boundaries"]
        self.experiment.optim_list=initilialisation_dict["optim_list"]
    def __getattr__(self, name):
        return getattr(self.experiment, name)
class ChangeTechnique(sci.SingleExperiment):
    def __init__(self, json_path, new_experiment,input_parameters):
        with open(json_path, "r") as f:
            initilialisation_dict=json.load(f)
        general_setup_keys=["N_elec", "Temp", "area", "Surface_coverage"]
        for key in initilialisation_dict["Experiment_parameters"]:
            if key not in input_parameters:
                if key in sci.experimental_input_params[new_experiment] or key in general_setup_keys:
                    input_parameters[key]=initilialisation_dict["Experiment_parameters"][key]
        if initilialisation_dict["experiment_type"]==new_experiment:
            raise ValueError("New experiment is the same as the old experiment - raising error")
        super().__init__(new_experiment, 
                        input_parameters,
                        **initilialisation_dict["Options"])
        self.fixed_parameters=initilialisation_dict["fixed_parameters"]
        self.boundaries=initilialisation_dict["boundaries"]
        self.optim_list=initilialisation_dict["optim_list"]
        
