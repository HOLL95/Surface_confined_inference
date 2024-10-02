import numpy as np
import json
import Surface_confined_inference as sci
class LoadSingleExperiment:
    @staticmethod
    def get_factory_class(json_path, **kwargs):
        allowed_args = ["base", "slurm", "mcmc"]
        classes = dict(zip(allowed_args, 
                           [sci.SingleExperiment, sci.SingleSlurmSetup, sci.RunSingleExperimentMCMC]))
        
        if "class_type" not in kwargs:
            kwargs["class_type"] = "base"
        elif kwargs["class_type"] not in allowed_args:
            raise KeyError(f"class_type needs to be one of {allowed_args}, not {kwargs['class_type']}")
        
        class FactoryClass(classes[kwargs["class_type"]]):
            def __init__(self, json_path):
                with open(json_path, "r") as f:
                    initialization_dict = json.load(f)
                
                super().__init__(initialization_dict["experiment_type"], 
                                 initialization_dict["Experiment_parameters"],
                                 **initialization_dict["Options"])
                self.fixed_parameters = initialization_dict["fixed_parameters"]
                self.boundaries = initialization_dict["boundaries"]
                self.optim_list = initialization_dict["optim_list"]
        
        return FactoryClass

    def __new__(cls, json_path, **kwargs):
        FactoryClass = cls.get_factory_class(json_path, **kwargs)
        return FactoryClass(json_path)

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
        
