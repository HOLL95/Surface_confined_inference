# base_experiment.py - Contains the base class
import json
import Surface_confined_inference as sci

class BaseExperiment:
    """Base class for all experiment types"""
    
    @classmethod
    def from_json(cls, json_path):
        """Factory method to create experiment from JSON"""
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Import all experiment modules to ensure registration
        import importlib
        import sys
        

        class_name=data["class"]["name"]
        experiment_class =  getattr(importlib.import_module(data["class"]["module"]),class_name)
        experiment_type= data["Options"]["experiment_type"]
        data["Options"].pop("experiment_type")
        if data["Options_handler"] is None:
            handler_arg=None
        else:
            module=importlib.import_module(data["Options_handler"]["module"])
            handler_arg=getattr(module, data["Options_handler"]["name"])
        
        instance = experiment_class(
            experiment_type,
            data["Experiment_parameters"],
            options_handler=handler_arg,
            **data["Options"]
        )
        # Add additional attributes
        for key in ["fixed_parameters", "boundaries", "optim_list"]:
            if key in data:
                setattr(instance, key, data[key])
        return instance