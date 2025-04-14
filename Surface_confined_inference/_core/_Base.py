# base_experiment.py - Contains the base class
import json
import Surface_confined_inference as sci

class BaseExperiment:
    """Base class for all experiment types"""
    
    @classmethod
    def from_json(cls, json_path, **kwargs):
        """Factory method to create experiment from JSON"""
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Import all experiment modules to ensure registration
        import importlib
        import sys
        
        # Determine experiment class to instantiate
        if "class_type" not in kwargs:
            kwargs["class_type"]="single"

        experiment_class =  sci.LoadExperiment.get_class(kwargs["class_type"])
        
        # Create instance
        instance = experiment_class(
            data["experiment_type"],
            data["Experiment_parameters"],
            **data["Options"]
        )
        
        # Add additional attributes
        for key in ["fixed_parameters", "boundaries", "optim_list"]:
            if key in data:
                setattr(instance, key, data[key])
                
        return instance