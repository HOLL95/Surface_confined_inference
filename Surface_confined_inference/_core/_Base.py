# base_experiment.py - Contains the base class
import json


class BaseExperiment:
    """Base class for all experiment types"""
    
    @classmethod
    def from_json(cls, json_path):
        """
        Factory method to create experiment instances from JSON configuration files.
        
        This method reads a JSON file containing experiment configuration and creates the appropriate
        experiment class instance based on the specifications in the file, where the .json file has been generated using the "save_class" function in SingleExperiment or its child classes. 
        Allows for sharing instances of the simulator class between machines or e.g. slurm processes. 
        
        Args:
            json_path (str): Path to the JSON configuration file containing experiment specifications.
                           The JSON file should contain:
                           - "class": Dictionary with "name" and "module" keys specifying the experiment class
                           - "Options": Dictionary containing experiment options including "experiment_type"
                           - "Experiment_parameters": Dictionary of experiment parameters
                           - "Options_handler": Optional handler configuration (can be None), will default to SingleExperimentOptions.
                           - Optional keys: "fixed_parameters", "boundaries", "optim_list"
        
        Returns:
            BaseExperiment: An instance of the appropriate experiment class (e.g., SingleExperiment, or any of its child classes)
                          configured with the parameters and options from the JSON file.
        
        Raises:
            FileNotFoundError: If the specified json_path does not exist
            json.JSONDecodeError: If the JSON file is malformed
            KeyError: If required keys are missing from the JSON configuration
            ImportError: If the specified module or class cannot be imported
            AttributeError: If the specified class or handler cannot be found in the module
            
        Usage:
            experiment = BaseExperiment.from_json("config.json")
            
        """
        with open(json_path) as f:
            data = json.load(f)
        
        # Import all experiment modules to ensure registration
        import importlib
        

        class_name=data["class"]["name"]

            
        if data["class"]["module"]=="Surface_confined_inference.infer._RunMCMC":
            data["class"]["module"]="Surface_confined_inference._core._Voltammetry"
            class_name="SingleExperiment"
            data["Options_handler"]=None
            if "phase_flag"  in data["Experiment_parameters"]:
                data["Experiment_parameters"].pop("phase_flag")
            if "num_cpu" in data["Options"]:
                data["Options"]["parallel_cpu"]=data["Options"]["num_cpu"]
                data["Options"].pop("num_cpu")
            
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