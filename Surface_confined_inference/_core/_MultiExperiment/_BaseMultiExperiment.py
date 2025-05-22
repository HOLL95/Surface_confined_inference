
import Surface_confined_inference as sci
import importlib
import sys
import os
import json
from numpy import loadtxt
class BaseMultiExperiment:
    """Base class for all experiment types"""
    
    @classmethod
    def from_directory(cls, directory_path):
        class_path=os.path.join(directory_path, "individual_classes", "classes")
        jsons=os.listdir(class_path)
        total_experiment_dict={}
        with open(os.path.join(directory_path,"multi_options.json"), "r") as f:
            global_options=json.load(f)
        for json_file in jsons:
            cls=sci.BaseExperiment.from_json(os.path.join(class_path, json_file))
            ext=json_file.find(".json")
            classname=json_file[:ext]
            keys=classname.split("-")
            options=cls._internal_options.as_dict()
            input_parameters=cls._internal_memory["input_parameters"]
            for key in ["fixed_parameters", "boundaries", "optim_list"]:
                if hasattr(cls, key):
                    options[key]=getattr(cls, key)
            try:
                path=os.path.join(directory_path, "individual_classes", "data", classname, "Zero_params.json")
                with open(path, "r") as f:
                    zero_params=json.load(f)
                
                experiment_dict={**{"Parameters":input_parameters}, **{"Options":options}, "Zero_params":zero_params}
            except:
                experiment_dict={**{"Parameters":input_parameters}, **{"Options":options}}
            
            total_experiment_dict=sci.construct_experimental_dictionary(total_experiment_dict, experiment_dict, *keys)
        instance=sci.MultiExperiment(total_experiment_dict)
        excluded_options=["common", "include_data"]
        if global_options["include_data"]==True:
            instance._internal_options.file_list=global_options["file_list"]
            excluded_options+=["file_list"]
            for json_file in jsons:
                ext=json_file.find(".json")
                classname=json_file[:ext]
                data_loc=os.path.join(directory_path, "individual_classes", "data", classname)
                files=os.listdir(data_loc)
                for file in files:
                    if ".txt" in file:
                        values=loadtxt(os.path.join(data_loc, file))
                        file_key=file[:file.find(".txt")]
                        instance.classes[classname][file_key]=values
        instance.group_list=global_options["group_list"]
        global_options.pop("group_list")
        for key in global_options:
            if key not in excluded_options:
                setattr(instance, key, global_options[key])
        return instance
                

