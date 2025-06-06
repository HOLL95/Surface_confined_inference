
import Surface_confined_inference as sci
import importlib
import sys
import os
import json
from numpy import loadtxt
from ._Plotting import PlotManager
from scipy.signal import decimate
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
    @classmethod
    def results_loader(cls, directory_path):
        instance=sci.BaseMultiExperiment.from_directory(os.path.join(directory_path, "evaluator"))
        results_array=[]
        for filename in ["parameters.txt", "scores.txt"]:
            
            with open(os.path.join(directory_path, "pareto_points", filename), "r") as f:
                file=loadtxt(f, skiprows=1)
                f.seek(0)
                header = f.readline().strip().split()[1:]

            
            if len(results_array)==0:
                results_array=[{} for x in range(0, file.shape[0])]
            key=filename.split(".")[0]
            if key=="parameters":
                item="parameter(s)"
                element=instance._all_parameters
            elif key=="scores":
                item="group(s)"
                element=instance.grouping_keys
            expected=set(element)
            present=set(header)
            too_many=present-expected
            not_enough=expected-present
            if len(too_many)!=0:
                raise ValueError("Additional {1} in parameter file versus class instance -{0}".format(too_many, item))
            if len(not_enough)!=0:
                raise ValueError("{1} missing from parameter file versus class instance -{0}".format(not_enough, item.title()))
            for i in range(0, file.shape[0]):
                results_array[i][key]=dict(zip(header, file[i,:]))
        sim_files=os.path.join(directory_path, "simulations")
        best={groupkey:{"score":1e23} for groupkey in instance.grouping_keys}
        saved=False
        for classkey in instance.class_keys:
            path=os.path.join(sim_files, classkey)
            if os.path.isdir(path) is True:
                saved=True
                files=os.listdir(path)
                for file in files:
                    filesplit=file.split("_")
                    first_idx=int(filesplit[1])
                    second_idx=int(filesplit[2].split(".")[0])
                    counter=1
                   
                    for j in range(first_idx, second_idx):
                        if j+1>len(results_array):
                            continue
                        if "saved_simulation" not in results_array[j]:
                            results_array[j]["saved_simulation"]={}
                        results_array[j]["saved_simulation"][classkey]={"col":counter, "address":os.path.join(path, file)}
                        counter+=1  
        if saved==True:
            for classkey in instance.class_keys:
                if instance.classes[classkey]["class"].experiment_type in ["FTACV","PSV"]:
                    with open(os.path.join(directory_path, "decimation.txt"), "r") as f:
                        dec_amount=loadtxt(f)
                    time=instance.classes[classkey]["times"]
                    current=instance.classes[classkey]["data"]
                    instance.classes[classkey]["times"]=decimate(time, 13)
                    instance.classes[classkey]["data"]=decimate(current, 13)
        for j in range(0 ,len(results_array)):
            for groupkey in instance.grouping_keys:
                putative=results_array[j]
                if putative["scores"][groupkey]<best[groupkey]["score"]:
                    best[groupkey]["score"]=putative["scores"][groupkey]
                    best[groupkey]["parameters"]=putative["parameters"]
        instance._best_results=best
        instance._results_array=sci.exclude_copies(results_array)
        instance._plot_manager=PlotManager(instance, instance._results_array)
        client_files=os.listdir(os.path.join(directory_path, "clients"))
        with open(os.path.join(directory_path, "clients", client_files[0]),"r") as f:
            vals=json.load(f)
        thresholds={}
        for i in range(0, len(vals["experiment"]["optimization_config"]["objective_thresholds"])):
            elem=vals["experiment"]["optimization_config"]["objective_thresholds"][i]
            thresholds[elem["metric"]["name"]]=elem["bound"]
           
        instance._thresholds=thresholds
        
        
        

        return instance
             