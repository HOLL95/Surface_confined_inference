import Surface_confined_inference as sci
def validate_input_dict(dictionary, path):
        keylist=list(dictionary.keys())
        if "Parameters" in keylist:
            return
        else:
            validator=["_" in x for x in keylist]
            if all(validator):
                values=[x.split("_") for x in keylist]
                units=set([x[1] for x in values])
                if len(units)>1:
                    raise ValueError("All units in node {1} need to be the same (currently {0} in {2})".format(set(units), "".join(["[{0}]".format(x) for x in path]), keylist))
                for i in range(0, len(values)):
                    try:
                        float(values[i][0])
                    except:
                        raise ValueError("Units need to be convertable to numbers  -> {0} {1}".format(values[i][0], values[i][1]))
            elif not any(validator):
                pass
            else:
                raise ValueError("All labels in node {1} need to be the same (currently {0})".format(keylist, "".join(["[`{0}`]".format(x) for x in path])))
            for key in keylist:
                updated_path=path+[key]
                validate_input_dict(dictionary[key], updated_path)

class InitialiseMultiExperiment:
    def __init__(self,experiment_dict,**kwargs):
        if "common" not in kwargs:
            kwargs["common"]={}
        if "boundaries" not in kwargs:
            kwargs["boundaries"]=None
        if "normalise" not in kwargs:
            kwargs["normalise"]=False
        if "sim_class" not in kwargs:
            kwargs["sim_class"]=sci.SingleExperiment
        if "file_list" not in kwargs:
            kwargs["file_list"]=None
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        if self.normalise==True and self.boundaries is None:
            raise ValueError("If `normalise` is True, then boundaries have to be set")
        
        self.all_keys=[]
        self.classes={}
        for experiment in experiment_dict.keys():
            if experiment=="SWV":
                exp_class="SquareWave"
            else:
                exp_class=experiment
            self._process_experiment_conditions(exp_class, experiment_dict[experiment], [experiment])
    def _process_experiment_conditions(self,experiment, conditions_dict,
                                    current_key_parts):
            """
            Recursively process nested experiment conditions to initialize experiment classes.
            
            Parameters:
            -----------
            experiment : str
                The experiment type (e.g., 'FTACV', 'SWV').
            conditions_dict : dict
                Dictionary containing conditions at the current level.
            files : list
                List of files in the data location.
            location : str
                Path to the data location.
            current_key_parts : list
                Parts of the experiment key accumulated so far.
            current_path : list
                Path through the conditions dict to reach the current point.
            """
            # Check if we've reached a leaf node (actual experiment parameters)
            keys=conditions_dict.keys()
            missing=set(["Parameters", "Options", "Zero_params"])-set(keys)
            if len(missing)>0 and len(missing)<3:
                raise KeyError("Missing the following elements from the experimental dictionary: {0}".format(" / ".join(list(missing))))
            if "Parameters" in keys and "Options"  in keys and "Zero_params" in keys:
                experiment_params=conditions_dict["Parameters"]
                # We've reached experiment parameters, create the experiment class
                experiment_key = "-".join(current_key_parts)
                self.all_keys.append(experiment_key)
                # Add extra parameters
                for key in self.common:
                    experiment_params[key] = self.common[key]
 
                
                # Create experiment class
                self.classes[experiment_key] = {
                    "class": self.sim_class(
                        experiment,
                        experiment_params,
                        problem="forwards",
                        normalise_parameters=self.normalise
                    )
                }
                self.classes[experiment_key]["Zero_params"]=conditions_dict["Zero_params"]
                try:
                    if self.boundaries is not None:
                        self.classes[experiment_key]["class"].boundaries = self.boundaries
                    if experiment=="SquareWave":
                        self.classes[experiment_key]["class"].fixed_parameters = {"Cdl": 0}
                    for key in conditions_dict["Options"].keys():
                        if key !="input_params":
                            self.classes[experiment_key]["class"].__setattr__(key, conditions_dict["Options"][key])
                except Exception as e:
                    raise KeyError("Error processing {0}:{1}".format(experiment_key, str(e)))

             
            else:
                # We haven't reached a leaf node, continue recursion
                for key in conditions_dict.keys():
                    new_key_parts = current_key_parts + [key]
                    if isinstance(conditions_dict[key], dict):
                        self._process_experiment_conditions(
                            experiment=experiment,
                            conditions_dict=conditions_dict[key],
                            current_key_parts=new_key_parts,
                        )
