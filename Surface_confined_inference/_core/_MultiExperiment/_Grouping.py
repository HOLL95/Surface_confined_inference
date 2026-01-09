from decimal import Decimal, getcontext
import numpy as np
getcontext().prec = 28

def count_dp(num):

    dec = Decimal(str(num))
    normalized = dec.normalize()
    return max(0, -normalized.as_tuple().exponent)

def initialise_grouping(group_list, classes):
    """
      Parse and validate experiment groupings for multi-experiment  optimization.
      This function processes a list of grouping specifications to determine which experiments should be grouped together based on experiment type, numeric 
      conditions, and match patterns.
      It validates that all grouping criteria are valid and that scaling functions are compatible
      with the grouped experiment classes. Operates over class keys, which are generated from filenames. 
      
      Args:
        group_list (list): List of dictionaries, where each dictionary specifies grouping criteria:
              - "experiment" (str): Experiment type identifier (e.g., "FTACV", "DCV", "PSV")
              - "type" (str): Sub-type specification for the experiment (either "ts" or "ft")
              - "numeric" (dict, optional): Numeric filtering conditions with structure:
                {parameter_name: {qualifier: value}} 
                where qualifier is one of:
                      - "lesser": Matches experiments where parameter < value
                      - "geq": Matches experiments where parameter >= value
                      - "between": Matches experiments where value[0] <= parameter <= value[1]
                      - "equals": Matches experiments where parameter == value
                parameter_name will generally be a unit (e.g. Hz) that exists in at least one loaded filename and consequently experiment key
                in the format unit_parameter (e.g. "FTACV-3_Hz-150_mV.txt")
              - "match" (list, optional): List of strings that must be present in experiment keys
              - "scaling" (dict): Scaling functions to apply, with structure:
                  {function_name: [parameter_list]} where function_name is "divide" or "multiply", and [parameter_list] are values from the input_parameters dict
          classes (dict): Dictionary mapping experiment keys to class instances and metadata.
              Keys are experiment identifiers (e.g., "100.0-mV-s_E_start-FTACV"),
              values are dicts containing "class" and other experiment-specific data.
      
      Returns:
          tuple: A 2-tuple containing:
              - group_to_conditions (dict): Maps generated group keys to their original grouping 
                specifications from group_list
              - group_to_class (dict): Maps generated group keys to lists of experiment keys that 
                match the grouping criteria
      
      Raises:
          ValueError: If a numeric parameter key contains '-' (reserved character)
          ValueError: If a numeric parameter is not found in any experiment keys
          ValueError: If a numeric qualifier has more than one condition specified
          ValueError: If a qualifier is not in the allowed set (lesser, geq,between, equals)
          ValueError: If a scaling function is not in allowed functions (divide, multiply)
          ValueError: If a scaling parameter is not present in an experiment's input_params
          ValueError: If no experiments match the specified grouping conditions
      
      Behavior:
          1. Iterates through each grouping specification in group_list
          2. Filters experiments by experiment type and sub-type
          3. Applies numeric filtering based on parameter values extracted from experiment keys
          4. Applies match filtering to require specific strings in experiment identifiers
          5. Generates unique group keys encoding all filtering criteria with appropriate precision
          6. Validates that scaling functions and parameters are compatible with grouped experiments
          7. Returns mapping from group keys to both conditions and matchingexperiment classes
      
      Key Format:
          Generated group keys have the format:
          "experiment:{exp_type}-type:{type}-{qualifier}:{value}{param}-match:{pattern}"
          Example: "experiment:FTACV-type:dcv-geq:100.0E_start-match:anodic"
      
      """

    class_keys=list(classes.keys())
    
    numeric_qualifiers=["lesser", "geq", "between", "equals"]
    group_to_class={}
    for i in range(0, len(group_list)):
        experiment_list=[x.split("-") for x in list(class_keys) if group_list[i]["experiment"] in x]
        naughty_list=[]
        empty_key=[f"{x}:{group_list[i][x]}" for x in ["experiment", "type"]]
        if "numeric" in group_list[i]:
            
            for key in group_list[i]["numeric"].keys():
                found_unit=False
                in_key=f"_{key}-"
                for expkey in class_keys:
                    if in_key in expkey or expkey[-len(in_key)+1:]==in_key[:-1]:
                        found_unit=True
                if found_unit==False:
                    raise ValueError(f"{key} not found in any experiments")

                    
                if "-" in key:
                    raise ValueError(f"'-'is not allowed in groupkeys ({key})")
                qualifiers=list(group_list[i]["numeric"][key].keys())
                if len(qualifiers)>1:
                    raise ValueError(f"{key} in {empty_key[0]}, {empty_key[1]} has more than one qualifier")
                else:
                    qualifier=qualifiers[0]
                    
                if qualifier not in numeric_qualifiers:
                    raise ValueError(f"{qualifiers[0]} not in allowed qualifiers - lesser, geq, between, equals")
                if qualifier!="between":
                    qualifier_value=float(group_list[i]["numeric"][key][qualifier])
                    if int(qualifier_value)==qualifier_value:
                        empty_key+=["%s:%d%s" % (qualifier, qualifier_value, key)] 
                    else:
                        precision=count_dp(qualifier_value)
                        empty_key+=["%s:%.*f%s" % (qualifier, precision, qualifier_value, key)]
                else:
                    qualifier_value=[float(x) for x in group_list[i]["numeric"][key][qualifier]]
                    groupstr=[f"{qualifier}:"]
                    qval_strs=[]
                    for qval in qualifier_value:
                        if int(qval)!=qval:
                            precision=count_dp(qval)
                            qval_strs.append("%.*f" % (precision, qval))
                        else:
                            qval_strs.append("%d" % qval)
                    groupstr+=["~".join(qval_strs), "%s"%key]
                    empty_key+=["".join(groupstr)]
                for j in range(0, len(experiment_list)):
                    current_exp=experiment_list[j]
                    get_numeric=float([x for x in current_exp if key in x][0].split("_")[0])
                    if qualifier=="lesser":
                        if get_numeric>=qualifier_value:
                            naughty_list.append(j)
                    elif qualifier =="geq":
                        if get_numeric<qualifier_value:
                            naughty_list.append(j)
                    elif qualifier=="between":
                        if get_numeric>qualifier_value[1] or get_numeric<qualifier_value[0]:
                            naughty_list.append(j)
                    elif qualifier=="equals":
                        if get_numeric!=qualifier_value:
                            naughty_list.append(j)
        if "match" in group_list[i]:
            
            for match in group_list[i]["match"]:
                empty_key+=[f"match:{match}"] 
                for j in range(0, len(experiment_list)):
                    if match not in experiment_list[j]:
                        naughty_list.append(j)
        final_key="-".join(empty_key)
        
        final_experiment_list=[experiment_list[i] for i in range(0, len(experiment_list)) if i not in naughty_list]
        
        group_to_class[final_key]=["-".join(x) for x in final_experiment_list]
        grouping_keys=list(group_to_class.keys())     
        group_to_condtions=dict(zip(grouping_keys, group_list)) 
        allowed_functions=["divide", "multiply"]
        for key in group_to_class.keys():
            if len(group_to_class[key])==0:
                raise ValueError(f"No experiments match condtions in {key}")
            for classkey in group_to_class[key]:
                cls=classes[classkey]["class"]
                for function in group_to_condtions[key]["scaling"]:
                    if function not in allowed_functions:
                        raise ValueError(f"Scaling function `{function}` not found in allowed functions {allowed_functions} (group {key})")
                    for parameter in group_to_condtions[key]["scaling"][function]:
                        if parameter not in cls._internal_options.input_params:
                            raise ValueError(f"Scaling parameter {parameter} not present for experiment class {classkey} (group {key})")
    return group_to_condtions, group_to_class
