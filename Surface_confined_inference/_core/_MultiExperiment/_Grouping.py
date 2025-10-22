from decimal import Decimal
import numpy as np
def count_dp(num):
    floor=np.floor(num)
    dp=num-floor
    numstr=str(dp)
    return len(Decimal(numstr).normalize().as_tuple().digits)
def initialise_grouping(group_list, classes):
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
                                precision=count_dp(qualifier_value)
                                qval_strs.append("%.*f" % (precision, qualifier_value))
                            else:
                                qval_strs.append("%d" % qualifier_value)
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
