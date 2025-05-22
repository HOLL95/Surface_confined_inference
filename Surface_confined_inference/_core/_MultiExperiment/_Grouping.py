import Surface_confined_inference as sci
def initialise_grouping(group_list, classes):
        class_keys=list(classes.keys())
       
        numeric_qualifiers=["lesser", "geq", "between", "equals"]
        group_to_class={}
        for i in range(0, len(group_list)):
            experiment_list=[x.split("-") for x in list(class_keys) if group_list[i]["experiment"] in x]
            naughty_list=[]
            empty_key=["{0}:{1}".format(x, group_list[i][x]) for x in ["experiment", "type"]]
            if "numeric" in group_list[i]:
                
                for key in group_list[i]["numeric"].keys():
                    found_unit=False
                    in_key="_{0}-".format(key)
                    for expkey in class_keys:
                        if in_key in expkey or expkey[-len(in_key)+1:]==in_key[:-1]:
                            found_unit=True
                    if found_unit==False:
                        raise ValueError("{0} not found in any experiments".format(key))

                        
                    if "-" in key:
                        raise ValueError("'-'is not allowed in groupkeys ({0})".format(key))
                    qualifiers=list(group_list[i]["numeric"][key].keys())
                    if len(qualifiers)>1:
                        raise ValueError("{0} in {1}, {2} has more than one qualifier".format(key, empty_key[0], empty_key[1]))
                    else:
                        qualifier=qualifiers[0]
                        
                    if qualifier not in numeric_qualifiers:
                        raise ValueError("{0} not in allowed qualifiers - lesser, geq, between, equals".format(qualifiers[0]))
                    if qualifier!="between":
                        qualifier_value=float(group_list[i]["numeric"][key][qualifier])
                        empty_key+=["%s:%d%s" % (qualifier, qualifier_value, key)] 
                    else:
                        qualifier_value=[float(x) for x in group_list[i]["numeric"][key][qualifier]]
                        empty_key+=["%s:%d~%d%s" % (qualifier, qualifier_value[0],qualifier_value[1], key)]
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
                    empty_key+=["match:{0}".format(match)] 
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
                raise ValueError("No experiments match condtions in {0}".format(key))
            for classkey in group_to_class[key]:
                cls=classes[classkey]["class"]
                for function in group_to_condtions[key]["scaling"]:
                    if function not in allowed_functions:
                        raise ValueError("Scaling function `{0}` not found in allowed functions {1} (group {2})".format(function, allowed_functions, key))
                    for parameter in group_to_condtions[key]["scaling"][function]:
                        if parameter not in cls._internal_memory["input_parameters"]:
                            raise ValueError("Scaling parameter {0} not present for experiment class {1} (group {2})".format(parameter, classkey, key))
        return group_to_condtions, group_to_class