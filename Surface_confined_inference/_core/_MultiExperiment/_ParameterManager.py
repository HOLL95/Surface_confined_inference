import numpy as np
import itertools
import copy
import tabulate
class ParameterManager:
    def __init__(self, all_parameters, grouping_keys, classes, SWV_e0_shift, group_to_class):
        self.all_parameters=all_parameters
        self.grouping_keys=grouping_keys
        self.SWV_e0_shift=SWV_e0_shift
        self.classes=classes
        self.group_to_class=group_to_class
    def initialise_simulation_parameters(self, seperated_param_dictionary={}):
        group_to_parameters={x:copy.deepcopy(self.all_parameters) for x in self.grouping_keys}
        if len(seperated_param_dictionary)==0:
            self.group_to_parameters=group_to_parameters
            return group_to_parameters, self.all_parameters
        new_all_parameters=[]
        for key in seperated_param_dictionary:
            if key not in self.all_parameters:
                raise ValueError("{0} not in optim_list of any class".format(key))
            all_idx=list(itertools.chain(*seperated_param_dictionary[key]))
            set_idx=set(all_idx)#existing_values
            required_idx=list(range(0, len(self.grouping_keys)))#required_values
            if len(set_idx)>len(required_idx):
                raise ValueError("More grouping indices ({0}) than number of groups ({1})".format(set_idx, required_idx))

            if len(set_idx)<len(required_idx):
                missing_values=list(set(required_idx).difference(set_idx))
                raise ValueError("{0} in parameter grouping assignment missing indexes for {1}".format(key, " ".join(["{0} (index {1})".format(self.grouping_keys[x],x) for x in missing_values])))

            if len(all_idx)!=len(required_idx):
                diffsum=sum(np.diff(list(set_idx)))
               
                if (diffsum-1)!=required_idx[-1]:
                    raise ValueError("{0} (in {1}) in parameter grouping assignment contains duplicates".format(all_idx, key))
                else:
                    raise ValueError("{0} (in {1}) in parameter grouping assignment contains more indexes that then number of groups ({2})".format(all_idx, key, len(required_idx)))

            new_all_parameters+=["{0}_{1}".format(key, x+1) for x in range(0, len(seperated_param_dictionary[key]))]
            for m in range(0,len(seperated_param_dictionary[key])):
                element=seperated_param_dictionary[key][m]
                for j in range(0, len(element)):
                    group_key=self.grouping_keys[element[j]]
                    p_idx=group_to_parameters[group_key].index(key)
                    group_to_parameters[group_key][p_idx]="{0}_{1}".format(key, m+1)
        common_params=[x for x in self.all_parameters if x not in seperated_param_dictionary]
        
        self.all_parameters=new_all_parameters+common_params
        if self.SWV_e0_shift==True:
            if "E0_mean" not in seperated_param_dictionary and "E0" not in seperated_param_dictionary:
                if "E0_mean" in self.all_parameters:
                    self.all_parameters+=["E0_mean_offset"]
                elif "E0" in self.all_parameters:
                    self.all_parameters+=["E0_offset"]
            else:
                if "E0_mean" in seperated_param_dictionary:
                    target="E0_mean"
                elif "E0" in seperated_param_dictionary:
                    target="E0"
                for groupkey in self.grouping_keys:
                    exp=[self.classes[x]["class"].experiment_type=="SquareWave" for x in self.experiment_grouping[groupkey]]
                    if all(exp)==True:
                        optim_list=group_to_parameters[group_key]
                        param=[x for x in optim_list if re.search(target+r"_\d", x)][0]+"_offset"
                        if param not in self.all_parameters:
                            self.all_parameters+=[param]
                    elif any(exp)==True:
                        raise ValueError("If SWV_e0_shift is set to True, all members of a SWV group have to be SquareWave experiments")
        self.group_to_parameters=group_to_parameters
        return group_to_parameters, self.all_parameters
    def parse_input(self, parameters):
        in_optimisation=False
        try:
            values=copy.deepcopy([parameters.get(x) for x in self.all_parameters])
            valuedict=dict(zip(self.all_parameters, values))
            in_optimisation=True
        except:
            valuedict=dict(zip(self.all_parameters, copy.deepcopy(parameters)))
        optimisation_parameters={}
        for group_key in self.grouping_keys:
            parameter_list=self.group_to_parameters[group_key]
            sim_values={}
            for classkey in self.group_to_class[group_key]:
                cls=self.classes[classkey]["class"]
                for param in parameter_list:
                    if param in cls.optim_list:
                        sim_values[param]=valuedict[param]
                    elif "_offset" in param:
                        continue
                    else:
                        found_parameter=False
                        for param2 in cls.optim_list:
                            changed_param=param2+"_"
                            if changed_param in param:
                                sim_values[param2]=valuedict[param]
                                found_parameter=True
                                break
                for param in self.all_parameters:
                    if self.classes[classkey]["class"].experiment_type!="SquareWave":
                        continue
                    elif self.SWV_e0_shift==True:
                        if "offset" in param:
                            idx=param.find("_offset")
                            true_param=param[:idx]
                            if true_param not in cls.optim_list:
                                for param2 in cls.optim_list:
                                    changed_param=param2+"_"
                                    if changed_param in param:
                                        true_param=param2
                            if "anodic" in classkey:
                                sim_values[true_param]+=valuedict[param]
                            elif "cathodic" in classkey:
                                sim_values[true_param]-=valuedict[param]
                            else:
                                raise ValueError("If SWV_e0_shift is set to True, then all SWV experiments must be identified as anodic or cathodic, not {0}".format(key))
                optimisation_parameters[classkey]=[sim_values[x] for x in cls.optim_list]
        for key in self.classes.keys():
            if key not in optimisation_parameters:
                raise KeyError("{0} not added to optimisation list, check that at least one group includes it".format(key))
        return optimisation_parameters   
    def results_table(self, parameters, class_keys,**kwargs):
        if "mode" not in kwargs:
            kwargs["mode"]="table"
        if kwargs["mode"]=="save":
            if "filename" not in kwargs:
                kwargs["filename"]="results_table.txt"
        mode=kwargs["mode"]
        simulation_values=self.parse_input(parameters)
        un_normed_values={}
        l_optim_list=0
        for classkey in class_keys:
            
            cls=self.classes[classkey]["class"]
            current_len=max(len(cls.optim_list), l_optim_list)
            if current_len>l_optim_list:
                l_optim_list=current_len
                longest_list=cls.optim_list
            normed_params_list=simulation_values[classkey]
            un_normed_values[classkey]=dict(zip(cls.optim_list, cls.change_normalisation_group(normed_params_list, "un_norm")))
            if mode=="simulation":
                print(classkey)
                print(un_normed_values)
        if mode=="simulation":
            return
        for classkey in class_keys:
            cls=self.classes[classkey]["class"]
            for param in cls.optim_list:
                if param not in longest_list:
                    longest_list+=[param]
        header_list=["Parameter"]+longest_list
        table_data=[
            [classkey]+[sci._utils.format_values(un_normed_values[classkey][x],3)+","
                if x in un_normed_values[classkey] else "*"
                for x in longest_list]
            for classkey in self.class_keys
        ]
        table=tabulate.tabulate(table_data, headers=header_list, tablefmt="grid")
        if mode=="table":
            print(table)
        elif mode=="save":
            with open(kwargs["filename"], "w") as f:
                f.write(table)
        
        return table