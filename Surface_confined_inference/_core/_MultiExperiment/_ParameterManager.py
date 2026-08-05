import copy
import itertools
import re

import numpy as np
import tabulate

import Surface_confined_inference as sci


class ParameterManager:
    def __init__(self, all_parameters, grouping_keys, classes, SWV_e0_shift, group_to_class):
        #The parameters as defined by the optim_lists of the individual classes. Kept separate
        #from all_parameters, which is derived from them, so that the derivation always starts
        #from the same place and can be repeated whenever the options it depends on change
        self._base_parameters=list(all_parameters)
        self.all_parameters=list(all_parameters)
        self.grouping_keys=grouping_keys
        self.SWV_e0_shift=SWV_e0_shift
        self.classes=classes
        self.group_to_class=group_to_class
    def initialise_simulation_parameters(self, seperated_param_dictionary={}):
        base_parameters=self._base_parameters
        group_to_parameters={x:copy.deepcopy(base_parameters) for x in self.grouping_keys}
        new_all_parameters=[]
        for key in seperated_param_dictionary:
            if key not in base_parameters:
                raise ValueError(f"{key} not in optim_list of any class")
            all_idx=list(itertools.chain(*seperated_param_dictionary[key]))
            set_idx=set(all_idx)#existing_values
            required_idx=list(range(0, len(self.grouping_keys)))#required_values
            if len(set_idx)>len(required_idx):
                raise ValueError(f"More grouping indices ({set_idx}) than number of groups ({required_idx})")

            if len(set_idx)<len(required_idx):
                missing_values=list(set(required_idx).difference(set_idx))
                raise ValueError("{0} in parameter grouping assignment missing indexes for {1}".format(key, " ".join([f"{self.grouping_keys[x]} (index {x})" for x in missing_values])))

            if len(all_idx)!=len(required_idx):
                diffsum=sum(np.diff(list(set_idx)))
               
                if (diffsum-1)!=required_idx[-1]:
                    raise ValueError(f"{all_idx} (in {key}) in parameter grouping assignment contains duplicates")
                else:
                    raise ValueError(f"{all_idx} (in {key}) in parameter grouping assignment contains more indexes that then number of groups ({len(required_idx)})")

            new_all_parameters+=[f"{key}_{x+1}" for x in range(0, len(seperated_param_dictionary[key]))]
            for m in range(0,len(seperated_param_dictionary[key])):
                element=seperated_param_dictionary[key][m]
                for j in range(0, len(element)):
                    group_key=self.grouping_keys[element[j]]
                    p_idx=group_to_parameters[group_key].index(key)
                    group_to_parameters[group_key][p_idx]=f"{key}_{m+1}"
        common_params=[x for x in base_parameters if x not in seperated_param_dictionary]

        all_parameters=new_all_parameters+common_params
        if self.SWV_e0_shift==True:
            all_parameters+=self._e0_offset_parameters(seperated_param_dictionary, group_to_parameters, all_parameters)
        self.all_parameters=all_parameters
        self.group_to_parameters=group_to_parameters
        return group_to_parameters, self.all_parameters
    def _e0_offset_parameters(self, seperated_param_dictionary, group_to_parameters, all_parameters):
        """
        The offset parameters required to shift the anodic and cathodic SWV experiments in
        opposite directions. If E0 is common to every group then one offset is enough, but if
        it has been separated then each group needs the offset belonging to its own copy of E0.
        """
        offsets=[]
        if "E0_mean" not in seperated_param_dictionary and "E0" not in seperated_param_dictionary:
            if "E0_mean" in all_parameters:
                offsets+=["E0_mean_offset"]
            elif "E0" in all_parameters:
                offsets+=["E0_offset"]
            return offsets
        if "E0_mean" in seperated_param_dictionary:
            target="E0_mean"
        else:
            target="E0"
        for groupkey in self.grouping_keys:
            exp=[self.classes[x]["class"].experiment_type=="SquareWave" for x in self.group_to_class[groupkey]]
            if all(exp)==True:
                optim_list=group_to_parameters[groupkey]
                param=[x for x in optim_list if re.search(target+r"_\d", x)][0]+"_offset"
                if param not in offsets:
                    offsets+=[param]
            elif any(exp)==True:
                raise ValueError("If SWV_e0_shift is set to True, all members of a SWV group have to be SquareWave experiments")
        return offsets
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
                            #Only the offset belonging to this group's copy of E0 applies here,
                            #any others belong to a different group
                            if true_param not in parameter_list:
                                continue
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
                                raise ValueError(f"If SWV_e0_shift is set to True, then all SWV experiments must be identified as anodic or cathodic, not {classkey}")
                optimisation_parameters[classkey]=[sim_values[x] for x in cls.optim_list]
        for key in self.classes.keys():
            if key not in optimisation_parameters:
                raise KeyError(f"{key} not added to optimisation list, check that at least one group includes it")
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
        longest_list=[]
        for classkey in class_keys:

            cls=self.classes[classkey]["class"]
            current_len=max(len(cls.optim_list), l_optim_list)
            if current_len>l_optim_list:
                l_optim_list=current_len
                #copied, otherwise the padding below extends the class's own optim_list
                longest_list=list(cls.optim_list)
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
            for classkey in class_keys
        ]
        table=tabulate.tabulate(table_data, headers=header_list, tablefmt="grid")
        if mode=="table":
            print(table)
        elif mode=="save":
            with open(kwargs["filename"], "w") as f:
                f.write(table)
        
        return table