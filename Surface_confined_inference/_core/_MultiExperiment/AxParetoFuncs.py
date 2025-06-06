from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.plot.pareto_utils import get_observed_pareto_frontiers
import os
from numpy import savetxt

def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))
def internal_domination(front, keys):
    non_dominated = []
    for i, candidate in enumerate(front):
        candidate_point = [candidate["scores"][key] for key in keys]
        is_dominated = False
        for j, other in enumerate(front):
            if i == j:
                continue  
            other_point = [other["scores"][key] for key in keys]
            if dominates(other_point, candidate_point):
                is_dominated = True
                break  
        if not is_dominated:
            non_dominated.append(candidate)
    return non_dominated
def exclude_copies(front):
    excluded_idx=[]
    keys=front[0]["parameters"].keys()
    for i in range(0, len(front)):
        if i not in excluded_idx:
            params=[front[i]["parameters"][key] for key in keys]
            for j in range(0, len(front)):
                if i==j:
                    continue
                if j not in excluded_idx:
                    target_params=[front[j]["parameters"][key] for key in keys]
                    equality=[x==y for x,y in zip(params, target_params)]
                    if all(equality) is True:
                        excluded_idx.append(j)
    return [front[x] for x in range(0, len(front)) if x not in excluded_idx]
            
def is_dominated(existing_front, proposed_front, keys, test_print=False):
    removal_idx=set()
    added_idx=list(range(0, len(proposed_front)))
    for i in range(0, len(existing_front)):
            front_point=[existing_front[i]["scores"][key] for key in keys]
            for j in range(0, len(proposed_front)):
                if j in added_idx:
                    proposed_point=[proposed_front[j]["scores"][key] for key in keys]
                    front_dominated=dominates(proposed_point, front_point)
                    proposed_dominated=dominates(front_point, proposed_point)
                    if proposed_dominated is True:
                        added_idx.remove(j)
                    elif front_dominated is True:
                        removal_idx.add(i)
                        break 
    
    final_frontier=[existing_front[x] for x in range(0, len(existing_front)) if x not in removal_idx]+[proposed_front[x] for x in added_idx]
    return exclude_copies(internal_domination(final_frontier, keys))

def pool_pareto(directory, grouping_keys, parameters, savepath):
    total_front_dict={}
    files=os.listdir(directory)
    for m in range(0, len(files)):
        file=files[m]
        path=os.path.join(directory, file)
        try:
            client=AxClient.load_from_json_file(filepath=path)
        except:
            continue
        front=get_observed_pareto_frontiers(client.experiment)
        frontdict={}
        for i in range(0, len(front)):
            key="&".join([front[i].primary_metric, front[i].secondary_metric])
            frontdict[key]=[{"parameters":front[i].param_dicts[z], "scores":{key:front[i].means[key][z] for key in grouping_keys }}
                            for z in range(0, len(front[i].param_dicts))]
        frontdict["thresholds"]=front[i].objective_thresholds
        loaded_frontier=[]
    
        keys=frontdict.keys()
        
        for key in keys:
            if "thresholds" not in key:
                loaded_frontier+=frontdict[key]
        if m==0:
            
            total_frontier=loaded_frontier

        else:
            total_frontier=is_dominated(total_frontier, loaded_frontier, grouping_keys)
    parameter_array=[]
    score_array=[]
    for front_point in total_frontier:
        parameter_array.append([front_point["parameters"][x] for x in parameters])
        score_array.append([front_point["scores"][x] for x in grouping_keys])
    arrays=[parameter_array, score_array]
    headers=[parameters, grouping_keys]
    exts=["parameters.txt", "scores.txt"]
    for i in range(0,len(exts)):
        with open(os.path.join(savepath, exts[i]), "w") as f:
            savetxt(f,arrays[i], header=" ".join(headers[i]))
    with open(os.path.join(savepath, "num_points.txt"), "w") as f:
        f.write(str(len(parameter_array)))
