import numpy as np
from functools import wraps
import re
import Surface_confined_inference as sci
experimental_input_params={"FTACV":["E_start", "E_reverse", "omega" ,"phase", "delta_E", "v"],
                "DCV":["E_start", "E_reverse",  "v"],
                "PSV":["Edc", "omega", "phase", "delta_E"]}
def add_noise(series, sd, **kwargs):
        if "method" not in kwargs:
            kwargs["method"]="simple"
        if kwargs["method"]=="simple":
            return np.add(series, np.random.normal(0, sd, len(series)))
        elif kwargs["method"]=="proportional":
            noisy_series=copy.deepcopy(series)
            for i in range(0, len(series)):
                noisy_series[i]+=(np.random.normal()*series[i]*sd)
            return noisy_series
def moving_avg(x,n):
    mv =  np.convolve(x,np.ones(n)/n,mode='valid')
    vals=np.concatenate(([np.nan for k in range(n-1)],mv))
    window_len=(n//2)+1
    for k in range(0, n-1):
        vals[k]=np.mean(vals[max(k-window_len, 0):k+window_len])
    return vals
def RMSE(simulation, data):
    """
    Args:
        simulation (list): list of simlation points
        data (list): list of data points to be compared to - needs to be the same length as simulation
    Returns:
        float: root mean squared error between the simulation and data
    """
    if len(simulation)!=len(data):
        raise ValueError("Simulation and data needs to be the same length simulation={0}, data={1}".format(len(simulation), len(data)))
    return np.sqrt(np.mean(np.square(simulation-data)))
def temporary_options(**func_kwargs):
    def decorator(func):
        @wraps(func)
        def wrapper_temporary_options(self, *args, **local_kwargs):
            kwargs={**func_kwargs, **local_kwargs}
            kwargs_set=set(kwargs)
            options_set= set(self._internal_options.get_option_names())
            accepted_keys=list(options_set.intersection(kwargs_set))
            current_options={key:getattr(self, key) for key in accepted_keys}
            for key in accepted_keys:
                setattr(self, key, kwargs[key])
            other_kwargs={key:kwargs[key] for key in kwargs_set-options_set}
            return_arg=func(self,*args, **other_kwargs)
            
            for key in accepted_keys:
                setattr(self, key, current_options[key])
            return return_arg
        return wrapper_temporary_options
    return decorator

def normalise(norm, boundaries):
    """
    Args:
        norm (number): value to be normalised
        boundaries (list): upper and lower bounds, in the format [lower, upper]
    Returns:
        number: Value normalised to value between 0 and 1 relative to the provided boundaries
    """
    return (norm - boundaries[0]) / (boundaries[1] - boundaries[0])

def un_normalise(norm, boundaries):
    """
    Args:
        norm (number): value to be un-normalised
        boundaries (list): upper and lower bounds, in the format [lower, upper] used in the original normalisation
    Returns:
        number: Normalised value is returned to its original value, given the same boundaries used in the initial normalisation
    """
    return (norm * (boundaries[1] - boundaries[0])) + boundaries[0]
def read_param_table(loc, **kwargs):
    return_arg=[]
    if "get_titles" not in kwargs:
        kwargs["get_titles"]=False
    with open(loc, "r")as f:
        lines=f.readlines()
        for line in lines[1:]:
            linelist=re.split(r",\s*", line.strip())
            try:
             float(linelist[-1])
            except:
             linelist=linelist[:-1]
            numeric_line=[float(x) for x in linelist[1:]]
            return_arg.append(numeric_line)
    if kwargs["get_titles"]==False:
        return return_arg
    elif kwargs["get_titles"]==True:
        titles=re.split(r",\s*", lines[0].strip())[1:]
        return return_arg, titles

def custom_linspace(start, end, custom_value, num_points):
  
    values = np.linspace(start, end, num_points)
    
    closest_index = np.argmin(np.abs(values - custom_value))
    
    offset = custom_value - values[closest_index]
    
    
    adjusted_values = values + offset
    
    return adjusted_values

def custom_logspace(start, end, custom_value, num_points):
    values = np.logspace(np.log10(start), np.log10(end), num_points)
    
    closest_index = np.argmin(np.abs(values - custom_value))
    
    offset = custom_value - values[closest_index]
    
    
    adjusted_values = values + offset
    
    return adjusted_values
def construct_experimental_dictionary(existing_dictionary,terminal_entry, *args):
    if len(args)==1:
        existing_dictionary[args[0]]=terminal_entry
        return existing_dictionary
    else:
        if args[0] not in existing_dictionary:
            existing_dictionary[args[0]]={}
        sci.construct_experimental_dictionary(existing_dictionary[args[0]], terminal_entry, *args[1:])
        return existing_dictionary
    



        
