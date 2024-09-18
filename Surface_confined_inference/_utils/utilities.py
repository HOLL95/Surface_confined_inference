import numpy as np
from functools import wraps
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
            options_set=set(self._internal_options.options.accepted_arguments.keys())
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
def read_param_table(loc):
    return_arg=[]
    with open(loc, "r")as f:
        lines=f.readlines()
        for line in lines[1:]:
            linelist=re.split(r",\s+", line)
            numeric_line=[float(x) for x in linelist[1:-1]]
            return_arg.append(numeric_line)
    return return_arg
