import numpy as np
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