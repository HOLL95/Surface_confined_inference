import numpy as np
import Surface_confined_inference as sci
from pathlib import Path
import argparse
import copy
import datetime
parser=argparse.ArgumentParser("Slurm processor")
parser.add_argument("datafile", help="time-current-potential data filename", type=str)
parser.add_argument("simulator", help="Json filename that initilises simulator class", type=str)
parser.add_argument("JobIds", help="JobID file location", type=str)
parser.add_argument("resultsLoc", help="path to results files", type=str)

args = parser.parse_args()
datafile=np.loadtxt(args.datafile)
time=datafile[:,0]
potential=datafile[:,2]
current=datafile[:,1]
with open(args.JobIds, "r") as f:
    ids=f.readlines()
simulator=sci.LoadSingleExperiment(args.simulator)
loc=args.resultsLoc
param_array=[]
for i in range(0, len(ids)):
    parameter_file=loc+"/Results_run_{0}.npy".format(ids[i].strip())
    parameters=np.load(parameter_file)
    print(parameters)
    param_array.append(parameters)
param_array=np.array(param_array)
scores=param_array[:,-1]
sorted_idx=np.flip(np.argsort(scores))
sorted_params=np.array([list(param_array[x,:]) for x in sorted_idx])
sim_currents=np.zeros((len(sorted_params), len(current)))
print(scores)
for i in range(0, len(sorted_params)):
    sim_currents[i, :]=simulator.dim_i(simulator.Dimensionalsimulate(sorted_params[i,:-1], time))
sim_voltage=simulator.get_voltage(time, dimensional=True)
if simulator._internal_options.experiment_type=="FTACV":
    DC_params=copy.deepcopy(simulator._internal_memory["input_parameters"])
    DC_params["delta_E"]=0
    DC_voltage=simulator.get_voltage(time, dimensional=True, input_parameters=DC_params)
else:
    DC_voltage=None
date=datetime.datetime.today().strftime('%Y-%m-%d')
savepath=loc+"/PooledResults_{0}".format(date)
Path(savepath).mkdir(parents=True, exist_ok=True)
sci.plot.save_results(time, 
                    sim_voltage, 
                    current, 
                    sim_currents, 
                    savepath, 
                    simulator._internal_options.experiment_type, 
                    simulator._internal_memory["boundaries"],
                    save_csv=True,
                    optim_list=simulator._optim_list, 
                    fixed_parameters=simulator.fixed_parameters,
                    score=np.flip(sorted(scores))
                    parameters=param_array,
                    DC_voltage=DC_voltage
                    )

