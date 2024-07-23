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
parser.add_argument("--checkfiles", help="path to files to check against inference results", default=["none"], nargs="+")
parser.add_argument("--checkfile_types", help="types of experiment_files", default=["none"],  nargs="+")

args = parser.parse_args()
datafile=np.loadtxt(args.datafile)
if len(args.checkfiles)!=len(args.checkfile_types):
    raise ValueError("Misconfigured checkfiles - {0} and {1}".format(args.checkfiles, args.checkfile_types))
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
    param_array.append(parameters)
param_array=np.array(param_array)
scores=param_array[:,-1]
sorted_idx=np.flip(np.argsort(scores))
sorted_params=np.array([list(param_array[x,:]) for x in sorted_idx])
sim_voltage=simulator.get_voltage(time, dimensional=True)
sim_dict=simulator.parameter_array_simulate(sorted_params, time)
sim_currents=sim_dict["Current_array"]
DC_voltage=sim_dict["DC_voltage"]
date=datetime.datetime.today().strftime('%Y-%m-%d')
savepath=loc.split("/")
savepath="/".join(savepath[:-1])+"/PooledResults_{0}".format(date)
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
                    score=np.flip(sorted(scores)),
                    parameters=param_array,
                    DC_voltage=DC_voltage
                    )
if "none" not in args.checkfiles:

    for i in range(0, len(args.checkfiles))
        checkloc=savepath+"/"+args.checkfile_types[i]+"_check"
        new_technique=sci.CheckOtherExperiment(args.checkfile_types[i], 
                                args.simulator, 
                                datafile=args.checkfiles[i]
                                )
        Path(checkloc).mkdir(parents=True, exist_ok=True)
        sim_dict=new_technique.parameter_array_simulate(sorted_params, new_technique.time)
        sim_currents=sim_dict["Current_array"]
        DC_voltage=sim_dict["DC_voltage"]
        new_voltage=new_technique.get_voltage(new_technique.time, dimensional=True)
        sci.plot.save_results(new_technique.time, 
                    new_voltage, 
                    new_technique.current, 
                    sim_currents, 
                    checkloc, 
                    new_technique._internal_options.experiment_type, 
                    new_technique._internal_memory["boundaries"],
                    save_csv=True,
                    optim_list=new_technique._optim_list, 
                    fixed_parameters=new_technique.fixed_parameters,
                    score=np.flip(sorted(scores)),
                    parameters=param_array,
                    DC_voltage=DC_voltage,
                    table=False
                    )
        
