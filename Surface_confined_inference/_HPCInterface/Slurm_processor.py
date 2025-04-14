import numpy as np
import Surface_confined_inference as sci
from pathlib import Path
import argparse
import copy
import datetime
import matplotlib.pyplot as plt
import os
parser=argparse.ArgumentParser("Slurm processor")
parser.add_argument("datafile", help="time-current-potential data filename", type=str)
parser.add_argument("simulator", help="Json filename that initilises simulator class", type=str)

parser.add_argument("resultsLoc", help="path to results files", type=str)
parser.add_argument("method", help="Sampling or Optimisation", type=str)
parser.add_argument("--JobIds", help="JobID file location", type=str)
parser.add_argument("--checkfiles", help="path to files to check against inference results", default=["none"], nargs="+")
parser.add_argument("--checkfile_types", help="types of experiment_files", default=["none"],  nargs="+")
parser.add_argument("--check_parameters", help="types of experiment_files", default=["none"],  nargs="+")
parser.add_argument("--save_csv", help="whether to save csv files or not", default=False,  type=bool)

args = parser.parse_args()

if args.method=="optimisation":
    datafile=np.loadtxt(args.datafile)
    if len(args.checkfiles)!=len(args.checkfile_types):
        raise ValueError("Misconfigured checkfiles - {0} and {1}".format(args.checkfiles, args.checkfile_types))
    time=datafile[:,0]
    potential=datafile[:,2]
    current=datafile[:,1]
    with open(args.JobIds, "r") as f:
        ids=f.readlines()
    simulator=sci.BaseExperiment.from_json(args.simulator)
    loc=args.resultsLoc
    param_array=[]
    for i in range(0, len(ids)):
        curr_id=ids[i].strip()

        try:
            int(curr_id[0])
            parameter_file=loc+"/Results_run_{0}.npy".format(curr_id)
        except:
            continue
        parameters=np.load(parameter_file)
        param_array.append(parameters)
    param_array=np.array(param_array)
    print(param_array)
    scores=param_array[:,-1]
    sorted_idx=np.argsort(scores)
    sorted_params=np.array([list(param_array[x,:]) for x in sorted_idx])

    if simulator._internal_options.transient_removal!=0:
        dim_transient=simulator.dim_t(simulator._internal_options.transient_removal)
        time_idx=np.where(time>dim_transient)
        subtime=time[time_idx]
        potential=potential[time_idx]
        current=current[time_idx]
    else:
        subtime=time
    if simulator._internal_options.experiment_type=="SquareWave":
        sim_voltage=simulator._internal_memory["SW_params"]["E_p"]
    else:
        sim_voltage=simulator.get_voltage(time, dimensional=True)

    date=datetime.datetime.today().strftime('%Y-%m-%d')
    savepath=loc.split("/")
    savepath="/".join(savepath[:-1])+"/PooledResults_{0}".format(date)
    Path(savepath).mkdir(parents=True, exist_ok=True)


    sim_dict=simulator.parameter_array_simulate(sorted_params, time)
    sim_currents=sim_dict["Current_array"]
    DC_voltage=sim_dict["DC_voltage"]
    if simulator._internal_options.transient_removal!=0:
        sim_voltage=np.array(sim_voltage)[time_idx]
        sim_currents=np.array([x[time_idx] for x in sim_dict["Current_array"]])
    if DC_voltage is not None:
        if simulator._internal_options.transient_removal!=0:
                DC_voltage=DC_voltage[time_idx]
    sci.plot.save_results(time, 
                        sim_voltage, 
                        current, 
                        sim_currents, 
                        savepath, 
                        simulator._internal_options.experiment_type, 
                        simulator._internal_memory["boundaries"],                    
                        optim_list=simulator._optim_list, 
                        fixed_parameters=simulator.fixed_parameters,
                        score=sorted_params[:,-1],
                        parameters=sorted_params,
                        DC_voltage=DC_voltage,
                        save_csv=args.save_csv
                        )

    if "none" not in args.checkfiles:
        check_jsons=args.check_parameters
        for i in range(0, len(args.checkfiles)):
            checkloc=savepath+"/"+args.checkfile_types[i]+"_check"
        
            new_technique=sci.BaseExperiment.from_json(check_jsons[i])
            data=np.loadtxt(args.checkfiles[i])
            n_time=data[:,0]
            n_current=data[:,1]
            Path(checkloc).mkdir(parents=True, exist_ok=True)
            sim_dict=new_technique.parameter_array_simulate(sorted_params, n_time)
            sim_currents=sim_dict["Current_array"]
            DC_voltage=sim_dict["DC_voltage"]
            new_voltage=new_technique.get_voltage(n_time, dimensional=True)
            sci.plot.save_results(n_time, 
                        new_voltage, 
                        n_current, 
                        sim_currents, 
                        checkloc, 
                        new_technique._internal_options.experiment_type, 
                        new_technique._internal_memory["boundaries"],
                        save_csv=args.save_csv,
                        optim_list=new_technique._optim_list, 
                        fixed_parameters=new_technique.fixed_parameters,
                        score=sorted_params[:,-1],
                        parameters=sorted_params,
                        DC_voltage=DC_voltage,
                        table=False
                        )
            
else:
 files=os.listdir(args.resultsLoc)
 from pints.plot import trace
 simulator=sci.BaseExperiment.from_json(args.simulator)
 chain_data=[np.load(os.path.join(args.resultsLoc, file)) for file in files]
 full_chain=np.concatenate(chain_data, axis=0)
 try:
  trace(full_chain, parameter_names=simulator._optim_list+["Noise"])
 except:
  trace(full_chain, parameter_names=simulator._optim_list)
 fig=plt.gcf()
 fig.set_size_inches(9,9)
 up_one=args.resultsLoc.split("/")
 up_one="/".join(up_one[:-1])
 fig.savefig(up_one+"/Trace.png", dpi=300)
  
