import datetime
import os
import uuid
from pathlib import Path

import numpy as np

import Surface_confined_inference as sci


class SingleSlurmSetup(sci.SingleExperiment):
    def __init__(self, experiment_type, experiment_parameters, **kwargs):
        super().__init__(experiment_type, experiment_parameters, **kwargs)
        
    def setup(self,**kwargs):
        user=os.environ.get('USER')
        if "datafile" not in kwargs:
            raise ValueError("Need to provide path to datafile")
        else:
            if os.path.isfile(kwargs["datafile"]) is False:
                raise ValueError(kwargs["datafile"]+" not found")
            fileloc=os.path.abspath(kwargs["datafile"])
        path_to_submitter=(sci.__file__).split("/")[:-1]

        hpc_loc="/".join(path_to_submitter)
        submitter_loc=hpc_loc+"/_HPCInterface/Slurm_submitter.py"
        processor_loc=hpc_loc+"/_HPCInterface/Slurm_processor.py"
        RDS_loc=hpc_loc+"/_HPCInterface/RDS.sh"
        os.path.isfile(submitter_loc)
        if "method" not in kwargs:
            kwargs["method"]="optimisation"
        if "email" not in kwargs:
            kwargs["email"]=user+"@york.ac.uk"
        if "cpu_ram" not in kwargs:
            kwargs["cpu_ram"]="8G"
        if "time" not in kwargs:
            kwargs["time"]="0-01:30:00"
        if "runs" not in kwargs:
            kwargs["runs"]=3
        defaults_cmaes={"threshold":1e-6, "unchanged_iterations":200, "check_experiments":{}, "save_csv":False}
        defaults_mcmc={"samples":10000, "CMAES_results_dir":None}
        default_dict={"optimisation":defaults_cmaes, "sampling":defaults_mcmc}
        for key in default_dict[kwargs["method"]].keys():
            if key not in kwargs:
                kwargs[key]=default_dict[kwargs["method"]][key]
            
        if "run" not in kwargs:
            kwargs["run"]=False
        if "results_directory" not in kwargs:
            kwargs["results_directory"]="Results"
        Path("Submission").mkdir(parents=True, exist_ok=True)
        cwd=os.getcwd()
        loc=cwd+"/Submission"
        if kwargs["debug"]!=True:
            Path("{0}/Individual_runs".format(kwargs["results_directory"])).mkdir(parents=True, exist_ok=False)
        if kwargs["method"]=="optimisation":
            num_cpu=4 + int(3 * np.log(self.n_parameters()+self.n_outputs()))            
        else:
            num_cpu=np.prod(self._internal_options.dispersion_bins)
        identifier=uuid.uuid4().hex
        job_name=self._internal_options.experiment_type+"_inference_"+user+"_"+identifier
        ntasks=1
        project="chem-electro-2024"
        output=r"slurm_logs/%x-%j.log"
        error=r"slurm_logs/%x-%j.err"
        runs="1-{0}".format(kwargs["runs"])
        master_dict={
            "--job-name":job_name,
            "--ntasks":1,            # Number of MPI tasks to request
            "--cpus-per-task":num_cpu,               # Number of CPU cores per MPI task
            "--mem":kwargs["cpu_ram"],                       # Total memory to request
            "--time":kwargs["time"],             # Time limit (DD-HH:MM:SS)
            "--account":project,      # Project account to use
            "--mail-type":"END,FAIL",            # Mail events (NONE, BEGIN, END, FAIL, ALL)
            "--mail-user":kwargs["email"],  # Where to send mail
            "--output":output,            # Standard output log
            "--error":error,               # Standard error log
            "--array":runs,                  # Array range"
            "--partition":"nodes"
        }
        save_json=identifier+"_Slurm_Json.json"
        self.save_class("Submission/"+save_json)
        submission_file=f"{identifier}_Automated_slurm_submission.job"
        
        with open("Submission/"+submission_file, "w") as f:
            f.write("#!/usr/bin/env bash\n")
            for key in master_dict.keys():
                write_str=f"#SBATCH {key}={master_dict[key]}\n"
                f.write(write_str)
            f.write("set -e \n")
            f.write('SLURM_LOG_DIR=\"slurm_logs\"\n')
            f.write("mkdir -p $SLURM_LOG_DIR\n")
            if kwargs["method"]=="optimisation":
                f.write('echo \"${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}\" >> '+ kwargs["results_directory"]+ '/Individual_runs/job_ids.txt\n')

                python_command=["python",
                                submitter_loc,
                                fileloc,
                                cwd+"/Submission/"+save_json,
                                kwargs["results_directory"],
                                kwargs["method"],
                                "--threshold={0}".format(kwargs["threshold"]),
                                "--unchanged_iterations={0}".format(kwargs["unchanged_iterations"])
                ]
            else:
                python_command=["python",
                                submitter_loc,
                                fileloc,
                                cwd+"/Submission/"+save_json,
                                kwargs["results_directory"],
                                kwargs["method"],
                                "--chain_samples={0}".format(kwargs["samples"]),
                                
                ]
                if kwargs["CMAES_results_dir"] is not False:
                    python_command+=["--init_point_dir={0}".format(kwargs["CMAES_results_dir"])]
                if isinstance(kwargs["starting_point"], list):
                    python_command+=["--init_point=\"{0}\"".format(" ".join([str(x) for x in kwargs["starting_point"]]))]       

            f.write(" ".join(python_command))
       
      
        process_time=kwargs["runs"]*3
        str_process_time=str(datetime.timedelta(minutes=process_time))
        cleanup_dict={
             
            "--job-name":identifier+"_cleanup",
            "--ntasks":1,            # Number of MPI tasks to request
            "--cpus-per-task":1,               # Number of CPU cores per MPI task
            "--mem":kwargs["cpu_ram"],                       # Total memory to request
            "--time":"0-0"+str_process_time,             # Time limit (DD-HH:MM:SS)
            "--account":project,      # Project account to use
            "--mail-type":"END,FAIL",            # Mail events (NONE, BEGIN, END, FAIL, ALL)
            "--mail-user":kwargs["email"],  # Where to send mail
            "--partition":"nodes"
        }
        cleanup_file=identifier+"_Cleanup.job"
        with open("Submission/"+cleanup_file, "w") as f:
            f.write("#!/usr/bin/env bash\n")
            for key in cleanup_dict.keys():
                write_str=f"#SBATCH {key}={cleanup_dict[key]}\n"
                f.write(write_str)
            f.write("set -e \n")
            if kwargs["method"]=="optimisation":
                python_command=["python",
                                processor_loc,
                                fileloc,
                                cwd+"/Submission/"+save_json,
                                cwd+"/{0}/Individual_runs".format(kwargs["results_directory"]),
                                kwargs["method"],
                                "--JobIds="+cwd+"/{0}/Individual_runs/job_ids.txt".format(kwargs["results_directory"]),

                ]
                check_keys=kwargs["check_experiments"].keys()
                if len(check_keys)>0:
                    
                    checkfiles=["--checkfiles"]
                    checkfile_types=["--checkfile_types"]
                    json_addresses=["--check_parameters"]
                    for key in check_keys:
                        if os.path.isfile(kwargs["check_experiments"][key]["file"]) is False:
                            raise ValueError(kwargs["check_experiments"][key]["file"]+" not found")
                        checkfiles.append(kwargs["check_experiments"][key]["file"])
                        checkfile_types.append(key)
                        if "parameters" in kwargs["check_experiments"][key]:
                            check_technique=sci.SingleExperiment(key, kwargs["check_experiments"][key]["parameters"])
                            
                            check_json_path= cwd+"/Submission/"+identifier+f"_Check_{key}.json"
                            self.save_class(check_json_path, switch_type={"experiment":key, "parameters":kwargs["check_experiments"][key]["parameters"]})
                            json_addresses.append(check_json_path)
                        else:
                            json_addresses.append("none")
                    python_command+=[" ".join(checkfiles)]+[" ".join(checkfile_types)]+[" ".join(json_addresses)]+[" --save_csv {0}".format(kwargs["save_csv"])]
                f.write(" ".join(python_command))
            else:
                python_command=["python",
                                processor_loc,
                                fileloc,
                                cwd+"/Submission/"+save_json,
                                cwd+"/{0}/Individual_runs".format(kwargs["results_directory"]),
                                kwargs["method"],

                ]
                f.write(" ".join(python_command))

        controller_file=identifier+"_Controller.sh"
        with open("Submission/"+controller_file, "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write("array_job_id=$(sbatch Submission/"+submission_file+" | awk '{print $4}')\n")
            f.write(f"sbatch --dependency=afterok:$array_job_id Submission/{cleanup_file}\n" )
        with open("Submission/RemoteDesktopSetup.sh", "w") as f:
            with open(RDS_loc ) as readfile:
             for line in readfile:
              f.write(line)

        if kwargs["debug"]==True:

            datafile=np.loadtxt(fileloc)
            time=datafile[:,0]
            current=datafile[:,1]
            potential=datafile[:,2]
            if kwargs["method"]=="optimisation":
                class_type="base"
                debug_class=sci.FittingDebug("Submission/"+save_json, time, current, potential, dimensional=True, Fourier_fitting=self._internal_options.Fourier_fitting, class_type=class_type)
            elif kwargs["method"]=="sampling":
                class_type="mcmc"
                debug_class=sci.FittingDebug("Submission/"+save_json, time, current, potential, dimensional=True, Fourier_fitting=self._internal_options.Fourier_fitting, class_type=class_type,starting_point=kwargs["starting_point"], CMAES_results_dir=kwargs["CMAES_results_dir"], num_cpu=num_cpu,)
            print(debug_class._internal_options.transient_removal, "slurm",self._internal_options.Fourier_fitting)
            
            #debug_class.simulate(kwargs["starting_point"][:-1], debug_class.nondim_t(time))
            #raise Exception()

            debug_class.go()
        elif kwargs["run"]==True:
            date=datetime.datetime.today().strftime('%Y-%m-%d')
            saveloc="{0}/{2}/PooledResults_{1}".format(os.getcwd(), date, kwargs["results_directory"])
            print("")
            print(f"Results will be written to {saveloc}")
            print(f"To copy this to your personal filestore (when the run is complete) from this terminal window, I think you should run:\n\n scp -r {saveloc} scp.york.ac.uk:/home/userfs/{user[0]}/{user}")
            print("")
            print("To start the remote desktop service and view the results, run")
            print("")
            print(f"source {os.getcwd()}/Submission/RemoteDesktopSetup.sh")
            print("")
            print("And follow the instructions there")
            import subprocess
            subprocess.call(["bash", "Submission/"+controller_file])

        
