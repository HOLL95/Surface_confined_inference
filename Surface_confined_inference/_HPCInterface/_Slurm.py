import numpy as np
import Surface_confined_inference as sci
import os
import uuid
from pathlib import Path
import datetime
class SingleSlurmSetup(sci.SingleExperiment):
    def __init__(self, experiment_type, experiment_parameters, **kwargs):
        super().__init__(experiment_type, experiment_parameters, **kwargs)
        
    def setup(self,**kwargs):
        user=os.environ.get('USER')
        if "datafile" not in kwargs:
            raise ValueError("Need to provide path to datafile")
        else:
            os.path.isfile(kwargs["datafile"])
            fileloc=os.path.abspath(kwargs["datafile"])
        path_to_submitter=(sci.__file__).split("/")[:-1]

        hpc_loc="/".join(path_to_submitter)
        submitter_loc=hpc_loc+"/_HPCInterface/Slurm_submitter.py"
        processor_loc=hpc_loc+"/_HPCInterface/Slurm_processor.py"
        os.path.isfile(submitter_loc)
        if "email" not in kwargs:
            kwargs["email"]=user+"@york.ac.uk"
        if "cpu_ram" not in kwargs:
            kwargs["cpu_ram"]="8G"
        if "time" not in kwargs:
            kwargs["time"]="0-01:30:00"
        if "runs" not in kwargs:
            kwargs["runs"]=2
        if "threshold" not in kwargs:
            kwargs["threshold"]=1e-6
        if "unchanged_iterations" not in kwargs:
            kwargs["unchanged_iterations"]=200
        if "run" not in kwargs:
            kwargs["run"]=False
        Path("Submission").mkdir(parents=True, exist_ok=True)
        Path("Results").mkdir(parents=True, exist_ok=True)
        cwd=os.getcwd()
        loc=cwd+"/Submission"
        num_cpu=4 + int(3 * np.log(self.n_parameters()+self.n_outputs()))            
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
        
        self.save_class("Submission/Slurm_Json.json")
        with open("Submission/Automated_slurm_submission.job", "w") as f:
            f.write("#!/usr/bin/env bash\n")
            for key in master_dict.keys():
                write_str="#SBATCH {0}={1}\n".format(key, master_dict[key])
                f.write(write_str)
            f.write("set -e \n")
            f.write('SLURM_LOG_DIR=\"slurm_logs\"\n')
            f.write("mkdir -p $SLURM_LOG_DIR\n")
            f.write('echo \"${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}\" >> Results/job_ids.txt\n')

            python_command=["python",
                            submitter_loc,
                            fileloc,
                            cwd+"/Submission/Slurm_Json.json",
                            "--threshold={0}".format(kwargs["threshold"]),
                            "--unchanged_iterations={0}".format(kwargs["unchanged_iterations"])
            ]
            f.write(" ".join(python_command))
       
        
        process_time=kwargs["runs"]*2
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
        with open("Submission/Cleanup.job", "w") as f:
            f.write("#!/usr/bin/env bash\n")
            for key in cleanup_dict.keys():
                write_str="#SBATCH {0}={1}\n".format(key, cleanup_dict[key])
                f.write(write_str)
            f.write("set -e \n")
            python_command=["python",
                            processor_loc,
                            fileloc,
                            cwd+"/Submission/Slurm_Json.json",
                            cwd+"/Results/job_ids.txt",
                            cwd+"/Results"

            ]
            f.write(" ".join(python_command))
        with open("Submission/Controller.sh", "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write("rm -f Results/job_ids.txt\n")
            f.write("array_job_id=$(sbatch Submission/Automated_slurm_submission.job | awk '{print $4}')\n")
            f.write("sbatch --dependency=afterok:$array_job_id Submission/Cleanup.job && rm -f Results/*.npy" )

        if kwargs["run"]==True:
            import subprocess
            subprocess.call(["bash", "Submission/Controller.sh"])
        
