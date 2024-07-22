import numpy as np
import Surface_confined_inference as sci
import os
import uuid
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


        submitter_loc="/".join(path_to_submitter)+"/_HPCInterface/Slurm_submitter.py"
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
        num_cpu=4 + int(3 * np.log(self.n_parameters()+self.n_outputs()))            
        job_name=self._internal_options.experiment_type+"_inference_"+user+"_"+uuid.uuid4().hex
        ntasks=1
        project="chem-electro-2024"
        output=r"%x-%j.log"
        error=r"%x-%j.err"
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
        }
        self.save_class("Slurm_Json.json")
        with open("Automated_slurm_submission.job", "w") as f:
            f.write("#!/usr/bin/env bash\n")
            for key in master_dict.keys():
                write_str="#SBATCH {0}={1}\n".format(key, master_dict[key])
                f.write(write_str)
            f.write("set -e \n")
            python_command=["python",
                            submitter_loc,
                            fileloc,
                            os.getcwd()+"/"+"Slurm_Json.json",
                            "--threshold={0}".format(kwargs["threshold"]),
                            "--unchanged_iterations={0}".format(kwargs["unchanged_iterations"])
            ]
            f.write(" ".join(python_command))
        