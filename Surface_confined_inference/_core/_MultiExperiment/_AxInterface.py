from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

from submitit import AutoExecutor
import numpy as np
import Surface_confined_inference as sci
from pathlib import Path
import os
import numpy as np
import submitit
from .AxParetoFuncs import pool_pareto
import time
from scipy.signal import decimate
import subprocess
class AxInterface(sci.OptionsAwareMixin):
    def __init__(self,**kwargs):
        self._internal_options = sci.AxInterfaceOptions(**kwargs)
        dirs=["clients","evaluator","pareto_points"]
        for dir in dirs:
            Path(os.path.join(self._internal_options.results_directory, dir)).mkdir(exists_ok=False)
        

    def run(self,job_number):
        cls=sci.BaseMultiExperiment.from_directory(os.path.join(self._internal_options.results_directory,"evaluator"))
        for i in range(0, self._internal_options.num_iterations):
            parameters, trial_index = self.ax_client.get_next_trial()
            ax_client.complete_trial(trial_index=trial_index, raw_data=cls.optimise_simple_score(parameters))
            ax_client.save_to_json_file(filepath=os.path.join(self._internal_options.results_directory, "clients", "ax_client_run_{0}.npy".format(job_number)))
    def init_sim_executor(self, name, timeout=None, dependency=None):
        executor=submitit.AutoExecutor(folder=self._internal_options.log_directory)
        if timeout is None:
            timeout=self._internal_options.max_run_time*60
        executor.update_parameters(timeout_min=self._internal_options.max_run_time*60) 

        executor.update_parameters(cpus_per_task=self._internal_options.num_cpu)
        executor.update_parameters(slurm_partition="nodes")
        executor.update_parameters(slurm_job_name=self._internal_options.name+"_"+"name")
        executor.update_parameters(slurm_account=self._internal_options.project)
        executor.update_parameters(mem_gb=self._internal_options.GB_ram)
        if self._internal_options.email != "":
            executor.update_parameters(mail_user=self._internal_options.email)
            executor.update_parameters(mail_type="END, FAIL")
        return executor
    def init_process_executor(self, name, timeout=20, dependency=None):
        executor=submitit.AutoExecutor(folder=self._internal_options.log_directory)
        executor.update_parameters(
            cpus_per_task=1,
            slurm_partition="nodes",
            slurm_job_name=self._internal_options.name+"_"+name,
            slurm_account=self._internal_options.project,
            mem_gb=self._internal_options.GB_ram,
            timeout_min=timeout,
            )
        if self._internal_options.email != "":
            executor.update_parameters(
                mail_user=self._internal_options.email,
                mail_type="END, FAIL"
            )
        if dependency is not None:
            executor.update_parameters(slurm_additional_parameters={
                            "dependency": f"afterok:{':'.join(dependency)}"
                            })
        return executor
    def experiment(self,):
        if self._internal_options.in_cluster==True:
            
            exp_exectutor=self.init_sim_executor("experiment")
            run_experiment=exp_exectutor.map_array(self.run, range(0, self._internal_options.independent_runs))
            exp_job_ids = [job.job_id for job in run_experiment]
            pool_handler=self.init_process_executor("pool", timeout=30, dependency=exp_job_ids)
            
            job=pool_handler.submit(pool_pareto, 
                                    os.path.join(self._internal_options.results_directory, "clients"), 
                                    self._cls.grouping_keys, 
                                    os.path.join(self._internal_options.results_directory, "pareto_points"))
            if self._internal_options.simulate_front==True:
                spawn_handler=self.init_process_executor("spawn_simulations", timeout=2, dependency=job.job_id)
                submitted_sim_job=spawn_handler.submit(self.spawn_bulk_simulation)
            if self._internal_options.rclone_directory!="":
                if self._internal_options.simulate_front:
                    depend = [submitted_sim_job.job_id]
                else:
                    depend = [job.job_id]
                spawn_handler=self.init_process_executor("spawn_rclone", timeout=2, dependency=depend)
                spawn_handler.submit(self.spawn_rclone, self._internal_options.simulate_front, depend)
        
            
            
    def setup_client(self, MultiExperimentInstance):
        if isinstance(MultiExperimentInstance, str):
            try:
                self._cls=sci.BaseMultiExperiment.from_directory(MultiExperimentInstance)
            except Exception as e:
                raise ValueError("Failed to load MultiExperiment class: {0}".format(str(e)))
        elif MultiExperimentInstance.__class__.__name__ !="MultiExperiment":
            raise ValueError("Instance needs to be Surface_confined_inference.MultiExperiment, not {0}".format(MultiExperimentInstance.__class__.__name__))
        else:
            self._cls=MultiExperimentInstance
        max_cpu=self._internal_options.num_cpu
        for classkey in self._cls.class_keys:
            cls=self._cls.classes[classkey]["class"]
            if hasattr(cls._internal_options, "num_cpu"):
                max_cpu=max(max_cpu, cls._internal_options.num_cpu)
        self._internal_options.num_cpu=max_cpu
        thresholds=self.get_zero_point_scores()
        self.ax_client=AxClient()
        param_arg=[
                    {
                        "name": x,
                        "type": "range",
                        "value_type":"float",
                        "bounds": [0.0, 1.0],
                    }
                    if "offset" not in x else 
                    
                    {
                        "name": x,
                        "type": "range",
                        "value_type":"float",
                        "bounds": [0.0, 0.2],
                    }
                    
                    for x in self._cls._all_parameters 
                ]

        objectives={key:ObjectiveProperties(minimize=True, threshold=thresholds[key]) for key in self._cls.grouping_keys}
        self.ax_client.create_experiment(
                                    name=self._internal_options.name,
                                    parameters=param_arg,
                                    objectives=objectives,
                                    overwrite_existing_experiment=False,
                                    is_test=False,
                                )
        self._cls.save_class(dir_path=os.path.join(self._internal_options.results_directory,"evaluator"), include_data=True)
        if self._internal_options.simulate_front==True:
            for classkey in self._cls.class_keys:
                Path(os.path.join(self._internal_options.results_directory, "simulations", classkey)).mkdir(exists_ok=False,parents=True)
    def get_zero_point_scores(self):
        zero_dict={}
        for classkey in self._cls.class_keys:
            zero_dict[classkey]=self._cls.classes[classkey]["zero_sim"]
        return self._cls.simple_score(zero_dict)
    def spawn_bulk_simulation(self, ):
        
        node_chunks=300
        with open(os.path.join(self._internal_options.results_directory, "pareto_points", "num_points.txt"), "r") as f:
            num_points=int(f.readline())
        num_processes=(num_points//node_chunks)+1
        start=time.time()
        cls.evaluate(np.random.rand(len(cls._all_parameters)))
        dummy_time=time.time()-start
        total_time=int((int(dummy_time/60)+5)*num_processes)
        simulation_executor=self.init_sim_executor("simulation", timeout=total_time)
        jobs = simulation_executor.map_array(run_bulk_simulation, range(0, num_points))
        job_ids = [job.job_id for job in jobs]

   
        id_path = os.path.join(self._internal_options.results_directory, "pareto_points", "jobids_bulk_sim.txt")
        with open(id_path, "w") as f:
            f.write(":".join(job_ids))

    def run_bulk_simulation(self, index, chunk_size):
        cls=sci.BaseMultiExperiment.from_directory(os.path.join(self._internal_options.results_directory,"evaluator"))
        with open(os.path.join(self._internal_options.results_directory, "pareto_points", "parameters.txt"), "r") as f:
            param_values = np.loadtxt(f, skiprows=1)
            f.seek(0)
            params = f.readline().strip().split()
        dec_factor=13
        save_dict={}
        for classkey in cls.class_keys:
            if cls.classes[classkey]["class"].experiment_type in ["FTACV","PSV"]:
                dec_time=decimate(cls.classes[classkey]["class"]["times"], dec_factor)
            else:
                dec_time=cls.classes[classkey]["class"]["times"]
            save_dict[classkey]=np.zeros((len(dec_time), chunk_size+1))
            save_dict[classkey][:,0]=dec_time
        counter=1
        for i in range(index*chunk_size, ((index+1)*chunk_size)):
            parameters=dict(zip(params, param_values[i,:]))
            param_values=[parameters[x] for x in cls._all_parameters]
            simulation_dict=cls.evaluate(param_values)
            for classkey in cls.class_keys:
                if cls.classes[classkey]["class"].experiment_type in ["FTACV","PSV"]:
                    dec_current=decimate(simulation_dict[classkey], dec_factor)
                else:
                    dec_current=simulation_dict[classkey]
                save_dict[classkey][:,counter]=dec_current
            counter+=1
        for key in cls.class_keys:
            filepath=os.path.join(self._internal_options.results_directory, "simulations", classkey, "simulations_%d_%d.txt" % (index*chunk_size, ((index+1)*chunk_size)))
            with open(filepath, "w") as f:
                np.savetxt(f, save_dict[classkey])
    def spawn_rclone(self, simulated_front, dependency=None):
        if simulated_front==True:
            jobid_path = os.path.join(self._internal_options.results_directory, "pareto_points", "jobids_bulk_sim.txt")
            with open(jobid_path) as f:
                sim_job_ids = f.read().strip()
            dependency=sim_job_ids
        elif dependency is not None:
            pass
        else:
            raise ValueError("Either simulated front needs to be True, or dependency needs to not be None")
        rclone_executor=self.init_process_executor("rclone", timeout=60, dependency=dependency)
        rclone_executor.submit(self.rclone)
    def rclone(self):
       
        command=["rclone", "copy", self._internal_options.results_directory, self._internal_options.rclone_directory]
        subprocess.run("module load rclone && "+ " ".join(command), shell=True, executable="/bin/bash")
        

        
