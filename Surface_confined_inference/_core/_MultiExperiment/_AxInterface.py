from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import logging
import copy
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
import torch
class AxInterface(sci.OptionsAwareMixin):
    def __init__(self,**kwargs):
        self._internal_options = sci.AxInterfaceOptions(**kwargs)
        dirs=["clients","evaluator","pareto_points"]
        for dir in dirs:
            Path(os.path.join(self._internal_options.results_directory, dir)).mkdir(exist_ok=True)
        if self._internal_options.GPU !="none":
            optimal_threads = min(8, self._internal_options.num_cpu)
            torch.set_num_threads(optimal_threads)
            os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
            os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
            os.environ['OPENBLAS_NUM_THREADS'] = str(optimal_threads)
            
        environs=["IN_ARC", "IN_VIKING"]
        self._environ=None
        for environ in environs:
            if  os.environ.get(environ, '').lower() in ('true', '1', 'yes'):
                self._environ=environ
        if self._environ is not None:
            if self._environ=="IN_VIKING" or self._environ=="IN_ARC":
                self._environ_args={
                    "cpus_per_task": "cpus_per_task",
                    "slurm_partition": "slurm_partition", 
                    "slurm_job_name": "slurm_job_name",
                    "slurm_account": "slurm_account",
                    "mem_gb": "mem_gb",
                    "timeout_min": "timeout_min",
                    "slurm_mail_user": "slurm_mail_user",
                    "slurm_mail_type": "slurm_mail_type",
                    "slurm_qos":"slurm_qos"
                }
            if self._environ=="IN_ARC":
             self._environ_args["mem_gb"]="slurm_mem_per_cpu"
    def set_memory(self, memory):
        if self._environ=="IN_VIKING":
            return memory
        elif self._environ=="IN_ARC":
            return int((memory * 1024) // self._internal_options.num_cpu)
    def run(self,job_number):
        cls=sci.BaseMultiExperiment.from_directory(os.path.join(self._internal_options.results_directory,"evaluator"))
        for i in range(0, self._internal_options.num_iterations):
            parameters, trial_index = self.ax_client.get_next_trial()
            self.ax_client.complete_trial(trial_index=trial_index, raw_data=cls.optimise_simple_score(parameters))
            self.ax_client.save_to_json_file(filepath=os.path.join(self._internal_options.results_directory, "clients", "ax_client_run_{0}.json".format(job_number)))
    def init_sim_executor(self, name, timeout=None, dependency=None):
        executor=submitit.AutoExecutor(folder=self._internal_options.log_directory)
        if timeout is None:
            timeout=self._internal_options.max_run_time*60
        arg_dict = {
            self._environ_args["timeout_min"]: self._internal_options.max_run_time*60,
            self._environ_args["cpus_per_task"]: self._internal_options.num_cpu,
            self._environ_args["slurm_partition"]: "nodes",
            self._environ_args["slurm_job_name"]: self._internal_options.name+"_"+"name",
            self._environ_args["slurm_account"]: self._internal_options.project,
            self._environ_args["mem_gb"]: self.set_memory(self._internal_options.GB_ram)
        }
        if self._internal_options.QOS !="none":
            arg_dict[self._environ_args["slurm_qos"]]=self._internal_options.QOS
        if self._environ=="IN_ARC":
            if arg_dict[self._environ_args["timeout_min"]]<12*60:
                arg_dict[self._environ_args["slurm_partition"]]="short"
            if arg_dict[self._environ_args["timeout_min"]]<48*60:
                arg_dict[self._environ_args["slurm_partition"]]="medium"
            else:
                arg_dict[self._environ_args["slurm_partition"]]="long"

        if self._internal_options.email != "":
            arg_dict.update({
                self._environ_args["slurm_mail_user"]: self._internal_options.email,
                self._environ_args["slurm_mail_type"]: "END, FAIL"
            })
        
        executor.update_parameters(**arg_dict)
        return executor

    def init_process_executor(self, name, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"]=20
        if "dependency" not in kwargs:
            kwargs["dependency"]=None
        if "criteria" not in kwargs:
            kwargs["criteria"]="afterok"
        if "cpu_count" not in kwargs:
            kwargs["cpu_count"]=1
        timeout=kwargs["timeout"]
        dependency=kwargs["dependency"]
        criteria=kwargs["criteria"]
        executor=submitit.AutoExecutor(folder=self._internal_options.log_directory)
        arg_dict = {
            self._environ_args["cpus_per_task"]: kwargs["cpu_count"],
            self._environ_args["slurm_partition"]: "nodes",
            self._environ_args["slurm_job_name"]: self._internal_options.name + "_" + name,
            self._environ_args["slurm_account"]: self._internal_options.project,
            self._environ_args["mem_gb"]: self.set_memory(self._internal_options.GB_ram),
            self._environ_args["timeout_min"]: timeout,
        }
        if self._environ=="IN_ARC":
            arg_dict[self._environ_args["slurm_partition"]]= "short"
        if self._internal_options.email != "":
            arg_dict.update({
                self._environ_args["slurm_mail_user"]: self._internal_options.email,
                self._environ_args["slurm_mail_type"]: "END, FAIL"
            })
        executor.update_parameters(**arg_dict)
        if dependency is not None:
            executor.update_parameters(slurm_additional_parameters={
                            "dependency": f"{criteria}:{':'.join(dependency)}"
                            })
        return executor
    def experiment(self,):
        if self._internal_options.in_cluster==True:
            if os.path.isdir(self._internal_options.results_directory) is True:
                for directory in ["clients", "pareto_points"]:
                    path=os.path.join(self._internal_options.results_directory, directory)
                    if len(os.listdir(path))>0:
                        raise ValueError(f"Results directory '{path}' must be empty (contains ({os.listdir(path)}))")
            with open(os.path.join(self._internal_options.results_directory, "decimation.txt"), "w") as f:
                f.write(str(self._internal_options.front_decimation))
            exp_job_ids=self.run_inference()
            pool_job=self.pool_inference_results(dependency=exp_job_ids, criteria="afterany")
            if self._internal_options.simulate_front==True:
                submitted_sim_job=self.simulate_fronts(dependency=pool_job)
            if self._internal_options.rclone_directory!="":
                if self._internal_options.simulate_front is True:
                    depend = submitted_sim_job
                else:
                    depend = pool_job
                self.rclone_results(dependency=depend)
    def restart(self, start_point):
        valid_start_points=["pool", "simulate", "rclone"]
        if start_point not in valid_start_points:
            raise ValueError("{0} not in allowed restart point ({1})".format(start_point, valid_start_points))
        dependency=None
        logging.basicConfig(filename='restart.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
        try:
         logging.info("Starting the experiment.")
         if start_point == "pool":
            dependency = self.pool_inference_results(dependency=dependency)
         if start_point in ["pool", "simulate"] and self._internal_options.simulate_front is True:
            dependency = self.simulate_fronts(dependency=dependency)
         if self._internal_options.rclone_directory != "":
            self.rclone_results(dependency=dependency)
         logging.info("Experiment finished successfully.")

        except Exception as e:
         logging.error("An unhandled exception occurred!", exc_info=True)
         print(f"FATAL ERROR: {e}. Check job_debug.log for details.", file=sys.stderr)
         sys.exit(1) 


    def run_inference(self, ):
        exp_exectutor=self.init_sim_executor("experiment")
        if self._internal_options.GPU !="none":
            exp_exectutor.update_parameters(slurm_gres=self._internal_options.GPU)
        run_experiment=exp_exectutor.map_array(self.run, range(0, self._internal_options.independent_runs))
        exp_job_ids = [job.job_id for job in run_experiment]
        return exp_job_ids
    def pool_inference_results(self,dependency=None, criteria="afterok"):
        pool_handler=self.init_process_executor("pool", timeout=30, dependency=dependency, criteria=criteria)
        job=pool_handler.submit(pool_pareto, 
                                os.path.join(self._internal_options.results_directory, "clients"), 
                                self._cls.grouping_keys, 
                                self._cls._all_parameters,
                                os.path.join(self._internal_options.results_directory, "pareto_points"))
        return [job.job_id]
    def simulate_fronts(self, dependency=None):
        spawn_handler=self.init_process_executor("spawn_simulations", timeout=10, dependency=dependency, cpu_count=self._internal_options.num_cpu)
        submitted_sim_job=spawn_handler.submit(self.spawn_bulk_simulation)
        return [submitted_sim_job.job_id]
    def rclone_results(self, dependency=None):
        spawn_handler=self.init_process_executor("spawn_rclone", timeout=2, dependency=dependency)
        spawn_handler.submit(self.spawn_rclone, self._internal_options.simulate_front, dependency)   
        return None
            
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
        if self._internal_options.GPU !="none":
         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         self.ax_client=AxClient(torch_device=torch.device("cuda"))
        else:
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
        input_dict=dict(
            name=self._internal_options.name,
            parameters=param_arg,
            objectives=objectives,
            overwrite_existing_experiment=False,
            is_test=False,
        )
        if len(self._internal_options.input_constraints)>0:
            input_dict["parameter_constraints"]=self._internal_options.input_constraints
        self.ax_client.create_experiment(**input_dict)
        self._cls.save_class(dir_path=os.path.join(self._internal_options.results_directory,"evaluator"), include_data=True)
        if self._internal_options.simulate_front==True:
            for classkey in self._cls.class_keys:
                Path(os.path.join(self._internal_options.results_directory, "simulations", classkey)).mkdir(exist_ok=True,parents=True)
    def get_zero_point_scores(self):
        zero_dict={}
        for classkey in self._cls.class_keys:
            zero_dict[classkey]=self._cls.classes[classkey]["zero_sim"]
        return self._cls.simple_score(zero_dict)
    def spawn_bulk_simulation(self, ):
        cls=sci.BaseMultiExperiment.from_directory(os.path.join(self._internal_options.results_directory,"evaluator"))
        with open(os.path.join(self._internal_options.results_directory, "pareto_points", "num_points.txt"), "r") as f:
            num_points=int(f.readline())
        node_chunks=min(num_points, 300)
        process_per_chunk=int(np.floor(num_points//node_chunks))+1
        quit_point=int(np.ceil(node_chunks/process_per_chunk))
        start=time.time()
        cls.evaluate(np.random.rand(len(cls._all_parameters)))
        dummy_time=time.time()-start
        total_time=int((int(dummy_time/60)+5)*process_per_chunk)
        simulation_executor=self.init_sim_executor("simulation", timeout=total_time)
        jobs = simulation_executor.map_array(self.run_bulk_simulation, range(0, quit_point), [process_per_chunk]*quit_point)
        job_ids = [job.job_id for job in jobs]

   
        id_path = os.path.join(self._internal_options.results_directory, "pareto_points", "jobids_bulk_sim.txt")
        with open(id_path, "w") as f:
            f.write(":".join(job_ids))

    def run_bulk_simulation(self, index, chunk_size):
        print("spawn3")
        cls=sci.BaseMultiExperiment.from_directory(os.path.join(self._internal_options.results_directory,"evaluator"))
        with open(os.path.join(self._internal_options.results_directory, "pareto_points", "parameters.txt"), "r") as f:
            param_values = np.loadtxt(f, skiprows=1)
            f.seek(0)
            params = f.readline().strip().split()[1:]
        dec_factor=self._internal_options.front_decimation

        save_dict={}
        for classkey in cls.class_keys:
            if cls.classes[classkey]["class"].experiment_type in ["FTACV","PSV"]:
                dec_time=decimate(copy.deepcopy(cls.classes[classkey]["times"]), dec_factor)
                size=chunk_size+1
                save_dict[classkey]=np.zeros((len(dec_time), size))
                save_dict[classkey][:,0]=dec_time
            else:
                save_dict[classkey]=None

        counter=1

        for i in range(index*chunk_size, ((index+1)*chunk_size)):
            if param_values.shape[0]-1<i:
             break
            parameters=dict(zip(params, param_values[i,:]))
            param_value_list=[parameters[x] for x in cls._all_parameters]
            simulation_dict=cls.evaluate(param_value_list)
            for classkey in cls.class_keys:
                if cls.classes[classkey]["class"].experiment_type in ["FTACV","PSV"]:
                    dec_current=decimate(simulation_dict[classkey], dec_factor)
                else:                 
                    dec_current=simulation_dict[classkey]
                if save_dict[classkey] is not None:
                 save_dict[classkey][:,counter]=dec_current
                else:
                 save_dict[classkey]=np.zeros((len(dec_current), chunk_size+1))
                 save_dict[classkey][:,0]=list(range(0, len(dec_current)))
                 save_dict[classkey][:,counter]=dec_current
            counter+=1
        for classkey in cls.class_keys:
            filepath=os.path.join(self._internal_options.results_directory, "simulations", classkey, "simulations_%d_%d.txt" % (index*chunk_size, ((index+1)*chunk_size)))
            with open(filepath, "w") as f:
                np.savetxt(f, save_dict[classkey])
    def spawn_rclone(self, simulated_front, dependency=None):
        if simulated_front==True:
            jobid_path = os.path.join(self._internal_options.results_directory, "pareto_points", "jobids_bulk_sim.txt")
            with open(jobid_path) as f:
                sim_job_ids = f.read().strip()
            dependency=[sim_job_ids]
        elif dependency is not None:
            pass
        else:
            raise ValueError("Either simulated front needs to be True, or dependency needs to not be None")
        rclone_executor=self.init_process_executor("rclone", timeout=60, dependency=dependency)
        rclone_executor.submit(self.rclone)
    def rclone(self):
       
        command=["rclone", "copy", self._internal_options.results_directory, self._internal_options.rclone_directory]
        subprocess.run("module load rclone && "+ " ".join(command), shell=True, executable="/bin/bash")
        

        
