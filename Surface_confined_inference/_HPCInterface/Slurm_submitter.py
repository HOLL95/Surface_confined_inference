import numpy as np
import matplotlib.pyplot as plt
import Surface_confined_inference as sci
import argparse
import os
parser = argparse.ArgumentParser("Slurm submitter")
parser.add_argument("datafile", help="time-current-potential data filename", type=str)
parser.add_argument("simulator", help="Json filename that initilises simulator class", type=str)
parser.add_argument("--threshold", help="Inference will terminated after the score doesn't change by this amount for unchanged iterations", default=1e-6, type=float)
parser.add_argument("--unchanged_iterations",  help="Inference will terminated after the score doesn't change after this number of unchanged iterations", default=200, type=int)


args = parser.parse_args()

datafile=np.loadtxt(args.datafile)
time=datafile[:,0]
potential=datafile[:,2]
current=datafile[:,1]
simclass=sci.LoadSingleExperiment(args.simulator)
results=simclass.Current_optimisation(time, current,
                                parallel=True,
                                Fourier_filter=simclass._internal_options.Fourier_fitting, 
                                runs=1, 
                                threshold=args.threshold, 
                                unchanged_iterations=args.unchanged_iterations,
                                starting_point="random",
                                sigma0=0.075,
                                dimensional=True)
job_id=os.environ.get('SLURM_JOB_ID')
np.save("Results_run_{0}.npy".format(job_id), results)





