import SurfaceODESolver as sos
import Surface_confined_inference as sci
import copy
import numpy as np
class SWVStepwise(sci.SingleExperiment):
    def __init__(self, experiment_parameters, **kwargs):

        

        super().__init__("SquareWave", experiment_parameters, **kwargs)
        num_steps=experiment_parameters["delta_E"]/experiment_parameters["scan_increment"]
        times=self.calculate_times()
        potential=self.get_voltage(times)
        classes=[]
        dc_input_dict={key:experiment_parameters[key] for key in ["Surface_coverage", "area", "Temp", "N_elec"]}
        potential_vals=np.zeros(2*int(num_steps))
        potential_vals[1]=experiment_parameters["E_start"]
        for i in range(1, 2*int(num_steps)):

    
            if i%2==1:
                potential_vals[i]=potential_vals[i-1]+experiment_parameters["v"]*(2*experiment_parameters["SW_amplitude"])
            else:
                potential_vals[i]=potential_vals[i-1]-experiment_parameters["v"]*(2*experiment_parameters["SW_amplitude"])+(experiment_parameters["v"]*experiment_parameters["scan_increment"])
                curr_dict=copy.deepcopy(dc_input_dict)
                curr_dict["Edc"]=potential_vals[i]
                curr_dict["phase"]=0
                curr_dict["delta_E"]=0
                curr_dict["omega"]=experiment_parameters["omega"]
                classes.append(sci.SingleExperiment("PSV", curr_dict))
        self.potential_vals=potential_vals
        self.exp_time=np.linspace(0, 0.5, int(experiment_parameters["sampling_factor"])//2)
        self.classes=classes
        
        
        
    def simulate(self, parameters, times):
        if self._optim_list is None:
            raise Exception(
                "optim_list variable needs to be set, even if it is an empty list"
            )
        if len(parameters) != len(self._optim_list):
            raise Exception(
                f"Parameters and optim_list need to be the same length, currently parameters={len(parameters)} and optim_list={len(self._optim_list)}"
            )
        if self._internal_options.normalise_parameters == True:
            sim_params = dict(
                zip(
                    self._optim_list,
                    self.change_normalisation_group(parameters, "un_norm"),
                )
            )
        
        else:
            sim_params = dict(zip(self._optim_list, parameters))
     
      
       
        
        timelen=len(self.exp_time)
        final_current=np.zeros(len(self.potential_vals)*timelen)
        final_potential= np.zeros(len(self.potential_vals)*timelen)
        final_theta= np.zeros(len(self.potential_vals)*timelen)
        
            
        for i in range(0, len(self.classes)):
            self.classes[i].optim_list=self.optim_list
            nd_dict = self.classes[i].nondimensionalise(sim_params)
            if i==0: 
                nd_dict["theta"]=1
            else:
                nd_dict["theta"]=previous_theta
            values = np.array(sos.ODEsimulate(self.exp_time, nd_dict))
            
            final_current[i*timelen:(i+1)*timelen]=values[0,:]
            final_theta[i*timelen:(i+1)*timelen]=values[1,:]
            final_potential[i*timelen:(i+1)*timelen]=self.potential_vals[i]
            
            previous_theta=values[1, -1]
            
        sw_dict={"total":final_current}   
        sw_dict["forwards"], sw_dict["backwards"], sw_dict["net"], sw_dict["E_p"]=self.SW_peak_extractor(final_current)
        sw_dict["theta"]=final_theta

        return sw_dict
   