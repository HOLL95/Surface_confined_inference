import Surface_confined_inference as sci
import numpy as np
import matplotlib.pyplot as plt
import pints
import scipy as sp
import tabulate
import copy

class DummyVoltageSimulator(sci.SingleExperiment):
    def __init__(self, experiment_type, input_parameters, **kwargs):
        

        param_dict=copy.deepcopy(sci.experimental_input_params)
        if "sinusoidal_phase" not in kwargs:
            kwargs["sinusoidal_phase"]=False
            
        if kwargs["sinusoidal_phase"]==True:
            phase_func="sinusoidal"
            sinusoidal_phase_params=["phase_delta_E", "phase_omega", "phase_phase"]
            param_dict[experiment_type]+=sinusoidal_phase_params
            for key in sinusoidal_phase_params:
                input_parameters[key]=1
            
        else:
            phase_func="constant"
        
        initialisation_parameters={x:input_parameters[x] for x in param_dict[experiment_type]}
        initialisation_parameters=self.add_dummy_params(initialisation_parameters)
        super().__init__(experiment_type, initialisation_parameters, phase_function=phase_func)
        self.experiment_type=experiment_type
        self.param_dict=param_dict
    def add_dummy_params(self, param_dict):
        param_dict["Temp"]=298
        param_dict["area"]=0.07
        param_dict["Surface_coverage"]=1e-10
        param_dict["N_elec"]=1
        return param_dict
    def n_outputs(self):
        return 1
    def n_parameters(self):
        return len(self.param_dict[self.experiment_type])
    def dummy_normalise(self, params):
        names=list(self.param_dict[self.experiment_type])
        return_dict={}
        for i in range(0, len(names)):
            name=names[i]
            return_dict[name]=self.normalise(params[i], self._internal_memory["param_boundaries"][name])
        return return_dict
    def dummy_un_normalise(self, params):
        
        names=list(self.param_dict[self.experiment_type])
        return_dict={}
        for i in range(0, len(names)):
            name=names[i]
            return_dict[name]=self.un_normalise(params[i], self._internal_memory["param_boundaries"][name])
        return return_dict
    def simulate(self,parameters, times):
        param_dict=self.dummy_un_normalise(parameters)
        updated_params=self.add_dummy_params(param_dict)
        simulated_voltage=super().get_voltage(times, dimensional=True, input_parameters=updated_params)


        return np.array(simulated_voltage)
class CheckOtherExperiment(sci.ChangeTechnique):
    def __init__(self, experiment_type, simulator,**kwargs):

        if "input_parameters" in kwargs:
            super().__init__(simulator, experiment_type, kwargs["input_parameters"])
            if "datafile" in kwargs:
                print("Warning: ignoring input data in place of desired parameters")
                datafile=np.loadtxt(kwargs["datafile"])
                self.time=datafile[:,0]
                self.potential=datafile[:,2]
                self.current=datafile[:,1]
        else:
            if "datafile" not in kwargs:
                raise Exception("Need either a datafile or a set of input parameters to change technique")
            datafile=np.loadtxt(kwargs["datafile"])
            self.time=datafile[:,0]
            self.potential=datafile[:,2]
            self.current=datafile[:,1]
            estimated, optimised=sci.infer.get_input_parameters(self.time, self.potential, self.current,experiment_type, optimise=True)
            super().__init__(simulator, experiment_type, optimised)
        if "datafile" not in kwargs:
            self.save_results_possible=False
        else:
            self.save_results_possible=True
       
    def save_results(self,save_loc, parameter_array,**kwargs):
        if self.save_results_possible==False:
            raise Exception("Need a datafile to save inference results")
        sim_voltage=self.get_voltage(self.time, dimensional=True)
        sim_dict=self.parameter_array_simulate(parameter_array, self.time, contains_noise=True)                
        sci.plot.save_results(self.time, 
                sim_voltage, 
                self.current, 
                sim_dict["Current_array"], 
                save_loc, 
                self._internal_options.experiment_type, 
                self._internal_memory["boundaries"],
                save_csv=kwargs["save_csv"],
                optim_list=self._optim_list, 
                fixed_parameters=self.fixed_parameters,
                score=parameter_array[:,-1],
                parameters=parameter_array,
                DC_voltage=sim_dict["Current_array"]
                )


def get_input_parameters(time, voltage,current, experiment_type, **kwargs):
    time=np.array(time)
    voltage=np.array(voltage)
    current=np.array(current)
    len_ts=len(time)
    if "plot_results" not in kwargs:
        kwargs["plot_results"]=False
    if "optimise" not in kwargs:
        kwargs["optimise"]=True
    if "runs" not in kwargs:
        kwargs["runs"]=5
    
    if "return_sim_values" not in kwargs:
        kwargs["return_sim_values"]=False
    if "sinusoidal_phase" not in kwargs:
        kwargs["sinusoidal_phase"]=False
    if "sigma" not in kwargs:
        kwargs["sigma"]=0.01
    if experiment_type in ["DCV", "FTACV"]:
        if experiment_type=="FTACV":
            DC_voltage=sci.get_DC_component(time, voltage, current)
        else:
            DC_voltage=voltage
        
        if DC_voltage[len_ts//2]>DC_voltage[-1]:
            reversed=False
        else:
            reversed=True
        if "E_reverse" not in kwargs:
            if reversed==False:
                kwargs["E_reverse"]=max(DC_voltage)
            else:
                kwargs["E_reverse"]=min(DC_voltage)
        if "E_start" not in kwargs:
            if reversed==False:
                kwargs["E_start"]=min(DC_voltage)
            else:
                kwargs["E_start"]=max(DC_voltage)
        if "v" not in kwargs:
            y1=DC_voltage[0]
            y2=DC_voltage[(len_ts//2)-10]
            x1=time[0]
            x2=time[(len_ts//2)-10]
            kwargs["v"]=(y2-y1)/(x2-x1)

    if experiment_type in ["PSV", "FTACV"]:
        sneak_freq=sci.get_frequency(time, current)
        window=sp.signal.windows.flattop(len_ts)
        pot_fft=np.fft.fft(window*voltage)
        fft_freq=np.fft.fftfreq(len_ts, time[1]-time[0])
        pos_freq=fft_freq[np.where(fft_freq>0)]
        subbed_freq=abs(pos_freq-sneak_freq)
        closest_bin=pos_freq[np.where(subbed_freq==min(subbed_freq))]
        peak_amplitude=abs(pot_fft[np.where(fft_freq==closest_bin)][0])
        sneak_amp=2*peak_amplitude/(sum(window))
        if "phase" not in kwargs:
            kwargs["phase"]=0
        if "omega" not in kwargs:
            kwargs["omega"]=sneak_freq
        
        if "delta_E" not in kwargs:
            
            kwargs["delta_E"]=sneak_amp
        if experiment_type=="PSV":
            if "Edc" not in kwargs:
                kwargs["Edc"]=np.mean(voltage)
            if "num_peaks" not in kwargs:
                kwargs["num_peaks"]=time[-1]*sneak_freq
                
    simulator=DummyVoltageSimulator(experiment_type, kwargs, sinusoidal_phase=kwargs["sinusoidal_phase"])
    
    boundaries={"E_start":[-2, 2],
                    "E_reverse":[-2, 2],
                    "E_dc":[-2, 2],
                    "delta_E":[0.8*sneak_amp, 1.2*sneak_amp],
                    "omega":[0.95*sneak_freq, 1.05*sneak_freq],
                    "v":[1e-4, 100],
                    "phase":[0,2*np.pi],
                    "Edc":[-2,2],
                    "phase_phase":[0, 2*np.pi],
                    "phase_delta_E":[-1.5, 1.5],
                    "phase_omega":[0.1, 400]
                    }
    simulator._internal_memory["param_boundaries"]=boundaries
    estimated_parameters={x:kwargs[x] for x in simulator.param_dict[experiment_type]}
    if kwargs["optimise"]==True:
        if "aliasing" not in kwargs:
            if experiment_type=="DCV":
                max_points=1e4
                if len_ts<max_points:
                    kwargs["aliasing"]=1
                else:
                    kwargs["aliasing"]=int(len_ts//max_points)
            elif experiment_type in ["PSV", "FTACV"]:
                points_per_osc=20
                period=1/sneak_freq
                end_time=time[-1]
                total_sine_waves=end_time/period
                desired_points=(points_per_osc*total_sine_waves)
                if desired_points>len_ts:
                    kwargs["aliasing"]=1
                else:
                    kwargs["aliasing"]=int(len_ts//desired_points)
        
        if isinstance(kwargs["aliasing"], int) is False:
            raise TypeError("Aliasing keyword needs to be of type int")
        aliased_time=time[::kwargs["aliasing"]]
        aliased_voltage=voltage[::kwargs["aliasing"]]
        
        if "fixed_parameters" not in kwargs:
            kwargs["fixed_parameters"]=[]
        for param in kwargs["fixed_parameters"]:
            bounds=[kwargs[param]*0.95, 1.05*kwargs[param]]
            boundaries[param]=[min(bounds), max(bounds)]
        
            

        
        for key in estimated_parameters:
            if estimated_parameters[key]>boundaries[key][1]:
                print("Warning: {0} is larger than reasonable experimental value {1} for the parameter {2}. Would recommend checking your data".format(estimated_parameters[key], boundaries[key][1], key))
            if estimated_parameters[key]<boundaries[key][0]:
                print("Warning: {0} is smaller than reasonable experimental value {1} for the parameter {2}. Would recommend checking your data".format(estimated_parameters[key], boundaries[key][1], key))

        problem = pints.SingleOutputProblem(simulator, aliased_time, aliased_voltage)
        error = pints.SumOfSquaresError(problem)
        bounds=np.array([boundaries[key] for key in simulator.param_dict[experiment_type]])
        boundaries=pints.RectangularBoundaries(np.zeros(len(simulator.param_dict[experiment_type])), 
                                                np.ones(len(simulator.param_dict[experiment_type])))
        print([estimated_parameters[x] for x in simulator.param_dict[experiment_type]])
        x0=simulator.dummy_normalise([estimated_parameters[x] for x in simulator.param_dict[experiment_type]])
        
        #x0=[0.5 for x in simulator.param_dict[experiment_type]]
        best=1e12
        
        for i in range(0, kwargs["runs"]):
            x0=np.random.rand(len(simulator.param_dict[experiment_type]))
            opt= pints.OptimisationController(
                error,
                x0,
                sigma0=[kwargs["sigma"] for x in simulator.param_dict[experiment_type]],
                boundaries=boundaries,
                method=pints.CMAES,
                )
            #opt.set_max_iterations(kwargs["optimisation_iterations"])
            found_parameters, found_value =opt.run()
            if found_value<best:
                best=found_value
                best_parameters=found_parameters
        found_parameters=best_parameters
        inferred_params=dict(zip(simulator.param_dict[experiment_type], found_parameters))
        table_inferred=simulator.dummy_un_normalise(found_parameters)
    normed_estimated=simulator.dummy_normalise([estimated_parameters[key] for key in simulator.param_dict[experiment_type]])
    normed_estimated_list=[normed_estimated[key] for key in simulator.param_dict[experiment_type]]
    estimated_simulated=simulator.simulate(normed_estimated_list, time)
    estimated_error=sci._utils.RMSE(estimated_simulated, voltage)*100
    
    if kwargs["optimise"]==False:
        tabulate_list=[[key, estimated_parameters[key]] for key in simulator.param_dict[experiment_type]]
        headers=["Parameter", "Estimated (Mean Error=%.1e mV)"% estimated_error ]
        
    else:
        inferred_simulated=simulator.simulate([inferred_params[key] for key in simulator.param_dict[experiment_type]], time)
        inferred_error=sci._utils.RMSE(inferred_simulated, voltage)*100
        tabulate_list=[[key, estimated_parameters[key], table_inferred[key]] for key in simulator.param_dict[experiment_type]]
        headers=["Parameter", 
                "Estimated (Mean Error=%.1e mV)"% estimated_error, 
                "Inferred (Mean Error=%.1e mV)"% inferred_error,]
    print(tabulate.tabulate(tabulate_list, headers=headers))

    if kwargs["plot_results"]==True:
        colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax=plt.subplots(1,2)
        
        ax[0].plot(time, voltage, label="Data")
        ax[0].plot(time, estimated_simulated, linestyle="--", label="Estimated")
        ax[1].plot(time, voltage-estimated_simulated, color=colours[1], label="Estimated")
        if kwargs["optimise"]==True:
            inferred_simulated=simulator.simulate([inferred_params[key] for key in simulator.param_dict[experiment_type]], time)
            ax[0].plot(time, inferred_simulated, linestyle="--", label="Inferred")
            ax[1].plot(time, voltage-inferred_simulated, color=colours[2], label="Inferred")
        ax[1].set_ylabel("Residual (V)")
        ax[0].set_ylabel("Potential (V)")
        for axes in ax:
            axes.set_xlabel("Time (s)")
            axes.legend()
        plt.show()
    if kwargs["optimise"]==False:
        if kwargs["return_sim_values"]==False:
            return estimated_parameters
        else:
            return estimated_parameters, estimated_simulated
    else:
        if kwargs["return_sim_values"]==False:
            return estimated_parameters, table_inferred
        else:
            
            return estimated_parameters, table_inferred, estimated_simulated,inferred_simulated
       
