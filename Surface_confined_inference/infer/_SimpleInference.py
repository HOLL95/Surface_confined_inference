import Surface_confined_inference as sci
import numpy as np
import matplotlib.pyplot as plt
import pints
import scipy as sp
import tabulate
class DummyVoltageSimulator(sci.SingleExperiment):
    def __init__(self, experiment_type, input_parameters):
        
        
        param_dict={"FTACV":["E_start", "E_reverse", "omega" ,"phase", "delta_E", "v"],
                        "DCV":["E_start", "E_reverse",  "v"],
                        "PSV":["Edc", "omega", "phase", "delta_E"]}
        initialisation_parameters={x:input_parameters[x] for x in param_dict[experiment_type]}
        initialisation_parameters=self.add_dummy_params(initialisation_parameters)
        super().__init__(experiment_type, initialisation_parameters)
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
    def simulate(self,parameters, times):
        param_dict=dict(zip(self.param_dict[self.experiment_type], parameters))
        updated_params=self.add_dummy_params(param_dict)
        simulated_voltage=super().get_voltage(times, dimensional=True, input_parameters=updated_params)
        return np.array(simulated_voltage)
def get_input_parameters(time, voltage,current, experiment_type, **kwargs):
    time=np.array(time)
    voltage=np.array(voltage)
    current=np.array(current)
    len_ts=len(time)
    if "plot_results" not in kwargs:
        kwargs["plot_results"]=False
    if "optimise" not in kwargs:
        kwargs["optimise"]=True
    if "optimsation_iterations" not in kwargs:
        kwargs["optimisation_iterations"]=300
    if experiment_type in ["DCV", "FTACV"]:
        if experiment_type=="FTACV":
            DC_voltage=sci.get_DC_component(time, voltage, current)
        else:
            DC_voltage=voltage
        if DC_voltage[len_ts//2]>DC_voltage[-1]:
            reversed=False
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
    elif experiment_type=="PSV":
        if "Edc" not in kwargs:
            kwargs["Edc"]=np.mean(voltage)
    if experiment_type in ["PSV", "FTACV"]:
        sneak_freq=sci.get_frequency(time, current)
        if "phase" not in kwargs:
            kwargs["phase"]=0
        if "omega" not in kwargs:
            kwargs["omega"]=sneak_freq
        
        if "delta_E" not in kwargs:
            window=sp.signal.windows.flattop(len_ts)
            pot_fft=np.fft.fft(window*voltage)
            fft_freq=np.fft.fftfreq(len_ts, time[1]-time[0])
            pos_freq=fft_freq[np.where(fft_freq>0)]
            subbed_freq=abs(pos_freq-sneak_freq)
            closest_bin=pos_freq[np.where(subbed_freq==min(subbed_freq))]
            peak_amplitude=abs(pot_fft[np.where(fft_freq==closest_bin)][0])
            kwargs["delta_E"]=2*peak_amplitude/(sum(window))
    simulator=DummyVoltageSimulator(experiment_type, kwargs)
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
        boundaries={"E_start":[-2, 2],
                    "E_reverse":[-2, 2],
                    "E_dc":[-2, 2],
                    "delta_E":[1e-3, 0.5],
                    "omega":[0.1, 1e4],
                    "v":[1e-4, 100],
                    "phase":[0,2*np.pi]}
        
        
        for key in estimated_parameters:
            if estimated_parameters[key]>boundaries[key][1]:
                print("Warning: {0} is larger than reasonable experimental value {1} for the parameter {2}. Would recommend checking your data".format(estimated_parameters[key], boundaries[key][1], key))
            if estimated_parameters[key]<boundaries[key][0]:
                print("Warning: {0} is smaller than reasonable experimental value {1} for the parameter {2}. Would recommend checking your data".format(estimated_parameters[key], boundaries[key][1], key))

        problem = pints.SingleOutputProblem(simulator, aliased_time, aliased_voltage)
        error = pints.SumOfSquaresError(problem)
        bounds=np.array([boundaries[key] for key in simulator.param_dict[experiment_type]])
        boundaries=pints.RectangularBoundaries(bounds[:,0], bounds[:,1])
        x0=[estimated_parameters[x] for x in simulator.param_dict[experiment_type]]
        opt= pints.OptimisationController(
            error,
            x0,
            sigma0=[0.01 for x in simulator.param_dict[experiment_type]],
            boundaries=boundaries,
            method=pints.CMAES,
            )
        opt.set_max_iterations(kwargs["optimisation_iterations"])
        found_parameters, found_value =opt.run()
        inferred_params=dict(zip(simulator.param_dict[experiment_type], found_parameters))
    estimated_simulated=simulator.simulate([estimated_parameters[key] for key in simulator.param_dict[experiment_type]], time)
    estimated_error=sci._utils.RMSE(estimated_simulated, voltage)*100
    if kwargs["optimise"]==False:
        tabulate_list=[[key, estimated_parameters[key]] for key in simulator.param_dict[experiment_type]]
        headers=["Parameter", "Estimated (Mean Error=%.1e mV)"% estimated_error ]
        
    else:
        inferred_simulated=simulator.simulate([inferred_params[key] for key in simulator.param_dict[experiment_type]], time)
        inferred_error=sci._utils.RMSE(inferred_simulated, voltage)*100
        tabulate_list=[[key, estimated_parameters[key], inferred_params[key]] for key in simulator.param_dict[experiment_type]]
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
        return estimated_parameters
    else:
        return estimated_parameters, inferred_params
