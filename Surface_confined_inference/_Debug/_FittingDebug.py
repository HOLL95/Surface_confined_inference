import numpy as np
import matplotlib.pyplot as plt
import Surface_confined_inference as sci
class FittingDebug:
    def __new__(self, json_path, time, current, potential, **kwargs):
        if "class_type" not in kwargs:
            kwargs["class_type"]="base"
        FactoryClass = sci.LoadSingleExperiment.get_factory_class(json_path, class_type=kwargs["class_type"])
        class RunTimeFactory(FactoryClass):
            def __init__(self, json_path, time, current, potential, **kwargs):
                super().__init__(json_path)
                if "dimensional" not in kwargs:
                    kwargs["dimensional"]=True
                if "Fourier_fitting" not in kwargs:
                    kwargs["Fourier_fitting"]=False
                self.Fourier_fitting=kwargs["Fourier_fitting"]
                
                if kwargs["dimensional"]==True:
                    self.time=self.nondim_t(time)
                    self.current=self.nondim_i(current)
                    self.potential=self.nondim_e(potential)
                else:
                    self.time=time
                    self.current=current
                    self.potential=potential
                if self.Fourier_fitting==True:
                    
                    self.Spectrum=sci.top_hat_filter(self.time, self.current, Fourier_window=self._internal_options.Fourier_window,
                                                                   top_hat_width=self._internal_options.top_hat_width,
                                                                   Fourier_function=self._internal_options.Fourier_function, 
                                                                   Fourier_harmonics=self._internal_options.Fourier_harmonics,
                                                              )
                self.kwargs=kwargs

            def simulate(self,parameters, times):
                print(dict(zip(self._optim_list, parameters)))
                print(dict(zip(self._optim_list, self.change_normalisation_group(parameters, "un_norm"))))
                
                timeseries=super().simulate(parameters, times)
                ts_fig, ts_ax=plt.subplots()
                ts_ax.plot(self.time, self.current, label="Data")
                ts_ax.plot(times, timeseries, alpha=0.5, label="Simulation")
                ts_ax.legend()
                if self.Fourier_fitting==True:
                    sim_spectrum=sci.top_hat_filter(times, timeseries, Fourier_window=self._internal_options.Fourier_window,
                                                                   top_hat_width=self._internal_options.top_hat_width,
                                                                   Fourier_function=self._internal_options.Fourier_function, 
                                                                   Fourier_harmonics=self._internal_options.Fourier_harmonics,
                                                                   )
                    f_fig, f_ax=plt.subplots()
                    f_ax.plot(self.Spectrum, label="Data")
                    f_ax.plot(sim_spectrum, alpha=0.5, label="Simulation")
                    f_ax.legend()
                    if self._internal_options.experiment_type=="PSV":
                        hanning=False
                        xaxis="potential"
                        func=np.real
                    else:
                        hanning=True
                        xaxis="time"
                        func=np.abs
                    sci.plot.plot_harmonics(Data_data={"time":self.time, "current":self.current, "potential":self.potential, "harmonics":self._internal_options.Fourier_harmonics,},
                                            Sim_data={"time":times, "current":timeseries, "potential":self.potential, "harmonics":self._internal_options.Fourier_harmonics,},
                                            
                                            xaxis=xaxis, 
                                            hanning=hanning,
                                            plot_func=func)  
                plt.show()
                return timeseries
            def go(self,):
                if self.kwargs["class_type"]=="base":
                    self.Current_optimisation(self.time, self.current, Fourier_filter=self.Fourier_fitting, parallel=False, dimensional=False)
                elif  self.kwargs["class_type"]=="mcmc":
                    print("starts here")
                    self.run(self.time, self.current, starting_point=self.kwargs["starting_point"],fourier_fitting=self.kwargs["Fourier_fitting"], CMAES_results_dir=self.kwargs["CMAES_results_dir"], num_cpu=self.kwargs["num_cpu"],dimensional=False)
            #def __setattr__(self, name, value):
            #    super().__setattr__(name, value, silent_flag=True)

        
        return RunTimeFactory(json_path, time, current, potential, **kwargs)
