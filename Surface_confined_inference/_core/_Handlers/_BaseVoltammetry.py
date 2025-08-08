import Surface_confined_inference as sci
import SurfaceODESolver as sos
#context manager
#options
#parameter handling
#dispersion handling
class BaseHandler:
    def __init__(self, options, cls):
        self.options=options
        self.cls=cls
    def simulate(self, parameters, times):
        """
        Args:
            times (list): list of times to pass to the solver
            parameters (list): list of parameters to be passed to the solver - needs to be the same length as _optim_list, will throw an error if not. The variable `_optim_list` needs to be intiialised
            before this function is called to allow for various santity checks to take place.
        Returns:
            list: A list of current values at the times passed to the `simulate` function
        Raises:
            Exception: If `_optim_list` and the arg `parameters` are not of the same length
            Exception: if the variable `_optim_list` has not been set then no checking of the parameters has occured, and so an error is thrown.

        """
        if self.options._optim_list is None:
            raise Exception(
                "optim_list variable needs to be set, even if it is an empty list"
            )
        if len(parameters) != len(self._optim_list):
            raise Exception(
                f"Parameters and optim_list need to be the same length, currently parameters={len(parameters)} and optim_list={len(self._optim_list)}"
            )
        if self.options.normalise_parameters == True:
            sim_params = dict(
                zip(
                    self.options._optim_list,
                    self.cls.change_normalisation_group(parameters, "un_norm"),
                )
            )
        else:
            sim_params = dict(zip(self._optim_list, parameters))
        nd_dict = self.nondimensionalise(sim_params)
        return nd_dict
    def get_voltage(self, times, **kwargs):
        """
        Args:
            times (list): list of timepoints
            input_parameters (dict, optional): dictionary of input parameters
            dimensional (bool, optional): whether or not the times are in dimensional format or not
        Returns:
            list: list of potential values at the provided time points. By default, these values will be in dimensional format (V)
        """
        if "input_parameters" not in kwargs:
            input_parameters=None
        else:
            input_parameters=kwargs["input_parameters"]
        if "dimensional" not in kwargs:
            kwargs["dimensional"]=False
        if input_parameters == None:
            input_parameters = self._internal_memory["input_parameters"]
        
        
        input_parameters["phase_flag"]=0
        checking_dict=copy.deepcopy(self._internal_memory["input_parameters"])
        checking_dict["phase_flag"]=input_parameters["phase_flag"]
        sci.check_input_dict(
            input_parameters, checking_dict, optional_arguments=[]
        )
        input_parameters=copy.deepcopy(input_parameters)
        input_parameters = self.validate_input_parameters(input_parameters)
        if self._internal_options.experiment_type!="SquareWave":
            if "tr" in input_parameters:
                if kwargs["dimensional"]==True:
                    input_parameters["tr"]*=self._NDclass.c_T0
            input_parameters["omega"] *= 2 * np.pi
            return sos.potential(times, input_parameters)
        else:
           
            voltages=np.zeros(len(times))

            for i in times:
                
                i=int(i)
                voltages[i-1]=sos.SW_potential(i,input_parameters["sampling_factor"],input_parameters["scan_increment"],input_parameters["SW_amplitude"],input_parameters["E_start"],input_parameters["v"])
            voltages[-1]=voltages[-2]
        return voltages
class ContinuousHandler(BaseHandler):
    def __init__(self, options, cls):
        super().__init__(options, cls)
    def simulate(self, parameters, times):
        nd_dict=super().simulate(parameters, times)
        if self.options.Faradaic_only==True:
            def solver(times, nd_dict):
                return sos.ODEsimulate(times, nd_dict[2,:])
        else:
            def solver(times, nd_dict):
                return sos.ODEsimulate(times, nd_dict[0,:])
        if self.options.dispersion==True:
             current = self.dispersion_simulator(solver,  sim_params, times)
        else:
            solver(times, nd_dict)
        return current
class SquareWaveHandler(BaseHandler):
        def __init__(self, options, cls):
            super().__init__(options, cls)
            self.SW_sampling()
        
    def SW_sampling(self,**kwargs):
        self.SW_params={}
        if "parameters" not in kwargs:
            parameters=self.cls._internal_memory["input_parameters"]
        else:
            parameters=kwargs["parameters"]
        sampling_factor=parameters["sampling_factor"]
        self.SW_params["end"]=int(abs(parameters['delta_E']//parameters['scan_increment']))

        p=np.array(range(0, self.SW_params["end"]))
        self.SW_params["b_idx"]=((sampling_factor*p)+(sampling_factor/2))-1
        self.SW_params["f_idx"]=(p*sampling_factor)-1
        Es=parameters["E_start"]#-parameters["E_0"]
        self.SW_params["E_p"]=(Es+parameters["v"]*(p*parameters['scan_increment']))
        self.SW_params["sim_times"]=self.calculate_times()
    
    def SW_peak_extractor(self, current, **kwargs):
        if "mean" not in kwargs:
            kwargs["mean"]=0
        if "window_length" not in kwargs:
            kwargs["window_length"]=1
        j=np.array(range(1, self.SW_params["end"]*self._internal_memory["input_parameters"]["sampling_factor"]))
        if kwargs["mean"]==0:
            forwards=np.zeros(len(self.SW_params["f_idx"]))
            backwards=np.zeros(len(self.SW_params["b_idx"]))
            forwards=np.array([current[x] for x in self.SW_params["f_idx"]])
            backwards=np.array([current[int(x)] for x in self.SW_params["b_idx"]])
        else:
            raise NotImplementedError
            indexes=[self.SW_params["f_idx"], self.SW_params["b_idx"]]
            sampled_currents=[np.zeros(len(self.SW_params["f_idx"])), np.zeros(len(self.SW_params["b_idx"]))]
            colours=["red", "green"]
            mean_idx=copy.deepcopy(sampled_currents)
            for i in range(0, len(self.SW_params["f_idx"])):
                for j in range(0, len(sampled_currents)):
                    x=indexes[j][i]
                    data=self.rolling_window(current[int(x-kwargs["mean"]-1):int(x-1)], kwargs["window_length"])
                    #plt.scatter(range(int(x-kwargs["mean"]-1),int(x-1)), data, color=colours[j])
                    #mean_idx[j][i]=np.mean(range(int(x-kwargs["mean"]-1),int(x-1)))
                    sampled_currents[j][i]=np.mean(data)

            forwards=np.zeros(len(self.SW_params["f_idx"]))
            backwards=np.zeros(len(self.SW_params["b_idx"]))
            forwards=np.array([current[x-1] for x in self.SW_params["f_idx"]])
            backwards=np.array([current[int(x)-1] for x in self.SW_params["b_idx"]])
        return forwards, backwards, backwards-forwards, self.SW_params["E_p"]
        def simulate(self, parameters, times):
                nd_dict=super().simulate(parameters, times)
                times=self.SW_params["sim_times"]
                solver=sos.SW_current
                
                if self._internal_options.dispersion == True:
                    current = self.dispersion_simulator(solver,  sim_params, t)
                else:
                    current = solver(t, nd_dict)
                sw_dict={"total":current}
                sw_dict["forwards"], sw_dict["backwards"], sw_dict["net"], E_p=self.SW_peak_extractor(current)
                if self.options.square_wave_return!="total":
                    polynomial_cap=nd_dict["Cdl"]*np.ones(len(E_p))
                    keys=["CdlE1", "CdlE2", "CdlE3"]
                    for i in range(0, len(keys)):
                        cdl=keys[i]
                        if nd_dict[cdl]!=0:
                            ep_power=np.power(E_p, i+1)
                            polynomial_cap=np.add(polynomial_cap, nd_dict[cdl]*ep_power)
                    current=np.add(polynomial_cap, sw_dict[self._internal_options.square_wave_return])
        return current



class ExperimentHandler:
    @staticmethod
    def create(options, cls):
        #Handle conditional initialisation here
        pass


