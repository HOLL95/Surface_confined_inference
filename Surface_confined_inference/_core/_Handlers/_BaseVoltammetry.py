import Surface_confined_inference as sci
import SurfaceODESolver as sos
import itertools
#context manager
#options
#parameter handling
#dispersion handling
def _parallel_ode_simulation(nd_dict, times, weight):
    return weight * np.array(sos.ODEsimulate(times, nd_dict))[0, :]

def _parallel_faradaic_simulation(nd_dict, times, weight):
    return weight * np.array(sos.ODEsimulate(times, nd_dict))[2, :]

def _parallel_sw_simulation(nd_dict, times, weight):
    return weight * np.array(sos.SW_current(times, nd_dict))
def _funcswitch(tuple_arg):
    if tuple_arg[0]=="SquareWave":
        return _parallel_sw_simulation
    elif tuple_arg[1] is True:
        return _parallel_faradaic_simulation
    else:
        return _parallel_ode_simulation

FUNCTION_REGISTRY = {
    {tuple(x):_funcswitch(x) for x in itertools.product(["FTACV", "PSV", "SWV", "SquareWave"], [True, False])}
}

class ParameterInterface:
    def __init__(self,func_dict=None, simulation_dict=None, disp_class=None,  gh_values=None, self.optim_list=None):
        self.sim_dict=simulation_dict
        self.func_dict=func_dict
        self.disp_class = disp_class
        self.gh_values = gh_values
        self.optim_list=optim_list
     def nondimensionalise(self, sim_params, simulation_dict, function_dict):
        """
        Args:
            sim_params (dict): dictionary of parameter values with the same keys as _optim_list
        Modifies:
            _internal_memory["simulation_dict"] with the values from sim_params
        Returns:
            dict: Dictionary of appropriately non-dimensionalised parameters
        """
        nd_dict = {}
        for key in self.optim_list:
            self.sim_dict[key] = sim_params[key]
        for key in self.sim_dict.keys():
            nd_dict[key] = self.func_dict[key](
                self.sim_dict[key]
            )
        return nd_dict

class BaseSimulation:
    """Base strategy for simulation execution"""
    def __init__(self, options, param_interface):
        self.options = options
        self.param_interface = param_interface
    
    def get_solver_function(self):
        """Return the appropriate solver function"""
        raise NotImplementedError
    
    def get_parallel_function(self):
        """Return the appropriate parallel function for multiprocessing"""
        raise NotImplementedError
    
    def simulate_single(self, solver, nd_dict, times):
        """Execute single simulation"""
        return solver(times, nd_dict)
    
    def simulate_parallel(self, sim_params, times):
        """Execute parallel simulation with dispersion"""
       


class BaseHandler:
    def __init__(self, options, param_interface):
        self.options=options
        self.param_interface=param_interface
    def simulate(self, sim_params, times):
        nd_dict = self.param_interface.nondimensionalise(sim_params)
        return nd_dict
    def dispersion_simulator(self, sim_params, times):#######################################Need to add parallel####################################
        """
        Args:
            Solver (function): function that takes time (list) and simualtion_params (dictionary) arguemnts and returns a current
            times (list): list of time points for the solver
            sim_params (dict): dictionary of simulation parameters, with the same keys as _optim_list
        Returns:
            list: list of current values at times

        Initially passes `sim_params` to the `nondimensionalise` function so all parameters are correctly represented in `_internal_memory["simulation_dict"]`,
        then caluclates bin values and associated weights by calling `disp_class.generic_dispersion`
        This returns a list of parameter values of size b^n (n dispersed parameters, b bins) and a list of weights, which is b^n tuples each of length n.
        For each weight, the `simulation_dict` is updated with the correct parameters, nondimensionalised and the resulting dictionary passed to the solver
        The resulting current is multiplied by the product of the appropriate weight tuple, and added to the dispersed current
        This dispersed current is then returned

        """
        
        nd_dict = self.param_interface.nondimensionalise(sim_params)
        
        disp_params, values, weights = self.param_interface.disp_class.generic_dispersion(
            self.param_interface.sim_dict, self.param_interface.gh_values
        )
        dictionaries = []
        weight_products = []
        
        for i in range(len(weights)):
            temp_dict = copy.deepcopy(self.param_interface.sim_dict)
            for j, param in enumerate(disp_params):
                temp_dict[param] = values[i][j]
            old_sim_dict = self.param_interface.sim_dict
            self.param_interface.sim_dict = temp_dict
            nd_dict_dispersed = self.param_interface.nondimensionalise(sim_params)
            self.param_interface.sim_dict = old_sim_dict
            
            dictionaries.append(nd_dict_dispersed)
            weight_products.append(np.prod(weights[i]))
        
        parallel_func = self.get_parallel_function()
        iterable = [(nd_dict, times, weight) for nd_dict, weight in zip(dictionaries, weight_products)]
        
        pool = None
        try:
            pool = mp.Pool(processes=num_cpu)
            results = pool.starmap(parallel_func, iterable)
            np_results = np.array(results)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        
        return np.sum(np_results, axis=0)
    def get_voltage(self, times, input_parameters, validation_parameters, **kwargs):
        """
        Args:
            times (list): list of timepoints
            input_parameters (dict, optional): dictionary of input parameters
            dimensional (bool, optional): whether or not the times are in dimensional format or not
        Returns:
            list: list of potential values at the provided time points. By default, these values will be in dimensional format (V)
        """

        if "dimensional" not in kwargs:
            kwargs["dimensional"]=False 
        
        
        sci.check_input_dict(
            input_parameters, copy.deepcopy(validation_parameters), optional_arguments=[]
        )
        input_parameters=copy.deepcopy(input_parameters)
        input_parameters = self.validate_input_parameters(input_parameters)
        return input_parameters
    def calculate_times(self, params, **kwargs):
        """
        Calculates the time of an experiment e.g. for synthetic data. Each experiment has an end time, and a timestep.
        FTV/DCV: Defined by the voltage window (V) and scan rate (V s^-1), doubled as you go up and down
        PSV: Defined by the number of oscillations as part of the initialisation of the Experiment class
        SquareWave: Defined per square wave "oscillation"

        Args:
            sampling_factor (int, default=200): defines the timestep dt. For the experiments involving a sinusoid (PSV, FTACV, SquareWave), this is taken to mean
            X points per oscillation. For DCV, it is taken to mean X points per second.
            input_parameters (dict, default=self._internal_memory["input_parameters"]): Defines the parameters used to calculate the time
            Dimensional: controls whether the time is returned in dimensional or non-dimensional form
        Returns:
            np.ndarray: Array of times
        """
        print("Not implemented in base")
        return
      
class ContinuousHandler(BaseHandler):
    def __init__(self, options, param_interface):
        super().__init__(options, param_interface)
    def simulate(self, simulation_params, times):
        if self.options.dispersion==True:
            current = self.dispersion_simulator(sim_params, times)
        else:
            nd_dict=super().simulate(simulation_params, times)
            solver=FUNCTION_REGISTRY.get(self.options.experiment_type, self.options.Faradaic_only)
            solver(times, nd_dict)
        return current
    def get_voltage(self, times, input_parameters, validation_parameters, **kwargs):
        inputs=super().get_voltage(times, input_parameters, validation_parameters, **kwargs)
        if "tr" in inputs:
            if kwargs["dimensional"]==True:
                input_parameters["tr"]*=self._NDclass.c_T0
            input_parameters["omega"] *= 2 * np.pi
        return sos.potential(times, input_parameters)
    def calculate_times(self, params, **kwargs):
        if "sampling_factor" not in kwargs:
            sampling_factor=200
        else:
            sampling_factor=kwargs["sampling_factor"]
        if "dimensional" not in kwargs:
            dimensional=False
        else:
            dimensional=kwargs["dimensional"]
        if self._internal_options.experiment_type == "FTACV" or self._internal_options.experiment_type == "DCV":
            end_time = 2 * abs(params["E_start"] - params["E_reverse"]) / params["v"]
        elif self._internal_options.experiment_type == "PSV":
            if "PSV_num_peaks" not in kwargs:
                kwargs["PSV_num_peaks"]=50
            end_time = kwargs["PSV_num_peaks"]/ params["omega"]
        if self._internal_options.experiment_type == "DCV":
            dt = 1 / (sampling_factor * params["v"])
        elif self._internal_options.experiment_type == "FTACV" or self._internal_options.experiment_type == "PSV":
            dt = 1 / (sampling_factor * params["omega"])
        times = np.arange(0, end_time, dt)
        if dimensional == False:
            times = self.nondim_t(times)
        return times
class SquareWaveHandler(BaseHandler):
    def __init__(self, options, param_interface):
        super().__init__(options, param_interface)
        self.SW_sampling(param_interface.sim_dict)
    def get_voltage(self, times, input_parameters, validation_parameters, **kwargs):
        inputs=super().get_voltage(times, input_parameters, validation_parameters, **kwargs)
        voltages=np.zeros(len(times))
        for i in times:
            i=int(i)
            voltages[i-1]=sos.SW_potential(i,inputs["sampling_factor"],inputs["scan_increment"],inputs["SW_amplitude"],inputs["E_start"],inputs["v"])
        voltages[-1]=voltages[-2]
        return voltages
    def calculate_times(self,params, **kwargs):
        end_time=(abs(params["delta_E"]/params["scan_increment"])*params["sampling_factor"])
        dt=1
        return  np.arange(0, end_time, dt)
    def SW_sampling(self,parameters, **kwargs):
        self.SW_params={}
        sampling_factor=parameters["sampling_factor"]
        self.SW_params["end"]=int(abs(parameters['delta_E']//parameters['scan_increment']))
        p=np.array(range(0, self.SW_params["end"]))
        self.SW_params["b_idx"]=((sampling_factor*p)+(sampling_factor/2))-1
        self.SW_params["f_idx"]=(p*sampling_factor)-1
        Es=parameters["E_start"]#-parameters["E_0"]
        self.SW_params["E_p"]=(Es+parameters["v"]*(p*parameters['scan_increment']))
        self.SW_params["sim_times"]=self.calculate_times(parameters)

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
            if self.options.dispersion==True:   
                current = self.dispersion_simulator(sim_params, times)
            else:
                nd_dict=super().simulate(parameters, times)
                times=self.SW_params["sim_times"]
                solver=FUNCTION_REGISTRY.get(options.experiment_type, False)
                current = solver(times, nd_dict)
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
    def create(options, param_interface):
        if options.experiment_type=="SquareWave":
            return SquareWaveHandler(options, param_interface)
        else:
            return ContinuousHandler(options,param_interface)

