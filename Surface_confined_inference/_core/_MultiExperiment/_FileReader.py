import Surface_confined_inference as sci
import numpy as np
from scipy.interpolate import CubicSpline
import os
def _process_data(addresses, class_list, class_keys):
    if len(addresses)==0:
        return class_list
    for experiment_key in class_keys:
        loc=class_list[experiment_key]
        cls=loc["class"]
        key_parts = experiment_key.split("-")
        matching_files = [
            x for x in addresses if all(part in x for part in key_parts)
        ]
        
        if not matching_files:
            raise FileNotFoundError("No matching file containing {0} found in provided file_list".format(key_parts))
            
        
        file = matching_files[0]
        
        print(f"Reading file for: {experiment_key}")
        try:
            try:
                data = np.loadtxt(file, delimiter=",")
            except:
                data = np.loadtxt(file)
        except:
            raise
        process_dict={"FTACV":_process_ts_data,"DCV":_process_ts_data,"PSV":_process_ts_data, "SWV":_process_swv_data, "SquareWave":_process_swv_data}     
        if cls.experiment_type not in process_dict:
            raise KeyError("Experiment {0} needs to contain one of {1}".format(experiment_key, process_dict.keys()))
        else:
            class_list[experiment_key]=process_dict[cls.experiment_type](experiment_key, data,loc)
    return class_list
def _process_ts_data(experiment_key, data,  loc):
        """
        Process FTACV data.
        
        Parameters:
        -----------
        experiment_key : str
            Unique experiment identifier.
        data : numpy.ndarray
            Experimental data array.
        """        
        cls=loc["class"]
        zero_option=loc["Zero_params"]
        time=data[:,0]
        current=data[:,1]
        norm_current = cls.nondim_i(current)
        norm_time = cls.nondim_t(time)
        
        # Store in class dictionary
        loc["data"] = norm_current
        loc["times"] = norm_time
        if zero_option==None:
            return loc
        elif isinstance(zero_option, list):
            zero_params=zero_option
        elif hasattr(cls, "boundaries") is False:
            raise ValueError("Class doesn't contain boundaries for normalisation required for method {0}".format(zero_option))
        elif zero_option=="midpoint":
            zero_params=cls.change_normalisation_group([0.5 for x in cls.optim_list], "un_norm")
        elif zero_option=="random":
            zero_params=cls.change_normalisation_group(np.random.rand(len(cls.optim_list)), "un_norm")
        else:
            raise ValueError("Zero params for {0} must be one of midpoint, random, a list of parameters or None, not {1}".format(experiment_key, zero_option))
        # Generate zero point for error calculation
        dummy_zero_class = sci.SingleExperiment(
            cls.experiment_type,
            cls._internal_options.input_params,
            problem="forwards",
            normalise_parameters=False,
            model=cls.model
        )
        dummy_zero_class.fixed_parameters=cls.fixed_parameters
        dummy_zero_class.dispersion_bins = [1]
        dummy_zero_class.optim_list = cls.optim_list
        

        worst_case = dummy_zero_class.simulate(
            zero_params, 
            norm_time
        )
       
        
        loc["zero_point"] = sci._utils.RMSE(worst_case, norm_current)
        loc["zero_sim"]=worst_case
        if cls.experiment_type!="DCV":
            loc["FT"] = cls.experiment_top_hat(norm_time, norm_current)#
            ft_worst_case = cls.experiment_top_hat(norm_time, worst_case)
            loc["zero_point_ft"] = sci._utils.RMSE(ft_worst_case, loc["FT"])
        return loc
def _process_swv_data(experiment_key, data, loc):
        """
        Process SWV data.
        
        Parameters:
        -----------
        experiment_key : str
            Unique experiment identifier.
        data : numpy.ndarray
            Experimental data array.
        """
        current = data[:-1, 1]
        cls = loc["class"]
        zero_params=loc["Zero_params"]
        # Configure class
        
        
        
        
        # Calculate times and voltages
        times = cls.calculate_times()
        voltage = cls.get_voltage(times)
        pot = np.array([voltage[int(x)] for x in cls.SW_params["b_idx"]])
        
        # Apply baseline correction
        if zero_params is not None:
            signal_region = zero_params["potential_window"]
            before = np.where((pot < signal_region[0]))
            after = np.where((pot > signal_region[1]))
            
            noise_data = []
            noise_spacing = zero_params["thinning"]
            roll = zero_params["smoothing"]
            midded_current = sci._utils.moving_avg(current, roll)
            
            for sequence in [pot, midded_current]:
                catted_sequence = np.concatenate([
                    sequence[before][roll+10-1::noise_spacing],
                    sequence[after][roll+10-1::noise_spacing]
                ])
                noise_data.append(catted_sequence)
            
            sort_args = np.argsort(noise_data[0])
            sorted_x = [noise_data[0][x] for x in sort_args]
            sorted_y = [noise_data[1][x] for x in sort_args]
            
            # Apply cubic spline for baseline correction
            CS = CubicSpline(sorted_x, sorted_y)
            
            # Store normalized data
            loc["data"] = cls.nondim_i(current - CS(pot))
        else:
             loc["data"] = cls.nondim_i(current)
        loc["times"] = times
        loc["zero_sim"]=np.zeros(len(current))
        loc["zero_point"] = sci._utils.RMSE(np.zeros(len(current)), loc["data"])
        return loc