import unittest
from unittest.mock import patch
import pytest
import numpy as np
from scipy.signal import decimate
from Surface_confined_inference import SingleExperiment, NDParams, Dispersion, top_hat_filter, BaseExperiment
import Surface_confined_inference
from Surface_confined_inference._utils import RMSE

import os

@pytest.mark.parametrize("experiment_type", ["FTACV", "PSV", "SquareWave", "DCV"])
class TestSingleExperiment:

    def setup_method(self, method):
        cwd=os.getcwd()
        self.memory={}
        input_dict={"FTACV":
            {
                "E_start": -0.4,
                "E_reverse": 0.3,
                "omega": 10,
                "delta_E": 0.15,
                "area": 0.07,
                "Temp": 298,
                "N_elec": 1,
                "phase": 0,
                "Surface_coverage": 1e-10,
                "v": 25e-3,
            },
        "PSV":
            {
                "Edc": -0.05,
                "omega": 10,
                "delta_E": 0.15,
                "area": 0.07,
                "Temp": 298,
                "N_elec": 1,
                "phase": 0,
                "Surface_coverage": 1e-10,
            },
        "DCV":
            {
                "E_start": -0.4,
                "E_reverse": 0.3,
                "area": 0.07,
                "Temp": 298,
                "N_elec": 1,
                "Surface_coverage": 1e-10,
                "v": 25e-3,
            },
        "SquareWave":
            {
                "scan_increment":2e-3,
                "omega":10,
                "delta_E":0.7,
                "SW_amplitude":10e-3,
                "sampling_factor":200,
                "E_start":-0.4,
                "Temp":278,
                "v":1,
                "area":0.07,
                "N_elec":1,
                "Surface_coverage":1e-10
            }
        }
        self.decs={key:8 if key in ["FTACV","DCV", "PSV"] else 1 for key in input_dict.keys()}
        for key in ["FTACV", "PSV", "SquareWave", "DCV"]:
            self.memory[key]={}

            if "Surface_confined_inference/tests" in cwd:
                self.memory[key]["data_loc"]=os.path.join("testdata", key)
            else:
                self.memory[key]["data_loc"]=os.path.join("tests","testdata", key)
            self.memory[key]["class"] = SingleExperiment(key, input_dict[key])
            self.memory[key]["class"].fixed_parameters={
                                            "Cdl": 1e-4,
                                            "gamma": 1e-10,
                                            "alpha": 0.5,
                                            "Ru": 100,
                                            }
            self.memory[key]["class"].boundaries={"E0":[0,0.2], "k0":[1e-3, 1000]}
            self.memory[key]["class"].optim_list=["E0", "k0"]
            self.memory[key]["times"]=self.memory[key]["class"].calculate_times(sampling_factor=200, dimensional=False)
            current=self.memory[key]["class"].simulate([0.1, 100],self.memory[key]["times"])
            self.memory[key]["dec_current"]=decimate(current, self.decs[key])        
            file="Current.npy"
            if file in os.listdir(self.memory[key]["data_loc"]):
                self.memory[key]["test_current"]=np.load(os.path.join(self.memory[key]["data_loc"], file))
            else:
                self.memory[key]["test_current"]=self.memory[key]["dec_current"]
                np.save((os.path.join(self.memory[key]["data_loc"], file)), self.memory[key]["dec_current"])
            
    def test_init(self, experiment_type):
        assert self.memory[experiment_type]["class"]._internal_options.experiment_type == experiment_type

    def test_calculate_times(self,experiment_type):
        times=self.memory[experiment_type]["class"].dim_t(self.memory[experiment_type]["times"])
        val_dict={"FTACV":{"dt":0.0005, "t_end":56},
                "PSV":{"dt":0.0005, "t_end":5},
                "DCV":{"dt":0.2, "t_end":56}}
        if experiment_type in val_dict:
            assert isinstance(times, np.ndarray)
            assert abs((val_dict[experiment_type]["dt"]) - (times[1]-times[0])) < 1e-10
            assert abs(times[-1] -val_dict[experiment_type]["t_end"]+val_dict[experiment_type]["dt"]) < 1e-6
        

    def test_optim_list_setter(self, experiment_type):
        self.memory[experiment_type]["class"].fixed_parameters={
                                        "Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        self.memory[experiment_type]["class"].boundaries = {"E0": [0, 1], "k0": [0, 2]}
        self.memory[experiment_type]["class"].optim_list = ["E0", "k0"]
        assert self.memory[experiment_type]["class"].optim_list == ["E0", "k0"]

    def test_boundaries_setter(self,experiment_type):
        self.memory[experiment_type]["class"].fixed_parameters={
                                        "Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        boundaries = {"param1": [0, 1], "param2": [0, 2]}
        self.memory[experiment_type]["class"].boundaries = boundaries
        assert self.memory[experiment_type]["class"].boundaries == boundaries

    def test_fixed_parameters_setter(self,experiment_type):
        fixed_params={
                                        "Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        self.memory[experiment_type]["class"].fixed_parameters = fixed_params
        assert self.memory[experiment_type]["class"].fixed_parameters == fixed_params
    def test_dispersion_checking(self,experiment_type):
        E0Normal=["E0_mean", "E0_std"]
        self.memory[experiment_type]["class"].optim_list=[]
        self.memory[experiment_type]["class"].fixed_parameters={
                                        "k0":1,
                                        "Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        self.memory[experiment_type]["class"].dispersion_bins=[16]
        self.memory[experiment_type]["class"].optim_list=E0Normal
        self.memory[experiment_type]["class"].boundaries={"E0_mean": [-0.1, 0.1], "E0_std": [1e-3, 0.05]}
        self.memory[experiment_type]["class"].GH_quadrature=True
        assert isinstance(self.memory[experiment_type]["class"]._disp_class, Dispersion)
        self.memory[experiment_type]["class"].optim_list=[]
    def test_normalise_unnormalise(self,experiment_type):
        value = 0.5
        boundaries = [0, 2]
        normalised = Surface_confined_inference._utils.normalise(value, boundaries)
        unnormalised = Surface_confined_inference._utils.un_normalise(normalised, boundaries)
        assert abs(value - unnormalised) < 1e-7
    def test_change_norm_group(self,experiment_type):
        self.memory[experiment_type]["class"].fixed_parameters={"Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        self.memory[experiment_type]["class"].boundaries={"k0": [1e-3, 200], "E0": [1e-3, 0.06]}
        self.memory[experiment_type]["class"].optim_list=["k0", "E0"]
        values=[10, 0.03]
        normed=self.memory[experiment_type]["class"].change_normalisation_group(values, "norm")
        unnormed=self.memory[experiment_type]["class"].change_normalisation_group(normed, "un_norm")
        
        for elem in zip(values, unnormed):
            assert abs(elem[0] - elem[1]) < 1e-7
    
    def test_simulate(self,experiment_type):
        
        error=RMSE(self.memory[experiment_type]["test_current"], self.memory[experiment_type]["dec_current"])
        assert error < 1e-4
    def test_save_class(self,experiment_type):
        self.memory[experiment_type]["class"].save_class("Test_json.json")
        save_class=BaseExperiment.from_json("Test_json.json")
        assert isinstance(save_class, SingleExperiment)
        old_options=self.memory[experiment_type]["class"]._internal_options.as_dict()
        saved_options=save_class._internal_options.as_dict()
        for key in old_options.keys():
            assert old_options[key] == saved_options[key]
    def test_top_hat_filter(self,experiment_type):
        if experiment_type in ["PSV", "DCV"]:
            FT=top_hat_filter(decimate(self.memory[experiment_type]["times"], self.decs[experiment_type]), self.memory[experiment_type]["dec_current"], Fourier_function="abs", Fourier_harmonics=list(range(0, 10)), Fourier_window="hanning", top_hat_width=0.5)
            file="CurrentFT.npy"
            if file in os.listdir(self.memory[experiment_type]["data_loc"]):
                test_FT=np.load(os.path.join(self.memory[experiment_type]["data_loc"], file))
            else:
                test_FT=FT
                np.save((os.path.join(self.memory[experiment_type]["data_loc"], file)), test_FT)
            error=RMSE(FT, test_FT)
            assert error < 0.3
    def test_get_voltage(self,experiment_type):

        times=self.memory[experiment_type]["class"].dim_t(self.memory[experiment_type]["times"])
        voltage=self.memory[experiment_type]["class"].get_voltage(times)
        voltage=decimate(voltage, self.decs[experiment_type])
        file="Potential.npy"
        if file in os.listdir(self.memory[experiment_type]["data_loc"]):
            test_voltage=np.load(os.path.join(self.memory[experiment_type]["data_loc"], file))
        else:
            test_voltage=voltage
            np.save((os.path.join(self.memory[experiment_type]["data_loc"], file)), voltage)
        error=RMSE(voltage, test_voltage)
        assert error < 1e-9
    def test_dispersion_simulate(self,experiment_type):
        self.memory[experiment_type]["class"].dispersion_bins = [16]
        self.memory[experiment_type]["class"].GH_quadrature = True
        self.memory[experiment_type]["class"].optim_list=[]
        self.memory[experiment_type]["class"].fixed_parameters={
                                        "E0_mean":0.1,
                                        "E0_std":0.03,
                                        "k0":100,
                                        "Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        
        times=self.memory[experiment_type]["times"]
        predicted_current=self.memory[experiment_type]["class"].simulate([], times)
        decimated_current=decimate(predicted_current, self.decs[experiment_type])
        file="DispersedCurrent.npy"
        if file in os.listdir(self.memory[experiment_type]["data_loc"]):
            test_current=np.load(os.path.join(self.memory[experiment_type]["data_loc"], file))
        else:
            test_current=decimated_current
            np.save((os.path.join(self.memory[experiment_type]["data_loc"], file)), decimated_current)
        error=RMSE(test_current, decimated_current)
        assert error < 1e-4


    


if __name__ == '__main__':
    #v=TestSingleExperiment()
    #v.setup_method(None)
    unittest.main()