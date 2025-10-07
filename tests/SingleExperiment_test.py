import unittest
from unittest.mock import patch
import numpy as np
from scipy.signal import decimate
from Surface_confined_inference import SingleExperiment, NDParams, Dispersion, top_hat_filter, BaseExperiment
import Surface_confined_inference
from Surface_confined_inference._utils import RMSE

import os
class TestSingleExperiment(unittest.TestCase):

    def setUp(self):
        self.experiment_type = "FTACV"
        cwd=os.getcwd()
        
        if "Surface_confined_inference/tests" in cwd:
            self.data_loc=cwd+"/testdata"
        else:
            self.data_loc=cwd+"/tests/testdata"
        self.experiment_parameters = {
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
        }
        self.experiment = SingleExperiment(self.experiment_type, self.experiment_parameters)
        self.experiment.fixed_parameters={
                                        "Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        self.experiment.boundaries={"E0":[0,0.2], "k0":[1e-3, 1000]}
        self.experiment.optim_list=["E0", "k0"]
        self.times=self.experiment.calculate_times(sampling_factor=200, dimensional=False)
        self.predicted_current=self.experiment.simulate([0.1, 100],self.times)
        self.decimated_current=decimate(self.predicted_current, 8)        
        self.test_current=np.load(self.data_loc+"/Current.npy")

    def test_init(self):
        self.assertEqual(self.experiment._internal_options.experiment_type, "FTACV")

    def test_calculate_times(self):
        times=self.experiment.dim_t(self.times)
        self.assertIsInstance(times, np.ndarray)
        self.assertEqual((times[1]-self.times[0]),0.0005)
        self.assertAlmostEqual(times[-1],55.9995)
        

    def test_optim_list_setter(self):
        self.experiment.fixed_parameters={
                                        "Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        self.experiment.boundaries = {"E0": [0, 1], "k0": [0, 2]}
        self.experiment.optim_list = ["E0", "k0"]
        self.assertEqual(self.experiment.optim_list, ["E0", "k0"])

    def test_boundaries_setter(self):
        self.experiment.fixed_parameters={
                                        "Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        boundaries = {"param1": [0, 1], "param2": [0, 2]}
        self.experiment.boundaries = boundaries
        self.assertEqual(self.experiment.boundaries, boundaries)

    def test_fixed_parameters_setter(self):
        fixed_params={
                                        "Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        self.experiment.fixed_parameters = fixed_params
        self.assertEqual(self.experiment.fixed_parameters, fixed_params)
    def test_dispersion_checking(self):
        E0Normal=["E0_mean", "E0_std"]
        self.experiment.optim_list=[]
        self.experiment.fixed_parameters={
                                        "k0":1,
                                        "Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        self.experiment.dispersion_bins=[16]
        self.experiment.optim_list=E0Normal
        self.experiment.boundaries={"E0_mean": [-0.1, 0.1], "E0_std": [1e-3, 0.05]}
        self.experiment.GH_quadrature=True
        self.assertIsInstance(self.experiment._disp_class, Dispersion)

    def test_normalise_unnormalise(self):
        value = 0.5
        boundaries = [0, 2]
        normalised = Surface_confined_inference._utils.normalise(value, boundaries)
        unnormalised = Surface_confined_inference._utils.un_normalise(normalised, boundaries)
        self.assertAlmostEqual(value, unnormalised)
    def test_change_norm_group(self):
        self.experiment.fixed_parameters={"Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        self.experiment.boundaries={"k0": [1e-3, 200], "E0": [1e-3, 0.06]}
        self.experiment.optim_list=["k0", "E0"]
        values=[10, 0.03]
        normed=self.experiment.change_normalisation_group(values, "norm")
        unnormed=self.experiment.change_normalisation_group(normed, "un_norm")
        
        for elem in zip(values, unnormed):
            self.assertAlmostEqual(elem[0], elem[1])
    
    def test_simulate(self):
        
        error=RMSE(self.test_current, self.decimated_current)
        self.assertTrue(error<1e-4)
    def test_save_class(self):
        self.experiment.save_class("Test_json.json")
        save_class=BaseExperiment.from_json("Test_json.json")
        self.assertIsInstance(save_class,SingleExperiment)
        old_options=self.experiment._internal_options.as_dict()
        saved_options=save_class._internal_options.as_dict()
        for key in old_options.keys():
            self.assertEqual(old_options[key], saved_options[key])
    def test_top_hat_filter(self):
        FT=top_hat_filter(decimate(self.times, 8), self.decimated_current, Fourier_function="abs", Fourier_harmonics=list(range(0, 10)), Fourier_window="hanning", top_hat_width=0.5)
        #np.save(self.data_loc+"/CurrentFT.npy", FT)
        test_FT=np.load(self.data_loc+"/CurrentFT.npy")
        
        error=RMSE(FT, test_FT)
        self.assertTrue(error<0.3)
    def test_get_voltage(self):
        test_voltage=np.load(self.data_loc+"/Potential.npy")
        times=self.experiment.dim_t(self.times)
        voltage=self.experiment.get_voltage(times)
        voltage=decimate(voltage, 8)
        #np.save(self.data_loc+"/Potential.npy", voltage)
        error=RMSE(voltage, test_voltage)
        self.assertTrue(error<1e-9)
    def test_dispersion_simulate(self):
        test_current=np.load(self.data_loc+"/DispersedCurrent.npy")
        self.experiment.dispersion_bins = [16]
        self.experiment.GH_quadrature = True
        self.experiment.optim_list=[]
        self.experiment.fixed_parameters={
                                        "E0_mean":0.1,
                                        "E0_std":0.03,
                                        "k0":100,
                                        "Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        
        times=self.times
        predicted_current=self.experiment.simulate([], times)
        decimated_current=decimate(predicted_current, 8)
        #np.save(self.data_loc+"/DispersedCurrent.npy", decimated_current)
        error=RMSE(test_current, decimated_current)
        self.assertTrue(error<1e-4)


    


if __name__ == '__main__':
    unittest.main()