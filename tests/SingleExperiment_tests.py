import unittest
from unittest.mock import patch
import numpy as np
from scipy.signal import decimate
from Surface_confined_inference import SingleExperiment, NDParams, Dispersion
from Surface_confined_inference._core import OptionsDecorator
import os
class TestSingleExperiment(unittest.TestCase):

    def setUp(self):
        self.experiment_type = "FTACV"
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
                                        "E0":0.1,
                                        "k0":100,
                                        "Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        self.experiment.optim_list=[]
        self.times=self.experiment.calculate_times()
        predicted_current=self.experiment.simulate(self.times, [])
        self.decimated_current=decimate(predicted_current, 8)
        cwd=os.getcwd()
        if "Surface_confined_inference/tests" in cwd:
            self.data_loc=cwd+"/testdata"
        else:
            self.data_loc=cwd+"/tests/testdata"

    def test_init(self):
        self.assertEqual(self.experiment._internal_options.experiment_type, "FTACV")
        self.assertIsInstance(self.experiment._NDclass, NDParams)
        self.assertIsInstance(self.experiment._internal_options, OptionsDecorator)
        Nondim_constants=[0.0256796443598039, 6.575220832579881e-07, 1.027185774392156]
        Nonddim_variables=[self.experiment._NDclass.c_E0, self.experiment._NDclass.c_I0, self.experiment._NDclass.c_T0]
        for elem in zip(Nondim_constants, Nonddim_variables):
            self.assertEqual(elem[0], elem[1])


    def test_calculate_times(self):
        times = self.experiment.calculate_times(dimensional=True)
        self.assertIsInstance(times, np.ndarray)
        self.assertAlmostEqual(times[-1],55.9995)
        self.assertEqual((times[1]-times[0]),0.0005)


    def test_optim_list_setter(self):
        self.experiment.fixed_parameters={
                                        "E0":0.1, 
                                        "k0":1,
                                        "Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        self.experiment.boundaries = {"param1": [0, 1], "param2": [0, 2]}
        self.experiment.optim_list = ["param1", "param2"]
        self.assertEqual(self.experiment.optim_list, ["param1", "param2"])

    def test_boundaries_setter(self):
        self.experiment.fixed_parameters={
                                        "E0":0.1, 
                                        "k0":1,
                                        "Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        boundaries = {"param1": [0, 1], "param2": [0, 2]}
        self.experiment.boundaries = boundaries
        self.assertEqual(self.experiment.boundaries, boundaries)

    def test_fixed_parameters_setter(self):
        fixed_params = {"param1": 0.5, "param2": 1.5}
        self.experiment.fixed_parameters = fixed_params
        self.assertEqual(self.experiment.fixed_parameters, fixed_params)
    def test_dipsersion_checking(self):
        self.experiment.fixed_parameters={
                                        "k0":1,
                                        "Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        E0Normal=["E0_mean", "E0_std"]
        self.experiment.boundaries={"E0_mean": [-0.1, 0.1], "E0_std": [1e-3, 0.05]}
        self.experiment.dispersion_bins=[16]
        self.experiment.GH_quadrature=True
        self.experiment.optim_list=E0Normal
        self.assertIsInstance(self.experiment._disp_class, Dispersion)

    def test_normalise_unnormalise(self):
        value = 0.5
        boundaries = [0, 1]
        normalised = self.experiment.normalise(value, boundaries)
        unnormalised = self.experiment.un_normalise(normalised, boundaries)
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
        test_current=np.load(self.data_loc+"/Current.npy")
        error=self.experiment.RMSE(test_current, self.decimated_current)
        self.assertTrue(error<1e-4)
    def test_top_hat_filter(self):
        test_FT=np.load(self.data_loc+"/CurrentFT.npy")
        self.experiment.Fourier_filtering=True
        self.experiment.Fourier_function="abs"
        FT=self.experiment.top_hat_filter(decimate(self.times, 8), self.decimated_current)
        error=self.experiment.RMSE(FT, test_FT)
        self.assertTrue(error<0.3)
    def test_get_voltage(self):
        test_voltage=np.load(self.data_loc+"/Potential.npy")
        times=self.experiment.dim_t(self.times)
        voltage=self.experiment.get_voltage(times, dimensional=True)
        voltage=decimate(voltage, 8)
        error=self.experiment.RMSE(voltage, test_voltage)
        self.assertTrue(error<1e-9)
    def test_dispersion_simulate(self):
        test_current=np.load(self.data_loc+"/DispersedCurrent.npy")
        self.experiment.dispersion_bins = [16]
        self.experiment.GH_quadrature = True
        self.experiment.fixed_parameters={
                                        "E0_mean":0.1,
                                        "E0_std":0.03,
                                        "k0":100,
                                        "Cdl": 1e-4,
                                        "gamma": 1e-10,
                                        "alpha": 0.5,
                                        "Ru": 100,
                                        }
        self.experiment.optim_list=[]
        times=self.times
        predicted_current=self.experiment.simulate(times, [])
        decimated_current=decimate(predicted_current, 8)
        error=self.experiment.RMSE(test_current, decimated_current)
        self.assertTrue(error<1e-4)
    


if __name__ == '__main__':
    unittest.main()