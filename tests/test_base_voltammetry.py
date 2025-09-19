import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from Surface_confined_inference._core._Handlers._BaseVoltammetry import (
    ParameterInterface,
    ContinuousHandler,
    SquareWaveHandler,
    ExperimentHandler,
    _funcswitch,
    _parallel_ode_simulation,
    _parallel_faradaic_simulation,
    _parallel_sw_simulation
)


class TestParameterInterface(unittest.TestCase):

    def setUp(self):
        # Mock dispersion function
        def mock_disp_function(params, gh_values):
            return ["E0"], [[0.1]], [[1.0]]

        # Create test data
        self.sim_dict = {
            "E0": 0.1,
            "k0": 100,
            "Cdl": 1e-4,
            "gamma": 1e-10
        }

        self.func_dict = {
            "E0": lambda x: x * 2,  # Simple transformation
            "k0": lambda x: x / 10,
            "Cdl": lambda x: x * 1000,
            "gamma": lambda x: x * 1e6
        }

        self.optim_list = ["E0", "k0"]
        self.gh_values = [1.0, 2.0]

        self.param_interface = ParameterInterface(
            disp_function=mock_disp_function,
            sim_dict=self.sim_dict,
            func_dict=self.func_dict,
            optim_list=self.optim_list,
            gh_values=self.gh_values
        )

    def test_nondimensionalise_basic(self):
        """Test basic nondimensionalisation functionality"""
        sim_params = {"E0": 0.2, "k0": 50}

        result = self.param_interface.nondimensionalise(sim_params)

        # Check that result is a dictionary
        self.assertIsInstance(result, dict)

        # Check that all parameters from sim_dict are present
        for key in self.sim_dict.keys():
            self.assertIn(key, result)

        # Check that transformations are applied correctly
        self.assertEqual(result["E0"], 0.2 * 2)  # Updated value * 2
        self.assertEqual(result["k0"], 50 / 10)  # Updated value / 10
        self.assertEqual(result["Cdl"], 1e-4 * 1000)  # Original value * 1000
        self.assertEqual(result["gamma"], 1e-10 * 1e6)  # Original value * 1e6

    def test_immutability(self):
        """Test that ParameterInterface is immutable"""
        with self.assertRaises(Exception):
            self.param_interface.sim_dict = {"new": "dict"}

        with self.assertRaises(Exception):
            self.param_interface.optim_list = ["new_param"]


class TestContinuousHandler(unittest.TestCase):

    def setUp(self):
        # Mock options
        self.mock_options = Mock()
        self.mock_options.experiment_type = "FTACV"
        self.mock_options.Faradaic_only = False
        self.mock_options.parallel_cpu = 1
        self.mock_options.dispersion = False

        # Mock parameter interface
        self.mock_param_interface = Mock()
        self.mock_param_interface.nondimensionalise.return_value = {
            "E0": 0.2, "k0": 10, "Cdl": 0.1, "gamma": 0.1
        }

        self.handler = ContinuousHandler(self.mock_options, self.mock_param_interface)

    @patch('Surface_confined_inference._core._Handlers._BaseVoltammetry.sos.ODEsimulate')
    def test_get_thetas(self, mock_ode_simulate):
        """Test get_thetas method"""
        # Mock ODE simulation result
        mock_result = np.array([
            [1, 2, 3],  # theta_ox
            [4, 5, 6],  # theta_red
            [7, 8, 9],  # i_far
            [10, 11, 12],  # i_cap
            [13, 14, 15],  # potential
            [16, 17, 18]   # current
        ])
        mock_ode_simulate.return_value = mock_result

        sim_params = {"E0": 0.1, "k0": 100}
        times = np.array([0, 1, 2])

        result = self.handler.get_thetas(sim_params, times)

        # Verify that ODEsimulate was called
        mock_ode_simulate.assert_called_once()

        # Verify result shape and content
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (6, 3))
        np.testing.assert_array_equal(result, mock_result)
    def test_get_parallel_function(self):
        """Test _get_parallel_function method"""
        # Test with Faradaic_only=False
        self.mock_options.Faradaic_only = False
        result = self.handler._get_parallel_function()
        self.assertEqual(result, _parallel_ode_simulation)

        # Test with Faradaic_only=True
        self.mock_options.Faradaic_only = True
        result = self.handler._get_parallel_function()
        self.assertEqual(result, _parallel_faradaic_simulation)


class TestSquareWaveHandler(unittest.TestCase):

    def setUp(self):
        # Mock options with required SW parameters
        self.mock_options = Mock()
        self.mock_options.experiment_type = "SquareWave"
        self.mock_options.input_params = {
            "E_start": -0.4,
            "delta_E": 0.5,
            "scan_increment": 0.005,
            "SW_amplitude": 0.05,
            "v": 0.025,
            "sampling_factor": 50
        }
        self.mock_options.square_wave_return = "total"
        self.mock_options.dispersion = False

        # Mock parameter interface
        self.mock_param_interface = Mock()

        # Create handler (this will call SW_sampling automatically)
        self.handler = SquareWaveHandler(self.mock_options, self.mock_param_interface)

    def test_get_parallel_function(self):
        """Test _get_parallel_function method"""
        result = self.handler._get_parallel_function()
        self.assertEqual(result, _parallel_sw_simulation)

    def test_sw_sampling_initialization(self):
        """Test that SW_sampling properly initializes SW_params"""
        # Check that SW_params was created
        self.assertIsInstance(self.handler.SW_params, dict)

        # Check required keys
        required_keys = ["end", "b_idx", "f_idx", "E_p", "sim_times"]
        for key in required_keys:
            self.assertIn(key, self.handler.SW_params)

        # Check data types
        self.assertIsInstance(self.handler.SW_params["end"], int)
        self.assertIsInstance(self.handler.SW_params["b_idx"], np.ndarray)
        self.assertIsInstance(self.handler.SW_params["f_idx"], np.ndarray)
        self.assertIsInstance(self.handler.SW_params["E_p"], np.ndarray)
        self.assertIsInstance(self.handler.SW_params["sim_times"], np.ndarray)

    def test_sw_peak_extractor(self):
        """Test SW_peak_extractor with basic functionality"""
        # Create test current data
        current_length = int(self.handler.SW_params["end"] * self.mock_options.input_params["sampling_factor"])
        test_current = np.sin(np.linspace(0, 2*np.pi, current_length)) * 1e-6

        forwards, backwards, net, E_p = self.handler.SW_peak_extractor(test_current)

        # Check return types
        self.assertIsInstance(forwards, np.ndarray)
        self.assertIsInstance(backwards, np.ndarray)
        self.assertIsInstance(net, np.ndarray)
        self.assertIsInstance(E_p, np.ndarray)

        # Check shapes
        expected_length = len(self.handler.SW_params["f_idx"])
        self.assertEqual(len(forwards), expected_length)
        self.assertEqual(len(backwards), expected_length)
        self.assertEqual(len(net), expected_length)
        self.assertEqual(len(E_p), expected_length)

        # Check that net = backwards - forwards
        np.testing.assert_array_almost_equal(net, backwards - forwards)


class TestExperimentHandler(unittest.TestCase):

    def test_create_continuous_handler(self):
        """Test creation of ContinuousHandler"""
        mock_options = Mock()
        mock_options.experiment_type = "FTACV"
        mock_param_interface = Mock()

        result = ExperimentHandler.create(mock_options, mock_param_interface)

        self.assertIsInstance(result, ContinuousHandler)

    def test_create_square_wave_handler(self):
        """Test creation of SquareWaveHandler"""
        mock_options = Mock()
        mock_options.experiment_type = "SquareWave"
        mock_options.input_params = {
            "E_start": -0.4,
            "delta_E": 0.5,
            "scan_increment": 0.005,
            "SW_amplitude": 0.05,
            "v": 0.025,
            "sampling_factor": 50
        }
        mock_param_interface = Mock()

        result = ExperimentHandler.create(mock_options, mock_param_interface)

        self.assertIsInstance(result, SquareWaveHandler)

    def test_create_other_experiment_types(self):
        """Test creation with other experiment types"""
        experiment_types = ["PSV", "DCV"]
        mock_param_interface = Mock()

        for exp_type in experiment_types:
            mock_options = Mock()
            mock_options.experiment_type = exp_type

            result = ExperimentHandler.create(mock_options, mock_param_interface)

            self.assertIsInstance(result, ContinuousHandler)


class TestParallelFunctions(unittest.TestCase):

    def test_funcswitch_square_wave(self):
        """Test _funcswitch for SquareWave experiments"""
        result = _funcswitch("SquareWave", False)
        self.assertEqual(result, _parallel_sw_simulation)

        result = _funcswitch("SquareWave", True)
        self.assertEqual(result, _parallel_sw_simulation)

    def test_funcswitch_faradaic_only(self):
        """Test _funcswitch for Faradaic only simulations"""
        result = _funcswitch("FTACV", True)
        self.assertEqual(result, _parallel_faradaic_simulation)

        result = _funcswitch("PSV", True)
        self.assertEqual(result, _parallel_faradaic_simulation)

    def test_funcswitch_normal_ode(self):
        """Test _funcswitch for normal ODE simulations"""
        result = _funcswitch("FTACV", False)
        self.assertEqual(result, _parallel_ode_simulation)

        result = _funcswitch("DCV", False)
        self.assertEqual(result, _parallel_ode_simulation)


if __name__ == '__main__':
    unittest.main()