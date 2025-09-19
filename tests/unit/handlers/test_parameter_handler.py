import unittest
import numpy as np
import copy
from unittest.mock import Mock, patch, MagicMock
import Surface_confined_inference as sci
from Surface_confined_inference._core._Handlers._ParameterHandler import (
    DispersionContext,
    ParameterContext,
    dispersion_checking,
    validate_boundaries,
    simulation_dict_construction,
    ParameterHandler
)


class TestDispersionContext(unittest.TestCase):
    """Test DispersionContext dataclass"""

    def test_dispersion_context_creation(self):
        """Test creating DispersionContext with all parameters"""
        context = DispersionContext(
            dispersion=True,
            dispersion_warning="test warning",
            dispersion_parameters=["param1", "param2"],
            dispersion_distributions=["normal", "uniform"],
            all_parameters=["param1", "param2", "param3"],
            GH_values=[[1, 2], [3, 4]]
        )

        self.assertTrue(context.dispersion)
        self.assertEqual(context.dispersion_warning, "test warning")
        self.assertEqual(context.dispersion_parameters, ["param1", "param2"])
        self.assertEqual(context.dispersion_distributions, ["normal", "uniform"])
        self.assertEqual(context.all_parameters, ["param1", "param2", "param3"])
        self.assertEqual(context.GH_values, [[1, 2], [3, 4]])

    def test_dispersion_context_defaults(self):
        """Test DispersionContext with default values"""
        context = DispersionContext(dispersion=False)

        self.assertFalse(context.dispersion)
        self.assertEqual(context.dispersion_warning, "")
        self.assertEqual(context.dispersion_parameters, [])
        self.assertEqual(context.dispersion_distributions, [])
        self.assertEqual(context.all_parameters, [])
        self.assertEqual(context.GH_values, [])


class TestParameterContext(unittest.TestCase):
    """Test ParameterContext dataclass"""

    def test_parameter_context_creation(self):
        """Test creating ParameterContext"""
        fixed_params = {"alpha": 0.5, "gamma": 1e-10}
        boundaries = {"E0": [0, 0.2], "k0": [1e-3, 1000]}
        optim_list = ["E0", "k0"]
        sim_dict = {"E0": None, "k0": None, "alpha": 0.5}

        context = ParameterContext(
            fixed_parameters=fixed_params,
            boundaries=boundaries,
            optim_list=optim_list,
            sim_dict=sim_dict
        )

        self.assertEqual(context.fixed_parameters, fixed_params)
        self.assertEqual(context.boundaries, boundaries)
        self.assertEqual(context.optim_list, optim_list)
        self.assertEqual(context.sim_dict, sim_dict)


class TestDispersionChecking(unittest.TestCase):
    """Test dispersion_checking function"""

    def test_no_dispersion_parameters(self):
        """Test when no dispersion parameters are present"""
        all_parameters = ["E0", "k0", "alpha", "gamma"]
        result = dispersion_checking(all_parameters, False, [])

        self.assertFalse(result.dispersion)
        self.assertEqual(result.dispersion_parameters, [])
        self.assertEqual(result.dispersion_distributions, [])
        self.assertEqual(result.all_parameters, all_parameters)
        self.assertEqual(result.GH_values, [])

    def test_normal_distribution_parameters(self):
        """Test detection of normal distribution parameters"""
        all_parameters = ["E0_mean", "E0_std", "k0", "alpha"]
        result = dispersion_checking(all_parameters, False, [10])

        self.assertTrue(result.dispersion)
        self.assertEqual(result.dispersion_parameters, ["E0"])
        self.assertEqual(result.dispersion_distributions, ["normal"])
        self.assertEqual(result.all_parameters, all_parameters)

    def test_multiple_distribution_types(self):
        """Test multiple distribution types"""
        all_parameters = ["E0_mean", "E0_std", "k0_lower", "k0_upper", "alpha"]
        result = dispersion_checking(all_parameters, False, [10, 5])

        self.assertTrue(result.dispersion)
        self.assertEqual(set(result.dispersion_parameters), {"E0", "k0"})
        self.assertIn("normal", result.dispersion_distributions)
        self.assertIn("uniform", result.dispersion_distributions)

    def test_incomplete_distribution_parameters(self):
        """Test exception when distribution parameters are incomplete"""
        all_parameters = ["E0_mean", "k0"]  # Missing E0_std for normal distribution

        with self.assertRaises(Exception) as context:
            dispersion_checking(all_parameters, False, [10])

        self.assertIn("requires", str(context.exception))

    def test_wrong_number_of_bins(self):
        """Test exception when number of bins doesn't match distributions"""
        all_parameters = ["E0_mean", "E0_std", "k0_lower", "k0_upper"]

        with self.assertRaises(ValueError) as context:
            dispersion_checking(all_parameters, False, [10])  # Need 2 bins, only have 1

        self.assertIn("Need one bin for each", str(context.exception))

    @patch('Surface_confined_inference._utils.GH_setup')
    def test_gauss_hermite_quadrature(self, mock_gh_setup):
        """Test Gauss-Hermite quadrature setup"""
        mock_gh_setup.return_value = [1, 2, 3]
        all_parameters = ["E0_mean", "E0_std", "k0_lower", "k0_upper"]

        result = dispersion_checking(all_parameters, True, [10, 5])

        self.assertTrue(result.dispersion)
        mock_gh_setup.assert_called_once_with(10)
        self.assertIsNotNone(result.GH_values[0])  # Should have GH values for normal distribution
        self.assertIsNone(result.GH_values[1])     # Should be None for uniform distribution


class TestValidateBoundaries(unittest.TestCase):
    """Test validate_boundaries function"""

    def test_valid_boundaries(self):
        """Test when all parameters have boundaries"""
        parameters = ["E0", "k0", "alpha"]
        boundaries = {"E0": [0, 0.2], "k0": [1e-3, 1000], "alpha": [0, 1]}

        # Should not raise an exception
        validate_boundaries(parameters, boundaries)

    def test_missing_boundaries(self):
        """Test exception when parameters are missing from boundaries"""
        parameters = ["E0", "k0", "alpha"]
        boundaries = {"E0": [0, 0.2], "k0": [1e-3, 1000]}  # Missing alpha

        with self.assertRaises(ValueError) as context:
            validate_boundaries(parameters, boundaries)

        self.assertIn("Need to define boundaries for", str(context.exception))
        self.assertIn("alpha", str(context.exception))

    def test_empty_parameters(self):
        """Test with empty parameters list"""
        parameters = []
        boundaries = {"E0": [0, 0.2]}

        # Should not raise an exception
        validate_boundaries(parameters, boundaries)


class TestSimulationDictConstruction(unittest.TestCase):
    """Test simulation_dict_construction function"""

    def setUp(self):
        self.mock_options = Mock()
        self.mock_options.experiment_type = "FTACV"
        self.mock_options.kinetics = "ButlerVolmer"
        self.mock_options.model = "simple"
        self.mock_options.input_params = {}

    def test_basic_construction(self):
        """Test basic simulation dictionary construction"""
        parameters = ["E0", "k0"]
        fixed_parameters = {"alpha": 0.5, "gamma": 1e-10}
        essential_parameters = ["E0", "k0", "alpha", "gamma", "Cdl", "CdlE1", "CdlE2", "CdlE3", "Ru", "phase"]
        dispersion_parameters = []
        dispersion = False
        simulation_dict = {}

        result = simulation_dict_construction(
            parameters, fixed_parameters, essential_parameters,
            dispersion_parameters, dispersion, simulation_dict, self.mock_options
        )

        self.assertIsNone(result["E0"])  # Optimization parameter
        self.assertIsNone(result["k0"])  # Optimization parameter
        self.assertEqual(result["alpha"], 0.5)  # Fixed parameter
        self.assertEqual(result["gamma"], 1e-10)  # Fixed parameter
        self.assertEqual(result["Cdl"], 0)  # Default value
        self.assertEqual(result["theta"], 0)  # Default value
        self.assertEqual(result["model"], 0)  # Simple model

    def test_missing_essential_parameters(self):
        """Test exception when essential parameters are missing"""
        parameters = ["E0"]
        fixed_parameters = {"alpha": 0.5}
        essential_parameters = ["E0", "k0", "alpha", "gamma"]  # k0 and gamma missing
        dispersion_parameters = []
        dispersion = False
        simulation_dict = {}

        with self.assertRaises(Exception) as context:
            simulation_dict_construction(
                parameters, fixed_parameters, essential_parameters,
                dispersion_parameters, dispersion, simulation_dict, self.mock_options
            )

        self.assertIn("following parameters either need to be set", str(context.exception))

    def test_parameter_in_both_fixed_and_optim(self):
        """Test warning when parameter is in both fixed and optimization lists"""
        parameters = ["E0", "alpha"]
        fixed_parameters = {"alpha": 0.5, "gamma": 1e-10}  # alpha in both
        essential_parameters = ["E0", "alpha", "gamma"]
        dispersion_parameters = []
        dispersion = False
        simulation_dict = {}

        with patch('warnings.warn') as mock_warn:
            result = simulation_dict_construction(
                parameters, fixed_parameters, essential_parameters,
                dispersion_parameters, dispersion, simulation_dict, self.mock_options
            )

            mock_warn.assert_called_once()
            self.assertIn("alpha", mock_warn.call_args[0][0])
            self.assertIsNone(result["alpha"])  # Should prioritize optimization

    def test_square_scheme_model(self):
        """Test square scheme model configuration"""
        self.mock_options.model = "square_scheme"

        parameters = ["E0_1"]
        fixed_parameters = {"alpha_1": 0.5}
        essential_parameters = ["E0_1", "alpha_1"]
        dispersion_parameters = []
        dispersion = False
        simulation_dict = {}

        result = simulation_dict_construction(
            parameters, fixed_parameters, essential_parameters,
            dispersion_parameters, dispersion, simulation_dict, self.mock_options
        )

        self.assertEqual(result["theta"], 1)  # Square scheme sets theta to 1
        self.assertEqual(result["model"], 1)  # Square scheme model flag

    def test_dcv_experiment_type(self):
        """Test DCV experiment type handling"""
        self.mock_options.experiment_type = "DCV"

        parameters = ["E0"]
        fixed_parameters = {"alpha": 0.5}
        essential_parameters = ["E0", "alpha", "phase"]
        dispersion_parameters = []
        dispersion = False
        simulation_dict = {}

        result = simulation_dict_construction(
            parameters, fixed_parameters, essential_parameters,
            dispersion_parameters, dispersion, simulation_dict, self.mock_options
        )

        self.assertEqual(result["phase"], 0)  # DCV defaults phase to 0


class TestParameterHandler(unittest.TestCase):
    """Test ParameterHandler class"""

    def setUp(self):
        self.mock_options = Mock()
        self.mock_options.experiment_type = "FTACV"
        self.mock_options.problem = "inverse"
        self.mock_options.model = "simple"
        self.mock_options.kinetics = "ButlerVolmer"
        self.mock_options.GH_quadrature = False
        self.mock_options.dispersion_bins = []
        self.mock_options.input_params = {}

        self.fixed_parameters = {"alpha": 0.5, "gamma": 1e-10, "Cdl": 1e-4, "Ru": 100}
        self.boundaries = {"E0": [0, 0.2], "k0": [1e-3, 1000]}
        self.parameters = ["E0", "k0"]

    def test_parameter_handler_init(self):
        """Test ParameterHandler initialization"""
        handler = ParameterHandler(
            options=self.mock_options,
            fixed_parameters=self.fixed_parameters,
            boundaries=self.boundaries,
            parameters=self.parameters
        )

        self.assertEqual(handler.options, self.mock_options)
        self.assertEqual(handler.fixed_parameters, self.fixed_parameters)
        self.assertEqual(handler.boundaries, self.boundaries)
        self.assertEqual(handler._optim_list, self.parameters)
        self.assertIsInstance(handler.context, ParameterContext)
        self.assertIsInstance(handler._disp_context, DispersionContext)

    def test_missing_boundaries_raises_error(self):
        """Test that missing boundaries raises ValueError"""
        incomplete_boundaries = {"E0": [0, 0.2]}  # Missing k0

        with self.assertRaises(ValueError):
            ParameterHandler(
                options=self.mock_options,
                fixed_parameters=self.fixed_parameters,
                boundaries=incomplete_boundaries,
                parameters=self.parameters
            )

    def test_change_normalisation_group_norm(self):
        """Test parameter normalization"""
        handler = ParameterHandler(
            options=self.mock_options,
            fixed_parameters=self.fixed_parameters,
            boundaries=self.boundaries,
            parameters=self.parameters
        )

        # Test normalization
        params = [0.1, 100]  # E0=0.1, k0=100
        normalized = handler.change_normalisation_group(params, "norm")

        # E0: 0.1 should be normalized to 0.5 (middle of [0, 0.2])
        # k0: 100 should be normalized to ~0.095 in log space
        self.assertAlmostEqual(normalized[0], 0.5, places=3)
        self.assertTrue(0 <= normalized[1] <= 1)

    def test_change_normalisation_group_unnorm(self):
        """Test parameter denormalization"""
        handler = ParameterHandler(
            options=self.mock_options,
            fixed_parameters=self.fixed_parameters,
            boundaries=self.boundaries,
            parameters=self.parameters
        )

        # Test denormalization
        normalized_params = [0.5, 0.5]
        denormalized = handler.change_normalisation_group(normalized_params, "un_norm")

        # E0: 0.5 should denormalize to 0.1 (middle of [0, 0.2])
        self.assertAlmostEqual(denormalized[0], 0.1, places=3)
        self.assertTrue(self.boundaries["k0"][0] <= denormalized[1] <= self.boundaries["k0"][1])

    def test_missing_boundary_key_error(self):
        """Test KeyError when parameter not in boundaries"""
        handler = ParameterHandler(
            options=self.mock_options,
            fixed_parameters=self.fixed_parameters,
            boundaries={"E0": [0, 0.2]},  # Missing k0
            parameters=["E0"]  # Only E0 to avoid validation error
        )

        # Add k0 to optim_list manually to test the error
        handler._optim_list = ["E0", "k0"]

        with self.assertRaises(KeyError):
            handler.change_normalisation_group([0.1, 100], "norm")

    def test_validate_input_parameters_dcv(self):
        """Test DCV parameter validation"""
        inputs = {
            "E_start": -0.4,
            "E_reverse": 0.3,
            "v": 0.025
        }

        result = ParameterHandler.validate_input_parameters(inputs, "DCV")

        self.assertEqual(result["tr"], abs(inputs["E_reverse"] - inputs["E_start"]) / inputs["v"])
        self.assertEqual(result["omega"], 0)
        self.assertEqual(result["delta_E"], 0)
        self.assertEqual(result["phase"], 0)

    def test_validate_input_parameters_ftacv(self):
        """Test FTACV parameter validation"""
        inputs = {
            "E_start": -0.4,
            "E_reverse": 0.3,
            "v": 0.025
        }

        result = ParameterHandler.validate_input_parameters(inputs, "FTACV")

        self.assertEqual(result["tr"], abs(inputs["E_reverse"] - inputs["E_start"]) / inputs["v"])
        # Should not modify omega, delta_E for FTACV

    def test_validate_input_parameters_psv(self):
        """Test PSV parameter validation"""
        inputs = {
            "Edc": 0.1
        }

        result = ParameterHandler.validate_input_parameters(inputs, "PSV")

        self.assertEqual(result["E_reverse"], inputs["Edc"])
        self.assertEqual(result["tr"], -1)
        self.assertEqual(result["v"], 0)

    def test_create_warning(self):
        """Test warning message creation"""
        warning_str = "Test warning message"
        formatted_warning = ParameterHandler.create_warning(warning_str)

        self.assertIn(warning_str, formatted_warning)
        self.assertTrue(formatted_warning.startswith("#"))
        self.assertTrue(formatted_warning.endswith("#"))


if __name__ == '__main__':
    unittest.main()