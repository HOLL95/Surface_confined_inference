import unittest
import io
import sys
from unittest.mock import Mock, patch, MagicMock
from Surface_confined_inference._core._Options._OptionsMixin import OptionsAwareMixin
from Surface_confined_inference._core._Options._OptionsDescriptor import (
    StringOption, NumberOption, BoolOption, EnumOption
)
from Surface_confined_inference._core._Options._OptionsMeta import OptionsManager


class TestOptionsAwareMixin(unittest.TestCase):
    """Test the OptionsAwareMixin class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock options manager for testing
        self.mock_options_manager = Mock()
        self.mock_options_manager.get_option_names.return_value = ["valid_option1", "valid_option2"]

    def test_core_options_defined(self):
        """Test that core options are properly defined."""
        expected_core_options = ["boundaries", "fixed_parameters", "optim_list"]
        self.assertEqual(OptionsAwareMixin._core_options, expected_core_options)

    def test_setattr_with_validated_option(self):
        """Test setting a validated option through _internal_options."""

        class TestClass(OptionsAwareMixin):
            def __init__(self):
                self._internal_options = self.mock_options_manager

        obj = TestClass()
        obj._internal_options = self.mock_options_manager

        # Set a valid option
        obj.valid_option1 = "test_value"

        # Should call setattr on the internal options manager
        self.mock_options_manager.__setattr__.assert_called_with("valid_option1", "test_value")
        # Should also set the attribute on the instance
        self.assertEqual(obj.valid_option1, "test_value")

    def test_setattr_with_manual_option(self):
        """Test setting a manual option that bypasses validation."""

        class TestClass(OptionsAwareMixin):
            def __init__(self):
                self._manual_options = ["manual_option"]

        obj = TestClass()

        # Set a manual option
        obj.manual_option = "manual_value"

        # Should set directly without validation
        self.assertEqual(obj.manual_option, "manual_value")

    def test_setattr_with_core_option(self):
        """Test setting a core option that bypasses validation."""

        class TestClass(OptionsAwareMixin):
            pass

        obj = TestClass()

        # Set core options
        obj.boundaries = {"param": [0, 1]}
        obj.fixed_parameters = {"fixed": 5.0}
        obj.optim_list = ["param1", "param2"]

        # Should set directly without validation or warnings
        self.assertEqual(obj.boundaries, {"param": [0, 1]})
        self.assertEqual(obj.fixed_parameters, {"fixed": 5.0})
        self.assertEqual(obj.optim_list, ["param1", "param2"])

    def test_setattr_with_private_attribute(self):
        """Test setting private attributes (starting with _)."""

        class TestClass(OptionsAwareMixin):
            pass

        obj = TestClass()

        # Set private attributes
        obj._private_attr = "private_value"
        obj._another_private = 42

        # Should set without warnings
        self.assertEqual(obj._private_attr, "private_value")
        self.assertEqual(obj._another_private, 42)

    @patch('builtins.print')
    def test_setattr_with_unknown_public_attribute(self, mock_print):
        """Test setting unknown public attributes triggers warning."""

        class TestClass(OptionsAwareMixin):
            pass

        obj = TestClass()

        # Set unknown public attribute
        obj.unknown_attr = "unknown_value"

        # Should print warning
        mock_print.assert_called_once()
        warning_call = mock_print.call_args[0][0]
        self.assertIn("Warning:", warning_call)
        self.assertIn("unknown_attr", warning_call)
        self.assertIn("TestClass", warning_call)
        self.assertIn("will not affect simulation behavior", warning_call)

        # Should still set the attribute
        self.assertEqual(obj.unknown_attr, "unknown_value")

    def test_setattr_without_internal_options(self):
        """Test behavior when _internal_options is not defined."""

        class TestClass(OptionsAwareMixin):
            pass

        obj = TestClass()

        # Should work normally without _internal_options
        obj.some_attr = "value"
        self.assertEqual(obj.some_attr, "value")

    def test_setattr_with_internal_options_no_get_option_names(self):
        """Test behavior when _internal_options exists but has no get_option_names method."""

        class TestClass(OptionsAwareMixin):
            def __init__(self):
                self._internal_options = Mock()
                # Don't give it get_option_names method

        obj = TestClass()

        # Should work normally without get_option_names
        obj.some_attr = "value"
        self.assertEqual(obj.some_attr, "value")

    @patch('builtins.print')
    def test_multiple_unknown_attributes_warnings(self, mock_print):
        """Test that warnings are printed for each unknown attribute."""

        class TestClass(OptionsAwareMixin):
            pass

        obj = TestClass()

        # Set multiple unknown attributes
        obj.unknown1 = "value1"
        obj.unknown2 = "value2"

        # Should print warning for each
        self.assertEqual(mock_print.call_count, 2)

        # Check that both warnings contain appropriate messages
        calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("unknown1" in call for call in calls))
        self.assertTrue(any("unknown2" in call for call in calls))

    def test_setattr_precedence_validated_over_manual(self):
        """Test that validated options take precedence over manual options."""

        class TestClass(OptionsAwareMixin):
            def __init__(self):
                self._internal_options = self.mock_options_manager
                self._manual_options = ["valid_option1"]  # Same name as validated option

        obj = TestClass()
        obj._internal_options = self.mock_options_manager

        # Set the option
        obj.valid_option1 = "test_value"

        # Should go through validation (not manual) since validated options have precedence
        self.mock_options_manager.__setattr__.assert_called_with("valid_option1", "test_value")

    def test_setattr_precedence_manual_over_core(self):
        """Test that manual options take precedence over core options."""

        class TestClass(OptionsAwareMixin):
            def __init__(self):
                self._manual_options = ["boundaries"]  # Same name as core option

        obj = TestClass()

        # Set the option
        obj.boundaries = {"param": [0, 1]}

        # Should be treated as manual option (no warning)
        self.assertEqual(obj.boundaries, {"param": [0, 1]})

    @patch('builtins.print')
    def test_setattr_core_option_no_warning(self, mock_print):
        """Test that core options don't trigger warnings."""

        class TestClass(OptionsAwareMixin):
            pass

        obj = TestClass()

        # Set all core options
        for core_option in OptionsAwareMixin._core_options:
            setattr(obj, core_option, f"value_for_{core_option}")

        # Should not print any warnings
        mock_print.assert_not_called()


class TestOptionsAwareMixinIntegration(unittest.TestCase):
    """Test OptionsAwareMixin integration with real options managers."""

    def test_integration_with_real_options_manager(self):
        """Test integration with actual OptionsManager instance."""

        class MockOptionsManager(OptionsManager):
            string_opt = StringOption("string_opt", default="default")
            number_opt = NumberOption("number_opt", default=42, min_value=0, max_value=100)
            bool_opt = BoolOption("bool_opt", default=True)

        class TestExperiment(OptionsAwareMixin):
            def __init__(self):
                self._internal_options = MockOptionsManager()
                self._manual_options = ["manual_param"]

        experiment = TestExperiment()

        # Test validated options
        experiment.string_opt = "new_string"
        experiment.number_opt = 75
        experiment.bool_opt = False

        # Check values are set on both instance and internal options
        self.assertEqual(experiment.string_opt, "new_string")
        self.assertEqual(experiment.number_opt, 75)
        self.assertEqual(experiment.bool_opt, False)

        self.assertEqual(experiment._internal_options.string_opt, "new_string")
        self.assertEqual(experiment._internal_options.number_opt, 75)
        self.assertEqual(experiment._internal_options.bool_opt, False)

        # Test manual option
        experiment.manual_param = "manual_value"
        self.assertEqual(experiment.manual_param, "manual_value")

        # Test core option
        experiment.boundaries = {"param": [0, 1]}
        self.assertEqual(experiment.boundaries, {"param": [0, 1]})

    def test_validation_errors_propagated(self):
        """Test that validation errors from descriptors are properly propagated."""

        class MockOptionsManager(OptionsManager):
            bounded_number = NumberOption("bounded_number", default=5, min_value=0, max_value=10)
            enum_choice = EnumOption("enum_choice", ["a", "b", "c"], default="a")

        class TestExperiment(OptionsAwareMixin):
            def __init__(self):
                self._internal_options = MockOptionsManager()

        experiment = TestExperiment()

        # Test validation error for number out of range
        with self.assertRaises(ValueError) as cm:
            experiment.bounded_number = 15
        self.assertIn("must be at most 10", str(cm.exception))

        # Test validation error for invalid enum value
        with self.assertRaises(ValueError) as cm:
            experiment.enum_choice = "invalid"
        self.assertIn("must be one of", str(cm.exception))

    @patch('builtins.print')
    def test_complex_class_hierarchy(self, mock_print):
        """Test OptionsAwareMixin in complex inheritance scenarios."""

        class BaseExperiment(OptionsAwareMixin):
            def __init__(self):
                self._manual_options = ["base_manual"]

        class SpecificExperiment(BaseExperiment):
            def __init__(self):
                super().__init__()
                self._manual_options.append("specific_manual")

        experiment = SpecificExperiment()

        # Test manual options from both base and derived classes
        experiment.base_manual = "base_value"
        experiment.specific_manual = "specific_value"

        self.assertEqual(experiment.base_manual, "base_value")
        self.assertEqual(experiment.specific_manual, "specific_value")

        # Test unknown option still triggers warning
        experiment.unknown_option = "unknown_value"
        mock_print.assert_called_once()

    def test_setattr_method_inheritance(self):
        """Test that __setattr__ method works correctly with inheritance."""

        class BaseClass:
            def __init__(self):
                self.base_attr = "base"

        class MixedClass(BaseClass, OptionsAwareMixin):
            def __init__(self):
                super().__init__()
                self._manual_options = ["manual_opt"]

        obj = MixedClass()

        # Should preserve base class functionality
        self.assertEqual(obj.base_attr, "base")

        # Should handle mixin functionality
        obj.manual_opt = "manual_value"
        self.assertEqual(obj.manual_opt, "manual_value")

        # Should handle core options
        obj.boundaries = {"test": [0, 1]}
        self.assertEqual(obj.boundaries, {"test": [0, 1]})


class TestOptionsAwareMixinEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for OptionsAwareMixin."""

    def test_empty_manual_options_list(self):
        """Test behavior with empty manual options list."""

        class TestClass(OptionsAwareMixin):
            def __init__(self):
                self._manual_options = []

        obj = TestClass()

        # Should work normally with empty list
        obj.some_attr = "value"
        self.assertEqual(obj.some_attr, "value")

    def test_none_manual_options(self):
        """Test behavior when _manual_options is None."""

        class TestClass(OptionsAwareMixin):
            def __init__(self):
                self._manual_options = None

        obj = TestClass()

        # Should handle None gracefully (hasattr check should prevent issues)
        obj.some_attr = "value"
        self.assertEqual(obj.some_attr, "value")

    def test_manual_options_not_list(self):
        """Test behavior when _manual_options is not a list."""

        class TestClass(OptionsAwareMixin):
            def __init__(self):
                self._manual_options = "not_a_list"

        obj = TestClass()

        # Should handle gracefully (string membership test should work)
        obj.some_attr = "value"
        self.assertEqual(obj.some_attr, "value")

    @patch('builtins.print')
    def test_class_name_in_warning(self, mock_print):
        """Test that class name appears correctly in warning messages."""

        class VerySpecificClassName(OptionsAwareMixin):
            pass

        obj = VerySpecificClassName()
        obj.unknown_attr = "value"

        # Check that the specific class name appears in the warning
        warning_call = mock_print.call_args[0][0]
        self.assertIn("VerySpecificClassName", warning_call)

    def test_setting_internal_options_attribute(self):
        """Test setting the _internal_options attribute itself."""

        class TestClass(OptionsAwareMixin):
            pass

        obj = TestClass()

        # Should be able to set _internal_options (private attribute)
        obj._internal_options = Mock()
        self.assertIsInstance(obj._internal_options, Mock)

    def test_setting_manual_options_attribute(self):
        """Test setting the _manual_options attribute itself."""

        class TestClass(OptionsAwareMixin):
            pass

        obj = TestClass()

        # Should be able to set _manual_options (private attribute)
        obj._manual_options = ["test"]
        self.assertEqual(obj._manual_options, ["test"])

    def test_core_options_modification(self):
        """Test that modifying core options list affects behavior."""

        # Save original core options
        original_core_options = OptionsAwareMixin._core_options.copy()

        try:
            # Add a new core option
            OptionsAwareMixin._core_options.append("test_core_option")

            class TestClass(OptionsAwareMixin):
                pass

            obj = TestClass()

            with patch('builtins.print') as mock_print:
                obj.test_core_option = "value"
                # Should not print warning since it's now a core option
                mock_print.assert_not_called()

        finally:
            # Restore original core options
            OptionsAwareMixin._core_options = original_core_options

    def test_internal_options_with_validation_side_effects(self):
        """Test behavior when internal options validation has side effects."""

        class SideEffectOptionsManager:
            def __init__(self):
                self.side_effect_called = False

            def get_option_names(self):
                return ["option_with_side_effect"]

            def __setattr__(self, name, value):
                if name == "option_with_side_effect":
                    self.side_effect_called = True
                super().__setattr__(name, value)

        class TestClass(OptionsAwareMixin):
            def __init__(self):
                self._internal_options = SideEffectOptionsManager()

        obj = TestClass()
        obj.option_with_side_effect = "test_value"

        # Should have triggered the side effect
        self.assertTrue(obj._internal_options.side_effect_called)
        self.assertEqual(obj.option_with_side_effect, "test_value")

    @patch('builtins.print')
    def test_unicode_attribute_names(self, mock_print):
        """Test behavior with unicode attribute names."""

        class TestClass(OptionsAwareMixin):
            pass

        obj = TestClass()

        # Test with unicode attribute name
        setattr(obj, "属性", "unicode_value")

        # Should handle unicode gracefully
        self.assertEqual(getattr(obj, "属性"), "unicode_value")

        # Should print warning with unicode name
        warning_call = mock_print.call_args[0][0]
        self.assertIn("属性", warning_call)

    def test_attribute_deletion(self):
        """Test that attribute deletion works normally."""

        class TestClass(OptionsAwareMixin):
            pass

        obj = TestClass()
        obj.test_attr = "value"

        # Should be able to delete attributes normally
        del obj.test_attr

        with self.assertRaises(AttributeError):
            _ = obj.test_attr


class TestOptionsAwareMixinWarningBehavior(unittest.TestCase):
    """Test warning behavior in detail."""

    @patch('builtins.print')
    def test_warning_message_format(self, mock_print):
        """Test the exact format of warning messages."""

        class TestExperiment(OptionsAwareMixin):
            pass

        obj = TestExperiment()
        obj.typo_option = "value"

        expected_warning = (
            "Warning: 'typo_option' is not in the list of accepted options for "
            "`TestExperiment` and will not affect simulation behavior"
        )
        mock_print.assert_called_once_with(expected_warning)

    @patch('builtins.print')
    def test_no_warning_for_known_patterns(self, mock_print):
        """Test that known attribute patterns don't trigger warnings."""

        class TestClass(OptionsAwareMixin):
            def __init__(self):
                self._manual_options = ["manual_opt"]

        obj = TestClass()

        # These should not trigger warnings
        obj._private_attr = "private"
        obj.boundaries = "core_option"
        obj.fixed_parameters = "core_option"
        obj.optim_list = "core_option"
        obj.manual_opt = "manual"

        mock_print.assert_not_called()

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_warning_output_destination(self, mock_stdout):
        """Test that warnings go to the correct output stream."""

        class TestClass(OptionsAwareMixin):
            pass

        obj = TestClass()
        obj.unknown_attr = "value"

        # The warning should appear in stdout
        output = mock_stdout.getvalue()
        self.assertIn("Warning:", output)
        self.assertIn("unknown_attr", output)


if __name__ == '__main__':
    unittest.main()