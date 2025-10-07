import unittest
from unittest.mock import Mock, patch
from Surface_confined_inference._core._Options._OptionsMeta import OptionsMeta, OptionsManager
from Surface_confined_inference._core._Options._OptionsDescriptor import (
    OptionDescriptor, StringOption, NumberOption, BoolOption, EnumOption
)


class TestOptionsMeta(unittest.TestCase):
    """Test the OptionsMeta metaclass."""

    def test_metaclass_creates_options_set(self):
        """Test that metaclass creates _options set for new classes."""
        class TestOptions(OptionsManager):
            test_option = StringOption("test_option", default="test")

        self.assertIsInstance(TestOptions._options, set)
        self.assertIn("test_option", TestOptions._options)

    def test_metaclass_inherits_options_from_base(self):
        """Test that metaclass inherits options from base classes."""
        class BaseOptions(OptionsManager):
            base_option = StringOption("base_option", default="base")

        class DerivedOptions(BaseOptions):
            derived_option = NumberOption("derived_option", default=42)

        # Derived class should have both base and derived options
        self.assertIn("base_option", DerivedOptions._options)
        self.assertIn("derived_option", DerivedOptions._options)
        self.assertEqual(len(DerivedOptions._options), 2)

    def test_metaclass_multiple_inheritance(self):
        """Test metaclass with multiple inheritance."""
        class OptionsA(OptionsManager):
            option_a = StringOption("option_a", default="a")

        class OptionsB(OptionsManager):
            option_b = NumberOption("option_b", default=1)

        class CombinedOptions(OptionsA, OptionsB):
            option_c = BoolOption("option_c", default=True)

        # Should have options from both parents plus its own
        self.assertIn("option_a", CombinedOptions._options)
        self.assertIn("option_b", CombinedOptions._options)
        self.assertIn("option_c", CombinedOptions._options)
        self.assertEqual(len(CombinedOptions._options), 3)

    def test_metaclass_deep_inheritance(self):
        """Test metaclass with deep inheritance hierarchies."""
        class Level1Options(OptionsManager):
            level1_option = StringOption("level1_option", default="level1")

        class Level2Options(Level1Options):
            level2_option = NumberOption("level2_option", default=2)

        class Level3Options(Level2Options):
            level3_option = BoolOption("level3_option", default=True)

        # Level3 should have all options from the hierarchy
        self.assertIn("level1_option", Level3Options._options)
        self.assertIn("level2_option", Level3Options._options)
        self.assertIn("level3_option", Level3Options._options)
        self.assertEqual(len(Level3Options._options), 3)

    def test_metaclass_ignores_non_descriptor_attributes(self):
        """Test that metaclass only registers OptionDescriptor instances."""
        class TestOptions(OptionsManager):
            option = StringOption("option", default="test")
            regular_attr = "not an option"
            _private_attr = "private"

        # Only the descriptor should be in _options
        self.assertIn("option", TestOptions._options)
        self.assertNotIn("regular_attr", TestOptions._options)
        self.assertNotIn("_private_attr", TestOptions._options)
        self.assertEqual(len(TestOptions._options), 1)

    def test_metaclass_empty_class(self):
        """Test metaclass with class that has no options."""
        class EmptyOptions(OptionsManager):
            pass

        self.assertIsInstance(EmptyOptions._options, set)
        self.assertEqual(len(EmptyOptions._options), 0)

    def test_metaclass_preserves_existing_attributes(self):
        """Test that metaclass preserves other class attributes."""
        class TestOptions(OptionsManager):
            option = StringOption("option", default="test")

            def custom_method(self):
                return "custom"

            class_var = "class_variable"

        # Check that non-option attributes are preserved
        self.assertTrue(hasattr(TestOptions, "custom_method"))
        self.assertTrue(hasattr(TestOptions, "class_var"))
        self.assertEqual(TestOptions.class_var, "class_variable")

        # Check that option is still registered
        self.assertIn("option", TestOptions._options)


class TestOptionsManager(unittest.TestCase):
    """Test the OptionsManager base class."""

    def setUp(self):
        """Set up test fixtures."""
        class TestOptions(OptionsManager):
            string_opt = StringOption("string_opt", default="default_string")
            number_opt = NumberOption("number_opt", default=42)
            bool_opt = BoolOption("bool_opt", default=True)
            enum_opt = EnumOption("enum_opt", ["option1", "option2"], default="option1")

        self.TestOptions = TestOptions

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        options = self.TestOptions()

        self.assertEqual(options.string_opt, "default_string")
        self.assertEqual(options.number_opt, 42)
        self.assertEqual(options.bool_opt, True)
        self.assertEqual(options.enum_opt, "option1")

    def test_init_with_provided_values(self):
        """Test initialization with provided values."""
        options = self.TestOptions(
            string_opt="custom_string",
            number_opt=100,
            bool_opt=False,
            enum_opt="option2"
        )

        self.assertEqual(options.string_opt, "custom_string")
        self.assertEqual(options.number_opt, 100)
        self.assertEqual(options.bool_opt, False)
        self.assertEqual(options.enum_opt, "option2")

    def test_init_partial_values(self):
        """Test initialization with some values provided, others default."""
        options = self.TestOptions(string_opt="custom", bool_opt=False)

        self.assertEqual(options.string_opt, "custom")
        self.assertEqual(options.number_opt, 42)  # default
        self.assertEqual(options.bool_opt, False)
        self.assertEqual(options.enum_opt, "option1")  # default

    def test_init_unknown_option_raises_error(self):
        """Test that unknown options raise AttributeError during initialization."""
        with self.assertRaises(AttributeError) as cm:
            self.TestOptions(unknown_option="value")
        self.assertIn("Unknown option: unknown_option", str(cm.exception))

    def test_setattr_known_option(self):
        """Test setting known options after initialization."""
        options = self.TestOptions()

        options.string_opt = "new_string"
        options.number_opt = 200

        self.assertEqual(options.string_opt, "new_string")
        self.assertEqual(options.number_opt, 200)

    def test_setattr_unknown_option_calls_handler(self):
        """Test that setting unknown options calls _handle_unknown_attribute."""
        options = self.TestOptions()

        # Should call _handle_unknown_attribute but not raise (default behavior)
        options.unknown_attr = "value"
        self.assertEqual(options.unknown_attr, "value")

    def test_setattr_private_attribute(self):
        """Test setting private attributes (starting with _)."""
        options = self.TestOptions()

        # Private attributes should be set normally
        options._private = "private_value"
        self.assertEqual(options._private, "private_value")

    def test_as_dict(self):
        """Test as_dict method returns all options."""
        options = self.TestOptions(string_opt="test", number_opt=99)

        result = options.as_dict()

        expected = {
            "string_opt": "test",
            "number_opt": 99,
            "bool_opt": True,  # default
            "enum_opt": "option1"  # default
        }
        self.assertEqual(result, expected)

    def test_update_known_options(self):
        """Test update method with known options."""
        options = self.TestOptions()

        options.update(string_opt="updated", number_opt=999, bool_opt=False)

        self.assertEqual(options.string_opt, "updated")
        self.assertEqual(options.number_opt, 999)
        self.assertEqual(options.bool_opt, False)

    def test_update_unknown_option_raises_error(self):
        """Test that update with unknown option raises AttributeError."""
        options = self.TestOptions()

        with self.assertRaises(AttributeError) as cm:
            options.update(unknown_option="value")
        self.assertIn("Unknown option: unknown_option", str(cm.exception))

    def test_get_option_names(self):
        """Test get_option_names class method."""
        option_names = self.TestOptions.get_option_names()

        expected_names = {"string_opt", "number_opt", "bool_opt", "enum_opt"}
        self.assertEqual(set(option_names), expected_names)

    def test_get_option_defaults(self):
        """Test get_option_defaults class method."""
        defaults = self.TestOptions.get_option_defaults()

        expected_defaults = {
            "string_opt": "default_string",
            "number_opt": 42,
            "bool_opt": True,
            "enum_opt": "option1"
        }
        self.assertEqual(defaults, expected_defaults)

    def test_handle_unknown_option_can_be_overridden(self):
        """Test that _handle_unknown_option can be overridden in subclasses."""
        class CustomOptionsManager(OptionsManager):
            option = StringOption("option", default="test")

            def _handle_unknown_option(self, key, value):
                # Custom handler that allows unknown options
                setattr(self, key, value)

        options = CustomOptionsManager(unknown_option="allowed")
        self.assertEqual(options.unknown_option, "allowed")

    def test_handle_unknown_attribute_can_be_overridden(self):
        """Test that _handle_unknown_attribute can be overridden in subclasses."""
        class StrictOptionsManager(OptionsManager):
            option = StringOption("option", default="test")

            def _handle_unknown_attribute(self, name, value):
                raise ValueError(f"Setting unknown attribute '{name}' is not allowed")

        options = StrictOptionsManager()

        with self.assertRaises(ValueError) as cm:
            options.unknown_attr = "value"
        self.assertIn("Setting unknown attribute 'unknown_attr' is not allowed", str(cm.exception))


class TestOptionsInheritance(unittest.TestCase):
    """Test option inheritance behavior."""

    def test_single_inheritance(self):
        """Test options inheritance with single inheritance."""
        class BaseOptions(OptionsManager):
            base_option = StringOption("base_option", default="base")

        class DerivedOptions(BaseOptions):
            derived_option = NumberOption("derived_option", default=42)

        # Test that derived class has both options
        options = DerivedOptions()
        self.assertEqual(options.base_option, "base")
        self.assertEqual(options.derived_option, 42)

        # Test that options can be set
        options.base_option = "modified_base"
        options.derived_option = 100
        self.assertEqual(options.base_option, "modified_base")
        self.assertEqual(options.derived_option, 100)

    def test_deep_inheritance_chain(self):
        """Test options inheritance through multiple levels."""
        class Level1(OptionsManager):
            level1_opt = StringOption("level1_opt", default="level1")

        class Level2(Level1):
            level2_opt = NumberOption("level2_opt", default=2)

        class Level3(Level2):
            level3_opt = BoolOption("level3_opt", default=True)

        options = Level3()
        self.assertEqual(options.level1_opt, "level1")
        self.assertEqual(options.level2_opt, 2)
        self.assertEqual(options.level3_opt, True)

        # Test as_dict includes all options
        result = options.as_dict()
        self.assertEqual(len(result), 3)
        self.assertIn("level1_opt", result)
        self.assertIn("level2_opt", result)
        self.assertIn("level3_opt", result)

    def test_override_parent_option(self):
        """Test that child classes can override parent options."""
        class BaseOptions(OptionsManager):
            option = StringOption("option", default="base_default")

        class DerivedOptions(BaseOptions):
            option = StringOption("option", default="derived_default")

        base_options = BaseOptions()
        derived_options = DerivedOptions()

        self.assertEqual(base_options.option, "base_default")
        self.assertEqual(derived_options.option, "derived_default")

    def test_multiple_inheritance_no_conflicts(self):
        """Test multiple inheritance with no option name conflicts."""
        class OptionsA(OptionsManager):
            option_a = StringOption("option_a", default="a")

        class OptionsB(OptionsManager):
            option_b = NumberOption("option_b", default=1)

        class CombinedOptions(OptionsA, OptionsB):
            option_c = BoolOption("option_c", default=True)

        options = CombinedOptions()
        self.assertEqual(options.option_a, "a")
        self.assertEqual(options.option_b, 1)
        self.assertEqual(options.option_c, True)

        # Test that all options are in the registry
        option_names = CombinedOptions.get_option_names()
        self.assertEqual(set(option_names), {"option_a", "option_b", "option_c"})


class TestOptionsManagerIntegration(unittest.TestCase):
    """Test integration scenarios with OptionsManager."""

    def test_validation_integration(self):
        """Test that descriptor validation works through OptionsManager."""
        class ValidatedOptions(OptionsManager):
            bounded_number = NumberOption("bounded_number", default=5, min_value=0, max_value=10)
            enum_choice = EnumOption("enum_choice", ["a", "b", "c"], default="a")

        options = ValidatedOptions()

        # Test valid values
        options.bounded_number = 7
        options.enum_choice = "b"
        self.assertEqual(options.bounded_number, 7)
        self.assertEqual(options.enum_choice, "b")

        # Test validation errors
        with self.assertRaises(ValueError):
            options.bounded_number = 15  # exceeds max

        with self.assertRaises(ValueError):
            options.enum_choice = "invalid"  # not in enum

    def test_initialization_with_validation_errors(self):
        """Test that validation errors are caught during initialization."""
        class ValidatedOptions(OptionsManager):
            bounded_number = NumberOption("bounded_number", default=5, min_value=0, max_value=10)

        # Should raise validation error during __init__
        with self.assertRaises(ValueError):
            ValidatedOptions(bounded_number=20)

    def test_complex_inheritance_with_validation(self):
        """Test complex inheritance scenarios with validation."""
        class BaseValidated(OptionsManager):
            base_str = StringOption("base_str", default="base", min_length=3)

        class MiddleValidated(BaseValidated):
            middle_num = NumberOption("middle_num", default=10, min_value=5)

        class FinalValidated(MiddleValidated):
            final_enum = EnumOption("final_enum", ["x", "y", "z"], default="x")

        # Test valid initialization
        options = FinalValidated(base_str="valid", middle_num=8, final_enum="y")
        self.assertEqual(options.base_str, "valid")
        self.assertEqual(options.middle_num, 8)
        self.assertEqual(options.final_enum, "y")

        # Test validation error from base class
        with self.assertRaises(ValueError):
            FinalValidated(base_str="xx")  # too short

        # Test validation error from middle class
        with self.assertRaises(ValueError):
            FinalValidated(middle_num=2)  # below minimum

    def test_as_dict_with_inheritance(self):
        """Test as_dict method with inherited options."""
        class Parent(OptionsManager):
            parent_opt = StringOption("parent_opt", default="parent")

        class Child(Parent):
            child_opt = NumberOption("child_opt", default=42)

        options = Child(parent_opt="modified")
        result = options.as_dict()

        expected = {"parent_opt": "modified", "child_opt": 42}
        self.assertEqual(result, expected)

    def test_update_with_inheritance(self):
        """Test update method with inherited options."""
        class Parent(OptionsManager):
            parent_opt = StringOption("parent_opt", default="parent")

        class Child(Parent):
            child_opt = NumberOption("child_opt", default=42)

        options = Child()
        options.update(parent_opt="updated_parent", child_opt=100)

        self.assertEqual(options.parent_opt, "updated_parent")
        self.assertEqual(options.child_opt, 100)

    def test_get_option_defaults_with_inheritance(self):
        """Test get_option_defaults with inherited options."""
        class Parent(OptionsManager):
            parent_opt = StringOption("parent_opt", default="parent_default")

        class Child(Parent):
            child_opt = NumberOption("child_opt", default=42)

        defaults = Child.get_option_defaults()
        expected = {"parent_opt": "parent_default", "child_opt": 42}
        self.assertEqual(defaults, expected)


class TestOptionsManagerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_empty_options_class(self):
        """Test OptionsManager subclass with no options."""
        class EmptyOptions(OptionsManager):
            pass

        options = EmptyOptions()
        self.assertEqual(options.as_dict(), {})
        self.assertEqual(EmptyOptions.get_option_names(), [])
        self.assertEqual(EmptyOptions.get_option_defaults(), {})

    def test_option_descriptor_without_default(self):
        """Test option descriptor without explicit default value."""
        class NoDefaultOptions(OptionsManager):
            no_default = OptionDescriptor("no_default")  # default=None

        options = NoDefaultOptions()
        self.assertIsNone(options.no_default)

    def test_dynamic_option_addition(self):
        """Test that dynamically added options are not registered."""
        class DynamicOptions(OptionsManager):
            static_opt = StringOption("static_opt", default="static")

        # Dynamically add option (should not be in _options)
        DynamicOptions.dynamic_opt = StringOption("dynamic_opt", default="dynamic")

        # The dynamic option won't be in _options since metaclass already ran
        self.assertIn("static_opt", DynamicOptions._options)
        self.assertNotIn("dynamic_opt", DynamicOptions._options)

    def test_class_without_metaclass(self):
        """Test that regular classes don't interfere with option inheritance."""
        class RegularClass:
            regular_attr = "not an option"

        class OptionsWithRegularBase(RegularClass, OptionsManager):
            option = StringOption("option", default="test")

        options = OptionsWithRegularBase()
        self.assertEqual(options.option, "test")
        self.assertTrue(hasattr(options, "regular_attr"))


if __name__ == '__main__':
    unittest.main()