import unittest
import tempfile
import os
import numbers
from pathlib import Path
from unittest.mock import patch, mock_open
import collections.abc

from Surface_confined_inference._core._Options._OptionsDescriptor import (
    OptionDescriptor,
    TypedOption,
    EnumOption,
    BoolOption,
    NumberOption,
    SequenceOption,
    ExclusiveSequenceOption,
    StringOption,
    DictOption,
    ExclusiveDictOption,
    ComposedOption,
    FileOption,
    DirectoryOption
)


class TestOptionDescriptor(unittest.TestCase):
    """Test the base OptionDescriptor class."""

    def setUp(self):
        self.descriptor = OptionDescriptor("test_option", default="default_value", doc="Test option")

    def test_init(self):
        """Test descriptor initialization."""
        self.assertEqual(self.descriptor.name, "test_option")
        self.assertEqual(self.descriptor.private_name, "_test_option")
        self.assertEqual(self.descriptor.default, "default_value")
        self.assertEqual(self.descriptor.__doc__, "Test option")

    def test_set_name(self):
        """Test __set_name__ method."""
        self.descriptor.__set_name__(None, "new_name")
        self.assertEqual(self.descriptor.name, "new_name")
        self.assertEqual(self.descriptor.private_name, "_new_name")

    def test_get_descriptor_from_class(self):
        """Test getting descriptor from class returns the descriptor itself."""
        class TestClass:
            option = OptionDescriptor("option", default="test")

        result = TestClass.option
        self.assertIsInstance(result, OptionDescriptor)

    def test_get_from_instance_with_value(self):
        """Test getting value from instance."""
        class TestClass:
            option = OptionDescriptor("option", default="default")

        obj = TestClass()
        obj._option = "set_value"
        result = TestClass.option.__get__(obj, TestClass)
        self.assertEqual(result, "set_value")

    def test_get_from_instance_with_default(self):
        """Test getting default value from instance."""
        class TestClass:
            option = OptionDescriptor("option", default="default")

        obj = TestClass()
        result = TestClass.option.__get__(obj, TestClass)
        self.assertEqual(result, "default")

    def test_set_value(self):
        """Test setting value on instance."""
        class TestClass:
            option = OptionDescriptor("option", default="default")

        obj = TestClass()
        TestClass.option.__set__(obj, "new_value")
        self.assertEqual(obj._option, "new_value")

    def test_validate_base_implementation(self):
        """Test that base validate method does nothing."""
        # Should not raise any exception
        self.descriptor.validate("any_value")


class TestTypedOption(unittest.TestCase):
    """Test the TypedOption descriptor."""

    def test_init_single_type(self):
        """Test initialization with single type."""
        descriptor = TypedOption("test", str, default="test")
        self.assertEqual(descriptor.allowed_types, [str])

    def test_init_multiple_types(self):
        """Test initialization with multiple types."""
        descriptor = TypedOption("test", [str, int], default="test")
        self.assertEqual(descriptor.allowed_types, [str, int])

    def test_validate_correct_type(self):
        """Test validation with correct type."""
        descriptor = TypedOption("test", str)
        descriptor.validate("valid_string")  # Should not raise

    def test_validate_multiple_correct_types(self):
        """Test validation with multiple allowed types."""
        descriptor = TypedOption("test", [str, int])
        descriptor.validate("valid_string")  # Should not raise
        descriptor.validate(42)  # Should not raise

    def test_validate_incorrect_type(self):
        """Test validation with incorrect type."""
        descriptor = TypedOption("test", str)
        with self.assertRaises(TypeError) as cm:
            descriptor.validate(42)
        self.assertIn("test must be of type str", str(cm.exception))

    def test_validate_incorrect_type_multiple_allowed(self):
        """Test validation error message with multiple allowed types."""
        descriptor = TypedOption("test", [str, int])
        with self.assertRaises(TypeError) as cm:
            descriptor.validate([])
        self.assertIn("test must be of type str or int", str(cm.exception))


class TestEnumOption(unittest.TestCase):
    """Test the EnumOption descriptor."""

    def setUp(self):
        self.descriptor = EnumOption("test", ["option1", "option2", "option3"])

    def test_validate_valid_value(self):
        """Test validation with valid enum value."""
        self.descriptor.validate("option1")  # Should not raise

    def test_validate_invalid_value(self):
        """Test validation with invalid enum value."""
        with self.assertRaises(ValueError) as cm:
            self.descriptor.validate("invalid_option")
        self.assertIn("test must be one of", str(cm.exception))
        self.assertIn("'option1', 'option2', 'option3'", str(cm.exception))

    def test_validate_with_mixed_types(self):
        """Test enum with mixed value types."""
        descriptor = EnumOption("test", ["string", 42, True])
        descriptor.validate("string")  # Should not raise
        descriptor.validate(42)  # Should not raise
        descriptor.validate(True)  # Should not raise

        with self.assertRaises(ValueError):
            descriptor.validate("other")


class TestBoolOption(unittest.TestCase):
    """Test the BoolOption descriptor."""

    def setUp(self):
        self.descriptor = BoolOption("test", default=False)

    def test_init(self):
        """Test initialization sets correct type."""
        self.assertEqual(self.descriptor.allowed_types, [bool])
        self.assertEqual(self.descriptor.default, False)

    def test_validate_bool_values(self):
        """Test validation with boolean values."""
        self.descriptor.validate(True)  # Should not raise
        self.descriptor.validate(False)  # Should not raise

    def test_validate_non_bool(self):
        """Test validation with non-boolean values."""
        with self.assertRaises(TypeError):
            self.descriptor.validate("true")
        with self.assertRaises(TypeError):
            self.descriptor.validate(1)
        with self.assertRaises(TypeError):
            self.descriptor.validate(0)


class TestNumberOption(unittest.TestCase):
    """Test the NumberOption descriptor."""

    def test_init_no_bounds(self):
        """Test initialization without bounds."""
        descriptor = NumberOption("test", default=0)
        self.assertEqual(descriptor.allowed_types, [numbers.Number])
        self.assertIsNone(descriptor.min_value)
        self.assertIsNone(descriptor.max_value)

    def test_init_with_bounds(self):
        """Test initialization with bounds."""
        descriptor = NumberOption("test", default=5, min_value=0, max_value=10)
        self.assertEqual(descriptor.min_value, 0)
        self.assertEqual(descriptor.max_value, 10)

    def test_validate_valid_numbers(self):
        """Test validation with valid numbers."""
        descriptor = NumberOption("test")
        descriptor.validate(42)  # int
        descriptor.validate(3.14)  # float
        descriptor.validate(1+2j)  # complex

    def test_validate_non_number(self):
        """Test validation with non-number values."""
        descriptor = NumberOption("test")
        with self.assertRaises(TypeError):
            descriptor.validate("42")

    def test_validate_within_bounds(self):
        """Test validation within specified bounds."""
        descriptor = NumberOption("test", min_value=0, max_value=10)
        descriptor.validate(5)  # Should not raise
        descriptor.validate(0)  # Should not raise (boundary)
        descriptor.validate(10)  # Should not raise (boundary)

    def test_validate_below_minimum(self):
        """Test validation below minimum value."""
        descriptor = NumberOption("test", min_value=0)
        with self.assertRaises(ValueError) as cm:
            descriptor.validate(-1)
        self.assertIn("test must be at least 0", str(cm.exception))

    def test_validate_above_maximum(self):
        """Test validation above maximum value."""
        descriptor = NumberOption("test", max_value=10)
        with self.assertRaises(ValueError) as cm:
            descriptor.validate(11)
        self.assertIn("test must be at most 10", str(cm.exception))


class TestSequenceOption(unittest.TestCase):
    """Test the SequenceOption descriptor."""

    def test_init_default_empty_list(self):
        """Test initialization creates empty list default."""
        descriptor = SequenceOption("test")
        self.assertEqual(descriptor.default, [])

    def test_init_with_constraints(self):
        """Test initialization with length and type constraints."""
        descriptor = SequenceOption("test", item_type=int, min_length=1, max_length=5)
        self.assertEqual(descriptor.item_type, int)
        self.assertEqual(descriptor.min_length, 1)
        self.assertEqual(descriptor.max_length, 5)

    def test_validate_valid_sequences(self):
        """Test validation with valid sequences."""
        descriptor = SequenceOption("test")
        descriptor.validate([1, 2, 3])  # list
        descriptor.validate((1, 2, 3))  # tuple
        descriptor.validate("abc")  # string is sequence

    def test_validate_non_sequence(self):
        """Test validation with non-sequence values."""
        descriptor = SequenceOption("test")
        with self.assertRaises(TypeError):
            descriptor.validate(42)

    def test_validate_item_types(self):
        """Test validation of item types within sequence."""
        descriptor = SequenceOption("test", item_type=int)
        descriptor.validate([1, 2, 3])  # Should not raise

        with self.assertRaises(TypeError) as cm:
            descriptor.validate([1, "2", 3])
        self.assertIn("Item 1 in test must be of type int", str(cm.exception))

    def test_validate_length_constraints(self):
        """Test validation of sequence length constraints."""
        descriptor = SequenceOption("test", min_length=2, max_length=4)

        descriptor.validate([1, 2])  # min length - Should not raise
        descriptor.validate([1, 2, 3, 4])  # max length - Should not raise

        with self.assertRaises(ValueError) as cm:
            descriptor.validate([1])  # too short
        self.assertIn("test must have at least 2 items", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            descriptor.validate([1, 2, 3, 4, 5])  # too long
        self.assertIn("test must have at most 4 items", str(cm.exception))


class TestExclusiveSequenceOption(unittest.TestCase):
    """Test the ExclusiveSequenceOption descriptor."""

    def setUp(self):
        self.descriptor = ExclusiveSequenceOption("test", target=["a", "b", "c"])

    def test_validate_exact_match(self):
        """Test validation when sequence exactly matches target."""
        self.descriptor.validate(["a", "b", "c"])  # Should not raise
        self.descriptor.validate(["c", "a", "b"])  # Order doesn't matter

    def test_validate_missing_values(self):
        """Test validation when values are missing."""
        with self.assertRaises(ValueError) as cm:
            self.descriptor.validate(["a", "b"])  # missing "c"
        self.assertIn("the following values are missing", str(cm.exception))
        self.assertIn("c", str(cm.exception))

    def test_validate_extra_values(self):
        """Test validation when extra values are present."""
        with self.assertRaises(ValueError) as cm:
            self.descriptor.validate(["a", "b", "c", "d"])  # extra "d"
        self.assertIn("the following values should not be present", str(cm.exception))
        self.assertIn("d", str(cm.exception))

    def test_validate_none_value(self):
        """Test validation with None value."""
        # Should handle None gracefully based on the implementation
        descriptor = ExclusiveSequenceOption("test", target=["a", "b"])
        # The validate method checks if value is not None
        descriptor.validate(None)  # Should not raise based on current implementation


class TestStringOption(unittest.TestCase):
    """Test the StringOption descriptor."""

    def test_init_default_empty_string(self):
        """Test initialization with default empty string."""
        descriptor = StringOption("test")
        self.assertEqual(descriptor.default, "")

    def test_init_with_length_constraints(self):
        """Test initialization with length constraints."""
        descriptor = StringOption("test", min_length=1, max_length=10)
        self.assertEqual(descriptor.min_length, 1)
        self.assertEqual(descriptor.max_length, 10)

    def test_validate_valid_strings(self):
        """Test validation with valid strings."""
        descriptor = StringOption("test")
        descriptor.validate("hello")  # Should not raise
        descriptor.validate("")  # Should not raise

    def test_validate_non_string(self):
        """Test validation with non-string values."""
        descriptor = StringOption("test")
        with self.assertRaises(TypeError):
            descriptor.validate(42)

    def test_validate_length_constraints(self):
        """Test validation of string length constraints."""
        descriptor = StringOption("test", min_length=3, max_length=6)

        descriptor.validate("abc")  # min length - Should not raise
        descriptor.validate("abcdef")  # max length - Should not raise

        with self.assertRaises(ValueError) as cm:
            descriptor.validate("ab")  # too short
        self.assertIn("test must have at least 3 characters", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            descriptor.validate("abcdefg")  # too long
        self.assertIn("test must have at most 6 characters", str(cm.exception))


class TestDictOption(unittest.TestCase):
    """Test the DictOption descriptor."""

    def test_init_default_empty_dict(self):
        """Test initialization with default empty dictionary."""
        descriptor = DictOption("test")
        self.assertEqual(descriptor.default, {})

    def test_init_with_constraints(self):
        """Test initialization with type and key constraints."""
        descriptor = DictOption("test", key_type=str, value_type=int, required_keys=["key1"])
        self.assertEqual(descriptor.key_type, str)
        self.assertEqual(descriptor.value_type, int)
        self.assertEqual(descriptor.required_keys, ["key1"])

    def test_validate_valid_dict(self):
        """Test validation with valid dictionary."""
        descriptor = DictOption("test")
        descriptor.validate({"key": "value"})  # Should not raise

    def test_validate_non_dict(self):
        """Test validation with non-dictionary values."""
        descriptor = DictOption("test")
        with self.assertRaises(TypeError):
            descriptor.validate("not a dict")

    def test_validate_required_keys(self):
        """Test validation of required keys."""
        descriptor = DictOption("test", required_keys=["key1", "key2"])

        descriptor.validate({"key1": "val1", "key2": "val2"})  # Should not raise
        descriptor.validate({"key1": "val1", "key2": "val2", "key3": "val3"})  # Should not raise

        with self.assertRaises(ValueError) as cm:
            descriptor.validate({"key1": "val1"})  # missing key2
        self.assertIn("test is missing required keys", str(cm.exception))
        self.assertIn("'key2'", str(cm.exception))

    def test_validate_key_types(self):
        """Test validation of key types."""
        descriptor = DictOption("test", key_type=str)

        descriptor.validate({"string_key": "value"})  # Should not raise

        with self.assertRaises(TypeError) as cm:
            descriptor.validate({42: "value"})  # invalid key type
        self.assertIn("Key 42 in test must be of type str", str(cm.exception))

    def test_validate_value_types(self):
        """Test validation of value types."""
        descriptor = DictOption("test", value_type=int)

        descriptor.validate({"key": 42})  # Should not raise

        with self.assertRaises(TypeError) as cm:
            descriptor.validate({"key": "not_int"})  # invalid value type
        self.assertIn("Value for key 'key' in test must be of type int", str(cm.exception))


class TestExclusiveDictOption(unittest.TestCase):
    """Test the ExclusiveDictOption descriptor."""

    def setUp(self):
        self.descriptor = ExclusiveDictOption("test", target=["key1", "key2", "key3"])

    def test_validate_exact_keys(self):
        """Test validation when dictionary has exact target keys."""
        self.descriptor.validate({"key1": "val1", "key2": "val2", "key3": "val3"})  # Should not raise

    def test_validate_missing_keys(self):
        """Test validation when keys are missing."""
        with self.assertRaises(ValueError) as cm:
            self.descriptor.validate({"key1": "val1", "key2": "val2"})  # missing key3
        self.assertIn("the following values are missing", str(cm.exception))
        self.assertIn("key3", str(cm.exception))

    def test_validate_extra_keys(self):
        """Test validation when extra keys are present."""
        with self.assertRaises(ValueError) as cm:
            self.descriptor.validate({"key1": "val1", "key2": "val2", "key3": "val3", "key4": "val4"})
        self.assertIn("the following values should not be present", str(cm.exception))
        self.assertIn("key4", str(cm.exception))

    def test_validate_value_types(self):
        """Test validation of value types in exclusive dict."""
        descriptor = ExclusiveDictOption("test", target=["key1"], value_type=int)

        descriptor.validate({"key1": 42})  # Should not raise

        with self.assertRaises(TypeError) as cm:
            descriptor.validate({"key1": "not_int"})
        self.assertIn("Value for key 'key1' in test must be of type int", str(cm.exception))


class TestComposedOption(unittest.TestCase):
    """Test the ComposedOption descriptor."""

    def setUp(self):
        self.str_validator = StringOption
        self.int_validator = NumberOption
        self.descriptor = ComposedOption("test", [self.str_validator, self.int_validator])

    def test_init(self):
        """Test initialization."""
        self.assertEqual(len(self.descriptor.validators), 2)
        self.assertEqual(self.descriptor.default, [])

    def test_validate_non_sequence(self):
        """Test validation with non-sequence value."""
        with self.assertRaises(TypeError) as cm:
            self.descriptor.validate("not_a_sequence")
        self.assertIn("test must be a sequence", str(cm.exception))

    def test_validate_valid_items(self):
        """Test validation when all items pass at least one validator."""
        # Strings should pass StringOption, numbers should pass NumberOption
        self.descriptor.validate(["hello", 42, "world", 3.14])  # Should not raise

    def test_validate_invalid_item(self):
        """Test validation when an item fails all validators."""
        with self.assertRaises(ValueError) as cm:
            self.descriptor.validate(["hello", []])  # list fails both string and number validation
        self.assertIn("Item 1 in test failed validation with all validators", str(cm.exception))

    def test_validate_empty_sequence(self):
        """Test validation with empty sequence."""
        self.descriptor.validate([])  # Should not raise


class TestFileOption(unittest.TestCase):
    """Test the FileOption descriptor."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
        self.existing_file = self.temp_file.name

    def tearDown(self):
        try:
            os.unlink(self.existing_file)
        except FileNotFoundError:
            pass

    def test_init(self):
        """Test initialization."""
        descriptor = FileOption("test", must_exist=True)
        self.assertTrue(descriptor.must_exist)

    def test_validate_existing_file(self):
        """Test validation with existing file."""
        descriptor = FileOption("test", must_exist=True)
        descriptor.validate(self.existing_file)  # Should not raise

    def test_validate_non_existing_file_required(self):
        """Test validation with non-existing file when existence is required."""
        descriptor = FileOption("test", must_exist=True)
        with self.assertRaises(ValueError) as cm:
            descriptor.validate("/non/existent/file.txt")
        self.assertIn("test must be a path to an existing file", str(cm.exception))

    def test_validate_non_existing_file_not_required(self):
        """Test validation with non-existing file when existence is not required."""
        descriptor = FileOption("test", must_exist=False)
        descriptor.validate("/non/existent/file.txt")  # Should not raise

    def test_validate_empty_string_required(self):
        """Test validation with empty string when existence is required."""
        descriptor = FileOption("test", must_exist=True)
        with self.assertRaises(ValueError) as cm:
            descriptor.validate("")
        self.assertIn("test cannot be empty when must_exist=True", str(cm.exception))

    def test_validate_empty_string_not_required(self):
        """Test validation with empty string when existence is not required."""
        descriptor = FileOption("test", must_exist=False)
        descriptor.validate("")  # Should not raise

    def test_validate_non_string(self):
        """Test validation with non-string value."""
        descriptor = FileOption("test")
        with self.assertRaises(TypeError):
            descriptor.validate(42)


class TestDirectoryOption(unittest.TestCase):
    """Test the DirectoryOption descriptor."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.existing_dir = self.temp_dir

        # Create a non-empty directory
        self.non_empty_dir = tempfile.mkdtemp()
        with open(os.path.join(self.non_empty_dir, "test_file.txt"), "w") as f:
            f.write("test")

    def tearDown(self):
        import shutil
        try:
            shutil.rmtree(self.existing_dir)
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree(self.non_empty_dir)
        except FileNotFoundError:
            pass

    def test_init(self):
        """Test initialization."""
        descriptor = DirectoryOption("test", must_exist=True, must_be_empty=False)
        self.assertTrue(descriptor.must_exist)
        self.assertFalse(descriptor.must_be_empty)

    def test_validate_existing_directory(self):
        """Test validation with existing directory."""
        descriptor = DirectoryOption("test", must_exist=True)
        descriptor.validate(self.existing_dir)  # Should not raise

    def test_validate_non_existing_directory_required(self):
        """Test validation with non-existing directory when existence is required."""
        descriptor = DirectoryOption("test", must_exist=True, can_create=False)
        with self.assertRaises(ValueError) as cm:
            descriptor.validate("/non/existent/directory")
        self.assertIn("test must be a path to an existing directory", str(cm.exception))

    def test_validate_non_existing_directory_not_required(self):
        """Test validation with non-existing directory when existence is not required."""
        descriptor = DirectoryOption("test", must_exist=False)
        descriptor.validate("/non/existent/directory")  # Should not raise

    def test_validate_empty_string_required(self):
        """Test validation with empty string when existence is required."""
        descriptor = DirectoryOption("test", must_exist=True)
        with self.assertRaises(ValueError) as cm:
            descriptor.validate("")
        self.assertIn("test cannot be empty when must_exist=True", str(cm.exception))

    def test_validate_empty_string_not_required(self):
        """Test validation with empty string when existence is not required."""
        descriptor = DirectoryOption("test", must_exist=False)
        descriptor.validate("")  # Should not raise

    def test_validate_directory_must_be_empty(self):
        """Test validation when directory must be empty."""
        descriptor = DirectoryOption("test", must_exist=True, must_be_empty=True)

        descriptor.validate(self.existing_dir)  # Should not raise (empty)

        with self.assertRaises(ValueError) as cm:
            descriptor.validate(self.non_empty_dir)  # has file
        self.assertIn("directory", str(cm.exception))
        self.assertIn("must be empty", str(cm.exception))

    def test_validate_can_create_directory(self):
        """Test validation with can_create=True for non-existing directory."""
        descriptor = DirectoryOption("test", must_exist=True, can_create=True)

        new_dir_path = os.path.join(tempfile.gettempdir(), "test_new_dir_12345")

        # Ensure directory doesn't exist
        if os.path.exists(new_dir_path):
            os.rmdir(new_dir_path)

        # Should create the directory and not raise
        descriptor.validate(new_dir_path)

        # Verify directory was created
        self.assertTrue(os.path.isdir(new_dir_path))

        # Clean up
        os.rmdir(new_dir_path)

    def test_validate_non_string(self):
        """Test validation with non-string value."""
        descriptor = DirectoryOption("test")
        with self.assertRaises(TypeError):
            descriptor.validate(42)


class TestDescriptorIntegration(unittest.TestCase):
    """Test descriptors working together in a class context."""

    def test_descriptor_in_class(self):
        """Test descriptors working within a class definition."""
        class TestClass:
            string_opt = StringOption("string_opt", default="default")
            number_opt = NumberOption("number_opt", default=0, min_value=0, max_value=100)
            bool_opt = BoolOption("bool_opt", default=True)
            enum_opt = EnumOption("enum_opt", ["option1", "option2"], default="option1")

        obj = TestClass()

        # Test defaults
        self.assertEqual(obj.string_opt, "default")
        self.assertEqual(obj.number_opt, 0)
        self.assertEqual(obj.bool_opt, True)
        self.assertEqual(obj.enum_opt, "option1")

        # Test setting valid values
        obj.string_opt = "new_value"
        obj.number_opt = 50
        obj.bool_opt = False
        obj.enum_opt = "option2"

        self.assertEqual(obj.string_opt, "new_value")
        self.assertEqual(obj.number_opt, 50)
        self.assertEqual(obj.bool_opt, False)
        self.assertEqual(obj.enum_opt, "option2")

        # Test validation errors
        with self.assertRaises(TypeError):
            obj.string_opt = 42

        with self.assertRaises(ValueError):
            obj.number_opt = 150  # exceeds max_value

        with self.assertRaises(ValueError):
            obj.enum_opt = "invalid_option"


if __name__ == '__main__':
    unittest.main()