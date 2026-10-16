import os
import sys
import unittest




class CategoryForEntryTests(unittest.TestCase):
    def test_category_for_entry_parallel(self):
        from mini_thalamus.evals.bfcl_eval import category_for_entry

        self.assertEqual(category_for_entry({"id": "parallel_0"}), "parallel")

    def test_category_for_entry_live_parallel(self):
        from mini_thalamus.evals.bfcl_eval import category_for_entry

        self.assertEqual(
            category_for_entry({"id": "live_parallel_0"}),
            "live_parallel",
        )

    def test_category_for_entry_simple_python(self):
        from mini_thalamus.evals.bfcl_eval import category_for_entry

        self.assertEqual(
            category_for_entry({"id": "simple_python_13"}),
            "simple_python",
        )

    def test_cast_properties_handles_nested_array_of_objects(self):
        from mini_thalamus.evals.bfcl_eval import cast_properties

        properties = {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"score": {"type": "float"}},
                },
            }
        }
        result = cast_properties(properties)
        self.assertEqual(
            result["events"]["items"]["properties"]["score"]["type"],
            "number",
        )

    def test_coerce_args_int_to_float_for_float_property(self):
        from mini_thalamus.evals.bfcl_eval import coerce_args_to_schema

        args = {"interval": [1, 3]}
        properties = {
            "interval": {
                "type": "array",
                "items": {"type": "float"},
            }
        }
        result = coerce_args_to_schema(args, properties)
        self.assertEqual(result["interval"], [1.0, 3.0])


if __name__ == "__main__":
    unittest.main()
