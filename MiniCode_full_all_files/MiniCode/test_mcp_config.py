import os
import unittest

from mcp_client import MCPClientManager


class MCPConfigTests(unittest.TestCase):
    def test_example_config_parses_stdio_server(self):
        example = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            ".mcp.json.example",
        )
        manager = MCPClientManager(example)

        loaded = manager.load_config()

        self.assertTrue(loaded)
        self.assertIn("example", manager.servers)
        self.assertEqual(manager.servers["example"].command, "npx")

    def test_missing_config_returns_false(self):
        manager = MCPClientManager("data/not_exists.json")
        self.assertFalse(manager.load_config())


if __name__ == "__main__":
    unittest.main()
