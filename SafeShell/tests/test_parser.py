"""
test_parser.py

Unit tests for SafeShell's real Bashlex-based command parser
(src/parser/command_parser.py), matching the team's existing
unittest.TestCase style (see test_rules_engine.py).

Verifies parse() produces the exact AST dict shape rules_engine.check()
and semantic_fusion.fuse() already expect, and that invalid/empty
input raises the correct structured exceptions rather than crashing.
"""

import sys
import unittest
from pathlib import Path

_SAFESHELL_ROOT = Path(__file__).resolve().parents[1]
if str(_SAFESHELL_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAFESHELL_ROOT))

from src.parser.command_parser import parse
from src.parser.ast_utils import EmptyCommandError, InvalidSyntaxError


class TestCommandParser(unittest.TestCase):
    """Tests ordered from simple structure checks through to the
    error-handling paths."""

    # ------------------------------------------------------------------ #
    #  BASIC STRUCTURE                                                    #
    # ------------------------------------------------------------------ #

    def test_01_simple_command(self):
        ast = parse("pwd")
        self.assertEqual(ast["command"], "pwd")
        self.assertEqual(ast["flags"], [])
        self.assertEqual(ast["args"], [])

    def test_02_command_with_flags(self):
        ast = parse("ls -la")
        self.assertEqual(ast["command"], "ls")
        self.assertIn("-l", ast["flags"])
        self.assertIn("-a", ast["flags"])

    def test_03_command_with_arguments(self):
        ast = parse("rm important.txt")
        self.assertEqual(ast["command"], "rm")
        self.assertEqual(ast["args"], ["important.txt"])
        self.assertEqual(ast["target_path"], "important.txt")

    # ------------------------------------------------------------------ #
    #  SUDO / PRIVILEGE                                                   #
    # ------------------------------------------------------------------ #

    def test_04_sudo_prefix_detected(self):
        ast = parse("sudo rm -rf /home/project")
        self.assertEqual(ast["command"], "rm")
        self.assertTrue(ast["is_sudo"])
        self.assertTrue(ast["is_recursive"])
        self.assertTrue(ast["is_force"])
        self.assertEqual(ast["target_path"], "/home/project")

    def test_05_no_sudo_by_default(self):
        ast = parse("ls -la")
        self.assertFalse(ast["is_sudo"])

    # ------------------------------------------------------------------ #
    #  FLAGS: RECURSIVE / FORCE DETECTION                                 #
    # ------------------------------------------------------------------ #

    def test_06_combined_short_flags_expanded(self):
        ast = parse("rm -rf project/")
        self.assertIn("-rf", ast["flags"])
        self.assertIn("-r", ast["flags"])
        self.assertIn("-f", ast["flags"])
        self.assertTrue(ast["is_recursive"])
        self.assertTrue(ast["is_force"])

    def test_07_chmod_recursive_flag(self):
        ast = parse("chmod -R 777 /etc")
        self.assertTrue(ast["is_recursive"])
        self.assertEqual(ast["target_path"], "/etc")

    # ------------------------------------------------------------------ #
    #  PIPES                                                              #
    # ------------------------------------------------------------------ #

    def test_08_pipe_to_target_captured(self):
        ast = parse("cat file.txt | grep hello")
        self.assertEqual(ast["command"], "cat")
        self.assertEqual(ast["pipe_to"], "grep")

    def test_09_curl_piped_to_bash_detected(self):
        ast = parse("curl http://example.com/install.sh | bash")
        self.assertEqual(ast["command"], "curl")
        self.assertEqual(ast["pipe_to"], "bash")

    def test_10_no_pipe_target_when_absent(self):
        ast = parse("ls -la")
        self.assertEqual(ast["pipe_to"], "")

    # ------------------------------------------------------------------ #
    #  ENVIRONMENT ASSIGNMENTS (correctness improvement over shlex stopgap)
    # ------------------------------------------------------------------ #

    def test_11_leading_env_assignment_not_mistaken_for_command(self):
        ast = parse("FOO=bar echo hi")
        self.assertEqual(ast["command"], "echo")
        self.assertEqual(ast["args"], ["hi"])

    # ------------------------------------------------------------------ #
    #  QUOTED ARGUMENTS / REDIRECTS                                       #
    # ------------------------------------------------------------------ #

    def test_12_quoted_argument_preserved(self):
        ast = parse('echo "hello world"')
        self.assertEqual(ast["args"], ["hello world"])

    def test_13_redirect_does_not_break_parsing(self):
        ast = parse("echo hello > output.txt")
        self.assertEqual(ast["command"], "echo")
        self.assertEqual(ast["args"], ["hello"])

    # ------------------------------------------------------------------ #
    #  TARGET PATH HEURISTIC                                              #
    # ------------------------------------------------------------------ #

    def test_14_target_path_prefers_absolute_path(self):
        ast = parse("rm -rf /")
        self.assertEqual(ast["target_path"], "/")

    def test_15_target_path_falls_back_to_last_arg(self):
        ast = parse("cp file1.txt file2.txt")
        self.assertEqual(ast["target_path"], "file2.txt")

    # ------------------------------------------------------------------ #
    #  ERROR HANDLING -- must raise, never crash silently                 #
    # ------------------------------------------------------------------ #

    def test_16_empty_input_raises_empty_command_error(self):
        with self.assertRaises(EmptyCommandError):
            parse("")

    def test_17_whitespace_only_input_raises(self):
        with self.assertRaises(EmptyCommandError):
            parse("    ")

    def test_18_unclosed_quote_raises_invalid_syntax(self):
        with self.assertRaises(InvalidSyntaxError):
            parse('echo "hello')

    # ------------------------------------------------------------------ #
    #  SECURITY: PARSING NEVER EXECUTES                                   #
    # ------------------------------------------------------------------ #

    def test_19_dangerous_commands_are_only_parsed_not_executed(self):
        """Parsing must never actually run these -- only structure them."""
        dangerous = [
            "sudo rm -rf /",
            "dd if=/dev/zero of=/dev/sda",
            "mkfs.ext4 /dev/sda1",
            "chmod 777 /etc/passwd",
        ]
        for cmd in dangerous:
            ast = parse(cmd)
            self.assertIsInstance(ast, dict)  # structured, not executed

    def test_20_ast_shape_matches_rules_engine_contract(self):
        """Confirms parse() output has exactly the keys rules_engine.py
        and semantic_fusion.py expect."""
        ast = parse("ls -la")
        expected_keys = {
            "command", "flags", "args", "target_path",
            "is_sudo", "is_recursive", "is_force", "raw", "pipe_to",
        }
        self.assertEqual(set(ast.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()