import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure scripts directory is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from swebench_inclusion import install_instance


class TestInstallInstance(unittest.TestCase):
    @patch("subprocess.run")
    def test_install_instance_no_shell(self, mock_run):
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_run.return_value = mock_res

        spec = {
            "pre_install": ["sed -i 's/foo/bar/g' test.txt", "sed -i s/foo/bar/g test.txt"],
            "install": "python -m pip install -e .[test]"
        }

        success, cmd, error = install_instance("/tmp/workspace", "/tmp/env", spec=spec)

        self.assertTrue(success)
        self.assertTrue(mock_run.called)

        # Check call arguments to ensure shell=False and commands are lists
        for call_args in mock_run.call_args_list:
            args, kwargs = call_args
            self.assertIsInstance(args[0], list)
            self.assertFalse(kwargs.get("shell", False))

        # Check pre_install call 1 (with quotes after -i)
        first_call_cmd = mock_run.call_args_list[0][0][0]
        self.assertEqual(first_call_cmd, ["sed", "-i", "s/foo/bar/g", "test.txt"])

        # Check pre_install call 2 (without quotes after -i)
        second_call_cmd = mock_run.call_args_list[1][0][0]
        self.assertEqual(second_call_cmd, ["sed", "-i", "", "s/foo/bar/g", "test.txt"])

        # Check install call
        third_call_cmd = mock_run.call_args_list[2][0][0]
        self.assertEqual(third_call_cmd, ["python", "-m", "pip", "install", "-e", ".[test]"])


if __name__ == "__main__":
    unittest.main()
