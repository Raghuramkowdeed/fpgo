"""Code execution oracle for LiveCodeBench problems."""

import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Optional

from ..datasets.base import Problem
from .base import Oracle


# Template for functional (LeetCode) test execution.
# Receives test data via environment variables to avoid escaping issues.
_FUNCTIONAL_TEMPLATE = '''\
import json, os
{code}
_raw_input = os.environ["_TEST_INPUT"]
_inputs = [json.loads(line) for line in _raw_input.strip().splitlines()]
_expected = json.loads(os.environ["_TEST_EXPECTED"])
try:
    _result = Solution().{fn_name}(*_inputs)
except NameError:
    _result = {fn_name}(*_inputs)
if _result == _expected:
    print("__PASS__")
else:
    print(f"__FAIL__Expected {{_expected}}, got {{_result}}")
'''


class CodeOracle(Oracle):
    """Evaluates code by running test cases via subprocess with timeout."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def extract_answer(self, response: str) -> Optional[str]:
        """Extract code from markdown code block or raw response."""
        # Try ```python ... ``` first
        match = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Try ``` ... ```
        match = re.search(r'```\s*\n(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: entire response (might be raw code)
        return response.strip()

    def _run_single_test(self, code: str, test_input: str,
                         expected_output: str, fn_name: Optional[str]) -> tuple:
        """Run one test case via subprocess. Returns (status, message)."""

        if fn_name:
            # Function-call format (LeetCode style)
            # Pass test data via env vars to avoid string-escaping bugs
            script = _FUNCTIONAL_TEMPLATE.format(code=code, fn_name=fn_name)
            env = {**os.environ, "_TEST_INPUT": test_input,
                   "_TEST_EXPECTED": expected_output}
            stdin_data = None
        else:
            # Stdio format (Codeforces/AtCoder) — run code with stdin
            script = code
            env = None
            stdin_data = test_input

        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return ("timeout", "Execution timed out")
        except Exception as e:
            return ("error", f"{type(e).__name__}: {e}")

        if result.returncode != 0:
            stderr = result.stderr.strip()
            lines = stderr.split('\n')
            short = '\n'.join(lines[-3:]) if len(lines) > 3 else stderr
            return ("error", short)

        actual = result.stdout.strip()

        if fn_name:
            # Check last line of output — model code may print extra lines
            last_line = actual.split('\n')[-1] if actual else ""
            if last_line.startswith("__PASS__"):
                return ("pass", "")
            elif last_line.startswith("__FAIL__"):
                return ("fail", last_line[8:])
            else:
                return ("error", f"Unexpected output: {actual[:200]}")
        else:
            expected = expected_output.strip()
            if actual == expected:
                return ("pass", "")
            else:
                return ("fail", f"Expected:\n{expected[:200]}\nGot:\n{actual[:200]}")

    def _parse_test_cases(self, problem: Problem):
        """Parse input/output test cases from problem ground truth."""
        gt = problem.ground_truth
        fn_name = gt.get("fn_name", None)
        testtype = gt.get("testtype", "stdin")

        public_tests = gt.get("public_test_cases", [])
        if isinstance(public_tests, str):
            try:
                public_tests = json.loads(public_tests)
            except (json.JSONDecodeError, TypeError):
                public_tests = []

        test_cases = []
        for tc in public_tests:
            inp = tc.get("input", "")
            out = tc.get("output", "")
            test_cases.append((str(inp), str(out)))

        return test_cases, fn_name if testtype != "stdin" else None

    def evaluate(self, response: str, problem: Problem,
                 fractional: bool = False) -> float:
        """Run all test cases and return a score.

        Args:
            response: Model-generated response containing code.
            problem: Problem with ground_truth test cases.
            fractional: If True, return passed/total (fractional reward).
                        If False (default), return 1.0 only if all pass.
        """
        code = self.extract_answer(response)
        if not code:
            return 0.0

        test_cases, fn_name = self._parse_test_cases(problem)
        if not test_cases:
            return 0.0

        passed = 0
        for test_input, expected_output in test_cases:
            status, _ = self._run_single_test(code, test_input, expected_output, fn_name)
            if status == "pass":
                passed += 1
            elif not fractional:
                return 0.0

        if fractional:
            return passed / len(test_cases)
        return 1.0

    def get_feedback(self, response: str, problem: Problem) -> str:
        """Run test cases and return detailed feedback on failures."""
        code = self.extract_answer(response)
        if not code:
            return "Could not extract code from your response. Please wrap your code in ```python ... ``` blocks."

        test_cases, fn_name = self._parse_test_cases(problem)
        if not test_cases:
            return "No test cases available for this problem."

        feedback_parts = []
        all_pass = True
        for i, (test_input, expected_output) in enumerate(test_cases):
            status, message = self._run_single_test(code, test_input, expected_output, fn_name)
            if status != "pass":
                all_pass = False
                feedback_parts.append(f"Test case {i+1} ({status}): {message}")
                if len(feedback_parts) >= 3:
                    break

        if all_pass:
            return "All test cases passed."

        return "Your code failed on the following test cases:\n" + "\n".join(feedback_parts)
