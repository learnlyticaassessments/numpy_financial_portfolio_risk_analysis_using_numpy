import importlib.util
import datetime
import os
import numpy as np
import contextlib
from io import StringIO

@contextlib.contextmanager
def suppress_output():
    with contextlib.redirect_stdout(StringIO()):
        yield

def test_student_code(solution_path):
    report_dir = os.path.join(os.path.dirname(__file__), "..", "student_workspace")
    report_path = os.path.join(report_dir, "report.txt")
    os.makedirs(report_dir, exist_ok=True)

    spec = importlib.util.spec_from_file_location("student_module", solution_path)
    student_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_module)

    report_lines = [f"\n=== RiskAnalyzer Test Run at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="]

    randomized_failures = set()

    randomized_tests = {
        "create_stock_array": ([-2.0, 6.4, -5.6, -1.4, 2.3], np.array([-2.0, 6.4, -5.6, -1.4, 2.3])),
        "validate_stock_array": (np.array([97.22228, -72.37739]), False),
        "compute_volatility": (np.array([-2.0, 6.4, -5.6, -1.4, 2.3]), (-0.06, 4.57, 6.4)),
        "flag_volatile_stocks": (np.array([0.5, 2.5, 6.1]), np.array(["Stable", "Moderate Risk", "High Risk"])),
        "longest_loss_streak": (np.array([-1.0, -0.5, -2.0, -3.5, 1.0, -0.1]), 4),
        "format_stock_report": (np.array([3.14159, -2.71828]), np.array(["3.14%", "-2.72%"]))
    }

    for func_name, (input_data, expected_output) in randomized_tests.items():
        try:
            method = getattr(student_module, func_name)
            with suppress_output():
                result = method(input_data)

            if isinstance(expected_output, np.ndarray):
                if not np.array_equal(result, expected_output):
                    randomized_failures.add(func_name)
            elif isinstance(expected_output, tuple):
                rounded_result = tuple(round(r, 2) for r in result)
                if rounded_result != expected_output:
                    randomized_failures.add(func_name)
            else:
                if result != expected_output:
                    randomized_failures.add(func_name)
        except:
            randomized_failures.add(func_name)

    test_cases = [
        ("Visible", {
            "id": "TC1",
            "desc": "Create stock array",
            "func": "create_stock_array",
            "input": [2.5, -1.2, 0.8, -3.5, 4.0],
            "expected": np.array([2.5, -1.2, 0.8, -3.5, 4.0])
        }),
        ("Visible", {
            "id": "TC2",
            "desc": "Validate stock array - invalid",
            "func": "validate_stock_array",
            "input": np.array([10, 70]),
            "expected": False
        }),
        ("Visible", {
            "id": "TC3",
            "desc": "Compute volatility metrics",
            "func": "compute_volatility",
            "input": np.array([2.5, -1.2, 0.8, -3.5, 4.0]),
            "expected": (0.52, 2.97, 4.0)
        }),
        ("Visible", {
            "id": "TC4",
            "desc": "Flagging volatility risk levels",
            "func": "flag_volatile_stocks",
            "input": np.array([1.9, 5.0, 5.1]),  # Updated input
            "expected": np.array(["Stable", "High Risk", "High Risk"])  # Updated expected
        }),
        ("Visible", {
            "id": "TC5",
            "desc": "Longest loss streak",
            "func": "longest_loss_streak",
            "input": np.array([-1.1, -2.3, -3.0, 0.2, -1.5]),
            "expected": 3
        }),
        ("Hidden", {
            "id": "HTC1",
            "desc": "Formatted stock report",
            "func": "format_stock_report",
            "input": np.array([1.2345, -5.6789]),
            "expected": np.array(["1.23%", "-5.68%"])
        }),
        ("Hidden", {
            "id": "HTC2",
            "desc": "Validation with empty array",
            "func": "validate_stock_array",
            "input": np.array([]),
            "expected": False
        }),
        ("Hidden", {
            "id": "HTC3",
            "desc": "Edge case in volatility flagging",
            "func": "flag_volatile_stocks",
            "input": np.array([2.0, 5.0]),
            "expected": np.array(["Moderate Risk", "High Risk"])
        }),
    ]

    for section, case in test_cases:
        try:
            method = getattr(student_module, case["func"])
            with suppress_output():
                if isinstance(case["input"], tuple):
                    result = method(*case["input"])
                else:
                    result = method(case["input"])

            if case["func"] in randomized_failures:
                raise AssertionError("Implementation did not match randomized logic.")

            if isinstance(case["expected"], np.ndarray):
                assert np.array_equal(result, case["expected"]), f"Expected: {case['expected']} | Got: {result}"
            elif isinstance(case["expected"], tuple):
                rounded_result = tuple(round(r, 2) for r in result)
                assert rounded_result == case["expected"], f"Expected: {case['expected']} | Got: {rounded_result}"
            else:
                assert result == case["expected"], f"Expected: {case['expected']} | Got: {result}"

            msg = f"✅ {case['id']}: {case['desc']} passed"
        except Exception as e:
            msg = f"❌ {case['id']}: {case['desc']} failed | Reason: {str(e)}"

        print(msg)
        report_lines.append(msg)

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

if __name__ == "__main__":
    solution_file = os.path.join(os.path.dirname(__file__), "..", "student_workspace", "solution.py")
    test_student_code(solution_file)
