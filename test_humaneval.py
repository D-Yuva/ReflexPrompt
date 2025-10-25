import ast
import tempfile
import os
import re
from typing import List, Dict, Any, Tuple
import json

class HumanEvalTestSuite:
    def __init__(self):
        self.results = []
    
    def extract_all_functions(self, solution: str, main_function_name: str) -> str:
        """Extract ALL functions and imports from the full solution"""
        try:
            tree = ast.parse(solution)
            imports = []
            functions = []
            
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(node))
                elif isinstance(node, ast.FunctionDef):
                    functions.append(ast.unparse(node))
            
            return '\n'.join(imports + [''] + functions)
        except Exception as e:
            print(f"Warning: Could not parse solution: {e}")
            return solution

    def _parse_humaneval_tests(self, test_code: str, function_name: str) -> Tuple[str, List[str]]:
        """
        Parses the HumanEval 'test' field to extract the check function and test calls.
        FIXED VERSION: Properly handles HumanEval test format
        """
        check_func_name = f"check_{function_name}"
        test_calls = []
        
        # Extract assert statements from the check function
        if "def check(candidate):" in test_code:
            # Extract the check function body
            check_start = test_code.find("def check(candidate):")
            check_body = test_code[check_start:]
            
            # Find all assert statements in the check function
            lines = check_body.split('\n')
            in_check_function = False
            
            for line in lines:
                stripped_line = line.strip()
                if stripped_line.startswith("def check(candidate):"):
                    in_check_function = True
                    continue
                elif in_check_function:
                    if stripped_line.startswith("def ") and not stripped_line.startswith("def check(candidate):"):
                        # Reached another function definition
                        break
                    if stripped_line.startswith("assert candidate("):
                        # Convert "assert candidate(args) == expected" to "assert function_name(args) == expected"
                        test_call = stripped_line.replace("candidate", function_name)
                        test_calls.append(test_call)
        
        # Generate the check function code
        if test_calls:
            test_body = "\n    ".join(test_calls)
            check_func_code = f"""
def {check_func_name}({function_name}):
    \"\"\"Test function for {function_name}\"\"\"
    {test_body}
"""
        else:
            check_func_code = f"""
def {check_func_name}({function_name}):
    \"\"\"Test function for {function_name}\"\"\"
    pass
"""
        
        print(f"    Found {len(test_calls)} test calls for {function_name}")
        if test_calls:
            print(f"    Sample test call: {test_calls[0][:100]}...")
        
        return check_func_code, test_calls

    def _generate_safe_test_runner(self, function_name: str, test_code: str) -> str:
        """Generate safe test runner code"""
        check_func_code, test_calls = self._parse_humaneval_tests(test_code, function_name)
        
        # Build the test code as direct exec statements
        test_exec_code = "\n    ".join(test_calls)
        
        test_runner_code = f'''
    {check_func_code}

    def run_all_tests():
        """Test runner for {function_name}"""
        try:
            {test_exec_code if test_exec_code else "pass  # No tests found"}
            print("✅ All tests passed!")
            return True
        except AssertionError as e:
            print(f"❌ Test failed: {{str(e)}}")
            return False
        except Exception as e:
            print(f"💥 Error: {{type(e).__name__}}: {{str(e)}}")
            return False

    if __name__ == "__main__":
        run_all_tests()
    '''
        
        return test_runner_code


    
    def test_generated_solution(self, solution: str, test_code: str, function_name: str) -> Dict[str, Any]:
        """
        Test the generated solution against HumanEval test cases.
        SIMPLIFIED VERSION
        """
        print(f"\n🧪 Testing {function_name} with HumanEval tests...")
        
        # First, validate the solution syntax
        try:
            ast.parse(solution)
        except SyntaxError as e:
            return {
                "function_name": function_name,
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": [],
                "test_details": [],
                "execution_successful": False,
                "error_message": f"SyntaxError: {e}"
            }
        
        results = {
            "function_name": function_name,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": [],
            "test_details": [],
            "execution_successful": False,
            "error_message": None
        }
        
        try:
            # Create a clean execution environment
            exec_globals = {}
            
            # Execute the solution code
            exec(solution, exec_globals)
            
            # Extract test cases from HumanEval format
            test_calls = self._extract_test_calls(test_code, function_name)
            results["total_tests"] = len(test_calls)
            
            passed_tests = 0
            test_details = []
            
            for i, test_call in enumerate(test_calls):
                try:
                    # Execute the test
                    exec(test_call, exec_globals)
                    passed_tests += 1
                    test_details.append({
                        "test_number": i + 1,
                        "success": True,
                        "error": None
                    })
                except AssertionError as e:
                    test_details.append({
                        "test_number": i + 1,
                        "success": False,
                        "error": f"Assertion failed: {e}"
                    })
                except Exception as e:
                    test_details.append({
                        "test_number": i + 1,
                        "success": False,
                        "error": f"Runtime error: {type(e).__name__}: {e}"
                    })
            
            results["passed_tests"] = passed_tests
            results["test_details"] = test_details
            results["execution_successful"] = True
            results["failed_tests"] = [t for t in test_details if not t['success']]
            
        except Exception as e:
            print(f"    ❌ Test execution failed: {type(e).__name__}: {e}")
            results["execution_successful"] = False
            results["error_message"] = str(e)
        
        return results

    def _extract_test_calls(self, test_code: str, function_name: str) -> List[str]:
        """Extract individual test calls from HumanEval test code"""
        test_calls = []
        
        # Look for assert candidate(...) patterns and replace with actual function name
        lines = test_code.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('assert candidate('):
                # Replace 'candidate' with the actual function name
                test_call = stripped.replace('candidate', function_name)
                test_calls.append(test_call)
        
        return test_calls
    
    def analyze_test_failures(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test failures to provide feedback for improvement"""
        if test_results["execution_successful"] and test_results["passed_tests"] == test_results["total_tests"]:
            return {
                "needs_improvement": False,
                "feedback": "All tests passed!",
                "critical_issues": []
            }
        
        issues = []
        feedback_parts = []
        
        if not test_results["execution_successful"]:
            error_message = test_results.get('error_message', 'Unknown error')
            issues.append(f"Execution failed: {error_message}")
            feedback_parts.append(f"Execution failed: {error_message}")
        else:
            for failed_test in test_results.get("failed_tests", []):
                if failed_test.get("error"):
                    issues.append(f"Test {failed_test['test_number']}: {failed_test['error']}")
                    feedback_parts.append(f"Test {failed_test['test_number']} failed: {failed_test['error']}")
        
        return {
            "needs_improvement": True,
            "feedback": "; ".join(feedback_parts),
            "critical_issues": issues,
            "passed_tests": test_results["passed_tests"],
            "total_tests": test_results["total_tests"]
        }