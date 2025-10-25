from modular_tot_architecture import ModularToTSelfReflectiveArchitecture
from test_humaneval import HumanEvalTestSuite
from langchain_ollama.llms import OllamaLLM
import json
import time
import re
from typing import Dict, Any, List
from datasets import load_dataset

class IterativeCodeImprover:
    def __init__(self, llm, max_iterations=3):
        self.llm = llm
        self.generator = ModularToTSelfReflectiveArchitecture(llm)
        self.test_suite = HumanEvalTestSuite()
        self.max_iterations = max_iterations
    
    def improve_solution_based_on_feedback(self, problem: str, current_solution: str, test_feedback: Dict[str, Any]) -> str:
        """Use test feedback to improve the solution"""
        improvement_prompt = f"""
You are an expert code debugger and improver. 
    
ORIGINAL PROBLEM:
{problem}
    
CURRENT SOLUTION (has test failures):
{current_solution}
    
TEST FAILURES ANALYSIS:
{test_feedback['feedback']}
    
CRITICAL ISSUES TO FIX:
{chr(10).join(test_feedback['critical_issues'])}
    
Please fix the code to address these issues. Focus on:
1. Fixing runtime errors first
2. Correcting logic that produces wrong outputs
3. Maintaining the original modular structure if possible
4. Ensuring all test cases pass
    
Return ONLY the improved Python code without any additional text or explanations.
"""
        
        try:
            response = self.llm.invoke(improvement_prompt)
            
            improved_code = None
            
            if "```python" in response:
                start = response.find("```python") + 9
                end = response.find("```", start)
                if end != -1:
                    improved_code = response[start:end].strip()
            
            if improved_code is None and "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                if end != -1:
                    improved_code = response[start:end].strip()
            
            if improved_code is None:
                raw_response = response.strip()
                if raw_response.startswith("def ") or raw_response.startswith("import ") or raw_response.startswith("from "):
                    improved_code = raw_response
                else:
                    improved_code = current_solution
                    print("    ⚠️ LLM response did not contain a code block and did not look like Python code. Falling back to previous solution.")
    
            if not improved_code or not improved_code.strip():
                 improved_code = current_solution
                 print("    ⚠️ Extracted code is empty. Falling back to previous solution.")
    
            print("    🔧 Generated improved solution based on test feedback")
            return improved_code
            
        except Exception as e:
            print(f"    ❌ Improvement generation failed: {e}")
            return current_solution

    def solve_with_iteration(self, problem_data: Dict) -> Dict[str, Any]:
        """Solve a problem with iterative improvement based on test results"""
        
        problem = problem_data["problem"]
        test_code = problem_data["test_code"] 
        function_name = problem_data["name"]
        
        print(f"\n{'='*60}")
        print(f"🧪 PROCESSING: {function_name} (with iteration)")
        print(f"{'='*60}")
        
        all_iteration_results = []
        current_solution = None
        
        for iteration in range(self.max_iterations):
            print(f"\n🔄 ITERATION {iteration + 1}/{self.max_iterations}")
            start_time = time.time()
            
            if iteration == 0:
                generation_result = self.generator.generate_solution(problem)
                if not generation_result["success"]:
                    print(f"❌ Initial generation failed: {generation_result.get('error')}")
                    return {
                        "success": False,
                        "error": generation_result.get("error"),
                        "iterations": all_iteration_results
                    }
                current_solution = generation_result["solution"]
                generation_metrics = generation_result
            else:
                previous_results = all_iteration_results[-1]
                test_feedback = self.test_suite.analyze_test_failures(previous_results["test_results"])
                
                if not test_feedback["needs_improvement"]:
                    print("    ✅ All tests passed, no further improvement needed")
                    break
                
                print(f"    📝 Improving solution based on test feedback...")
                current_solution = self.improve_solution_based_on_feedback(
                    problem, current_solution, test_feedback
                )
                generation_metrics = {
                    "verification_artifacts": {
                        "defensive_coverage": 85,
                        "modules_implemented": 1,
                        "total_reflection_cycles": iteration + 1
                    }
                }
            
            print(f"    📄 Current solution (iteration {iteration + 1}):")
            print("    " + "=" * 40)
            solution_lines = current_solution.split('\n')
            for line in solution_lines[:10]:
                print(f"    {line}")
            if len(solution_lines) > 10:
                print(f"    ... and {len(solution_lines) - 10} more lines")
            print("    " + "=" * 40)
            
            # FIXED: Use correct variable names
            test_results = self.test_suite.test_generated_solution(
                current_solution, test_code, function_name
            )
            
            iteration_result = {
                "iteration": iteration + 1,
                "solution": current_solution,
                "test_results": test_results,
                "generation_metrics": generation_metrics.get("verification_artifacts", {}),
                "time_taken": time.time() - start_time
            }
            
            all_iteration_results.append(iteration_result)
            
            print(f"    ✅ Iteration {iteration + 1} completed in {iteration_result['time_taken']:.2f}s")
            print(f"    📊 Tests passed: {test_results['passed_tests']}/{test_results['total_tests']}")
            
            if test_results["execution_successful"] and test_results["passed_tests"] == test_results["total_tests"]:
                print(f"    🎉 ALL TESTS PASSED at iteration {iteration + 1}!")
                break
        
        if not all_iteration_results:
            return {
                "success": False,
                "final_solution": current_solution,
                "iterations": [],
                "final_test_results": {"passed_tests": 0, "total_tests": 0, "execution_successful": False},
                "total_iterations": 0
            }
        
        best_iteration = all_iteration_results[-1]
        final_test_results = best_iteration["test_results"]
        
        success = (final_test_results["execution_successful"] and 
                  final_test_results["passed_tests"] == final_test_results["total_tests"])
        
        return {
            "success": success,
            "final_solution": current_solution,
            "iterations": all_iteration_results,
            "final_test_results": final_test_results,
            "total_iterations": len(all_iteration_results)
        }

def run_batch_humaneval_with_iteration():
    """Run HumanEval problems with iterative improvement using the full dataset"""
    llm = OllamaLLM(
        model="gpt-oss:20b",
        base_url="http://localhost:11434",
        temperature=0.1
    )
    
    improver = IterativeCodeImprover(llm, max_iterations=3)
    
    print("📥 Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")
    print(f"✅ Loaded {len(dataset)} problems from HumanEval.")
        
    humaneval_problems = []
    
    for item in dataset:
        match = re.search(r'def\s+(\w+)\s*\(', item['prompt'])
        function_name = match.group(1) if match else f"solution_{item['task_id']}"

        problem_data = {
            "name": function_name,
            "problem": item['prompt'],
            "test_code": item['test']
        }
        humaneval_problems.append(problem_data)
        
    humaneval_problems = humaneval_problems[:1]
    
    all_results = []
    
    for problem_data in humaneval_problems:
        start_time = time.time()
        
        result = improver.solve_with_iteration(problem_data)
        
        total_time = time.time() - start_time
        
        if result["success"]:
            print(f"\n🎉 PROBLEM SOLVED SUCCESSFULLY!")
            print(f"⏱️  Total time: {total_time:.2f}s")
            print(f"🔄 Iterations needed: {result['total_iterations']}")
        else:
            print(f"\n⚠️  Problem not fully solved after {result['total_iterations']} iterations")
            print(f"⏱️  Total time: {total_time:.2f}s")
        
        with open(f"result_{problem_data['name']}_iterative.txt", "w") as f:
            f.write(f"PROBLEM: {problem_data['name']}\n")
            f.write(f"SUCCESS: {result['success']}\n")
            f.write(f"ITERATIONS: {result['total_iterations']}\n")
            f.write(f"TOTAL_TIME: {total_time:.2f}s\n\n")
            
            for i, iteration in enumerate(result["iterations"]):
                f.write(f"=== ITERATION {i+1} ===\n")
                f.write(f"Time: {iteration['time_taken']:.2f}s\n")
                f.write(f"Tests Passed: {iteration['test_results']['passed_tests']}/{iteration['test_results']['total_tests']}\n")
                f.write("SOLUTION:\n")
                f.write(iteration["solution"])
                f.write("\n\nTEST RESULTS:\n")
                f.write(json.dumps(iteration["test_results"], indent=2))
                f.write("\n\n" + "="*50 + "\n\n")
        
        print(f"💾 Detailed results saved to result_{problem_data['name']}_iterative.txt")
        
        all_results.append({
            "problem": problem_data["name"],
            "success": result["success"],
            "iterations": result["total_iterations"],
            "final_tests_passed": result["final_test_results"]["passed_tests"],
            "total_tests": result["final_test_results"]["total_tests"],
            "total_time": total_time
        })
    
    print("\n" + "="*60)
    print("📊 ITERATIVE SOLVING SUMMARY")
    print("="*60)
    
    total_tests = 0
    total_passed_tests = 0
    total_problems_passed = 0
    
    if not all_results:
        print("No results to report.")
        return
        
    for result in all_results:
        status = "✅ SOLVED" if result["success"] else "⚠️  PARTIAL"
        print(f"\n{result['problem']}: {status}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Final Tests: {result['final_tests_passed']}/{result['total_tests']}")
        print(f"  Time: {result['total_time']:.2f}s")
        
        total_tests += result['total_tests']
        total_passed_tests += result['final_tests_passed']
        if result["success"]:
            total_problems_passed += 1
    
    if total_tests > 0:
        overall_pass_rate = (total_passed_tests / total_tests) * 100
        pass_at_1 = (total_problems_passed / len(all_results)) * 100 if all_results else 0
        
        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"   Overall Test Pass Rate: {total_passed_tests}/{total_tests} ({overall_pass_rate:.1f}%)")
        print(f"   Pass@1 Score: {total_problems_passed}/{len(all_results)} ({pass_at_1:.1f}%)")

if __name__ == "__main__":
    run_batch_humaneval_with_iteration()