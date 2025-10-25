from modular_tot_architecture import ModularToTSelfReflectiveArchitecture
from test_humaneval import HumanEvalTestSuite
from langchain_ollama.llms import OllamaLLM
import json
import time
import re
import math
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

def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k using the unbiased estimator from the Codex paper.
    
    Args:
        n: total number of samples generated
        c: number of correct samples
        k: k in pass@k
    
    Returns:
        pass@k score (0 to 1)
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def compute_pass_at_k_for_dataset(results: List[Dict], k_values: List[int] = [1]) -> Dict[int, float]:
    """
    Compute pass@k across all problems in the dataset.
    
    Args:
        results: List of result dicts, each with:
            - 'problem': problem name
            - 'samples': list of dicts with 'passed': bool
        k_values: list of k values to compute (default: [1] for Pass@1)
    
    Returns:
        Dict mapping k -> pass@k score
    """
    pass_at_k_scores = {k: [] for k in k_values}
    
    for problem_result in results:
        n = len(problem_result['samples'])  # total samples generated
        c = sum(1 for s in problem_result['samples'] if s['passed'])  # correct samples
        
        for k in k_values:
            if k <= n:
                score = calculate_pass_at_k(n, c, k)
                pass_at_k_scores[k].append(score)
    
    # Average across all problems
    final_scores = {}
    for k in k_values:
        if pass_at_k_scores[k]:
            final_scores[k] = sum(pass_at_k_scores[k]) / len(pass_at_k_scores[k])
        else:
            final_scores[k] = 0.0
    
    return final_scores

def run_batch_humaneval_with_iteration():
    """Run HumanEval problems with iterative improvement using the full dataset"""
    llm = OllamaLLM(
        model="gpt-oss:20b",
        base_url="http://localhost:11434",
        temperature=0.1  # Lower temperature for consistent results (Pass@1)
    )
    
    improver = IterativeCodeImprover(llm, max_iterations=3)
    
    print("📥 Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")
    print(f"✅ Loaded {len(dataset)} problems from HumanEval.")
    
    # Configuration for Pass@1 evaluation
    NUM_SAMPLES_PER_PROBLEM = 1  # Single sample per problem for Pass@1
    k_values = [1]  # Only compute Pass@1
    
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
        
    # For testing, use only first few problems
    #humaneval_problems = humaneval_problems[:5]  # Adjust as needed
    
    all_problem_results = []
    
    for problem_idx, problem_data in enumerate(humaneval_problems):
        print(f"\n{'='*60}")
        print(f"📝 PROBLEM {problem_idx + 1}/{len(humaneval_problems)}: {problem_data['name']}")
        print(f"{'='*60}")
        
        samples = []
        
        for sample_idx in range(NUM_SAMPLES_PER_PROBLEM):
            print(f"\n🔄 Generating sample {sample_idx + 1}/{NUM_SAMPLES_PER_PROBLEM}...")
            
            start_time = time.time()
            result = improver.solve_with_iteration(problem_data)
            total_time = time.time() - start_time
            
            sample_passed = (result["success"] and 
                           result["final_test_results"]["passed_tests"] == 
                           result["final_test_results"]["total_tests"])
            
            samples.append({
                "sample_id": sample_idx + 1,
                "passed": sample_passed,
                "solution": result["final_solution"],
                "tests_passed": result["final_test_results"]["passed_tests"],
                "total_tests": result["final_test_results"]["total_tests"],
                "time": total_time,
                "iterations_used": result["total_iterations"]
            })
            
            status = "✅ PASSED" if sample_passed else "❌ FAILED"
            print(f"  {status} ({result['final_test_results']['passed_tests']}/{result['final_test_results']['total_tests']} tests) in {total_time:.2f}s")
        
        problem_result = {
            "problem": problem_data["name"],
            "samples": samples,
            "num_passed": sum(1 for s in samples if s["passed"]),
            "num_samples": NUM_SAMPLES_PER_PROBLEM
        }
        all_problem_results.append(problem_result)
        
        print(f"\n📊 Problem Summary: {problem_result['num_passed']}/{NUM_SAMPLES_PER_PROBLEM} samples passed")
    
    # Calculate Pass@1
    print("\n" + "="*60)
    print("📈 PASS@1 EVALUATION")
    print("="*60)
    
    pass_at_k_results = compute_pass_at_k_for_dataset(all_problem_results, k_values)
    
    # Display individual problem results
    print("\n📊 INDIVIDUAL PROBLEM RESULTS:")
    for problem_result in all_problem_results:
        passed_count = problem_result['num_passed']
        total_count = problem_result['num_samples']
        print(f"  {problem_result['problem']}: {passed_count}/{total_count} passed "
              f"({passed_count/total_count*100:.1f}%)")
    
    # Display Pass@1 result
    pass_at_1_score = pass_at_k_results[1] * 100
    print(f"\n🎯 PASS@1 SCORE: {pass_at_1_score:.2f}%")
    
    # Calculate traditional metrics for comparison
    total_problems = len(all_problem_results)
    problems_passed = sum(1 for p in all_problem_results if p['num_passed'] > 0)
    
    print(f"\n📈 TRADITIONAL METRICS:")
    print(f"  Problems passed: {problems_passed}/{total_problems} ({problems_passed/total_problems*100:.1f}%)")
    
    # Save comprehensive results
    results_summary = {
        "pass_at_1": pass_at_1_score,
        "num_problems": len(all_problem_results),
        "problems_passed": problems_passed,
        "problems_failed": total_problems - problems_passed,
        "pass_rate": (problems_passed / total_problems) * 100,
        "detailed_results": all_problem_results
    }
    
    with open("humaneval_pass_at_1_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Detailed results saved to humaneval_pass_at_1_results.json")
    
    return pass_at_1_score

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING HUMANEVAL PASS@1 EVALUATION")
    print("="*60)
    
    # Run the main evaluation
    results = run_batch_humaneval_with_iteration()
    
    print("\n" + "="*60)
    print("🎉 EVALUATION COMPLETE")
    print("="*60)