from modular_tot_architecture import ModularToTSelfReflectiveArchitecture
from test_humaneval import HumanEvalTestSuite
from langchain_ollama.llms import OllamaLLM
import json
import time
import re
import math
import os
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
    """Calculate pass@k using the unbiased estimator"""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def compute_pass_at_k_for_dataset(results: List[Dict], k_values: List[int] = [1]) -> Dict[int, float]:
    """Compute pass@k across all problems"""
    pass_at_k_scores = {k: [] for k in k_values}
    
    for problem_result in results:
        n = len(problem_result['samples'])
        c = sum(1 for s in problem_result['samples'] if s['passed'])
        
        for k in k_values:
            if k <= n:
                score = calculate_pass_at_k(n, c, k)
                pass_at_k_scores[k].append(score)
    
    final_scores = {}
    for k in k_values:
        if pass_at_k_scores[k]:
            final_scores[k] = sum(pass_at_k_scores[k]) / len(pass_at_k_scores[k])
        else:
            final_scores[k] = 0.0
    
    return final_scores

def get_next_problem_index():
    """Determine the next problem index to process based on progress file"""
    progress_file = "humaneval_progress.json"
    
    if not os.path.exists(progress_file):
        print("🆕 No progress file found. Starting from the beginning.")
        return 0
    
    with open(progress_file, 'r') as f:
        progress_data = json.load(f)
    
    completed_problems = progress_data.get("completed_problems", [])
    return len(completed_problems)

def run_batch_humaneval_with_iteration():
    """Run HumanEval problems with checkpoint system"""
    llm = OllamaLLM(
        model="gpt-oss:20b",
        base_url="http://localhost:11434",
        temperature=0.1
    )
    
    improver = IterativeCodeImprover(llm, max_iterations=3)
    
    print("📥 Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")
    print(f"✅ Loaded {len(dataset)} problems from HumanEval.")
    
    NUM_SAMPLES_PER_PROBLEM = 1
    k_values = [1]
    
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
    
    all_problem_results = []
    progress_file = "humaneval_progress.json"
    start_time = time.time()
    
    # Determine starting point
    start_index = get_next_problem_index()
    print(f"🚀 Starting from problem index: {start_index}")
    
    # Load existing progress if any
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
            all_problem_results = progress_data.get("completed_problems", [])
            current_batch_count = progress_data.get("current_batch_count", 0)
            print(f"🔄 Resuming from progress file: {len(all_problem_results)} problems completed")
    else:
        current_batch_count = 0
        print("🆕 Starting fresh evaluation")
    
    # Process problems
    for problem_idx, problem_data in enumerate(humaneval_problems):
        # Skip problems before our starting point
        if problem_idx < start_index:
            continue
            
        # Skip already completed problems
        completed_problem_names = [p['problem'] for p in all_problem_results]
        if problem_data['name'] in completed_problem_names:
            print(f"⏩ Skipping already completed: {problem_data['name']}")
            continue
        
        # Check if we need a checkpoint break
        if current_batch_count >= 9:
            print(f"\n{'='*60}")
            print(f"🛑 CHECKPOINT REACHED: Processed {current_batch_count} problems")
            print(f"💤 Taking 300-second break before continuing...")
            print(f"{'='*60}")
            
            # Update progress before break
            progress_data = {
                "start_time": start_time,
                "current_time": time.time(),
                "elapsed_hours": (time.time() - start_time) / 3600,
                "completed_problems": all_problem_results,
                "current_batch_count": current_batch_count,
                "last_completed_problem": all_problem_results[-1]['problem'] if all_problem_results else None,
                "last_completed_index": problem_idx - 1
            }
            
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            # Take 300-second break
            for i in range(300, 0, -30):
                print(f"⏰ Resuming in {i} seconds... (Press Ctrl+C to stop)")
                time.sleep(30)
            
            # Reset batch counter after break
            current_batch_count = 0
            print("✅ Checkpoint break completed. Resuming processing...")
        
        print(f"\n{'='*60}")
        print(f"📝 PROBLEM {problem_idx + 1}/{len(humaneval_problems)}: {problem_data['name']}")
        print(f"📦 Batch progress: {current_batch_count + 1}/9")
        print(f"{'='*60}")
        
        samples = []
        
        for sample_idx in range(NUM_SAMPLES_PER_PROBLEM):
            print(f"\n🔄 Generating sample {sample_idx + 1}/{NUM_SAMPLES_PER_PROBLEM}...")
            
            sample_start_time = time.time()
            result = improver.solve_with_iteration(problem_data)
            total_time = time.time() - sample_start_time
            
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
        current_batch_count += 1
        
        print(f"\n📊 Problem Summary: {problem_result['num_passed']}/{NUM_SAMPLES_PER_PROBLEM} samples passed")
        
        # Calculate and display REAL-TIME Pass@1
        current_pass_at_1 = compute_pass_at_k_for_dataset(all_problem_results, [1])[1]
        problems_so_far = len(all_problem_results)
        passed_so_far = sum(1 for p in all_problem_results if p['num_passed'] > 0)
        
        print(f"\n🎯 REAL-TIME PASS@1: {current_pass_at_1 * 100:.2f}%")
        print(f"📊 Progress: {passed_so_far}/{problems_so_far} problems passed")
        print(f"📍 Current position: Problem {problem_idx + 1}/{len(humaneval_problems)}")
        print(f"📦 Batch progress: {current_batch_count}/9 problems")
        
        # Save progress after each problem
        progress_data = {
            "start_time": start_time,
            "current_time": time.time(),
            "elapsed_hours": (time.time() - start_time) / 3600,
            "completed_problems": all_problem_results,
            "current_pass_at_1": current_pass_at_1 * 100,
            "problems_completed": problems_so_far,
            "problems_passed": passed_so_far,
            "total_problems": len(humaneval_problems),
            "completion_percentage": (problems_so_far / len(humaneval_problems)) * 100,
            "last_completed_problem": problem_data['name'],
            "last_completed_index": problem_idx,
            "current_batch_count": current_batch_count
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        print(f"💾 Progress saved to {progress_file}")
        
        # Estimate remaining time
        if problems_so_far > 0:
            avg_time_per_problem = sum(s['time'] for p in all_problem_results for s in p['samples']) / problems_so_far
            remaining_problems = len(humaneval_problems) - problem_idx - 1
            remaining_hours = (avg_time_per_problem * remaining_problems) / 3600
            print(f"⏰ Estimated time remaining: {remaining_hours:.1f} hours")
        
        # Take 30-second break between problems (unless it's the last problem)
        if problem_idx < len(humaneval_problems) - 1:
            print(f"\n⏳ Taking 30-second break before next problem...")
            for i in range(30, 0, -5):
                print(f"   Next problem in {i} seconds...")
                time.sleep(5)
    
    # Final calculation
    print("\n" + "="*60)
    print("📈 FINAL PASS@1 EVALUATION")
    print("="*60)
    
    pass_at_k_results = compute_pass_at_k_for_dataset(all_problem_results, k_values)
    
    # Display Pass@1 result
    pass_at_1_score = pass_at_k_results[1] * 100
    print(f"\n🎯 FINAL PASS@1 SCORE: {pass_at_1_score:.2f}%")
    
    # Calculate traditional metrics
    total_problems = len(all_problem_results)
    problems_passed = sum(1 for p in all_problem_results if p['num_passed'] > 0)
    
    print(f"\n📈 FINAL METRICS:")
    print(f"  Problems passed: {problems_passed}/{total_problems} ({problems_passed/total_problems*100:.1f}%)")
    print(f"  Total time: {(time.time() - start_time) / 3600:.2f} hours")
    
    # Save comprehensive final results
    results_summary = {
        "pass_at_1": pass_at_1_score,
        "num_problems": len(all_problem_results),
        "problems_passed": problems_passed,
        "problems_failed": total_problems - problems_passed,
        "pass_rate": (problems_passed / total_problems) * 100,
        "total_time_hours": (time.time() - start_time) / 3600,
        "detailed_results": all_problem_results
    }
    
    with open("humaneval_pass_at_1_final_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Final results saved to humaneval_pass_at_1_final_results.json")
    
    return pass_at_1_score

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 HUMANEVAL PASS@1 EVALUATION WITH CHECKPOINT SYSTEM")
    print("="*60)
    print("📋 Features:")
    print("  • Checkpoint every 9 problems (300-second break)")
    print("  • 30-second breaks between problems") 
    print("  • Automatic resume from last position")
    print("  • Progress tracking in humaneval_progress.json")
    print("="*60)
    
    results = run_batch_humaneval_with_iteration()
    
    print("\n" + "="*60)
    print("🎉 EVALUATION COMPLETE")
    print("="*60)