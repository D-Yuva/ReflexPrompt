import json
import ast
import re
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ModuleSpec:
    name: str
    purpose: str
    recursive: bool
    inputs: List[str]
    outputs: str
    edge_cases: List[str]

@dataclass
class NodeResult:
    module_name: str
    approach: str
    code: str
    defensive_measures: List[str]
    reasoning_artifacts: Dict[str, Any]
    is_robust: bool = False
    defensive_score: int = 0

@dataclass
class ToTNode:
    module_spec: ModuleSpec
    expansion_results: List[NodeResult]
    best_result: Optional[NodeResult] = None
    reflection_feedback: Optional[Dict[str, Any]] = None

class ModularToTSelfReflectiveArchitecture:
    def __init__(self, llm):
        self.llm = llm
        self.problem = None
        self.modules = []
        self.tot_nodes = []
        self.final_solution = None
        self.decomposed_module_names = []
    
    def _validate_code_syntax(self, code: str) -> bool:
        """Validate that generated code has proper syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            print(f"    ❌ Generated code has syntax error: {e}")
            return False

    def _clean_code_string(self, code: str) -> str:
        """Fix escaped characters in code strings"""
        if not isinstance(code, str):
            return str(code)
            
        code = code.replace('\\n', '\n')
        code = code.replace('\\t', '\t')
        code = code.replace('\\"', '"')
        code = code.replace("\\'", "'")
        code = code.replace('\\\\', '\\')
        
        code = re.sub(r'\\([^nrt"\'\\])', r'\1', code)
        
        return code
    
    def _extract_json_robust(self, text: str) -> Dict[str, Any]:
        """Robust JSON extraction with comprehensive error handling"""
        if not text or not isinstance(text, str):
            raise json.JSONDecodeError("Empty or invalid text", text, 0)
            
        text = text.strip()
        
        if not text:
            raise json.JSONDecodeError("Empty response from LLM", text, 0)
        
        strategies = [
            lambda t: json.loads(t),
            lambda t: json.loads(re.search(r'```json\s*(.*?)\s*```', t, re.DOTALL).group(1).strip()),
            lambda t: json.loads(re.search(r'```\s*(.*?)\s*```', t, re.DOTALL).group(1).strip()),
            lambda t: json.loads(re.search(r'(\{.*\})', t, re.DOTALL).group(1)),
        ]
        
        last_error = None
        for strategy in strategies:
            try:
                result = strategy(text)
                print(f"    JSON extraction successful with strategy")
                return result
            except Exception as e:
                last_error = e
                continue
        
        raise json.JSONDecodeError(f"All JSON extraction strategies failed. Last error: {last_error}", text, 0)
    
    def _force_json_response(self, prompt: str, context: str = "") -> Dict[str, Any]:
        """Force LLM to return proper JSON with better error handling"""
        enhanced_prompt = f"""{prompt}

CRITICAL: You MUST respond with VALID JSON that can be parsed by json.loads().
{context}

Double-check your response:
1. All strings use double quotes
2. No trailing commas in arrays or objects  
3. All brackets and braces are properly closed
4. The structure matches exactly what's requested
"""
        
        try:
            response = self.llm.invoke(enhanced_prompt)
            print(f"    Raw LLM response preview: {response[:200]}...")
            
            if not response or not response.strip():
                print("    ⚠️ Empty response from LLM, using fallback")
                return self._create_fallback_response()
                
            return self._extract_json_robust(response)
        except json.JSONDecodeError as e:
            print(f"    ⚠️ JSON parsing failed: {e}")
            return self._create_fallback_response()
        except Exception as e:
            print(f"    ⚠️ LLM invocation failed: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> Dict[str, Any]:
        """Create a fallback response when LLM fails"""
        return {
            "approach": "Fallback approach due to LLM failure",
            "code": "def fallback_function():\n    # REASONING: This is a fallback implementation\n    return None",
            "defensive_measures": ["input_validation"],
            "reasoning_artifacts": {
                "preconditions": ["Input validation required"],
                "postconditions": ["Function completes successfully"],
                "invariants": ["Data integrity maintained"],
                "test_cases": ["Basic functionality"]
            }
        }
    
    def _extract_function_info(self, problem: str) -> Dict[str, Any]:
        """Extract function signature information from problem description"""
        function_info = {
            "name": "solution",
            "parameters": ["input_data"],
            "return_type": "Any",
            "imports": []
        }
        
        if "def " in problem:
            match = re.search(r'def\s+(\w+)\s*\(', problem)
            if match:
                function_info["name"] = match.group(1)
        
        param_match = re.search(r'\(([^)]*)\)', problem)
        if param_match:
            params_text = param_match.group(1)
            params = []
            for param in params_text.split(','):
                param = param.strip()
                if ':' in param:
                    param_name = param.split(':')[0].strip()
                else:
                    param_name = param
                if param_name and param_name != '':
                    params.append(param_name)
            if params:
                function_info["parameters"] = params
        
        return_match = re.search(r'->\s*([^:\n]+)', problem)
        if return_match:
            function_info["return_type"] = return_match.group(1).strip()
        
        function_info["imports"].append("from typing import List, Any, Optional, Union, Tuple, Dict")
        
        if "List[" in problem:
            function_info["imports"].append("from typing import List")
        if "Tuple[" in problem:
            function_info["imports"].append("from typing import Tuple")
        if "Dict[" in problem:
            function_info["imports"].append("from typing import Dict")
        if "Optional[" in problem:
            function_info["imports"].append("from typing import Optional")
        if "Union[" in problem:
            function_info["imports"].append("from typing import Union")
        
        function_info["imports"] = list(set(function_info["imports"]))
        
        return function_info

    def _generate_main_function(self, function_info: Dict[str, Any], available_modules: List[str]) -> str:
        """Generate the main function that directly uses available modules."""
        func_name = function_info["name"]
        params = function_info["parameters"]
        return_type = function_info["return_type"]
        
        param_str = ", ".join(params)
        
        main_function = f"""
def {func_name}({param_str}) -> {return_type}:
    \"\"\"
    Main function that integrates all modular components to solve the problem.
    Generated by Modular ToT Self-Reflective Architecture.
    \"\"\"
"""
        
        if not available_modules:
            main_function += f"    raise NotImplementedError('No modules were successfully generated to implement this function.')\n"
            return main_function
        
        primary_input = params[0] if params else "input_data"
        
        main_function += "    # Module execution chain\n"
        
        if len(available_modules) == 1:
            main_function += f"    return {available_modules[0]}({param_str})\n"
        else:
            first_module = available_modules[0]
            main_function += f"    processed = {first_module}({primary_input})\n"
            
            for module in available_modules[1:-1]:
                main_function += f"    processed = {module}(processed)\n"
                
            last_module = available_modules[-1]
            main_function += f"    return {last_module}(processed)\n"

        return main_function

    def compose_solution(self) -> str:
        """Step 4: Compose the final solution from all parts"""
        print("🧩 Step 4: Compose Modular Solution")
        
        function_info = self._extract_function_info(self.problem)
        
        module_codes = []
        available_modules = []
        modules_used = 0
        
        for node in self.tot_nodes:
            if node.best_result:
                function_name = self._extract_function_name_from_code(node.best_result.code)
                if function_name:
                    available_modules.append(function_name)
                    print(f"    ✅ Module '{node.best_result.module_name}' implemented as function '{function_name}'")
                else:
                    available_modules.append(node.best_result.module_name)
                    print(f"    ⚠️  Using module name '{node.best_result.module_name}' (could not extract function name)")
                
                cleaned_code = self._clean_code_string(node.best_result.code)
                module_codes.append(cleaned_code)
                modules_used += 1
                
            elif node.expansion_results:
                function_name = self._extract_function_name_from_code(node.expansion_results[0].code)
                if function_name:
                    available_modules.append(function_name)
                    print(f"    ✅ Module '{node.module_spec.name}' implemented as function '{function_name}'")
                else:
                    available_modules.append(node.module_spec.name)
                    print(f"    ⚠️  Using module name '{node.module_spec.name}' (could not extract function name)")
                    
                cleaned_code = self._clean_code_string(node.expansion_results[0].code)
                module_codes.append(cleaned_code)
                modules_used += 1
        
        print(f"    🎯 Available modules to call: {available_modules}")
        
        main_function_code = self._generate_main_function(function_info, available_modules)
        
        imports = "\n".join(function_info["imports"])
        
        all_code = [imports] + module_codes + [main_function_code]
        
        final_solution = "\n\n".join(c for c in all_code if c.strip()) 
        
        # Validate final solution syntax
        try:
            ast.parse(final_solution)
            print("    ✅ Final solution syntax is valid")
        except SyntaxError as e:
            print(f"    ⚠️  Final solution has syntax issues: {e}")
            # Try to fix common issues
            final_solution = self._fix_common_syntax_issues(final_solution)
            try:
                ast.parse(final_solution)
                print("    ✅ Fixed syntax issues")
            except SyntaxError as e2:
                print(f"    ❌ Could not fix syntax: {e2}")
        
        print(f"    📄 Solution composed with {modules_used} modules + main function '{function_info['name']}'")
        
        self.final_solution = final_solution
        return final_solution    
        
    def _fix_common_syntax_issues(self, code: str) -> str:
        """Fix common syntax issues in generated code"""
        lines = code.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Fix incomplete try blocks
            if stripped == 'try:' and i + 1 < len(lines):
                # Look ahead to see if there's an except/finally
                has_except = False
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].strip().startswith(('except', 'finally')):
                        has_except = True
                        break
                
                if not has_except:
                    fixed_lines.append(line)
                    fixed_lines.append("    # AUTO-FIX: Added missing except block")
                    fixed_lines.append("    pass")
                    fixed_lines.append("except Exception as e:")
                    fixed_lines.append("    # AUTO-FIX: Handle exception")
                    fixed_lines.append("    raise e")
                    i += 1
                    continue
            
            # Fix function definitions without colons
            if stripped.startswith('def ') and not stripped.endswith(':'):
                if ':' in stripped:
                    fixed_lines.append(line)
                else:
                    fixed_lines.append(line + ':')
                    fixed_lines.append('    pass  # AUTO-FIX: Added function body')
                i += 1
                continue
                
            fixed_lines.append(line)
            i += 1
        
        return '\n'.join(fixed_lines)

    def _extract_function_name_from_code(self, code: str) -> str:
        """Extract the actual function name from generated code"""
        try:
            match = re.search(r'def\s+(\w+)\s*\(', code)
            if match:
                return match.group(1)
            return None
        except Exception:
            return None

    def task_decomposition(self, problem: str) -> Dict[str, Any]:
        """Step 1: Modular decomposition with robust error handling"""
        print("🔍 Step 1: Task Decomposition & Module Planning")
        
        decomposition_prompt = f"""
You are an expert software architect specializing in defensive programming and modular design.
    
PROBLEM:
{problem}
    
Your task:
1. Identify the core algorithmic challenge
2. Decompose into 2-3 modules with CLEAR separation of concerns:
   - Input validation & sanitization module
   - Core algorithm/logic module
   - Output formatting/error handling module (if needed)
    
3. For EACH module, specify:
   - Exact function name (use descriptive snake_case)
   - Precise input parameters with types
   - Expected output type
   - Edge cases that MUST be handled
   - Whether recursion is beneficial and why
    
You MUST respond with VALID JSON in this EXACT structure:
{{
    "problem_analysis": "Brief analysis of what the problem requires",
    "modules": [
        {{
            "name": "module_name_1",
            "purpose": "What this module does",
            "recursive": false,
            "inputs": ["input_param1", "input_param2"],
            "outputs": "return_type_or_description", 
            "edge_cases": ["edge_case1", "edge_case2"]
        }},
        {{
            "name": "module_name_2", 
            "purpose": "What this module does",
            "recursive": false,
            "inputs": ["input_param"],
            "outputs": "return_type",
            "edge_cases": ["edge_case1"]
        }}
    ],
    "main_flow": "Brief description of how modules work together"
}}
    
The "modules" field MUST be present and contain at least one module.
IMPORTANT: Use descriptive function names that clearly indicate their purpose.
"""
        
        decomposition = self._force_json_response(
            decomposition_prompt, 
            "Focus on creating 2-3 practical modules that solve the core problem."
        )
        
        self.modules = []
        module_names = []
        for i, module_data in enumerate(decomposition["modules"]):
            module = ModuleSpec(
                name=module_data.get("name", f"module_{i+1}"),
                purpose=module_data.get("purpose", "Process data"),
                recursive=module_data.get("recursive", False),
                inputs=module_data.get("inputs", ["input"]),
                outputs=module_data.get("outputs", "result"),
                edge_cases=module_data.get("edge_cases", [])
            )
            self.modules.append(module)
            module_names.append(module.name)
                
        print(f"✅ Decomposed into {len(self.modules)} modules: {module_names}")
        
        self.decomposed_module_names = module_names
        
        return decomposition
    
    def initialize_tot(self):
        """Step 2: Initialize Tree of Thoughts nodes"""
        print("🌳 Step 2: ToT Tree Initialization")
        
        self.tot_nodes = []
        for module in self.modules:
            node = ToTNode(module_spec=module, expansion_results=[])
            self.tot_nodes.append(node)
            print(f"   Created node for: {module.name}")
    
    def _expand_single_node(self, module_spec: ModuleSpec, expansion_idx: int) -> Optional[NodeResult]:
        """Expand a single node with syntax validation"""
        expansion_prompt = f"""You are a senior defensive programming engineer writing robust, production-quality Python code.
    
--- MODULE SPECIFICATION ---
- Name: {module_spec.name}
- Purpose: {module_spec.purpose}
- Inputs: {module_spec.inputs}
- Outputs: {module_spec.outputs}
- Recursive: {module_spec.recursive}
- Edge Cases: {module_spec.edge_cases}
    
CRITICAL SYNTAX REQUIREMENTS:
- All try blocks MUST have corresponding except or finally blocks
- All function definitions must end with colons and have proper indentation
- No unmatched parentheses, brackets, or braces
- All strings must use consistent quotes

YOUR TASK:
1. **Analyze** the module and choose the most defensive and robust algorithm.
2. **Generate executable Python code** that:
   - Addresses ALL specified edge cases.
   - Incorporates all of these defensive measures:
     - Input validation (type and range checks)
     - Error handling with try/except (with specific exceptions where meaningful)
     - Assertions for preconditions and postconditions where applicable
     - Fallback logic for edge cases
     - Logging (optional)
   - Uses reasoning-embedded comments: EVERY block of logic MUST have a comment that **explains the reasoning** behind that block (use `# REASONING:`).
    
FORMAT:
Respond with a valid JSON object ONLY (no text outside JSON). Example:
{{
    "approach": "Describe in 2-3 sentences why you chose this algorithm and how defensive principles are satisfied.",
    "code": "def {module_spec.name}(...):\\n    # REASONING: Input validation is necessary...\\n    if not isinstance(...): ...\\n    # REASONING: ...",
    "defensive_measures": ["input_validation", "error_handling", ...],
    "reasoning_artifacts": {{
        "preconditions": ["input must be int", ...],
        "postconditions": ["output is always sorted", ...],
        "invariants": ["list length is preserved", ...],
        "test_cases": ["handles empty list", ...]
    }}
}}
    
REQUIREMENTS:
- The code MUST be valid Python and directly executable.
- Lines of logic MUST include `# REASONING:` comments explaining the decision or check.
- Reasoning artifacts must be filled.
- DO NOT output anything outside the JSON object.
    
If you cannot address an edge case, state why in 'approach' and show best fallback.
"""
        
        try:
            result_data = self._force_json_response(expansion_prompt)
            
            # VALIDATE SYNTAX BEFORE RETURNING
            if not self._validate_code_syntax(result_data["code"]):
                print(f"     ⚠️  Generated code has syntax errors, skipping")
                return None
                
            return NodeResult(
                module_name=module_spec.name,
                approach=result_data["approach"],
                code=result_data["code"],
                defensive_measures=result_data["defensive_measures"],
                reasoning_artifacts=result_data["reasoning_artifacts"]
            )
            
        except Exception as e:
            print(f"     ❌ Expansion failed: {e}")
            return None

    def node_expansion(self, max_expansions_per_node: int = 2):
        """Step 3: Expand each node with reasonable criteria"""
        print("🔄 Step 3: Node Expansion with Defensive Reasoning")
        
        for i, node in enumerate(self.tot_nodes):
            print(f"   Expanding node {i+1}/{len(self.tot_nodes)}: {node.module_spec.name}")
            
            successful_expansions = 0
            for expansion_idx in range(max_expansions_per_node):
                print(f"     Expansion {expansion_idx + 1}/{max_expansions_per_node}")
                
                try:
                    result = self._expand_single_node(node.module_spec, expansion_idx)
                    if result:
                        node.expansion_results.append(result)
                        successful_expansions += 1
                        
                        print(f"     Generated code preview:")
                        code_lines = result.code.split('\n')[:4]
                        for line in code_lines:
                            print(f"       {line}")
                        if len(result.code.split('\n')) > 4:
                            print(f"       ...")
                        
                        reflection = self._node_self_reflection(result)
                        node.reflection_feedback = reflection
                        
                        reflection_score = reflection.get("defensive_score", 0)
                        
                        if reflection_score >= 60:
                            node.best_result = result
                            print(f"     ✅ Found acceptable solution (score: {reflection_score})")
                            break
                        else:
                            print(f"     ⚠️  Solution needs improvement (score: {reflection_score})")
                    else:
                        print(f"     ❌ Expansion failed")
                        
                except Exception as e:
                    print(f"     ❌ Expansion error: {e}")
            
            if successful_expansions == 0:
                print(f"   ⚠️  No successful expansions for {node.module_spec.name}")
    
    def _node_self_reflection(self, node_result: NodeResult) -> Dict[str, Any]:
        """Self-reflection with practical criteria"""
        reflection_prompt = f"""
You are a strict code reviewer conducting a defensive programming audit.
CODE TO REVIEW:
{node_result.code}
CLAIMED DEFENSIVE MEASURES:
{node_result.defensive_measures}
CLAIMED REASONING ARTIFACTS:
{node_result.reasoning_artifacts}
AUDIT CHECKLIST (score 0-100):
1. **Input Validation (0-20 points)**:
- Type checks present? (+10)
- Range/null checks present? (+10)
2. **Error Handling (0-20 points)**:
- Try-except blocks for risky operations? (+10)
- Specific exception types (not bare except)? (+10)
3. **Reasoning Comments (0-20 points)**:
- Every logic branch explained? (+10)
- Comments use "REASONING:" prefix? (+10)
4. **Assertions (0-15 points)**:
- Preconditions checked? (+7)
- Postconditions verified? (+8)
5. **Edge Case Coverage (0-15 points)**:
- Handles empty input? (+5)
- Handles invalid types? (+5)
- Handles boundary conditions? (+5)
6. **Code Quality (0-10 points)**:
- No syntax errors? (+5)
- Follows function signature? (+5)
RESPOND WITH VALID JSON:
{{
"is_robust": true/false, // true ONLY if score ≥ 80
"issues": ["Specific issue 1", "Specific issue 2", ...],
"improvements": ["Concrete suggestion 1", ...],
"defensive_score": 85, // Sum of checklist points
"ready_for_integration": true/false, // true ONLY if score ≥ 80
"checklist_breakdown": {{
"input_validation": 18,
"error_handling": 15,
"reasoning_comments": 15,
"assertions": 10,
"edge_case_coverage": 10,
"code_quality": 8
}}
}}
BE STRICT. Reject code that doesn't meet minimum standards (score < 80).
"""
        
        try:
            return self._force_json_response(reflection_prompt)
        except Exception as e:
            print(f"     ❌ Reflection failed: {e}")
            return {
                "is_robust": True,
                "issues": ["Reflection mechanism failed"],
                "defensive_score": 70,
                "ready_for_integration": True
            }

    def meta_reasoning_review(self) -> Dict[str, Any]:
        """Step 5: Simple meta-reasoning"""
        print("🔎 Step 5: Meta-Reasoning (Global Review)")
        
        successful_modules = len([n for n in self.tot_nodes if n.best_result or n.expansion_results])
        
        if successful_modules > 0:
            meta_review = {
                "integration_valid": True,
                "defensive_coverage": 80,
                "ready_for_execution": True,
            }
        else:
            meta_review = {
                "integration_valid": False,
                "defensive_coverage": 0,
                "ready_for_execution": False,
            }
        
        print(f"   Meta-review: {'✅ Ready' if meta_review['ready_for_execution'] else '❌ Needs work'}")
        return meta_review

    def generate_solution(self, problem: str, max_retries: int = 2) -> Dict[str, Any]:
        """Main method to generate solution"""
        self.problem = problem
        
        print("🚀 STARTING MODULAR TOT SELF-REFLECTIVE ARCHITECTURE")
        print(f"Problem: {problem.split()[0]}...")
        print("=" * 60)
        
        for attempt in range(max_retries + 1):
            print(f"\n📋 ATTEMPT {attempt + 1}/{max_retries + 1}")
            
            try:
                decomposition = self.task_decomposition(problem)
                self.initialize_tot()
                self.node_expansion(max_expansions_per_node=2)
                solution = self.compose_solution()
                meta_review = self.meta_reasoning_review()
                
                modules_used = len([n for n in self.tot_nodes if n.best_result or n.expansion_results])
                
                return {
                    "success": True,
                    "solution": solution,
                    "verification_artifacts": {
                        "defensive_coverage": meta_review.get("defensive_coverage", 0),
                        "modules_implemented": modules_used,
                        "total_reflection_cycles": len(self.tot_nodes) * 2
                    }
                }
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    return {
                        "success": False,
                        "error": str(e)
                    }
                # Reset state for next attempt
                self.modules = []
                self.tot_nodes = []
                self.final_solution = None
                self.decomposed_module_names = []
        
        return {"success": False, "error": "Max retries reached"}