NODE_REFLECTION_PROMPT = """
You are a strict code reviewer conducting a defensive programming audit.
CODE TO REVIEW:
{code}
CLAIMED DEFENSIVE MEASURES:
{defensive_measures}
CLAIMED REASONING ARTIFACTS:
{reasoning_artifacts}
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

META_REASONING_PROMPT = """
You are a senior architect conducting final integration review.
FULL SOLUTION:
{full_solution}
MODULES SUMMARY:
{modules_summary}
INTEGRATED CODE:
{integrated_code}
VERIFICATION TASKS:
1. **Inter-Module Consistency**:
- Do output types of module N match input types of module N+1?
- Are all module function calls valid?
- Is the main orchestrator function correctly chaining modules?
2. **Global Defensive Coverage**:
- Is there at least ONE input validation module?
- Are exceptions propagated correctly?
- Are there logging/debugging aids?
3. **Completeness**:
- Are all required imports present?
- Are all edge cases from decomposition handled?
- Is the solution executable without modification?
4. **Execution Readiness**:
- Can ast.parse() succeed on the code?
- Are function signatures consistent with problem requirements?
RESPOND WITH VALID JSON:
{{
"integration_valid": true/false,
"defensive_coverage": 85, // 0-100 scale
"issues": ["Specific integration issue 1", ...],
"ready_for_execution": true/false,
"execution_risks": ["Potential runtime issue 1", ...],
"recommendations": ["Improvement 1", ...],
"module_consistency_check": {{
"type_chain_valid": true/false,
"function_calls_valid": true/false,
"imports_complete": true/false
}}
}}
Approve (ready_for_execution=true) ONLY if defensive_coverage ≥ 75 AND
integration_valid=true.
"""
