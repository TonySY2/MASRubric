from typing import Dict, Any
import itertools
from masrubric.prompt.prompt_set import PromptSet
from masrubric.prompt.prompt_set_registry import PromptSetRegistry
from masrubric.prompt.common import get_combine_materials


roles = itertools.cycle(['Math Solver',
                         'Mathematical Analyst',
                         'Programming Expert',
                         'Inspector',])


ROLE_DESCRIPTION = {
    "Math Solver": 
        "You are an expert mathematician specializing in high-level Olympiad competitions (IMO, USAMO). "
        "You will be given a challenging problem (Number Theory, Algebra, Geometry, or Combinatorics). "
        "1. Solve the problem step by step with rigorous proofs. "
        "2. Address all cases. If finding all solutions, ensure no solution is missed and no extraneous solution is included. "
        "3. Prove both necessity and sufficiency where required. "
        "4. Use LaTeX for mathematical expressions. "
        "The last line of your output must contain the final result boxed in LaTeX. "
        "For sets of solutions/tuples, format strictly as: \\boxed{(x_1, y_1), (x_2, y_2)}.\n",
        
    "Mathematical Analyst":
        "You are a mathematical analyst. "
        "1. Analyze the problem structure. Identify the subfield (e.g., Diophantine Equations, Functional Equations) and appropriate strategies (e.g., Modular Arithmetic, Bounding, Induction, Vieta Jumping). "
        "2. Break down complex conditions into manageable logical lemmas. "
        "3. Critique the approach: Are there edge cases? Is the bounding tight enough? "
        "Use LaTeX for mathematical expressions. "
        "The last line of your output must contain the final result boxed in LaTeX.\n",
        
    "Programming Expert":
        "You are a computational mathematics expert. "
        "Olympiad problems often cannot be solved purely by brute-force code, but code can find patterns or verify small cases to guide the proof. "
        "1. Write Python code to search for solutions in a reasonable range (e.g., check x, y < 100) to generate conjectures. "
        "2. Use SymPy for symbolic manipulation if algebraic expressions are messy. "
        "3. If the problem asks for a specific value, try to compute it numerically to verify the theoretical result. "
        "The last line of code should print or return the findings. "
        "Use a Python code block. Example:\n```python\ndef check_cases():\n  results = []\n  for x in range(1, 50):\n    if check(x): results.append(x)\n  return results\nprint(check_cases())\n```\n",
        
    "Inspector":
        "You are a rigorous Proof Auditor. "
        "Review the proposed solutions from other agents. "
        "1. Check for logical fallacies: Circular reasoning, division by zero, assuming what needs to be proved. "
        "2. Check for completeness: Did the solver find *all* solutions? Did they prove *uniqueness* if applicable? "
        "3. If specific examples were found by the programmer, ensure the mathematical solution accounts for them. "
        "The last line of your output must contain the corrected final answer boxed in LaTeX.\n",
}

DESCRIPTION = {
    "Math Solver": "An expert agent for solving Olympiad problems with rigorous proofs and complete solution sets.",
    "Mathematical Analyst": "An agent for strategic analysis, breaking down problems into lemmas and bounding conditions.",
    "Programming Expert": "An agent using Python to find patterns, verify small cases, or handle complex symbolic algebra.",
    "Inspector": "An agent for auditing logical gaps, ensuring completeness (all solutions found) and correctness.",
}

FEW_SHOT_DATA = {
"Math Solver":"""""",
"Mathematical Analyst":"""""",
"Programming Expert":
"""
Q: Find all pairs of integers (x, y) such that x^2 + y^2 = 25.
A:
```python
def find_pairs():
    solutions = []
    # Search in a reasonable range since x^2 <= 25 implies |x| <= 5
    for x in range(-5, 6):
        for y in range(-5, 6):
            if x**2 + y**2 == 25:
                solutions.append((x, y))
    return solutions
answer = find_pairs()
print(answer)
""",
"Inspector":"""""",
}

@PromptSetRegistry.register('olymMATH')
class OlymMATHPromptSet(PromptSet):
    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_constraint(role):
        return ROLE_DESCRIPTION[role]

    @staticmethod
    def get_description(role):
        return DESCRIPTION[role]

    @staticmethod
    def get_format():
        return "natural language with LaTeX"

    @staticmethod
    def get_answer_prompt(question, role="Mathematical Analyst"):
   
        prefix = FEW_SHOT_DATA.get(role, "")
        if prefix:
            return f"{prefix}\n\nQ:{question}"
        return f"Q:{question}"

    @staticmethod
    def get_decision_constraint():
   
        return (
        "You will be given a challenging Olympiad-level math problem and inputs from various agents. "
        "Synthesize the correct final answer based on the most rigorous proof and complete analysis. "
        "Ensure no solutions are missed and checks for 'necessary and sufficient' conditions are met. "
        "If the answer is a set of values or tuples, list them all inside the box. "
        "The last line of your output must contain ONLY the final answer boxed in LaTeX. "
        "Example: \\boxed{(1, 8, 19), (2, 7, 13)}"
        )

    @staticmethod
    def get_decision_role():
        return "You are the Head Judge of a Mathematical Olympiad. You synthesize proofs and computations to determine the logically perfect solution."

    @staticmethod
    def get_decision_few_shot():
        return """"""

    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Here is an unsuccessful attempt for solving the following Olympiad problem:

    Question:
    {question}
    Attempted Solution:
    {solution}
    Feedback:
    {feedback}
    Rewrite the solution based on the feedback. Address the logical gaps or missing cases identified.
    Ensure the final answer is boxed in LaTeX: \\boxed{{answer}}"""

 
    @staticmethod
    def get_query_prompt(question):
        return (
            "# Mathematical Strategy & Knowledge Retrieval\n\n"
            "Evaluate if the problem requires specific lemmas or theorems (e.g., LTE, Vieta Jumping, Pell's Equation). "
            f"## ❓ Target Problem:\n{question}\n\n"
            "## 🔍 Concepts to Verify:\n"
            "Identify advanced theorems or similar known problems that could provide a solution path.\n"
        )

    @staticmethod
    def get_file_analysis_prompt(query, file):
        return (
            "# Math Context Analysis\n\n"
            f"## 🔍 Concept/Theorem to verify:\n---\n{query}\n---\n\n"
            f"## 📄 Reference Content:\n---\n{file}\n---\n\n"
            "## 📝 Instructions:\n"
            "1. Extract the precise definition, conditions, or formula.\n"
            "2. Check specific conditions (e.g., 'p must be an odd prime') and apply to the current problem.\n"
        )

    @staticmethod
    def get_websearch_prompt(question, query):
        return (
            "# Math Web Search\n\n"
            f"## Original Problem: \n---\n{question}\n---\n\n"
            f"## 🔍 Search Objective:\n---\n{query}\n---\n\n"
            "## 🌐 Instructions:\n"
            "Generate specific search queries for similar Olympiad problems or specific theorem applications.\n"
        )

    @staticmethod
    def get_adversarial_answer_prompt(question):
        pass

    @staticmethod
    def get_distill_websearch_prompt(question, query, results):
        return (
            "# Summary of Mathematical Concepts\n\n"
            f"## Original Problem: \n---\n{question}\n---\n\n"
            f"## 🔍 Concept Searched:\n---\n{query}\n---\n\n"
            f"## 🌐 Search Results:\n---\n{results}\n---\n\n"
            "## 📝 Instructions:\n"
            "1. Extract the solution technique or theorem statement.\n"
            "2. If a similar problem is found, extract the key insight (the 'trick').\n"
        )

    @staticmethod
    def get_reflect_prompt(question, answer):
        return (
            "# Reflection on Olympiad Solution\n\n"
            f"## 🤔 Problem:\n---\n{question}\n---\n\n"
            f"## 💡 Proposed Answer:\n---\n{answer}\n---\n\n"
            "## ✏️ Instructions:\n"
            "1. **Verification**: Substitute the answer back into the original equation/condition. Does it satisfy it?\n"
            "2. **Completeness**: Did we find *all* solutions? Did we rule out other possibilities rigorously?\n"
            "3. **Format**: Is the final answer boxed correctly?\n"
        )

    @staticmethod
    def get_self_consistency(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Solution {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
            "# Self-Consistency & Consensus Evaluation\n\n"
            f"## 🤔 Problem:\n---\n{question}\n---\n\n"
            f"## 💡 Candidate Solutions:\n---\n{formatted_answers}\n---\n\n"
            "## 📋 Instructions:\n"
            "1. Compare the final boxed sets/values.\n"
            "2. Check which solution covers all cases and has the most rigorous logic.\n"
            "3. Select the most complete and correct solution.\n"
            "4. Output the final result in \\boxed{...} format.\n"
            f"5. Context/Constraint: {constraint}.\n"
        )

    @staticmethod
    def get_select_best(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Solution {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
            "# Best Math Solution Selection\n\n"
            f"## 🤔 Problem:\n---\n{question}\n---\n\n"
            f"## 💡 Candidate Solutions:\n---\n{formatted_answers}\n---\n\n"
            "## 📋 Instructions:\n"
            "1. Verify the logical steps. Are 'necessary' and 'sufficient' parts both present?\n"
            "2. Check for subtle errors (e.g., losing solutions during division, extraneous solutions during squaring).\n"
            "3. Choose the solution that is mathematically sound and complete.\n"
            "4. Copy the final answer strictly in \\boxed{...} format.\n"
            f"5. Context: {constraint}.\n"
        )

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)