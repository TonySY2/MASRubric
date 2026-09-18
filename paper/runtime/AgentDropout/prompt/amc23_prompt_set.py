from typing import Dict, Any
import itertools
from AgentDropout.prompt.prompt_set import PromptSet
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry
from AgentDropout.prompt.common import get_combine_materials
from AgentDropout.prompt.math_team_variant import apply_math_team_variant

roles = itertools.cycle(['Math Solver',
                         'Mathematical Analyst',
                         'Programming Expert',
                         'Inspector',])

# Revised descriptions for AMC 23 (Competitive Math, Trick-heavy, High Precision)
ROLE_DESCRIPTION = {
    "Math Solver": 
        "You are a competitive math expert specializing in AMC (American Mathematics Competitions) problems. "
        "You will be given a problem covering Algebra, Geometry, Number Theory, or Combinatorics. "
        "Hints from other agents may be provided. "
        "Give your own solving process step by step. Use specific competitive math techniques (e.g., Casework, Complementary Counting, Modular Arithmetic, Coordinate Geometry) where appropriate. "
        "Use LaTeX for mathematical expressions. "
        "The last line of your output must contain the final answer boxed in LaTeX, for example: \\boxed{answer}\n",
    "Mathematical Analyst":
        "You are a mathematical analyst for math competitions. "
        "You will be given an AMC-level problem, analysis, and code from other agents. "
        "Analyze the problem structure. Identify the domain (Algebra, Geometry, Number Theory, Combinatorics) and the most efficient strategy (e.g., 'Is there a symmetry?', 'Can we simplify small cases?', 'Is this a stars and bars problem?'). "
        "Perform symbolic derivations to simplify the problem before calculation. "
        "Use LaTeX for mathematical expressions. "
        "The last line of your output must contain the final answer boxed in LaTeX, for example: \\boxed{answer}\n",
    "Programming Expert":
        "You are a programming expert assisting in solving math competition problems. "
        "You will be given a math problem, analysis, and code from other agents. "
        "Use Python to solve the problem computationally or to verify analytical results. "
        "AMC problems often require: 1. Brute-force enumeration within limits. 2. Number theory checks (primality, gcd). 3. Simulation for probability. "
        "WARNING: Avoid floating-point errors. Use `fractions.Fraction` or integer arithmetic whenever possible. "
        "Write a function to solve the problem. The function should not take arguments and return the final result. "
        "The last line of code calls the function and assigns the return value to the variable `answer`. "
        "Use a Python code block. For example:\n```python\nimport sympy\ndef solve():\n # Logic here\n return result\nanswer = solve()\n```\n"
        "Do not include anything other than Python code blocks in your response.",
    "Inspector":
        "You are an Inspector and Proctor. "
        "You will be given an AMC math problem, analysis, and code from other agents. "
        "Check for logical fallacies, calculation errors, and missed edge cases (e.g., n=0, n=1, degenerate triangles). "
        "Verify if the solution considers all constraints (e.g., 'positive integers', 'distinct values'). "
        "Give your own solving process step by step if you detect errors. "
        "The last line of your output must contain the final answer boxed in LaTeX, for example: \\boxed{answer}\n",
}

DESCRIPTION = {
    "Math Solver": "An agent for solving competitive math problems using standard competition techniques and rigorous derivation.",
    "Mathematical Analyst": "An agent for identifying problem types, symmetries, and optimal strategies (Algebra/Geometry/Number Theory/Combinatorics).",
    "Programming Expert": "An agent for solving discrete math and number theory problems using Python, emphasizing integer/exact arithmetic.",
    "Inspector": "An agent for verifying logic, checking constraints, and avoiding common competitive math traps.",
}


FEW_SHOT_DATA = {
"Math Solver":"""""",

"Mathematical Analyst":"""""",

"Programming Expert":
"""
Q: Find the sum of all positive integers n such that n divides n^2 + 3.
A:
```python\n
def solve_divisibility():
    # Logic: n | n^2 + 3 implies n | 3 because n | n^2.
    # The divisors of 3 are 1 and 3.
    # Sum = 1 + 3 = 4
    possible_n = []
    for n in range(1, 1000): # Check a reasonable range
        if (n**2 + 3) % n == 0:
            possible_n.append(n)
    return sum(possible_n)
\n```

# Example call
answer = solve_divisibility()

Q: How many 3-digit numbers are divisible by 7?
A:
```python\n
def count_multiples():
    count = 0
    for i in range(100, 1000):
        if i % 7 == 0:
            count += 1
    return count
\n```

# Example call
answer = count_multiples()
""",
"Inspector":"""""",
}

apply_math_team_variant(
    ROLE_DESCRIPTION,
    DESCRIPTION,
    FEW_SHOT_DATA,
    task_label="AMC competition problems",
    final_answer_instruction=(
        "The last line of your output must contain the final answer boxed in LaTeX, "
        "for example: \\boxed{answer}\n"
    ),
)


@PromptSetRegistry.register('amc23') 
class AMC23PromptSet(PromptSet):

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
        # Format the question for the AI assistant to answer
        return f"{FEW_SHOT_DATA[role]}\n\nQ:{question}"

    @staticmethod
    def get_decision_constraint():
        return (
        "You will be given an AMC math problem, analysis, and code from other agents. "
        "Compare the approaches (Symbolic vs Computational). "
        "For Geometry/Algebra, trust rigorous derivation. For Number Theory/Combinatorics, cross-check code simulation with logic. "
        "Determine the correct answer. The answer is usually an integer or a simple fraction. "
        "The last line of your output must contain ONLY the final answer boxed in LaTeX, for example: \\boxed{answer}"
        )

    @staticmethod
    def get_decision_role():
        return "You are the head judge of a math competition. You synthesize results to ensure the final answer is robust and free of trap errors."

    @staticmethod
    def get_decision_few_shot():
        return """"""

    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Here is an unsuccessful attempt for solving the following AMC math problem:

    Question:
    {question}
    Attempted Solution:
    {solution}
    Feedback:
    {feedback}
    Rewrite the solution (and code if applicable) based on the feedback to correctly solve:
    {question}
    Ensure the final answer is boxed in LaTeX: \boxed{{answer}}"""

    @staticmethod
    def get_query_prompt(question):
        return (
            "# Mathematical Information Gathering\n\n"
            "Evaluate if the problem requires external knowledge (e.g., specific properties of 2023, theorems like Shoelace or Ptolemy) or if it is self-contained. "
            "AMC problems are usually self-contained but may rely on specific theorems.\n\n"
            f"## ❓ Target Problem:\n{question}\n\n"
            "## 🔍 Theorems/Concepts to Verify:\n"
            "Identify critical mathematical theorems or properties needed (e.g., 'Divisibility rules', 'Area of polygon coordinates').\n"
        )

    @staticmethod
    def get_file_analysis_prompt(query, file):
        return (
            "# Math Context Analysis\n\n"
            f"## 🔍 Concept/Theorem to verify:\n---\n{query}\n---\n\n"
            f"## 📄 Reference Content:\n---\n{file}\n---\n\n"
            "## 📝 Instructions:\n"
            "1. Extract the definition or formula relevant to the query.\n"
            "2. Ensure the conditions for applying the theorem match the problem state (e.g., cyclic quadrilateral, integer constraints).\n"
        )

    @staticmethod
    def get_websearch_prompt(question, query):
        return (
            "# Math Web Search\n\n"
            f"## Original Problem: \n---\n{question}\n---\n\n"
            f"## 🔍 Search Objective:\n---\n{query}\n---\n\n"
            "## 🌐 Instructions:\n"
            "Generate specific math search queries. Example: 'sum of coefficients of expansion', 'number of divisors of 2023'.\n"
            "Format: 'query1, query2, query3'."
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
            "1. Extract precise formulas, definitions, or constant values.\n"
            "2. Ignore general tutorial text; focus on the mathematical facts.\n"
            "3. If the results are irrelevant, state 'No useful mathematical info found'.\n"
        )

    @staticmethod
    def get_reflect_prompt(question, answer):
        return (
            "# Reflection on Math Solution\n\n"
            f"## 🤔 Problem:\n---\n{question}\n---\n\n"
            f"## 💡 Proposed Answer:\n---\n{answer}\n---\n\n"
            "## ✏️ Instructions:\n"
            "1. Check for calculation errors (especially signs and arithmetic).\n"
            "2. Verify if the answer format matches the requirement (boxed LaTeX).\n"
            "3. Check edge cases and constraints (e.g., 'positive integers', 'distinct').\n"
            "4. Ensure the answer makes sense (e.g., probability between 0 and 1, counts are integers)."
        )

    @staticmethod
    def get_self_consistency(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
            "# Self-Consistency Evaluation\n\n"
            f"## 🤔 Problem:\n---\n{question}\n---\n\n"
            f"## 💡 Candidate Answers:\n---\n{formatted_answers}\n---\n\n"
            "## 📋 Instructions:\n"
            "1. Compare the final boxed values in the answers.\n"
            "2. Select the answer that appears most frequently (majority vote) or has the most rigorous derivation.\n"
            "3. Ensure the selected answer is in \\boxed{...} format.\n"
            f"4. Constraints: {constraint}.\n"
        )

    @staticmethod
    def get_select_best(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
            "# Best Math Solution Selection\n\n"
            f"## 🤔 Problem:\n---\n{question}\n---\n\n"
            f"## 💡 Candidate Solutions:\n---\n{formatted_answers}\n---\n\n"
            "## 📋 Instructions:\n"
            "1. Verify the logical steps of each solution.\n"
            "2. Check for common algebraic or arithmetic mistakes.\n"
            "3. Choose the solution that is mathematically sound and complete.\n"
            "4. Copy the final answer strictly in \\boxed{...} format.\n"
            f"5. Constraints: {constraint}.\n"
        )

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
