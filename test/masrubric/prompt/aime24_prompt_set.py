from typing import Dict, Any
import itertools
from masrubric.prompt.prompt_set import PromptSet
from masrubric.prompt.prompt_set_registry import PromptSetRegistry
from masrubric.prompt.common import get_combine_materials

roles = itertools.cycle(['Math Solver',
                         'Mathematical Analyst',
                         'Programming Expert',
                         'Inspector',])

# Revised descriptions for AIME24 (High difficulty, Integer answers)
ROLE_DESCRIPTION = {
    "Math Solver": 
        "You are an expert math solver specializing in AIME competitions. "
        "You will be given a challenging math problem and hints from other agents. "
        "Solve the problem step by step with rigorous logic. "
        "Use LaTeX for mathematical expressions. "
        "IMPORTANT: The final answer must be a non-negative integer between 0 and 999. "
        "The last line of your output must contain the final answer boxed in LaTeX, for example: \\boxed{123}\n",
    "Mathematical Analyst":
        "You are a mathematical analyst. "
        "You will be given an AIME problem, analysis, and code from other agents. "
        "Analyze the problem-solving process. Identify core mathematical concepts (Number Theory, Geometry, Combinatorics, Algebra). "
        "Perform symbolic derivations to ensure the logic holds up under strict mathematical rules. "
        "Use LaTeX for mathematical expressions. "
        "IMPORTANT: The final answer must be a non-negative integer between 0 and 999. "
        "The last line of your output must contain the final answer boxed in LaTeX, for example: \\boxed{123}\n",
    "Programming Expert":
        "You are a programming expert assisting in AIME competition problems. "
        "You will be given a math problem and analysis from other agents. "
        "Use Python to computationally verify the math logic or solve complex combinatorial/number theory parts. "
        "Write a function to solve the problem. "
        "The function should return the final result (which should be an integer). "
        "The last line of code calls the function and assigns the return value to the variable `answer`. "
        "Use a Python code block to write your response. For example:\n```python\nimport sympy\ndef fun():\n x = sympy.Symbol('x')\n result = sympy.solve(x**2 - 4, x)\n return result\nanswer = fun()\n```\n"
        "Do not include anything other than Python code blocks in your response.",
    "Inspector":
        "You are an Inspector validating AIME solutions. "
        "You will be given a math problem, analysis, and code from other agents. "
        "Check for logical fallacies, calculation errors, and edge cases. "
        "Verify if the derivation leads to a valid AIME answer (Integer 0-999). "
        "Give your own solving process based on these checks. "
        "The last line of your output must contain the final answer boxed in LaTeX, for example: \\boxed{123}\n",
}

DESCRIPTION = {
    "Math Solver": "An expert agent for solving AIME problems with rigorous derivation, providing an integer result.",
    "Mathematical Analyst": "An agent for analyzing mathematical structures and theorems, ensuring the result fits AIME constraints.",
    "Programming Expert": "An agent for using Python (SymPy/NumPy) to compute or verify results for hard math problems.",
    "Inspector": "An agent for auditing the logic and ensuring the final answer is a valid AIME integer (0-999).",
}


FEW_SHOT_DATA = {
"Math Solver":"""""",

"Mathematical Analyst":"""""",

"Programming Expert":
"""
Q: Find the sum of squares of the first 5 natural numbers.
A:
```python\n
def sum_of_squares(n):
    sum_val = 0
    for i in range(1, n + 1):
        sum_val += i * i
    return sum_val
\n```
# Example call
answer = sum_of_squares(5)

Q: Check if "Racecar" is a palindrome.
A:
```python\n
def is_palindrome(s):
    clean_s = ''.join(char.lower() for char in s if char.isalnum())
    return clean_s == clean_s[::-1]
\n```
# Example call
answer = is_palindrome("Racecar")
""",
"Inspector":"""""",
}


@PromptSetRegistry.register('aime24')
class AIME24PromptSet(PromptSet):

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
        return f"{FEW_SHOT_DATA[role]}\n\nQ:{question}"

    @staticmethod
    def get_decision_constraint():
        return (
        "You will be given a challenging AIME math problem, analysis, and code from other agents. "
        "Compare the different approaches. "
        "Determine the correct answer based on the most rigorous derivation. "
        "IMPORTANT: The AIME answer format is always an integer between 000 and 999. "
        "If different agents disagree, prioritize the one with the soundest mathematical proof or successful code execution. "
        "The last line of your output must contain ONLY the final answer boxed in LaTeX, for example: \\boxed{123}"
        )

    @staticmethod
    def get_decision_role():
        return "You are the head judge of an AIME competition. You synthesize multiple viewpoints to determine the single correct integer answer."

    @staticmethod
    def get_decision_few_shot():
        return """"""

    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Here is an unsuccessful attempt for solving the following AIME problem:

    Question:
    {question}
    Attempted Solution:
    {solution}
    Feedback:
    {feedback}
    Rewrite the solution (and code if applicable) based on the feedback to correctly solve:
    {question}
    Ensure the final answer is a valid AIME integer (0-999) and boxed in LaTeX: \boxed{{answer}}"""

    @staticmethod
    def get_query_prompt(question):
        return (
            "# Mathematical Information Gathering\n\n"
            "Evaluate if the problem requires external knowledge (e.g., properties of specific numbers, advanced theorems). "
            f"## ❓ Target Problem:\n{question}\n\n"
            "## 🔍 Theorems/Concepts to Verify:\n"
            "Identify critical mathematical theorems (e.g., 'Ptolemy's Theorem', 'Lucas Theorem') that might be needed.\n"
        )

    @staticmethod
    def get_file_analysis_prompt(query, file):
        return (
            "# Math Context Analysis\n\n"
            f"## 🔍 Concept/Theorem to verify:\n---\n{query}\n---\n\n"
            f"## 📄 Reference Content:\n---\n{file}\n---\n\n"
            "## 📝 Instructions:\n"
            "1. Extract the definition or formula relevant to the query.\n"
            "2. Ensure the conditions for applying the theorem match the problem state.\n"
        )

    @staticmethod
    def get_websearch_prompt(question, query):
        return (
            "# Math Web Search\n\n"
            f"## Original Problem: \n---\n{question}\n---\n\n"
            f"## 🔍 Search Objective:\n---\n{query}\n---\n\n"
            "## 🌐 Instructions:\n"
            "Generate specific math search queries. Example: 'recurrence relation for problem X', 'properties of cyclic quadrilaterals'.\n"
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
        )

    @staticmethod
    def get_reflect_prompt(question, answer):
        return (
            "# Reflection on AIME Solution\n\n"
            f"## 🤔 Problem:\n---\n{question}\n---\n\n"
            f"## 💡 Proposed Answer:\n---\n{answer}\n---\n\n"
            "## ✏️ Instructions:\n"
            "1. Check for calculation errors.\n"
            "2. **CRITICAL**: Verify if the final answer is an integer between 0 and 999. If not, re-check the problem statement or calculations.\n"
            "3. Check edge cases (e.g., division by zero, domain restrictions).\n"
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
            "2. Select the answer that appears most frequently (majority vote).\n"
            "3. Ensure the selected answer is an integer [0, 999] in \\boxed{...} format.\n"
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
            "2. Check for common AIME pitfalls (e.g. counting errors, modulo arithmetic).\n"
            "3. Choose the solution that is mathematically sound and complete.\n"
            "4. Copy the final answer strictly in \\boxed{...} format (Must be integer 0-999).\n"
            f"5. Constraints: {constraint}.\n"
        )

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)