from typing import Dict, Any
import itertools
from masrubric.prompt.prompt_set import PromptSet
from masrubric.prompt.prompt_set_registry import PromptSetRegistry
from masrubric.prompt.common import get_combine_materials

roles = itertools.cycle(['Math Solver',
                         'Mathematical Analyst',
                         'Programming Expert',
                         'Inspector',])

# Revised descriptions for MATH500 (Open-ended, LaTeX heavy)
ROLE_DESCRIPTION = {
    "Math Solver": 
        "You are a math expert. "
        "You will be given a math problem (ranging from Algebra to Calculus) and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "Use LaTeX for mathematical expressions. "
        "The last line of your output must contain the final answer boxed in LaTeX, for example: \\boxed{answer}\n",
    "Mathematical Analyst":
        "You are a mathematical analyst. "
        "You will be given a math problem, analysis and code from other agents. "
        "You need to first analyze the problem-solving process step by step, identifying the core mathematical concepts and theorems required. "
        "Perform symbolic derivations where necessary. "
        "Use LaTeX for mathematical expressions. "
        "The last line of your output must contain the final answer boxed in LaTeX, for example: \\boxed{answer}\n",
    "Programming Expert":
        "You are a programming expert. "
        "You will be given a math problem, analysis and code from other agents. "
        "Integrate step-by-step reasoning and Python code (using libraries like sympy, numpy, math) to verify or solve the problem. "
        "Write a function to solve the problem computationally. "
        "The function should not take any arguments and use the final result as the return value. "
        "The last line of code calls the function you wrote and assigns the return value to the variable `answer`. "
        "Use a Python code block to write your response. For example:\n```python\nimport sympy\ndef fun():\n x = sympy.Symbol('x')\n result = sympy.solve(x**2 - 4, x)\n return result\nanswer = fun()\n```\n"
        "Do not include anything other than Python code blocks in your response.",
    "Inspector":
        "You are an Inspector. "
        "You will be given a math problem, analysis and code from other agents. "
        "Check whether the logic/calculation of the problem solving and analysis process is correct. "
        "Verify if the derivation steps strictly follow mathematical rules. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output must contain the final answer boxed in LaTeX, for example: \\boxed{answer}\n",
}

DESCRIPTION = {
    "Math Solver": "An agent for solving math problems step by step using rigorous derivation, providing a boxed LaTeX answer.",
    "Mathematical Analyst": "An agent for analyzing math problems symbolically, identifying theorems, and providing a boxed LaTeX answer.",
    "Programming Expert": "An agent for solving math problems with Python (often SymPy), writing code to compute or verify the result.",
    "Inspector": "An agent for verifying math solutions and code, checking for logical fallacies, and providing a corrected boxed LaTeX answer.",
}


FEW_SHOT_DATA = {
"Math Solver":"""""",

"Mathematical Analyst":"""""",

"Programming Expert":
"""
Q: Write a function to find the sum of squares of the first n natural numbers.
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

Q: Write a function to check if a given string is a palindrome.
A:
```python\n
def is_palindrome(s):
    # Remove non-alphanumeric characters and convert to lowercase
    clean_s = ''.join(char.lower() for char in s if char.isalnum())
    return clean_s == clean_s[::-1]
\n```

# Example call
answer = is_palindrome("Racecar")
""",
"Inspector":"""""",
}




@PromptSetRegistry.register('math500') # Registered as 'math' for general math tasks like MATH500
class MATH500PromptSet(PromptSet):


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
    def get_answer_prompt(question,role="Mathematical Analyst"):
        # Format the question for the AI assistant to answer
        return f"{FEW_SHOT_DATA[role]}\n\nQ:{question}"

    @staticmethod
    def get_decision_constraint():
        return (
        "You will be given a math problem, analysis, and code from other agents. "
        "Compare the different approaches (symbolic, computational, logical). "
        "Determine the correct answer based on the most rigorous derivation or verification. "
        "The last line of your output must contain ONLY the final answer boxed in LaTeX, for example: \\boxed{answer}"
        )

    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker. Good at verifying mathematical proofs, checking Python code results, and synthesizing a final correct answer formatted in LaTeX."


    @staticmethod
    def get_decision_few_shot():
        return """"""


    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Here is an unsuccessful attempt for solving the following math problem:

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
            "Evaluate if the problem requires external knowledge (e.g., specific physical constants, obscure theorems) or if it is self-contained. "
            "Most MATH500 problems are self-contained. If you need to verify a formula or theorem, outline the query.\n\n"
            f"## ❓ Target Problem:\n{question}\n\n"
            "## 🔍 Theorems/Concepts to Verify:\n"
            "Identify critical mathematical theorems (e.g., 'Shoelace Theorem', 'Euler's Totient Theorem') that might be needed.\n"
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
            "Generate specific math search queries. Example: 'formula for volume of tetrahedron given vertices', 'roots of unity properties'.\n"
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
            "1. Check for calculation errors.\n"
            "2. Verify if the answer format matches the requirement (boxed LaTeX).\n"
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
