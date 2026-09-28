from typing import Dict, Any
import itertools
from masrubric.prompt.prompt_set import PromptSet
from masrubric.prompt.prompt_set_registry import PromptSetRegistry
from masrubric.prompt.common import get_combine_materials

roles = itertools.cycle(['Project Manager',
                         'Algorithm Designer',
                         'Programming Expert',
                         'Test Analyst',
                         'Bug Fixer',])

# Revised descriptions for CodeContests (Competitive Programming)
ROLE_DESCRIPTION = {
    "Project Manager": 
        "You are a project manager for competitive programming tasks. "
        "You will be given a problem description, including input/output formats and constraints. "
        "Your goal is to analyze the problem requirements, constraints (time and memory limits), and input/output specifications. "
        "You need to determine the overall strategy and ensuring the solution handles the data scale correctly (e.g., O(N) vs O(N^2)). "
        "You do not need to write code, but clarify the logical flow and data structures required. "
        "I hope your reply will be more concise. Preferably within fifty words.",
        
    "Algorithm Designer":
        "You are an algorithm designer specializing in competitive programming. "
        "You will be given a problem description by the user. "
        "You need to design a specific algorithm (e.g., Dynamic Programming, Greedy, BFS/DFS, Binary Search) that solves the problem efficiently within the given constraints. "
        "Provide the logic or pseudocode. Explicitly state the time complexity of your approach. "
        "Identify necessary data structures (e.g., Priority Queue, Segment Tree). "
        "I hope your reply will be more concise. Preferably within fifty words.",
        
    "Programming Expert":
        "You are a competitive programming expert. "
        "You will be given a problem description and potentially an algorithm design. "
        "First identify the required interface. "
        "If the task explicitly says it is call-based or asks for a specific function name, implement exactly that top-level function, return the answer, and do not read stdin or print. "
        "For call-based tasks, do not use `class Solution`; define the requested function at module top level. Avoid undefined type annotations such as `List` unless you import them. "
        "Otherwise, write a **complete, executable Python script** that reads Standard Input (stdin) and prints to Standard Output (stdout). "
        "Use a Python code block to write your response. For example:\n```python\nimport sys\n\ndef solve():\n    # your code\n    pass\n\nif __name__ == '__main__':\n    solve()\n```"
        "Do not include anything other than Python code blocks in your response. "
        "Optimize for speed using `sys.stdin.read` where appropriate.",
        
    "Test Analyst":
        "You are a test analyst for algorithmic problems. "
        "You will be given the problem description and the current solution code. "
        "Your job is to identify potential edge cases (e.g., N=0, N=Max, negative numbers, disconnected graphs) where the code might fail or TLE (Time Limit Exceeded). "
        "Provide specific input examples that test these boundary conditions. "
        "Point out logical flaws or complexity issues. "
        "I hope your reply will be more concise. Preferably within fifty words.",
        
    "Bug Fixer":
        "You are a bug fixer for competitive programming code."
        "You will be given the problem description and the current code which may have errors or performance issues. "
        "Refine the code based on the feedback (e.g., fix logic errors, optimize loops for performance, fix input parsing). "
        "Preserve the required interface: for call-based tasks, output the exact requested top-level function with no `class Solution`; for standard-input tasks, output the complete executable script again. "
        "Use a Python code block to write your response. For example:\n```python\nimport sys\n# fixed code\n```"
        "Do not include anything other than Python code blocks in your response.",
        
    "Normal Programmer":
        "You are a programmer. "
        "You will be given a problem description. "
        "If the task is call-based or asks for a specific function name, implement that top-level function and return the answer without using `class Solution`. Otherwise write a complete Python script to solve it using stdin/stdout. "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not include anything other than Python code blocks in your response. ",
        
    "Stupid Programmer":
        "You are a novice programmer. "
        "You will be given a problem description. "
        "Give a code implementation that might have logical errors or be inefficient. "
        "Do not use comments for all errors. "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not include anything other than Python code blocks in your response. ",
}


DESCRIPTION = {
    "Project Manager": "Analyzes the problem statement, constraints, and I/O format. Decides the complexity requirements.",
    "Algorithm Designer": "Designs the specific algorithm (DP, Graph, etc.) and logic to solve the problem efficiently.",
    "Programming Expert": "Implements the full Python script, handling stdin/stdout and the core logic.",
    "Test Analyst": "Identifies corner cases, boundary conditions, and potential Time Limit Exceeded scenarios.",
    "Bug Fixer": "Optimizes or fixes the code based on feedback to ensure correctness and efficiency.",
    "Normal Programmer": "An agent that writes straightforward Python implementations for CP problems.",
    "Stupid Programmer": "An agent that deliberately produces erroneous code for robustness testing.",
}

@PromptSetRegistry.register('codecontest')
class CodecontestPromptSet(PromptSet):

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
        return "natural language"

    @staticmethod
    def get_answer_prompt(question):
        # Format the question for the AI assistant to answer
        return f"{question}"

    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Here is an unsuccessful attempt for solving the following competitive programming problem:
Question:
{question}
Attempted Solution:
{solution}
Feedback:\n{feedback}
Rewrite the complete code based on the feedback. Ensure it handles stdin/stdout correctly and meets time constraints:
{question}"""


    @staticmethod
    def get_query_prompt(question):
        return (
"# Information Gathering for Algorithm Design\n\n"
"Evaluate if the problem requires advanced algorithms or specific mathematical theorems. "
"If the problem involves specific topics (e.g., Number Theory, Computational Geometry), outline what formulas or templates might be needed.\n\n"
f"## ❓ Target Problem:\n{question}\n\n"
"## 🔍 Clues for Investigation:\n"
"Identify constraints, data types (integer overflow?), and algorithmic patterns (e.g., 'shortest path', 'knapsack').\n"
        )


    @staticmethod
    def get_file_analysis_prompt(query, file):
        return (
"# File Analysis Task\n\n"
f"## 🔍 Information Extraction Objective:\n---\n{query}\n---\n\n"
f"## 📄 File Under Analysis:\n---\n{file}\n---\n\n"
"## 📝 Instructions:\n"
"1. Identify the key sections in the file relevant to the query.\n"
"2. Extract and summarize the necessary information from these sections.\n"
"3. Ensure the response is focused and directly addresses the query.\n"
        )


    @staticmethod
    def get_websearch_prompt(question, query):
        return (
            "# Web Search Task\n\n"
            f"## Original Problem: \n---\n{question}\n---\n\n"
            f"## 🔍 Targeted Search Objective:\n---\n{query}\n---\n\n"
            "## 🌐 Simplified Search Instructions:\n"
            "Generate three specific search queries related to the algorithmic concepts in the question. "
            "Focus on finding standard algorithms or similar competitive programming problems.\n"
            "Example queries: 'fast exponentiation python', 'longest common subsequence optimization', 'Codeforces 1234A solution'."
        )



    @staticmethod
    def get_adversarial_answer_prompt(question):
        pass


    @staticmethod
    def get_distill_websearch_prompt(question, query, results):
        return (
"# Summarization of Search Results\n\n"
f"## Original question: \n---\n{question}\n---\n\n"
f"## 🔍 Required Information for Summary:\n---\n{query}\n---\n\n"
f"## 🌐 Analyzed Search Results:\n---\n{results}\n---\n\n"
"## 📝 Instructions for Summarization:\n"
"1. Identify relevant algorithms, data structures, or mathematical formulas from the search results.\n"
"2. Extract code snippets or logic explanations that help solve the specific constraints of the problem.\n"
"3. If the search results provide a direct solution to a similar problem, highlight the key logic.\n"
"4. If no useful info is found, state: \"No useful information from WebSearch\".\n"  
        )


    @staticmethod
    def get_reflect_prompt(question, answer):
        return (
"# Reflection on the Solution\n\n"
f"## 🤔 Problem:\n---\n{question}\n---\n\n"
f"## 💡 Your Solution:\n---\n{answer}\n---\n\n"
"## ✏️ Instructions:\n"
"Reflect on the correctness and efficiency. Does it handle the input format correctly? Is the time complexity acceptable for the given constraints?"
        )


    @staticmethod
    def get_self_consistency(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
"# Self-Consistency Evaluation Task\n\n"
f"## 🤔 Problem:\n---\n{question}\n---\n\n"
f"## 💡 Reviewable Code Solutions:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Instructions for Selection:\n"
"1. Read each Python script and check if it implements the logic correctly.\n"
"2. Verify if the code correctly handles standard input (stdin) and standard output (stdout).\n"
"3. Choose the code that is most likely to pass all test cases (correct logic + best time complexity).\n"
"4. Ignore answers that are just explanations without full code.\n"
"5. Copy the most suitable code as it is.\n"
f"6. Adhere to the constraints: {constraint}.\n"
        )

    @staticmethod
    def get_select_best(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
"# Best Solution Evaluation Task\n\n"
f"## 🤔 Problem:\n---\n{question}\n---\n\n"
f"## 💡 Candidate Solutions:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Evaluation Instructions:\n"
"1. Examine the problem constraints (Time/Memory limits).\n"
"2. Assess the algorithmic complexity of each candidate solution.\n"
"3. Choose the solution that correctly parses input, implements the efficient algorithm, and prints output correctly.\n"
"4. Ignore candidates that are incomplete or use placeholders.\n"
"5. Copy the chosen code exactly as it is presented.\n"
f"6. Adhere to the constraints: {constraint}.\n"
        )

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

    @staticmethod
    def get_decision_constraint():
        return (
"You will be given a competitive programming problem description."
"You may be given the algorithm analysis, code snippets, or test feedback."
"Write the final Python implementation in the interface required by the task."
"If the task explicitly says it is call-based or names a function to implement, define exactly that top-level function, return the answer, and do not read stdin or print."
"For call-based tasks, do not wrap the function in `class Solution` or any class. Avoid undefined type annotations such as `List`; either omit annotations or import required names."
"Otherwise write a full executable script that reads from `sys.stdin` or `input()` and prints to `sys.stdout` or `print()`."
"Do not invent function signatures when the task expects a standalone script."
"If the prompt contains correct code from other agents, choose the most reliable one."
"Use a Python code block to write your response. For example:\n```python\nimport sys\n# code...\n```"
"Do not include anything other than Python code blocks in your response"
)
    
    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker. You analyze inputs from the team and produce the final Python code in the required interface. You respond ONLY with the Python code block."
    
    @staticmethod
    def get_decision_few_shot():
        return ""
