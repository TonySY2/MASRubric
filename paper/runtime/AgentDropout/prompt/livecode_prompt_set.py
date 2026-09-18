from typing import Dict, Any
import itertools
from AgentDropout.prompt.prompt_set import PromptSet
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry
from AgentDropout.prompt.common import get_combine_materials

roles = itertools.cycle(['Project Manager',
                         'Algorithm Designer',
                         'Programming Expert',
                         'Test Analyst',
                         'Bug Fixer',])

# Revised descriptions for LiveCodeBench (Competitive Programming / LeetCode style)
ROLE_DESCRIPTION = {
    "Project Manager": 
        "You are a project manager for coding competitions. "
        "You will be given a problem description from platforms like LeetCode, AtCoder, or Codeforces. "
        "Your goal is to analyze the problem constraints (Time/Memory), input/output formats, and edge cases. "
        "Determine the complexity requirements (e.g., O(N log N)) to avoid Time Limit Exceeded (TLE). "
        "Outline the high-level strategy. "
        "I hope your reply will be more concise. Preferably within fifty words.",
        
    "Algorithm Designer":
        "You are an algorithm designer. "
        "You will be given a problem description. "
        "Design a specific algorithm (e.g., DP, Greedy, Graph, Binary Search) that fits the constraints. "
        "Provide the logic or pseudocode. Explicitly state the time complexity. "
        "Identify necessary data structures (e.g., Heap, Union-Find, Segment Tree). "
        "I hope your reply will be more concise. Preferably within fifty words.",
        
    "Programming Expert":
        "You are a competitive programming expert. "
        "You will be given a problem description and potentially a starter code template. "
        "Write a **complete, executable Python script**. "
        "IMPORTANT: "
        "1. If the problem requires a class `Solution`, implement it exactly as required. "
        "2. If the problem is stdin/stdout based, write the full script to read inputs and print outputs. "
        "3. Import all necessary libraries (e.g., `sys`, `collections`, `heapq`). "
        "Use a Python code block to write your response. "
        "Do not include anything other than Python code blocks in your response. "
        "Please think step by step.",
        
    "Test Analyst":
        "You are a test analyst. "
        "You will be given the problem description and the current solution code. "
        "Identify potential edge cases (e.g., empty arrays, max constraints, negative numbers) where the code might fail. "
        "Check for common pitfalls like integer overflow (though less relevant in Python), off-by-one errors, or infinite loops. "
        "I hope your reply will be more concise. Preferably within fifty words.",
        
    "Bug Fixer":
        "You are a bug fixer. "
        "You will be given the problem description and the current code which may have errors or TLE issues. "
        "Refine the code based on the feedback. Optimize loops, fix logic errors, or switch to a more efficient algorithm. "
        "Write the **complete, executable Python script** again. "
        "Use a Python code block. "
        "Do not include anything other than Python code blocks in your response.",
        
    "Normal Programmer":
        "You are a programmer. "
        "Write a complete Python solution for the given problem. "
        "Use a Python code block. "
        "Do not include anything other than Python code blocks in your response.",
        
    "Stupid Programmer":
        "You are a novice programmer. "
        "Give a code implementation that might have logical errors or be inefficient. "
        "Do not use comments for all errors. "
        "Use a Python code block.",
}

DESCRIPTION = {
    "Project Manager": "Analyzes constraints, I/O formats, and complexity requirements.",
    "Algorithm Designer": "Designs the specific algorithm (DP, Graph, etc.) to solve the problem efficiently.",
    "Programming Expert": "Implements the full Python script, handling class structures or stdin/stdout as required.",
    "Test Analyst": "Identifies corner cases, boundary conditions, and potential bugs.",
    "Bug Fixer": "Optimizes or fixes the code based on feedback to ensure correctness.",
    "Normal Programmer": "Writes straightforward Python implementations.",
    "Stupid Programmer": "Simulates mistakes for robustness testing.",
}

@PromptSetRegistry.register('livecode')
class LivecodePromptSet(PromptSet):

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
        return f"{question}"

    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Here is an unsuccessful attempt for solving the following problem:
Question:
{question}
Attempted Solution:
{solution}
Feedback:\n{feedback}
Rewrite the complete code based on the feedback. Ensure it handles the required input/output format correctly:
{question}"""

    @staticmethod
    def get_query_prompt(question):
        return (
"# Information Gathering for Algorithm Design\n\n"
"Evaluate if the problem requires advanced algorithms or specific mathematical theorems. "
f"## ❓ Target Problem:\n{question}\n\n"
"## 🔍 Clues for Investigation:\n"
"Identify constraints, specific data structures, and algorithmic patterns.\n"
        )

    @staticmethod
    def get_file_analysis_prompt(query, file):
        return (
"# File Analysis Task\n\n"
f"## 🔍 Information Extraction Objective:\n---\n{query}\n---\n\n"
f"## 📄 File Under Analysis:\n---\n{file}\n---\n\n"
"## 📝 Instructions:\n"
"1. Identify the key sections in the file relevant to the query.\n"
"2. Extract and summarize the necessary information.\n"
        )

    @staticmethod
    def get_websearch_prompt(question, query):
        return (
            "# Web Search Task\n\n"
            f"## Original Problem: \n---\n{question}\n---\n\n"
            f"## 🔍 Targeted Search Objective:\n---\n{query}\n---\n\n"
            "## 🌐 Simplified Search Instructions:\n"
            "Generate three specific search queries related to the algorithmic concepts. "
            "Example: 'LeetCode <Title> solution python', 'algorithm for <concept>'.\n"
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
"1. Identify relevant algorithms or code snippets.\n"
"2. Extract logic that helps solve the specific constraints.\n"
"3. If no useful info is found, state: \"No useful information from WebSearch\".\n"  
        )

    @staticmethod
    def get_reflect_prompt(question, answer):
        return (
"# Reflection on the Solution\n\n"
f"## 🤔 Problem:\n---\n{question}\n---\n\n"
f"## 💡 Your Solution:\n---\n{answer}\n---\n\n"
"## ✏️ Instructions:\n"
"Reflect on the correctness and efficiency. Does it handle the input format correctly? Is the time complexity acceptable?"
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
"2. Verify if the code adheres to the required class structure or stdin/stdout format.\n"
"3. Choose the code that is most likely to pass all test cases.\n"
"4. Copy the most suitable code as it is.\n"
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
"1. Examine the problem constraints.\n"
"2. Assess the algorithmic complexity of each candidate.\n"
"3. Choose the solution that correctly implements the efficient algorithm.\n"
"4. Copy the chosen code exactly as it is presented.\n"
f"6. Adhere to the constraints: {constraint}.\n"
        )

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

    @staticmethod
    def get_decision_constraint():
        return (
"You will be given a competitive programming or LeetCode-style problem description."
"You may be given the algorithm analysis, code snippets, or test feedback."
"Write your **full, executable Python implementation**."
"If the problem requires a class `Solution`, implement it."
"If the problem requires stdin/stdout, implement it."
"Use a Python code block to write your response. For example:\n```python\nimport sys\n# code...\n```"
"Do not include anything other than Python code blocks in your response"
)
    
    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker. You analyze inputs from the team and produce the final Python script for submission. You respond ONLY with the Python code block."
    
    @staticmethod
    def get_decision_few_shot():
        return ""