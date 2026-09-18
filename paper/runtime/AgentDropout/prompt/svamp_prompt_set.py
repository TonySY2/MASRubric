#--- START OF FILE svamp_prompt_set.py ---

from typing import Dict, Any
import itertools
from AgentDropout.prompt.prompt_set import PromptSet
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry
from AgentDropout.prompt.common import get_combine_materials

roles = itertools.cycle(['Math Solver',
                         'Mathematical Analyst',
                         'Programming Expert',
                         'Inspector',])

ROLE_DESCRIPTION = {
    "Math Solver": 
        "You are a math expert. "
        "You will be given a math word problem and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output must contain the final numerical answer, for example: The answer is 42\n",
    "Mathematical Analyst":
        "You are a mathematical analyst. "
        "You will be given a math word problem, analysis and code from other agents. "
        "You need to first analyze the problem-solving process step by step, identifying the arithmetic relationships between variables. "
        "Then perform the calculations carefully to get the result."
        "The last line of your output must contain the final numerical answer, for example: The answer is 42\n",
    "Programming Expert":
        "You are a programming expert. "
        "You will be given a math word problem, analysis and code from other agents. "
        "Integrate step-by-step reasoning and Python code to solve the problem. "
        "Analyze the question and write functions to solve the problem. "
        "The function should not take any arguments and use the final result as the return value. "
        "The last line of code calls the function you wrote and assigns the return value to the answer variable. "
        "Use a Python code block to write your response. For example:\n```python\ndef fun():\n x = 10\n y = 20\n return x + y\nanswer = fun()\n```\n"
        "Do not include anything other than Python code blocks in your response.",
    "Inspector":
        "You are an Inspector. "
        "You will be given a math word problem, analysis and code from other agents. "
        "Check whether the logic/calculation of the problem solving and analysis process is correct(if present). "
        "Check whether the code corresponds to the solution analysis(if present). "
        "Give your own solving process step by step based on hints. "
        "The last line of your output must contain the final numerical answer, for example: The answer is 42\n",
}

# Few-shot data cleared as requested, keeping structure to avoid key errors
FEW_SHOT_DATA_1 = {
    "Math Solver": "",
    "Mathematical Analyst": "",
    "Programming Expert": "",
    "Inspector": "",
}

FEW_SHOT_DATA = {
    "Math Solver": "",
    "Mathematical Analyst": "",
    "Programming Expert": "",
    "Inspector": "",
}

DESCRIPTION = {
    "Math Solver": "An agent for solving math word problems step by step, providing a clear final numeric answer.",
    "Mathematical Analyst": "An agent for analyzing math problems, reviewing arithmetic steps, and calculating the final numeric answer.",
    "Programming Expert": "An agent for solving math problems with Python, writing code to calculate the final result.",
    "Inspector": "An agent for verifying math solutions and logic, checking for arithmetic errors, and providing a corrected final numeric answer.",
}

@PromptSetRegistry.register('svamp')
class SVAMPPromptSet(PromptSet):

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
    def get_answer_prompt(question, role="Mathematical Analyst"):
        # Format the question for the AI assistant to answer
        # Added a check to only append few-shot if it exists to avoid empty newlines
        prefix = FEW_SHOT_DATA.get(role, "")
        if prefix:
            return f"{prefix}\n\nQ:{question}"
        return f"Q:{question}"

    @staticmethod
    def get_decision_constraint():
        return (
        "You will be given a math word problem, analysis and code from other agents. "
        "Please find the most reliable numerical answer based on the analysis and results of other agents. "
        "Give reasons for making decisions. "
        "The last line of your output must contain the final numerical answer, for example: The answer is 42")
    
    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker. " \
               "Good at analyzing and summarizing mathematical problems, judging other people's solutions, and giving the final numeric answer."
    
    @staticmethod
    def get_decision_few_shot_1():
        return ""

    @staticmethod
    def get_decision_few_shot():
        return ""

    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Here is an unsuccessful attempt for solving the folloing question:
Question:
{question}
Attempted Solution:
{solution}
Feedback:\n{feedback}
Rewrite the code based on the feedback and the following question:
{question}"""


    @staticmethod
    def get_query_prompt(question):
        return (
"# Information Gathering for Question Resolution\n\n"
"Evaluate if additional information is needed to answer the question. "
"If a web search or file analysis is necessary, outline specific clues or details to be searched for.\n\n"
f"## ❓ Target Question:\n{question}\n\n"
"## 🔍 Clues for Investigation:\n"
"Identify critical clues and concepts within the question that are essential for finding the answer.\n"
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
"Example: 'Identify the main theme in the text.'"
        )


    @staticmethod
    def get_websearch_prompt(question, query):
        return (
            "# Web Search Task\n\n"
            f"## Original Question: \n---\n{question}\n---\n\n"
            f"## 🔍 Targeted Search Objective:\n---\n{query}\n---\n\n"
            "## 🌐 Simplified Search Instructions:\n"
            "Generate three specific search queries directly related to the original question. Each query should focus on key terms from the question. Format the output as a comma-separated list.\n"
            "Remember to format the queries as 'query1, query2, query3'."
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
"1. Review the provided search results and identify the most relevant information related to the question and query.\n"
"2. Extract and highlight the key findings, facts, or data points from these results.\n"
"3. Organize the summarized information in a coherent and logical manner.\n"
"4. Ensure the summary is concise and directly addresses the query, avoiding extraneous details.\n"  
"5. If the information from web search is useless, directly answer: \"No useful information from WebSearch\".\n"  
        )


    @staticmethod
    def get_reflect_prompt(question, answer):
        return (
"# Reflection on the Task\n\n"
f"## 🤔 Reflection Question:\n---\n{question}\n---\n\n"
f"## 💡 Your Previous Answer:\n---\n{answer}\n---\n\n"
"## ✏️ Instructions:\n"
"Reflect on your answer process, considering the accuracy, method, and reasoning."
        )


    @staticmethod
    def get_self_consistency(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
"# Self-Consistency Evaluation Task\n\n"
f"## 🤔 Question for Review:\n---\n{question}\n---\n\n"
f"## 💡 Reviewable Answers:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Instructions for Selection:\n"
"1. Read each answer and assess how it addresses the question.\n"
"2. Compare the answers for their adherence to the given question's criteria and logical coherence.\n"
"3. Identify the answer that best aligns with the question's requirements and is the most logically consistent.\n"
"4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
"5. Copy the most suitable answer as it is, without modification, to maintain its original form.\n"
f"6. Adhere to the constraints: {constraint}.\n"
"Note: If no answer fully meets the criteria, choose and copy the one that is closest to the requirements."
        )

    @staticmethod
    def get_select_best(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
"# Best Answer Evaluation Task\n\n"
f"## 🤔 Question:\n---\n{question}\n---\n\n"
f"## 💡 Candidate Answers for Evaluation:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Evaluation Instructions:\n"
"1. Examine the question closely to understand its requirements.\n"
"2. Read each candidate answer thoroughly and assess its relevance and accuracy about the question.\n"
"3. Choose the answer that most accurately and completely addresses the question.\n"
"4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
"5. Copy the chosen answer exactly as it is presented, maintaining its original format.\n"
f"6. Adhere to the constraints: {constraint}.\n"
"Note: If none of the answers fully meet the question's criteria, select the one closest to fulfilling them."
        )

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)