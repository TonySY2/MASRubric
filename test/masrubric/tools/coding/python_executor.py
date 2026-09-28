#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import astunparse
from typing import List

from masrubric.tools.coding.executor_utils import function_with_timeout
from masrubric.tools.coding.executor_types import ExecuteResult, Executor
import timeout_decorator


import re
import traceback
from typing import List, Dict, Tuple, Optional, Any
import threading

import subprocess
import sys

def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left # type: ignore
    except:
        call_str = ast_parsed.body[0].test # type: ignore

    return astunparse.unparse(call_str).strip()

def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    try:
        exec(f"from typing import *\n{func}", globals())
        func_call = get_call_str(assert_statement)
        output = function_with_timeout(eval, (func_call, globals()), timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)

@timeout_decorator.timeout(5, timeout_exception=StopIteration)
def execute_code_get_return(code: str):
    local_vars = {}
    try:
        exec(code, {}, local_vars)
        if 'answer' in local_vars:
            return local_vars['answer']
        else:
            return None
    except StopIteration:
        return None  
    except Exception as e:
        return f"Error occurred: {e}"

class PyExecutor(Executor):
    def execute(self, func: str, tests: List[str], timeout: int = 5, verbose: bool = True) -> ExecuteResult:
        # Combine function code and assert statement
        imports = 'from typing import *'
        func_test_list = [f'{imports}\n{func}\n{test}' for test in tests]

        # Run the tests and collect the results
        success_tests = []
        failed_tests = []
        is_passing = True
        num_tests = len(func_test_list)
        for i in range(num_tests):
            try:
                function_with_timeout(exec, (func_test_list[i], globals()), timeout)
                success_tests.append(tests[i])
            except Exception:
                output = get_output(func, tests[i], timeout=timeout)
                failed_tests.append(f"{tests[i]} # output: {output}")
                is_passing = False

        state = [test in success_tests for test in tests]

        feedback = "Tests passed:\n" + "\n".join(success_tests) + "\n\nTests failed:"
        feedback += "\n" + "\n".join(failed_tests)
        return is_passing, feedback, tuple(state)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        
        code = f"""{func}

{test}

check({name})
    """
        try:
            function_with_timeout(exec, (code, globals()), timeout)
            return True
        except Exception:
            return False
        
class MBPPExecutor(Executor):

    
    class TimeoutError(Exception):
        pass

    def _run_with_timeout(self, func, timeout):
        
        result_container = []
        exception_container = []
        
        def target():
            try:
                result_container.append(func())
            except Exception as e:
                exception_container.append(e)

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
      
            raise self.TimeoutError(f"Execution timed out after {timeout} seconds")

        if exception_container:
            raise exception_container[0]

        return result_container[0] if result_container else None

    def execute(self, func: str, tests: List[str], timeout: int = 15, verbose: bool = True) -> ExecuteResult:
        
        if not tests or not tests[0]:
            return False, "Execution failed: No test assertions provided.", (False,)


        global_dict = {
            "math": __import__("math"),
            "re": __import__("re"),
            "sys": __import__("sys"),
            "os": __import__("os"),
            "random": __import__("random"),
            "datetime": __import__("datetime"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),
            "heapq": __import__("heapq"),
            "typing": __import__("typing"),
   
            "List": List, "Dict": Dict, "Tuple": Tuple, "Optional": Optional, "Any": Any, "Union": Any,
        }

     
        try:
            import numpy
            global_dict["np"] = numpy
            global_dict["numpy"] = numpy
        except ImportError:
       
            pass

        try:
  
            exec(func, global_dict)

      
            test_code = tests[0]
            
            self._run_with_timeout(lambda: exec(test_code, global_dict), timeout)

           
            is_passing = True
         
            preview = test_code[:200] + "..." if len(test_code) > 200 else test_code
            feedback = f"Tests passed.\nCode execution successful.\nTest Snippet:\n{preview}"
            
        except self.TimeoutError as e:
            is_passing = False
            feedback = f"Tests failed due to timeout ({timeout}s).\nError: {e}"
            
        except Exception as e:
            is_passing = False
            
            tb_list = traceback.format_tb(e.__traceback__)
          
            relevant_tb = "".join(tb_list[-2:]) if len(tb_list) > 0 else ""
            
            feedback = f"Tests failed.\nError Type: {type(e).__name__}\nError Message: {str(e)}\nTraceback:\n{relevant_tb}"
        
        return is_passing, feedback, (is_passing,)
    
    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
      
        return False
    
    

class HumanEvalExecutor(MBPPExecutor):
    def execute(self, func: str, tests: List[str], entry_point: str = None, timeout: int = 10, verbose: bool = True) -> ExecuteResult:
     
        if not tests or not tests[0]:
            return False, "Execution failed: No test assertions provided.", (False,)

  
        global_dict = {
            "math": __import__("math"),
            "hashlib": __import__("hashlib"), 
            "re": __import__("re"),
            "sys": __import__("sys"),
            "os": __import__("os"),
            "random": __import__("random"),
            "datetime": __import__("datetime"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),
            "heapq": __import__("heapq"),
            "typing": __import__("typing"),
            "List": List, "Dict": Dict, "Tuple": Tuple, "Optional": Optional, "Any": Any, "Union": Any,
        }

        try:
            import numpy
            global_dict["np"] = numpy
            global_dict["numpy"] = numpy
        except ImportError:
            pass

  
        if entry_point:
            if entry_point == "decode_cyclic":
                func = (
                    '\n\ndef encode_cyclic(s: str):\n    """\n    returns encoded string by cycling groups of three characters.\n    """\n    # split string to groups. Each of length 3.\n    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]\n    # cycle elements in each group. Unless group has fewer elements than 3.\n    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]\n    return "".join(groups)'
                    + "\n\n"
                    + func
                )
            elif entry_point == "decode_shift":
                func = (
                    '\n\ndef encode_shift(s: str):\n    """\n    returns encoded string by shifting every character by 5 in the alphabet.\n    """\n    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])\n\n\n'
                    + func
                )
            elif entry_point == "find_zero":
                func = (
                    "\n\ndef poly(xs: list, x: float):\n    return sum(coeff * (x ** i) for i, coeff in enumerate(xs))\n\n"
                    + func
                )

        try:
  
            exec(func, global_dict)

  
            test_code = tests[0]
            
  
            self._run_with_timeout(lambda: exec(test_code, global_dict), timeout)

            is_passing = True
            feedback = "Tests passed."
            
        except self.TimeoutError as e:
            is_passing = False
            feedback = f"Tests failed due to timeout ({timeout}s).\nError: {e}"
            
        except Exception as e:
            is_passing = False
  
            tb_list = traceback.format_tb(e.__traceback__)
            relevant_tb = "".join(tb_list[-2:]) if len(tb_list) > 0 else ""
            feedback = f"Tests failed.\nError Type: {type(e).__name__}\nError Message: {str(e)}\nTraceback:\n{relevant_tb}"
        
        return is_passing, feedback, (is_passing,)
    



class HumanEvalPlusExecutor:

    def execute(self, code: str, tests: List[str], entry_point: str = None, timeout: int = 10) -> Tuple[bool, str, Any]:
      
      
        header = "from typing import List, Tuple, Dict, Any, Optional\nimport math\nimport heapq\nimport sys\n\n"
        

        full_code = header + code + "\n\n"
  
        if tests and len(tests) > 0:
            full_code += "\n" + tests[0] 
     
        if "def check(" in full_code and "check(" not in full_code.split("def check(")[1]:
        
             full_code += f"\ncheck({entry_point})"


        
        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                return True, "Passed", None
            else:
                return False, result.stdout, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "Timeout", "Execution timed out"
        except Exception as e:
            return False, "Error", str(e)  
    
    
    
    
    
    
    