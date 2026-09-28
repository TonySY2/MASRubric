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
        return None  # 
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
            "datetime": __import__("datetime"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),
            "heapq": __import__("heapq"),
            "List": List, "Dict": Dict, "Tuple": Tuple, "Optional": Optional, "Any": Any,
        }
        

        match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", func)
        if not match:
            return False, "Execution failed: Cannot find function definition in the model's code.", (False,)
        entry_point = match.group(1)

        try:
    
            exec(func, global_dict)

            if entry_point not in global_dict:
                raise ValueError(f"Function '{entry_point}' is not defined after executing the solution code.")


            original_assertions = tests[0]
            original_func_name_match = re.search(r"assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", original_assertions)
            
            if original_func_name_match:
                original_func_name = original_func_name_match.group(1)
                test_code = original_assertions.replace(original_func_name, entry_point)
            else:
                test_code = original_assertions
            

            self._run_with_timeout(lambda: exec(test_code, global_dict), timeout)

   
            is_passing = True
            feedback = f"Tests passed:\n{test_code}"
            
        except self.TimeoutError as e:
            is_passing = False
            feedback = f"Tests failed due to timeout: {e}"
        except Exception as e:
            is_passing = False
            error_message = f"Error: {type(e).__name__}: {e}.\nTraceback:\n{traceback.format_exc()}"
            feedback = f"Tests failed:\n{error_message}"
        
        return is_passing, feedback, (is_passing,)
    
    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
      
        return False