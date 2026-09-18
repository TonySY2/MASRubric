#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import astunparse
from typing import List

from AgentDropout.tools.coding.executor_utils import function_with_timeout
from AgentDropout.tools.coding.executor_types import ExecuteResult, Executor
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
        return None  # 超时时返回 None
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
    """
    专为 MBPP 和 MBPP+ 设计的 Python 代码执行器。
    特点：
    1. 移除函数名强制替换逻辑，完全信任 Prompt 的签名约束。
    2. 支持 MBPP+ 的复杂测试代码（包含 numpy, helper functions, loops）。
    3. 线程级超时控制。
    """
    
    class TimeoutError(Exception):
        pass

    def _run_with_timeout(self, func, timeout):
        """
        在子线程中运行函数以实现超时控制。
        """
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
            # 虽然不能强制杀死线程，但可以抛出异常中断当前流程
            raise self.TimeoutError(f"Execution timed out after {timeout} seconds")

        if exception_container:
            raise exception_container[0]

        return result_container[0] if result_container else None

    def execute(self, func: str, tests: List[str], timeout: int = 15, verbose: bool = True) -> ExecuteResult:
        """
        执行模型生成的代码和测试用例。
        
        Args:
            func: 模型生成的完整代码字符串 (Solution).
            tests: 测试代码列表。对于 MBPP/MBPP+，通常是一个包含完整测试脚本的字符串列表 (len=1)。
            timeout: 超时时间 (秒)。
        """
        if not tests or not tests[0]:
            return False, "Execution failed: No test assertions provided.", (False,)

        # 1. 准备沙盒执行环境 (Global Scope)
        # 预加载常用库，防止 ImportError
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
            # 类型提示支持
            "List": List, "Dict": Dict, "Tuple": Tuple, "Optional": Optional, "Any": Any, "Union": Any,
        }

        # [MBPP+ 关键] 尝试加载 numpy，因为 MBPP+ 的测试代码大量使用了 numpy
        try:
            import numpy
            global_dict["np"] = numpy
            global_dict["numpy"] = numpy
        except ImportError:
            # 如果环境没装 numpy，MBPP+ 的测试大概率会挂，但我们允许继续尝试
            pass

        try:
            # 2. 执行模型生成的函数代码 (Definition Phase)
            # 这会将模型定义的函数 (e.g., def remove_Occ...) 注册到 global_dict 中
            exec(func, global_dict)

            # 3. 执行测试代码 (Testing Phase)
            # MBPP: 简单的 assert func(...)
            # MBPP+: 复杂的 def assertion... + for loop
            # 直接执行，不做任何正则替换，因为我们已经在 Prompt 阶段强制了函数签名。
            test_code = tests[0]
            
            self._run_with_timeout(lambda: exec(test_code, global_dict), timeout)

            # 4. 如果没有抛出异常，视为通过
            is_passing = True
            # 截断反馈日志，防止过长
            preview = test_code[:200] + "..." if len(test_code) > 200 else test_code
            feedback = f"Tests passed.\nCode execution successful.\nTest Snippet:\n{preview}"
            
        except self.TimeoutError as e:
            is_passing = False
            feedback = f"Tests failed due to timeout ({timeout}s).\nError: {e}"
            
        except Exception as e:
            is_passing = False
            # 获取详细的错误堆栈，只取最后几行更有用的信息
            tb_list = traceback.format_tb(e.__traceback__)
            # 过滤掉 executor 自身的堆栈，尽量只保留 exec 内部的
            relevant_tb = "".join(tb_list[-2:]) if len(tb_list) > 0 else ""
            
            feedback = f"Tests failed.\nError Type: {type(e).__name__}\nError Message: {str(e)}\nTraceback:\n{relevant_tb}"
        
        return is_passing, feedback, (is_passing,)
    
    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """占位符，满足基类接口需求"""
        return False
    
    
# ... (保留之前的 MBPPExecutor 代码) ...

# =========================================================================
# [核心修改] 增强版 HumanEvalExecutor
# 移植了 external humaneval.py 的特例处理和环境设置
# =========================================================================
class HumanEvalExecutor(MBPPExecutor):
    def execute(self, func: str, tests: List[str], entry_point: str = None, timeout: int = 10, verbose: bool = True) -> ExecuteResult:
        """
        Args:
            func: 模型生成的代码
            tests: 测试代码 (HumanEval 的 test string)
            entry_point: [新增] 函数入口名称 (用于特例注入)
        """
        if not tests or not tests[0]:
            return False, "Execution failed: No test assertions provided.", (False,)

        # 1. 准备沙盒环境 (增强版)
        global_dict = {
            "math": __import__("math"),
            "hashlib": __import__("hashlib"), # [新增] HumanEval 需要
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
        
        # 尝试加载 numpy
        try:
            import numpy
            global_dict["np"] = numpy
            global_dict["numpy"] = numpy
        except ImportError:
            pass

        # 2. [核心改进] 特例处理 (Helper Function Injection)
        # 某些题目依赖 Prompt 中定义的辅助函数，如果模型没重写，必须手动注入
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
            # 3. 执行模型生成的代码 (Definition)
            exec(func, global_dict)

            # 4. 执行测试代码 (Test)
            # HumanEval 的 tests[0] 包含 `def check(candidate): ...` 和 `check(entry_point)`
            # 我们直接执行这个脚本即可
            test_code = tests[0]
            
            # 使用父类的线程超时控制
            self._run_with_timeout(lambda: exec(test_code, global_dict), timeout)

            is_passing = True
            feedback = "Tests passed."
            
        except self.TimeoutError as e:
            is_passing = False
            feedback = f"Tests failed due to timeout ({timeout}s).\nError: {e}"
            
        except Exception as e:
            is_passing = False
            # 简化堆栈信息
            tb_list = traceback.format_tb(e.__traceback__)
            relevant_tb = "".join(tb_list[-2:]) if len(tb_list) > 0 else ""
            feedback = f"Tests failed.\nError Type: {type(e).__name__}\nError Message: {str(e)}\nTraceback:\n{relevant_tb}"
        
        return is_passing, feedback, (is_passing,)
    



class HumanEvalPlusExecutor:
    """
    专门用于 HumanEval+ 的代码执行器。
    HumanEval+ 的数据结构通常包含 'test' 字段，里面是完整的 assert 语句。
    """
    def execute(self, code: str, tests: List[str], entry_point: str = None, timeout: int = 10) -> Tuple[bool, str, Any]:
        """
        执行代码并验证。
        :param code: 模型生成的代码 (函数定义)
        :param tests: 测试代码列表 (通常是一个包含 assert 的大字符串)
        :return: (is_correct, output, error)
        """
        
        # 1. 构造完整的可执行脚本
        # 很多时候生成的代码只有函数体，需要加上 import
        header = "from typing import List, Tuple, Dict, Any, Optional\nimport math\nimport heapq\nimport sys\n\n"
        
        # 2. 拼接代码
        # 结构: Header -> Model Code -> Test Code -> Entry Point Check (optional)
        full_code = header + code + "\n\n"
        
        # HumanEval+ 的 tests 通常已经是可以直接运行的 assert 语句
        if tests and len(tests) > 0:
            full_code += "\n" + tests[0] # HumanEval+ 通常只有一个大的 test block
            
        # [可选] 如果有 check 函数调用，加上它
        if "def check(" in full_code and "check(" not in full_code.split("def check(")[1]:
             # 某些格式可能定义了 check 但没调用
             full_code += f"\ncheck({entry_point})"

        # 3. 在沙箱/子进程中执行
        # 为了安全和隔离，我们使用 multiprocessing 或 subprocess
        # 这里复用现有的 safe_execute 逻辑 (假设你有一个通用的执行函数)
        # 如果没有，下面是一个简化的子进程执行逻辑
        
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
    
    
    
    
    
    
    