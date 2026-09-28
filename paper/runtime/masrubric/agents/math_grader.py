import re
import regex
import math
import sys

# 尝试导入 sympy 进行高级数学判定
try:
    from sympy import simplify, N
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

class MathGrader:
    """
    全功能数学评测器。
    保留了 SymPy 符号计算、LaTeX 解析、数值近似等核心逻辑。
    增加了对非字符串输入（如 AIME 的 int 答案）的鲁棒性支持。
    """

    @staticmethod
    def extract_answer(text) -> str:
        """
        从文本中提取答案。
        优先级：
        1. \\boxed{...} (支持嵌套)
        2. 最后一句话/最后一行 (启发式)
        """
        # [修复] 强制类型转换，处理 int/float 输入
        if text is None: return ""
        text = str(text)
        
        # 1. 尝试提取 \boxed{...}
        # 贪婪匹配最后一个 boxed，支持花括号嵌套
        boxed_matches = []
        for m in re.finditer(r"\\boxed\{", text):
            start = m.end()
            balance = 1
            for i in range(start, len(text)):
                if text[i] == '{':
                    balance += 1
                elif text[i] == '}':
                    balance -= 1
                
                if balance == 0:
                    boxed_matches.append(text[start:i])
                    break
        
        if boxed_matches:
            return boxed_matches[-1].strip()

        # 2. 兜底：提取最后一行
        # 移除 markdown 代码块
        text = text.strip().replace("```", "")
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            last_line = lines[-1]
            # 移除常见前缀
            for prefix in ["Final Answer:", "Answer:", "The answer is"]:
                # 使用正则忽略大小写分割
                parts = re.split(f"{prefix}", last_line, flags=re.IGNORECASE)
                if len(parts) > 1:
                    last_line = parts[-1].strip()
            return last_line
            
        return ""

    @staticmethod
    def normalize_text(text) -> str:
        """
        标准化文本：移除格式控制符，统一符号。
        """
        if text is None: return ""
        text = str(text) # [修复] 类型安全
        
        # 1. 移除 LaTeX 样式命令
        commands_to_remove = [
            r"\left", r"\right", r"\text", r"\mathrm", r"\mathbf", r"\mbox", 
            r"\,", r"\:", r"\;", r"\!", r"\ "
        ]
        for cmd in commands_to_remove:
            text = text.replace(cmd, "")
            
        # 2. 移除空格和换行
        text = "".join(text.split())
        
        # 3. 统一符号
        text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
        text = text.replace(r"\div", "/").replace(r"\cdot", "*").replace(r"\times", "*")
        
        # 4. 移除美元符号
        text = text.replace("$", "")
        
        # 5. 处理结尾句号 (如 "42." -> "42")
        if text.endswith("."):
            text = text[:-1]
            
        return text

    @classmethod
    def check_correctness(cls, hypothesis, ground_truth) -> bool:
        """
        判定入口。
        """
        # [修复] 类型安全，防止 None
        if hypothesis is None: hypothesis = ""
        if ground_truth is None: ground_truth = ""
        
        # 1. 提取预测值
        pred_extracted = cls.extract_answer(hypothesis)
        
        # 2. 提取标准答案
        gt_extracted = cls.extract_answer(ground_truth)
        
        # [AIME 特别优化]
        # AIME 的 ground_truth 经常是纯数字 "204"，没有 boxed。
        # 此时 extract_answer 如果没找到 boxed 会返回 "204"。
        # 但为了保险，如果提取结果为空，而原 GT 不为空，强制使用原 GT。
        if not gt_extracted and str(ground_truth).strip():
            gt_extracted = str(ground_truth).strip()
            
        return cls.math_equal(pred_extracted, gt_extracted)

    @staticmethod
    def math_equal(prediction, reference) -> bool:
        """
        核心判定逻辑：
        1. 字符串归一化匹配
        2. SymPy 符号等价 (最强)
        3. 数值近似匹配
        """
        pred_norm = MathGrader.normalize_text(prediction)
        ref_norm = MathGrader.normalize_text(reference)

        if not pred_norm or not ref_norm:
            return False

        # 1. 字符串直接匹配
        if pred_norm == ref_norm:
            return True

        # 2. SymPy 符号解析 (处理 x+y = y+x 等情况)
        if HAS_SYMPY:
            try:
                if MathGrader.symbolic_equal(prediction, reference):
                    return True
            except Exception:
                pass 

        # 3. 数值比较 (处理 1/2 = 0.5)
        try:
            # 去掉逗号 (1,000 -> 1000)
            p_val = float(regex.sub(",", "", str(prediction)))
            r_val = float(regex.sub(",", "", str(reference)))
            # AIME 答案是整数，容差可以很小；但普通 Math 可能有小数
            if math.isclose(p_val, r_val, abs_tol=1e-4):
                return True
        except:
            pass

        return False

    @staticmethod
    def symbolic_equal(a, b):
        """利用 SymPy 判断数学等价性"""
        def _parse(s):
            s = str(s).replace("$", "")
            # 优先 LaTeX 解析
            try:
                return parse_latex(s)
            except:
                pass
            # 降级为 Python 表达式解析
            try:
                return parse_expr(s)
            except:
                return None

        expr_a = _parse(a)
        expr_b = _parse(b)

        if expr_a is None or expr_b is None:
            return False

        # 方法 A: 符号简化差值
        try:
            if simplify(expr_a - expr_b) == 0:
                return True
        except:
            pass

        # 方法 B: 数值评估 (处理 pi, e, sqrt 等)
        try:
            val_a = N(expr_a)
            val_b = N(expr_b)
            # 严格数值比对
            if math.isclose(float(val_a), float(val_b), abs_tol=1e-4):
                return True
        except:
            pass
            
        return False
    
    
class SvampGrader:
    """
    算术专用评测器 (SVAMP, GSM8K)。
    只关注数字，不关注 LaTeX 结构。
    """
    @staticmethod
    def extract_answer(text) -> str:
        if text is None: return ""
        text = str(text)
        clean_text = text.replace(',', '')
        # 匹配最后一个数字 (整数或小数)
        pattern = r"-?\d+\.\d+|-?\d+"
        matches = re.findall(pattern, clean_text)
        return matches[-1] if matches else ""

    @classmethod
    def check_correctness(cls, hypothesis, ground_truth) -> bool:
        pred_val = cls.extract_answer(hypothesis)
        gt_val = cls.extract_answer(ground_truth)
        
        if not pred_val or not gt_val:
            return False
            
        try:
            return math.isclose(float(pred_val), float(gt_val), abs_tol=1e-3)
        except ValueError:
            return False