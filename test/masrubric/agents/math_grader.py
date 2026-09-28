import re
import regex
import math
import sys


try:
    from sympy import simplify, N
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

class MathGrader:


    @staticmethod
    def extract_answer(text) -> str:

      
        if text is None: return ""
        text = str(text)
  
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

        text = text.strip().replace("```", "")
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            last_line = lines[-1]
  
            for prefix in ["Final Answer:", "Answer:", "The answer is"]:
             
                parts = re.split(f"{prefix}", last_line, flags=re.IGNORECASE)
                if len(parts) > 1:
                    last_line = parts[-1].strip()
            return last_line
            
        return ""

    @staticmethod
    def normalize_text(text) -> str:
       
        if text is None: return ""
        text = str(text) 

        commands_to_remove = [
            r"\left", r"\right", r"\text", r"\mathrm", r"\mathbf", r"\mbox", 
            r"\,", r"\:", r"\;", r"\!", r"\ "
        ]
        for cmd in commands_to_remove:
            text = text.replace(cmd, "")
            
   
        text = "".join(text.split())
 
        text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
        text = text.replace(r"\div", "/").replace(r"\cdot", "*").replace(r"\times", "*")
        
    
        text = text.replace("$", "")
        

        if text.endswith("."):
            text = text[:-1]
            
        return text

    @classmethod
    def check_correctness(cls, hypothesis, ground_truth) -> bool:
        
        if hypothesis is None: hypothesis = ""
        if ground_truth is None: ground_truth = ""
        
  
        pred_extracted = cls.extract_answer(hypothesis)
        
   
        gt_extracted = cls.extract_answer(ground_truth)
        
  
        if not gt_extracted and str(ground_truth).strip():
            gt_extracted = str(ground_truth).strip()
            
        return cls.math_equal(pred_extracted, gt_extracted)

    @staticmethod
    def math_equal(prediction, reference) -> bool:
      
        pred_norm = MathGrader.normalize_text(prediction)
        ref_norm = MathGrader.normalize_text(reference)

        if not pred_norm or not ref_norm:
            return False

        if pred_norm == ref_norm:
            return True


        if HAS_SYMPY:
            try:
                if MathGrader.symbolic_equal(prediction, reference):
                    return True
            except Exception:
                pass 

        try:
    
            p_val = float(regex.sub(",", "", str(prediction)))
            r_val = float(regex.sub(",", "", str(reference)))
       
            if math.isclose(p_val, r_val, abs_tol=1e-4):
                return True
        except:
            pass

        return False

    @staticmethod
    def symbolic_equal(a, b):
    
        def _parse(s):
            s = str(s).replace("$", "")
       
            try:
                return parse_latex(s)
            except:
                pass
       
            try:
                return parse_expr(s)
            except:
                return None

        expr_a = _parse(a)
        expr_b = _parse(b)

        if expr_a is None or expr_b is None:
            return False

     
        try:
            if simplify(expr_a - expr_b) == 0:
                return True
        except:
            pass

  
        try:
            val_a = N(expr_a)
            val_b = N(expr_b)
       
            if math.isclose(float(val_a), float(val_b), abs_tol=1e-4):
                return True
        except:
            pass
            
        return False
    
    
class SvampGrader:
 
    @staticmethod
    def extract_answer(text) -> str:
        if text is None: return ""
        text = str(text)
        clean_text = text.replace(',', '')
    
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