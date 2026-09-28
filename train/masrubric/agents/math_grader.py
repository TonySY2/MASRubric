import re

class MathGrader:
    

    @staticmethod
    def extract_answer(text: str) -> str:
        
        if not text:
            return ""
        

        boxed = MathGrader._extract_boxed_content(text, "\\boxed{")
        if boxed:
            return boxed

        fbox = MathGrader._extract_boxed_content(text, "\\fbox{")
        if fbox:
            return fbox
            
  
        last_line = text.strip().split('\n')[-1]
        
   
        marker = "The answer is"
        if marker in last_line:
            return last_line.split(marker)[-1].strip()
            

        marker = "Final Answer:"
        if marker in last_line:
            return last_line.split(marker)[-1].strip()

        if len(last_line) < 50:
            return last_line.strip()
            
        return ""

    @staticmethod
    def _extract_boxed_content(text: str, tag: str = "\\boxed{") -> str:
        
        idx = text.rfind(tag)
        if idx == -1:
            return ""
        
        start_idx = idx + len(tag)
        balance = 1
        end_idx = start_idx
        
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                balance += 1
            elif text[i] == '}':
                balance -= 1
                
            if balance == 0:
                end_idx = i
                break
        
 
        if balance == 0:
            return text[start_idx:end_idx]
            
        return ""

    @staticmethod
    def normalize_answer(s: str) -> str:
       
        if not s: 
            return ""
        s = str(s).strip()
        
 
        commands_to_remove = [
            r"\!", r"\,", r"\:", r"\;", r"\ ",  
            r"\left", r"\right",               
            r"\text", r"\mathrm", r"\mathbf", r"\mbox", 
            r"$"                              
        ]
        for cmd in commands_to_remove:
            s = s.replace(cmd, "")
            

        s = s.replace(r"\dfrac", r"\frac")
        s = s.replace(r"\tfrac", r"\frac")
        s = s.replace(r"\div", "/")
        s = s.replace(r"\cdot", "*")
        s = s.replace(r"\times", "*")
        
      
        s = "".join(s.split())
        
    
        if s.endswith("."):
            s = s[:-1]
            
        return s

    @classmethod
    def check_correctness(cls, hypothesis: str, ground_truth: str) -> bool:
        
      
        gt_core = cls._extract_boxed_content(ground_truth)
        if not gt_core:
            gt_core = ground_truth 
    
        hyp_core = cls.extract_answer(hypothesis)
        
   
        norm_gt = cls.normalize_answer(gt_core)
        norm_hyp = cls.normalize_answer(hyp_core)
        

        if not norm_gt:
            return False
            
        return norm_gt == norm_hyp