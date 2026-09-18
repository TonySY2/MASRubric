import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import time
import traceback
import json
import re
from typing import List, Tuple, Dict, Any, Union
import random
import builtins
import contextvars
import io
from tqdm.asyncio import tqdm_asyncio 
import numpy as np 
import asyncio
from openai import DefaultAsyncHttpxClient, Timeout

# --- AutoGen & AgentDropout 组件 ---
from autogen_agentchat.teams import SelectorGroupChat
from AgentDropout.usage_tracking import TrackedOpenAIChatCompletionClient
from AgentDropout.agents.agent_mode_helpers import (
    attach_agent_mode_state,
    configure_agent_mode_team,
    get_agent_mode_history,
    normalize_agent_mode,
    reset_agent_mode_state,
)
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from AgentDropout.agents import AgentRegistry
from AgentDropout.agents.supervisor_reasoning_pick_metric import Supervisor
from AgentDropout.agents.final_decision import FinalRefer

import random  # <--- 新增这行


DEFAULT_SELECTOR_URL = os.environ.get("AGENTDROPOUT_V2_SELECTOR_URL", "http://localhost:8001/v1")
DEFAULT_SELECTOR_API_KEY = os.environ.get(
    "AGENTDROPOUT_V2_SELECTOR_API_KEY",
    "EMPTY",
)

# ==========================================
# [核心升级] 强健的 AMC 判题器
# ==========================================
class LocalAMCGrader:
    @staticmethod
    def normalize_value(val: Union[str, float, int]) -> float:
        """
        将各种格式的答案（字符串、分数、LaTeX）统一转为 float
        """
        if isinstance(val, (float, int)):
            return float(val)
        
        text = str(val).strip().lower()
        
        # 1. 移除 LaTeX 干扰符
        text = text.replace('$', '').replace('\\', '').replace('degree', '').replace('deg', '')
        text = text.replace('boxed', '').replace('{', '').replace('}', '') # 暴力拆包
        text = text.replace('%', '')
        
        # 2. 移除常见的单位词
        units = ['miles', 'cm', 'm', 'km', 'units', 'ways', 'sq', 'square']
        for u in units:
            text = text.replace(u, '')
            
        # 3. 处理分数 (3/4 或 frac34 - 因为上面去了斜杠和括号)
        # 标准写法 3/4
        if '/' in text:
            try:
                parts = text.split('/')
                return float(parts[0].strip()) / float(parts[1].strip())
            except: pass
            
        # 处理 frac num den (虽然上面暴力去除了，但保险起见处理残留)
        # 如果是 "frac 1 2" 这种很难处理，通常 \boxed{\frac{1}{2}} 会被上面的 replace 变成 "frac12"，这很难搞
        # 所以我们需要更精细的提取
        
        try:
            # 尝试直接转换
            return float(text.replace(',', '')) # 去掉千位符
        except:
            return None

    @staticmethod
    def extract_answer(text: str) -> str:
        """
        多策略提取答案
        """
        if not text: return ""
        
        # --- 策略 A: 提取 \boxed{...} (最准) ---
        # 能够处理嵌套括号的简化版 (假设答案里没有复杂嵌套)
        boxed_pattern = r"\\boxed\s*\{([^}]+)\}"
        boxes = re.findall(boxed_pattern, text)
        if boxes:
            return boxes[-1].strip() # 取最后一个盒子
            
        # --- 策略 B: 提取 "The answer is ..." ---
        phrase_pattern = r"(?i)the\s+answer\s+is[:\s]*([^\n\.]+(?:\.[0-9]+)?)"
        match = re.search(phrase_pattern, text)
        if match:
            return match.group(1).strip()
            
        # --- 策略 C: (保底) 提取最后出现的数字 ---
        # 这一步比较激进，仅当上面都失败时尝试
        # 匹配 整数、小数、分数 (3/4)
        last_num_pattern = r"(-?\d+(?:\.\d+)?(?:/\d+)?)"
        nums = re.findall(last_num_pattern, text)
        if nums:
            return nums[-1]
            
        return ""

    @staticmethod
    def check_correctness(prediction: str, ground_truth: Union[str, float, int]) -> bool:
        """
        基于数值的鲁棒判题
        """
        if not prediction: return False
        
        # 1. 解析 Ground Truth
        gt_val = LocalAMCGrader.normalize_value(ground_truth)
        if gt_val is None:
            # 如果 GT 都解析不了，说明不是数字题，回退到字符串对比
            return str(prediction).strip() == str(ground_truth).strip()
            
        # 2. 解析 Prediction
        # 需要先处理 LaTeX 分数 \frac{a}{b} -> a/b
        # 因为 normalize_value 是暴力去除非数字字符，对 frac 处理不好，所以先用正则转一下
        pred_proc = re.sub(r"\\frac\s*\{(\d+)\}\s*\{(\d+)\}", r"\1/\2", prediction)
        
        pred_val = LocalAMCGrader.normalize_value(pred_proc)
        
        if pred_val is None:
            return False
            
        # 3. 比较 (允许 1e-4 误差)
        return abs(pred_val - gt_val) < 1e-4

# ==========================================
# [日志系统] 内存日志缓冲与锁
# ==========================================
_current_log_buffer = contextvars.ContextVar('current_log_buffer', default=None)
_global_log_store = {}
_original_print = builtins.print
_file_write_lock = asyncio.Lock()

def scoped_print(*args, **kwargs):
    buffer = _current_log_buffer.get()
    if buffer:
        kwargs['file'] = buffer
        _original_print(*args, **kwargs)
    else:
        _original_print(*args, **kwargs)

builtins.print = scoped_print

# ==========================================
# [资源加载]
# ==========================================
def load_global_resources(metric_file, cache_file):
    print(f"Loading Global Resources...")
    print(f" - Metrics: {metric_file}")
    print(f" - Cache:   {cache_file}")
    
    metrics = []
    if os.path.exists(metric_file):
        with open(metric_file, "r", encoding='utf-8') as f:
            metrics = json.load(f)
    
    emb_map = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        emb_map[record["name"]] = record["vector"]
                    except: pass
                
    vectors_list = []
    for m in metrics:
        name = m['name']
        if name in emb_map:
            vectors_list.append(np.array(emb_map[name], dtype=np.float32))
        else:
            if len(emb_map) > 0:
                vectors_list.append(np.zeros(len(next(iter(emb_map.values()))), dtype=np.float32))
            else:
                vectors_list.append(np.array([]))
            
    embeddings = np.stack(vectors_list) if vectors_list else np.array([])
    print(f"Global Resources Loaded. Embedding Shape: {embeddings.shape}")
    
    return metrics, embeddings

# ==========================================
# [初始化 Team]
# ==========================================
def init_team(preloaded_metrics, preloaded_embeddings) -> Tuple[SelectorGroupChat, FinalRefer, Dict[str, str], Supervisor]:
    
    use_llm = not args.force_direct_search
    
    supervisor = Supervisor(
        model=args.supervisor_model,
        api_key=os.environ.get("AGENTDROPOUT_SUPERVISOR_API_KEY", "EMPTY"), 
        base_url=args.supervisor_url,
        domain="amc23",
        metrics_retrieve_k=args.metrics_retrieve_k,
        pass_rate=args.pass_rate,
        prune_flag=True, 
        metric_pool_file=args.metric_pool_file, 
        embedding_cache_file=args.embedding_cache_file, 
        embedding_api_key=os.environ.get("AGENTDROPOUT_EMBEDDING_API_KEY", "EMPTY"),
        embedding_model=args.embedding_model,
        embedding_api_base=args.embedding_url,
        preloaded_metrics=preloaded_metrics,
        preloaded_embeddings=preloaded_embeddings,
        use_llm_rerank=use_llm, 
        max_metrics_count=args.max_metrics_count,
        lock_metrics_after_first_round=args.lock_metrics_after_first_round,
        use_simple_audit=args.use_simple_audit,
        force_direct_search=args.force_direct_search,
        retrieve_p=args.retrieve_p,
        select_q=args.select_q,
        direct_k=args.direct_k,
        
        # [新增] 传递随机参数
        random_k=args.random_k,
        
    )

    agent_resgistry = AgentRegistry()
    participants = [
        agent_resgistry.get(
            agent_name="MathSolver_amc23",
            name=f"Participant_{i + 1}",
            domain="amc23",
            model=args.reasoning_model,
            api_key=os.environ.get("AGENTDROPOUT_REASONING_API_KEY", "EMPTY"),
            base_url=args.reasoning_url,
            supervisor=supervisor,
            reflection_time=args.retries_times,
        ) for i in range(5)
    ]

    shared_score_board, predictor = configure_agent_mode_team(
        participants,
        agent_mode=args.agent_mode,
        prm_url=args.prm_url,
        prm_n_samples=args.prm_n_samples,
        model=args.reasoning_model,
        api_key=os.environ.get("AGENTDROPOUT_REASONING_API_KEY", "EMPTY"),
        base_url=args.reasoning_url,
    )

    role_map = {agent.name: agent.role for agent in participants}
    for agent in participants:
        agent.role_map = role_map
        
    selector_prompt = """Select an agent to perform task.
    {roles}
    Current conversation context:
    {history}
    Read the above conversation, then select an agent from {participants} to perform the next task.
    Make sure the planner agent has assigned tasks before other agents start working.
    Only select one agent.
    """

    model_client = TrackedOpenAIChatCompletionClient(
        model=args.selector_model,
        api_key=args.selector_api_key,
        base_url=args.selector_url,
        http_client=DefaultAsyncHttpxClient(
            trust_env=False,
            timeout=Timeout(120.0, connect=10.0),
        ),
        timeout=Timeout(120.0, connect=10.0),
        max_retries=5,
        usage_stage="selector",
        usage_source="SelectorGroupChat",
    )
    
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=args.max_turns)
    termination = text_mention_termination | max_messages_termination

    team = SelectorGroupChat(
        participants=participants,
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=True,  
    )
    attach_agent_mode_state(team, shared_score_board, predictor)

    decision_maker = AgentRegistry.get(
        agent_name="FinalRefer",
        name="DecisionMaker",
        domain="amc23",
        model=args.reasoning_model,
        api_key=os.environ.get("AGENTDROPOUT_REASONING_API_KEY", "EMPTY"),
        base_url=args.reasoning_url
    )
    
    return team, decision_maker, role_map, supervisor

# ==========================================
# [推理流程]
# ==========================================
async def reasoning(question, team: SelectorGroupChat, decision_maker: FinalRefer, role_map: Dict[str, str], supervisor: Supervisor):
    
    is_baseline_mode = getattr(args, 'baseline_only', False)

    agent_mode = normalize_agent_mode(getattr(args, 'agent_mode', 'supervisor'))

    if agent_mode != 'supervisor':
        print(f"\n>>> [Mode] Agent comparison mode: {agent_mode}")
        await team.reset()
        reset_agent_mode_state(team)
        supervisor.reset()
        supervisor.prune_flag = False
        await Console(team.run_stream(task=question))
        history_messages = get_agent_mode_history(team)
        print(f"[Agent Mode Result] collected messages: {len(history_messages)}")

    elif is_baseline_mode:
        print(f"\n>>> [Mode] Baseline Only (No Audit / No Pruning)")
        await team.reset()
        supervisor.reset()
        supervisor.prune_flag = False 
        await Console(team.run_stream(task=question))
        history_messages = supervisor.get_messages_above_threshold()

    else:
        print(f"\n>>> [Phase 1] 启动 Adversarial Audit 模式 (Task: {question[:30]}...)")
        
        await team.reset()
        supervisor.reset()
        supervisor.prune_flag = True 
        await Console(team.run_stream(task=question))
        
        history_messages = supervisor.get_messages_above_threshold()
        retained_count = len(history_messages)
        print(f"\n[Check] 本轮保留消息数: {retained_count}")
        
        # [保底机制]
        if retained_count <= 1: 
            print(f"\n⚠️ 触发保底机制 (Fallback Triggered)！")
            print(">>> [Phase 2] 放弃本轮操作，重新执行 Vanilla AutoGen...")
            
            await team.reset()
            supervisor.reset()
            original_prune_flag = supervisor.prune_flag
            supervisor.prune_flag = False 
            
            await Console(team.run_stream(task=question))
            
            history_messages = supervisor.get_messages_above_threshold()
            print(f"[Fallback Result] 保底运行结束。保留消息数: {len(history_messages)}")
            supervisor.prune_flag = original_prune_flag
    
    print("\n" + "="*50)
    print("--- [DEBUG] Final Decision 阶段开始 ---")
    
    if not history_messages:
        print("  >> 警告: 历史消息为空!")
    else:
        for i, msg in enumerate(history_messages):
            if msg.source != 'user':
                print(f"  - 消息 {i+1} | 来自: {msg.source}")

    raw_answer = await decision_maker.run_decision(history_messages=history_messages, role_map=role_map, task=question)
    raw_content = raw_answer.content.strip()
    
    print("\n[DEBUG] 2. Final Decision 的原始输出:")
    print(raw_content[:200] + "...") 

    print("="*50 + "\n")
    
    ret_scores = supervisor.get_scores(role_map)
    reflection_records = getattr(supervisor, 'reflection_records', [])
    
    return raw_content, ret_scores, reflection_records

# ==========================================
# [文件写入]
# ==========================================
async def write_to_file(out_file, data_id, data):
    try:
        await asyncio.wait_for(_file_write_lock.acquire(), timeout=10.0)
    except asyncio.TimeoutError:
        print(f"[WARNING] 🔒 Write Lock Timeout for Task {data_id}. Skipping save.")
        return

    try:
        exist_data = {}
        if os.path.exists(out_file):
            try:
                with open(out_file, 'r', encoding='utf-8') as f:
                    exist_data = json.load(f)
            except json.JSONDecodeError:
                pass
        
        exist_data[str(data_id)] = data
        try:
            sorted_keys = sorted(exist_data.keys(), key=lambda x: int(x))
        except:
            sorted_keys = sorted(exist_data.keys())
            
        sorted_data = {key: exist_data[key] for key in sorted_keys}
        
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] Failed to write file: {e}")
    finally:
        _file_write_lock.release()
    

async def run_sample(data, out_file, team, decision_maker, role_map, supervisor):
    # [核心适配] 数据字段映射
    question = data.get('question', data.get('problem', ''))
    instance_id = str(data.get('id', 'unknown')) 
    ground_truth = data.get('answer', data.get('solution', ''))
    
    if not args.disable_log_buffer:
        log_capture = io.StringIO()
        token = _current_log_buffer.set(log_capture)
    
    try:
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"--- [ {current_time} ] ---")
        print(f"开始处理: {instance_id}")

        raw_content, scores, reflection_records = await reasoning(question, team, decision_maker, role_map, supervisor)
        
        # [核心] 使用 LocalAMCGrader
        hypothesis = LocalAMCGrader.extract_answer(raw_content)
        is_correct = LocalAMCGrader.check_correctness(hypothesis, ground_truth)
        
        print(f"完成处理: {instance_id} | Correct: {is_correct} (GT: {ground_truth} vs Pred: {hypothesis})")
        
        await write_to_file(
            out_file, 
            instance_id, 
            {
                'id': instance_id, 
                'answer': ground_truth, 
                'hypothesis': hypothesis, 
                'question': question,
                'raw_response': raw_content, 
                'is_correct': is_correct,   
                'scores': scores,            
                'reflection_records': reflection_records 
            }
        )
    except Exception as e:
        _original_print(f"!!!!!! [CRITICAL ERROR] Task {instance_id}: {e} !!!!!!")
        _original_print(traceback.format_exc())
        traceback.print_exc()
        
    finally:
        if not args.disable_log_buffer:
            _global_log_store[instance_id] = log_capture.getvalue()
            log_capture.close()
            _current_log_buffer.reset(token)


async def main():
    if not os.path.exists(args.in_file):
        print(f"[ERROR] Input file not found: {args.in_file}")
        return

    # [核心适配] 数据加载逻辑
    print(f"[INFO] Loading data from {args.in_file}...")
    try:
        with open(args.in_file, 'r', encoding='utf-8') as f:
            try:
                raw_data = json.load(f)
                is_jsonl = False
            except json.JSONDecodeError:
                f.seek(0)
                raw_data = [json.loads(line) for line in f if line.strip()]
                is_jsonl = True
        
        input_data = []
        if is_jsonl:
            input_data = raw_data
            print(f"[INFO] Detected JSONL format. Loaded {len(input_data)} items.")
        else:
            if isinstance(raw_data, dict) and "test" in raw_data:
                input_data = raw_data["test"]
                print(f"[INFO] Detected AMC23 'test' key. Loaded {len(input_data)} items.")
            elif isinstance(raw_data, list):
                input_data = raw_data
                print(f"[INFO] Detected standard List. Loaded {len(input_data)} items.")
            else:
                print("[ERROR] Unknown JSON structure.")
                return

        if args.limit is not None and args.limit > 0:
            input_data = input_data[:args.limit]
            print(f"[INFO] Limit applied: {len(input_data)} tasks.")
            
    except Exception as e:
        print(f"[ERROR] Data load failed: {e}")
        return
        
    existing_ids: set[str] = set()
    if os.path.exists(args.out_file):
        try:
            with open(args.out_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            if isinstance(existing_data, dict):
                existing_ids = {str(k) for k in existing_data.keys()}
            elif isinstance(existing_data, list):
                for row in existing_data:
                    if isinstance(row, dict) and "id" in row:
                        existing_ids.add(str(row["id"]))
            if existing_ids:
                before_count = len(input_data)
                input_data = [
                    item for item in input_data
                    if str(item.get('id', 'unknown')) not in existing_ids
                ]
                print(
                    f"[INFO] Resume mode: detected {len(existing_ids)} existing results, "
                    f"remaining {len(input_data)}/{before_count} tasks."
                )
        except Exception as e:
            print(f"[WARNING] Failed to load existing results for resume: {e}")

    global_metrics, global_embeddings = load_global_resources(
        args.metric_pool_file, 
        args.embedding_cache_file
    )
    
    CONCURRENCY_LIMIT = args.concurrency_limit
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    if args.log_file:
        FINAL_LOG_FILE = args.log_file
    else:
        base_dir = os.path.dirname(args.out_file)
        base_name = os.path.basename(args.out_file).replace(".json", "_full.log")
        FINAL_LOG_FILE = os.path.join(base_dir, base_name)
    
    os.makedirs(os.path.dirname(FINAL_LOG_FILE), exist_ok=True)

    async def worker(instance):
        async with semaphore:
            team, decision_maker, role_map, supervisor = init_team(
                global_metrics, 
                global_embeddings
            )
            await run_sample(instance, args.out_file, team, decision_maker, role_map, supervisor)

    if not input_data:
        print("[INFO] No remaining AMC23 tasks. Resume target already complete.")
        return

    print(f"🚀 开始处理 {len(input_data)} 个任务 (并发: {CONCURRENCY_LIMIT})")
    
    start_time = time.time()
    await tqdm_asyncio.gather(*[worker(inst) for inst in input_data], desc="AMC23 Tasks")
    total_time = time.time() - start_time
    
    print(f"\n🎉 任务完成! 总耗时: {total_time:.2f}s")
    
    if not args.disable_log_buffer:
        print(f"💾 Writing logs to {FINAL_LOG_FILE}...")
        def sort_key(k):
            try: return int(k)
            except: return str(k)
        
        sorted_ids = sorted(_global_log_store.keys(), key=sort_key)
        with open(FINAL_LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== AMC23 Run Logs ===\n")
            f.write(f"Total: {len(input_data)}\n")
            for task_id in sorted_ids:
                f.write(f"\n{'='*40}\n=== TASK ID: {task_id} ===\n{'='*40}\n")
                f.write(_global_log_store[task_id])
                f.write("\n")
        print("✅ Logs saved.")
    else:
        print("[INFO] Log buffer disabled.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--reasoning_url', type=str)
    parser.add_argument('--reasoning_model', type=str)
    parser.add_argument('--supervisor_url', type=str) 
    parser.add_argument('--supervisor_model', type=str)
    parser.add_argument('--selector_model', type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument('--selector_url', type=str, default=DEFAULT_SELECTOR_URL)
    parser.add_argument('--selector_api_key', type=str, default=DEFAULT_SELECTOR_API_KEY)
    
    parser.add_argument("--embedding_url", type=str, required=True)
    parser.add_argument("--embedding_model", type=str, required=True)
    parser.add_argument("--metric_pool_file", type=str, required=True)
    parser.add_argument("--embedding_cache_file", type=str, required=True)
    
    parser.add_argument("--metrics_retrieve_k", type=int, default=20)
    parser.add_argument("--pass_rate", type=float, default=0.8)
    parser.add_argument('--max_turns', type=int, default=5)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--log_file', type=str)
    
    parser.add_argument("--baseline_only", action="store_true")
    parser.add_argument("--max_metrics_count", type=int, default=5)
    parser.add_argument("--lock_metrics_after_first_round", action="store_true")
    # [修改] 简单审计模式：0=关闭, 1=V1(通用), 2=V2(优化版)
    parser.add_argument("--use_simple_audit", type=int, default=0, help="0: Disable, 1: Simple V1, 2: Simple V2")
    parser.add_argument("--disable_log_buffer", action="store_true")
    parser.add_argument("--concurrency_limit", type=int, default=100)
    
    parser.add_argument("--force_direct_search", action="store_true")
    parser.add_argument("--retrieve_p", type=int, default=20)
    parser.add_argument("--select_q", type=int, default=5)
    parser.add_argument("--direct_k", type=int, default=5)
    
    # [新增] 随机对照参数
    parser.add_argument("--random_k", type=int, default=0, help="Randomly select k metrics (Random Mode)")
    parser.add_argument('--retries_times', type=int, default=3)
    parser.add_argument('--agent_mode', default='supervisor', choices=['supervisor', 'prm', 'self_refine', 'self-refine', 'multi_tag', 'multi-tag'])
    parser.add_argument('--prm_url', type=str, default=None)
    parser.add_argument('--prm_n_samples', type=int, default=3)

    args = parser.parse_args()
    
    if not args.disable_log_buffer:
        builtins.print = scoped_print
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
    asyncio.run(main())
