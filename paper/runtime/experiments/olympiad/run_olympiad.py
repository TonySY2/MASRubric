import sys
import os
# 确保能找到项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import time
import traceback
import json
import re
from AgentDropout.agents import AgentRegistry
from AgentDropout.agents.supervisor_reasoning_pick_metric import Supervisor
from AgentDropout.agents.final_decision import FinalRefer
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
import asyncio
from typing import List, Tuple, Dict
import random
import builtins
import contextvars
import io
from tqdm.asyncio import tqdm_asyncio 
from openai import Timeout
import numpy as np 

# [核心依赖] 导入判题器
# 确保 experiments/olympiad/ 目录下有 grader.py，或者 PYTHONPATH 能找到它
# 为了稳健，尝试多路径导入
try:
    from grader import math_equal
except ImportError:
    # 尝试添加当前脚本所在目录
    sys.path.append(os.path.dirname(__file__))
    try:
        from grader import math_equal
    except ImportError:
        # 尝试使用 AgentDropout 内置的 grader (如果有的话)
        try:
             from AgentDropout.agents.math_grader import MathGrader
        except:
             print("[FATAL] 找不到 math_equal 或 grader.py。")
             sys.exit(1)

# ==============================================================================
# [日志隔离与缓冲机制]
# ==============================================================================
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

# ==============================================================================
# [辅助函数]
# ==============================================================================
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

# [对齐 Single Agent] 简单的答案提取逻辑
def extract_boxed_answer(text: str) -> str:
    """
    提取 \boxed{...} 内容。处理简单的嵌套花括号。
    与 Single Agent 版完全对齐。
    """
    if not text: return ""
    idx = text.rfind("\\boxed{")
    if idx == -1: return ""
    
    content = ""
    balance = 0
    
    # 从 { 开始遍历
    for i in range(idx + 7, len(text)):
        char = text[i]
        if char == '{':
            balance += 1
            content += char
        elif char == '}':
            if balance == 0:
                return content.strip()
            balance -= 1
            content += char
        else:
            content += char
    return ""

# ==============================================================================
# [初始化 Team]
# ==============================================================================
def init_team(preloaded_metrics, preloaded_embeddings) -> Tuple[SelectorGroupChat, FinalRefer, Dict[str, str], Supervisor]:
    
    use_llm = not args.force_direct_search
    
    supervisor = Supervisor(
        model=args.supervisor_model,
        api_key=os.environ.get("AGENTDROPOUT_SUPERVISOR_API_KEY", "EMPTY"), 
        base_url=args.supervisor_url,
        domain="olympiad",
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
        direct_k=args.direct_k,
        retrieve_p=args.retrieve_p,
        select_q=args.select_q,
        random_k=args.random_k,
    )

    agent_resgistry = AgentRegistry()
    participants = [
        agent_resgistry.get(
            agent_name="MathSolver_olympiad", 
            name=f"Participant_{i + 1}",
            domain="olympiad",                
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
        if not hasattr(agent, 'description'):
             agent.description = f"An AI agent with the role of {agent.role}."
        
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
        http_client_args={"timeout": Timeout(120.0, connect=10.0)},
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
        domain="olympiad", 
        model=args.reasoning_model,
        api_key=os.environ.get("AGENTDROPOUT_REASONING_API_KEY", "EMPTY"),
        base_url=args.reasoning_url
    )
    
    return team, decision_maker, role_map, supervisor

# ==============================================================================
# [推理流程]
# ==============================================================================
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

# ==============================================================================
# [文件写入]
# ==============================================================================
async def write_to_file(out_file, data_id, data):
    async with _file_write_lock:
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
    

async def run_sample(data, out_file, team, decision_maker, role_map, supervisor):
    question = data.get('question', data.get('problem', ''))
    instance_id = str(data.get('id', 'unknown')) 
    
    # [对齐 Single Agent] Ground Truth 处理
    # OlympiadBench final_answer 是 list
    ground_truth_raw = data.get('final_answer', data.get('solution', ''))
    
    ground_truth = ""
    if isinstance(ground_truth_raw, list) and len(ground_truth_raw) > 0:
        ground_truth = str(ground_truth_raw[0]) 
    elif isinstance(ground_truth_raw, str):
        ground_truth = ground_truth_raw
    else:
        ground_truth = str(ground_truth_raw)
        
    # [对齐] 清洗掉 GT 中的 $ 符号
    ground_truth_clean = ground_truth.replace('$', '').strip()
    
    if not args.disable_log_buffer:
        log_capture = io.StringIO()
        token = _current_log_buffer.set(log_capture)
    
    try:
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"--- [ {current_time} ] ---")
        print(f"开始处理: {instance_id}")

        raw_content, scores, reflection_records = await reasoning(question, team, decision_maker, role_map, supervisor)
        
        # [对齐 Single Agent] 提取与判题
        hypothesis = extract_boxed_answer(raw_content)
        
        # [对齐] 调用 grader.py (timeout=True)
        # 兼容处理：如果本地找不到 math_equal，fallback 到简单的字符串比较
        if 'math_equal' in globals():
            is_correct = math_equal(hypothesis, ground_truth_clean, timeout=True)
        else:
            # 这里其实应该报错，但为了不崩，写个兼容
            print("[WARN] grader.math_equal not found, using MathGrader fallback.")
            try:
                 from AgentDropout.agents.math_grader import MathGrader
                 is_correct = MathGrader.check_correctness(hypothesis, ground_truth_clean)
            except:
                 is_correct = (hypothesis.strip() == ground_truth_clean.strip())

        print(f"完成处理: {instance_id} | Correct: {is_correct} (GT: {ground_truth_clean} vs Pred: {hypothesis})")
        
        await write_to_file(
            out_file, 
            instance_id, 
            {
                'id': instance_id, 
                'answer': ground_truth,      # 原始 GT (列表或字符串)
                'ground_truth_clean': ground_truth_clean, # 清洗后用于判题的 GT
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

    # 数据加载
    print(f"[INFO] Loading data from {args.in_file}...")
    try:
        input_data = []
        with open(args.in_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        input_data.append(json.loads(line))
                    except: pass
        
        if not input_data:
            with open(args.in_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)

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

    print(f"🚀 开始处理 {len(input_data)} 个任务 (并发: {CONCURRENCY_LIMIT})")
    
    start_time = time.time()
    await tqdm_asyncio.gather(*[worker(inst) for inst in input_data], desc="Olympiad Tasks")
    total_time = time.time() - start_time
    
    print(f"\n🎉 任务完成! 总耗时: {total_time:.2f}s")
    
    if not args.disable_log_buffer:
        print(f"💾 Writing logs to {FINAL_LOG_FILE}...")
        def sort_key(k):
            try: return int(k)
            except: return str(k)
        
        sorted_ids = sorted(_global_log_store.keys(), key=sort_key)
        with open(FINAL_LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== Olympiad Run Logs ===\n")
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
    parser.add_argument('--selector_url', type=str, default=None)
    parser.add_argument('--selector_api_key', type=str, default="EMPTY")
    
    parser.add_argument("--embedding_url", type=str, required=True)
    parser.add_argument("--embedding_model", type=str, required=True)
    parser.add_argument("--metric_pool_file", type=str, required=True)
    parser.add_argument("--embedding_cache_file", type=str, required=True)
    
    parser.add_argument("--metrics_retrieve_k", type=int, default=20)
    parser.add_argument("--pass_rate", type=float, default=0.8)
    parser.add_argument('--max_turns', type=int, default=10) 
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
    
    parser.add_argument("--random_k", type=int, default=0)
    parser.add_argument('--retries_times', type=int, default=3)
    parser.add_argument('--agent_mode', default='supervisor', choices=['supervisor', 'prm', 'self_refine', 'self-refine', 'multi_tag', 'multi-tag'])
    parser.add_argument('--prm_url', type=str, default=None)
    parser.add_argument('--prm_n_samples', type=int, default=3)

    args = parser.parse_args()
    if not args.selector_url:
        args.selector_url = args.reasoning_url
    
    if not args.disable_log_buffer:
        builtins.print = scoped_print
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
    asyncio.run(main())
