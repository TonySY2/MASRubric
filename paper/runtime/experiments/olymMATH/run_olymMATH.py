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

# ==============================================================================
# [核心逻辑移植] 判题与提取 (来自 local_tester.py)
# ==============================================================================

# 尝试导入 math_verify，如果没有则在判题时跳过
try:
    from math_verify import parse, verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False
    print("[WARNING] 'math_verify' library not found. Falling back to string comparison.")

def extract_boxed(text):
    """
    [移植] 提取 \boxed{...} 内容，支持嵌套花括号
    """
    if not text: return ""
    
    stack = []
    boxed_contents = []
    i = 0
    start_idx = -1

    while i < len(text):
        if text[i : i + 7] == "\\boxed{" and (i == 0 or text[i - 1] != "\\"):
            if not stack:
                start_idx = i + 7
            stack.append("{")
            i += 7
        elif text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
            stack.append("{")
            i += 1
        elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
            if stack:
                stack.pop()
                if not stack and start_idx != -1:
                    boxed_contents.append(text[start_idx:i])
                    start_idx = -1
            i += 1
        else:
            i += 1

    if boxed_contents:
        return boxed_contents[-1]

    pattern = r"\\boxed{((?:[^{}]|{(?:[^{}]|{[^{}]*})*})*?)}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)

    return ""

def format_for_math_verify(answer):
    """
    [移植] 格式化答案以供 math_verify 使用
    """
    if not answer:
        return "$.$"
    answer = str(answer).strip()
    if answer.startswith("$"):
        answer = answer[1:]
    if answer.endswith("$"):
        answer = answer[:-1]
    answer = answer.strip()
    if not answer:
        return "$.$"
    return f"${answer}$"

def string_compare_answers(extracted, gold):
    """
    [移植] 字符串归一化对比 (回退策略)
    """
    def normalize(text):
        if not text:
            return ""
        text = str(text)
        text = re.sub(r"\s+", "", text)
        text = text.replace("\\frac", "")
        text = text.replace("\\cdot", "*")
        text = text.replace("\\times", "*")
        text = re.sub(r"\\[a-zA-Z]+", "", text)
        return text

    normalized_extracted = normalize(extracted)
    normalized_gold = normalize(gold)

    return (
        normalized_extracted == normalized_gold
        or normalized_gold in normalized_extracted
        or normalized_extracted in normalized_gold
    )

def check_correctness_olym(extracted_answer, gold_answer):
    """
    统一判题入口：优先 math_verify，失败则 string compare
    """
    if not extracted_answer:
        return False
        
    # 1. Try math_verify
    if HAS_MATH_VERIFY:
        try:
            formatted_gold = format_for_math_verify(gold_answer)
            formatted_extracted = format_for_math_verify(extracted_answer)

            gold_parsed = parse(formatted_gold)
            extracted_parsed = parse(formatted_extracted)

            if verify(gold_parsed, extracted_parsed):
                return True
        except Exception:
            pass 

    # 2. Fallback
    try:
        if string_compare_answers(extracted_answer, gold_answer):
            return True
    except Exception as e:
        pass
    
    return False

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

# ==============================================================================
# [初始化 Team]
# ==============================================================================
def init_team(preloaded_metrics, preloaded_embeddings) -> Tuple[SelectorGroupChat, FinalRefer, Dict[str, str], Supervisor]:
    
    use_llm = not args.force_direct_search
    
    supervisor = Supervisor(
        model=args.supervisor_model,
        api_key=os.environ.get("AGENTDROPOUT_SUPERVISOR_API_KEY", "EMPTY"), 
        base_url=args.supervisor_url,
        domain="olymMATH", # [关键] 必须与 Solver 和 PromptSet 注册名一致
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
            agent_name="MathSolver_olymMATH", # [关键] 使用适配后的 Solver
            name=f"Participant_{i + 1}",
            domain="olymMATH",                # [关键] 必须与 PromptSetRegistry 一致
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

    # Selector 模型保持轻量级
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
        domain="olymMATH", 
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
        
        # 保底机制：如果保留消息太少，说明误杀严重，回退到无剪枝模式
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
            # 尝试按数字ID排序
            sorted_keys = sorted(exist_data.keys(), key=lambda x: int(x))
        except:
            sorted_keys = sorted(exist_data.keys())
            
        sorted_data = {key: exist_data[key] for key in sorted_keys}
        
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, indent=4, ensure_ascii=False)
    

async def run_sample(data, out_file, team, decision_maker, role_map, supervisor):
    # 解析 OlymMATH 字段
    question = data.get('question', data.get('problem', ''))
    instance_id = str(data.get('unique_id', data.get('id', 'unknown'))) 
    subject = data.get('subject', '')
    
    # Ground Truth 处理
    ground_truth = str(data.get('answer', data.get('final_answer', '')))
    
    if not args.disable_log_buffer:
        log_capture = io.StringIO()
        token = _current_log_buffer.set(log_capture)
    
    try:
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"--- [ {current_time} ] Processing Task {instance_id} ---")

        raw_content, scores, reflection_records = await reasoning(question, team, decision_maker, role_map, supervisor)
        
        # [提取与判题] 使用移植的逻辑
        hypothesis = extract_boxed(raw_content)
        is_correct = check_correctness_olym(hypothesis, ground_truth)

        print(f"Result: {is_correct} | GT: {ground_truth} | Pred: {hypothesis}")
        
        await write_to_file(
            out_file, 
            instance_id, 
            {
                'id': instance_id, 
                'unique_id': instance_id, # 增加 unique_id 字段
                'subject': subject,       # 增加学科字段
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

    # 数据加载：支持 JSON List 或 JSONL
    print(f"[INFO] Loading data from {args.in_file}...")
    try:
        input_data = []
        with open(args.in_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith("["):
                input_data = json.loads(content)
            else:
                input_data = [json.loads(line) for line in content.splitlines() if line.strip()]

        if args.limit is not None and args.limit > 0:
            input_data = input_data[:args.limit]
            print(f"[INFO] Limit applied: {len(input_data)} tasks.")
            
    except Exception as e:
        print(f"[ERROR] Data load failed: {e}")
        return
        
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
    await tqdm_asyncio.gather(*[worker(inst) for inst in input_data], desc="Olym Tasks")
    total_time = time.time() - start_time
    
    print(f"\n🎉 任务完成! 总耗时: {total_time:.2f}s")
    
    if not args.disable_log_buffer:
        print(f"💾 Writing logs to {FINAL_LOG_FILE}...")
        def sort_key(k):
            try: return int(k)
            except: return str(k)
        
        sorted_ids = sorted(_global_log_store.keys(), key=sort_key)
        with open(FINAL_LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== OlymMATH Run Logs ===\n")
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
